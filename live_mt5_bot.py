"""
XAU/USD Enterprise MT5 V26.2 Alert Bot — Full Backtest Parity
==============================================================
Strategy: Coiled EMA Breakout + Micro Fib Grid (two‑way)
Risk: 2% of virtual equity, $20 cap, 1:1 partial + 2:1 runner
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# ─── LOAD CREDENTIALS ───────────────────────────────────────────────────────
load_dotenv()

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "8851798524:AAGZTUnVeDJARKsH5CsETgBuXgwnTwF4u1g")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-5016015118")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    sys.exit("FATAL: Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID.")

# ─── STRATEGY CONSTANTS (V26.2) ─────────────────────────────────────────────
STARTING_CAP     = 500.0
MARGIN_PER_TRADE = 200.0
LEVERAGE         = 50
TRADE_POWER      = MARGIN_PER_TRADE * LEVERAGE   # $10,000

RISK_PERCENTAGE  = 0.02          # 2% of equity
MAX_RISK_PER_TRADE = 20.0        # dollar cap

PARTIAL_CLOSE_FRACTION = 0.5     # 50% at TP1
FINAL_RISK_REWARD      = 2.0     # remaining to 2:1

HARD_DD_PCT = 0.15
HALT_EQUITY = STARTING_CAP * (1 - HARD_DD_PCT)

ATR_MIN_RATIO = 0.70

# Sessions where Coiled EMA is blocked
COILED_EMA_BLOCKED = {"London (08-13 UTC)", "NY Only (17-22 UTC)"}

MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
POLL_INTERVAL = 1   # seconds

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=8)
        resp.raise_for_status()
        log.info("Telegram alert sent.")
    except requests.RequestException as e:
        log.warning(f"Telegram error: {e}")


def is_gold_market_open() -> bool:
    now = datetime.now(timezone.utc)
    if now.weekday() == 5:
        return False
    if now.weekday() == 6 and now.hour < 22:
        return False
    if now.weekday() == 4 and now.hour >= 21:
        return False
    if now.hour == 22:
        return False
    return True


def get_session(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asian (00-08 UTC)"
    elif 8 <= hour < 13:
        return "London (08-13 UTC)"
    elif 13 <= hour < 17:
        return "NY Overlap (13-17 UTC)"
    elif 17 <= hour < 22:
        return "NY Only (17-22 UTC)"
    else:
        return "Late / Sydney (22-00 UTC)"


class V26AlertBot:

    def __init__(self) -> None:
        log.info("Initializing MT5...")
        initialized = mt5.initialize(path=MT5_PATH) if os.path.exists(MT5_PATH) else mt5.initialize()
        if not initialized:
            sys.exit("MT5 init failed")
        account = mt5.account_info()
        log.info(f"Connected | Account: {account.login}")
        self.symbol = self._discover_gold_symbol()
        info = mt5.symbol_info(self.symbol)
        self.digits = info.digits
        self.contract_size = info.trade_contract_size if info else 100.0

        # Virtual account state
        self.capital = STARTING_CAP
        self.in_position = False
        self.pos_type = ""
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.tp1_hit = False
        self.qty_total = 0.0
        self.qty_remaining = 0.0
        self.trade_realized_pnl = 0.0
        self.risk_amount = 0.0
        self.active_setup = ""   # <-- new: track the setup name

        self.last_processed_15m_ts = None

    def _discover_gold_symbol(self) -> str:
        for name in ["XAUUSD", "GOLD", "XAUUSD.pro", "XAUUSD.", "XAUUSD.m", "XAUUSDm"]:
            info = mt5.symbol_info(name)
            if info:
                mt5.symbol_select(name, True)
                log.info(f"Symbol: {name}")
                return name
        sys.exit("Gold symbol not found")

    def get_frame(self, timeframe: int, count: int = 200) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
        return df

    def _compute_indicators(self, df_15m):
        """Compute V26.2 indicators on 15M data."""
        df = df_15m.copy()
        # EMAs
        df["EMA_25"] = df["Close"].ewm(span=25, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
        # RSI14
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = (100 - (100 / (1 + rs))).fillna(50)
        # ATR ratio
        hl = df["High"] - df["Low"]
        hcp = (df["High"] - df["Close"].shift(1)).abs()
        lcp = (df["Low"]  - df["Close"].shift(1)).abs()
        tr = pd.concat([hl, hcp, lcp], axis=1).max(axis=1)
        df["ATR_14"] = tr.ewm(com=13, adjust=False).mean()
        df["ATR_MA20"] = df["ATR_14"].rolling(20).mean()
        df["ATR_ratio"] = (df["ATR_14"] / df["ATR_MA20"]).fillna(1.0)
        return df

    def _compute_daily_levels(self, df_1d):
        """Calculate Fib levels and prior day bias."""
        df = df_1d.copy()
        df["Prev_High"] = df["High"].shift(1)
        df["Prev_Low"] = df["Low"].shift(1)
        df["Prev_Range"] = df["Prev_High"] - df["Prev_Low"]
        df["Prev_Green"] = df["Close"].shift(1) > df["Open"].shift(1)

        df["Fib_1618_Bull"] = df["Prev_High"] + (1.618 * df["Prev_Range"])
        df["Fib_0382_Bull"] = df["Prev_High"] - (0.382 * df["Prev_Range"])
        df["Fib_0618_Bull"] = df["Prev_High"] - (0.618 * df["Prev_Range"])

        df["Fib_1618_Bear"] = df["Prev_Low"] - (1.618 * df["Prev_Range"])
        df["Fib_0382_Bear"] = df["Prev_Low"] + (0.382 * df["Prev_Range"])
        df["Fib_0618_Bear"] = df["Prev_Low"] + (0.618 * df["Prev_Range"])
        return df

    def _compute_signal(self) -> dict:
        """Evaluate V26.2 entry conditions. Returns dict with keys:
        direction, entry, sl, tp1, tp2, qty, risk, setup_name
        """
        df_1d  = self.get_frame(mt5.TIMEFRAME_D1, 5)
        df_15m = self.get_frame(mt5.TIMEFRAME_M15, 200)
        if df_1d.empty or df_15m.empty:
            return {}

        df_15m = self._compute_indicators(df_15m)
        df_1d  = self._compute_daily_levels(df_1d)

        idx = -2
        row = df_15m.iloc[idx]
        c1 = float(row["Close"])
        c2 = float(df_15m["Close"].iloc[idx-1])
        ema_25 = float(row["EMA_25"])
        ema_50 = float(row["EMA_50"])
        ema_200 = float(row["EMA_200"])
        rsi    = float(row["RSI_14"])
        atr_ratio = float(row["ATR_ratio"])

        if np.isnan(atr_ratio) or atr_ratio < ATR_MIN_RATIO:
            return {}

        session = get_session(df_15m.index[idx].hour)

        ema_bull = ema_25 > ema_50
        ema_bear = ema_25 < ema_50
        ema_coil = abs(ema_25 - ema_50) <= 0.50
        bullish_stack = ema_25 > ema_50 and ema_50 > ema_200
        bearish_stack = ema_25 < ema_50 and ema_50 < ema_200

        yesterday = df_1d.iloc[-2]
        prev_high = float(yesterday["Prev_High"])
        prev_low  = float(yesterday["Prev_Low"])
        is_green  = bool(yesterday["Prev_Green"])
        fib_1618_bull = float(yesterday["Fib_1618_Bull"])
        fib_0382_bull = float(yesterday["Fib_0382_Bull"])
        fib_0618_bull = float(yesterday["Fib_0618_Bull"])
        fib_1618_bear = float(yesterday["Fib_1618_Bear"])
        fib_0382_bear = float(yesterday["Fib_0382_Bear"])
        fib_0618_bear = float(yesterday["Fib_0618_Bear"])

        direction = ""
        setup_name = ""

        # --- Strategy 1: Coiled EMA Breakout ---
        if ema_coil and session not in COILED_EMA_BLOCKED:
            if bullish_stack and (c2 > ema_25) and (c1 > ema_25) and rsi > 50:
                direction = "LONG"
                setup_name = "Coiled_EMA_Breakout"
            elif bearish_stack and (c2 < ema_25) and (c1 < ema_25) and rsi < 50:
                direction = "SHORT"
                setup_name = "Coiled_EMA_Breakout"

        # --- Strategy 2: Micro Fib Grid ---
        if not direction:
            bull_levels = [
                (fib_1618_bull, "1D_Fib_1.618"),
                (prev_high, "1D_Fib_1.0"),
                (fib_0382_bull, "1D_Fib_0.382"),
                (fib_0618_bull, "1D_Fib_0.618")
            ]
            bear_levels = [
                (fib_1618_bear, "1D_Fib_1.618"),
                (prev_low, "1D_Fib_1.0"),
                (fib_0382_bear, "1D_Fib_0.382"),
                (fib_0618_bear, "1D_Fib_0.618")
            ]

            # Continuation
            if is_green:
                for level, name in bull_levels:
                    if c2 > level and c1 > level and ema_bull and rsi > 50:
                        direction = "LONG"
                        setup_name = name
                        break
            else:
                for level, name in bear_levels:
                    if c2 < level and c1 < level and ema_bear and rsi < 50:
                        direction = "SHORT"
                        setup_name = name
                        break

            # Correction
            if not direction:
                if is_green:
                    for level, name in bull_levels:
                        if c2 < level and c1 < level and ema_bear and rsi < 50:
                            direction = "SHORT"
                            setup_name = name + " (Correction)"
                            break
                else:
                    for level, name in bear_levels:
                        if c2 > level and c1 > level and ema_bull and rsi > 50:
                            direction = "LONG"
                            setup_name = name + " (Correction)"
                            break

        if not direction:
            return {}

        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return {}
        entry = tick.ask if direction == "LONG" else tick.bid

        risk = min(MAX_RISK_PER_TRADE, self.capital * RISK_PERCENTAGE)
        qty = TRADE_POWER / entry
        sl_dist = risk / qty
        tp1_dist = sl_dist
        tp2_dist = sl_dist * FINAL_RISK_REWARD

        if direction == "LONG":
            sl  = round(entry - sl_dist, self.digits)
            tp1 = round(entry + tp1_dist, self.digits)
            tp2 = round(entry + tp2_dist, self.digits)
        else:
            sl  = round(entry + sl_dist, self.digits)
            tp1 = round(entry - tp1_dist, self.digits)
            tp2 = round(entry - tp2_dist, self.digits)

        return {
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "qty": qty,
            "risk": risk,
            "setup_name": setup_name
        }

    def _manage_virtual_position(self, bid: float, ask: float) -> None:
        if not self.in_position:
            return

        if self.pos_type == "LONG":
            exit_price = bid
            if not self.tp1_hit and exit_price >= self.tp1_price:
                close_qty = self.qty_total * PARTIAL_CLOSE_FRACTION
                pnl = (self.tp1_price - self.entry_price) * close_qty
                self.trade_realized_pnl += pnl
                self.capital += pnl
                self.qty_remaining -= close_qty
                self.tp1_hit = True
                self.sl_price = self.entry_price
                send_telegram(
                    f"🎯 *PARTIAL TP1 HIT (V26.2)*\n"
                    f"Setup: {self.active_setup}\n"
                    f"Direction: *LONG*\n"
                    f"Hit @ `${self.tp1_price:.2f}`\n"
                    f"Profit secured: `+${pnl:.2f}`\n"
                    f"Remaining {self.qty_remaining:.2f} units, SL → entry `${self.entry_price:.2f}`, TP2 @ `${self.tp2_price:.2f}`"
                )
            elif exit_price >= self.tp2_price:
                pnl = (self.tp2_price - self.entry_price) * self.qty_remaining
                self.trade_realized_pnl += pnl
                self.capital += pnl
                self.qty_remaining = 0
                self.in_position = False
                send_telegram(
                    f"🏆 *RUNNER TP2 HIT (V26.2)*\n"
                    f"Setup: {self.active_setup}\n"
                    f"Direction: *LONG*\n"
                    f"Hit @ `${self.tp2_price:.2f}`\n"
                    f"Runner P&L: `+${pnl:.2f}`\n"
                    f"Total Trade P&L: `+${self.trade_realized_pnl:.2f}`\n"
                    f"Equity: `${self.capital:.2f}`"
                )
            elif exit_price <= self.sl_price:
                pnl = (self.sl_price - self.entry_price) * self.qty_remaining
                self.trade_realized_pnl += pnl
                self.capital += pnl
                self.qty_remaining = 0
                self.in_position = False
                reason = "BE Stop" if self.sl_price == self.entry_price else "Hard SL"
                send_telegram(
                    f"❌ *STOP LOSS HIT (V26.2)*\n"
                    f"Setup: {self.active_setup}\n"
                    f"Direction: *LONG*\n"
                    f"Reason: `{reason}` @ `${self.sl_price:.2f}`\n"
                    f"P&L: `{fmt_usd(pnl, True)}`\n"
                    f"Equity: `${self.capital:.2f}`"
                )

        elif self.pos_type == "SHORT":
            exit_price = ask
            if not self.tp1_hit and exit_price <= self.tp1_price:
                close_qty = self.qty_total * PARTIAL_CLOSE_FRACTION
                pnl = (self.entry_price - self.tp1_price) * close_qty
                self.trade_realized_pnl += pnl
                self.capital += pnl
                self.qty_remaining -= close_qty
                self.tp1_hit = True
                self.sl_price = self.entry_price
                send_telegram(
                    f"🎯 *PARTIAL TP1 HIT (V26.2)*\n"
                    f"Setup: {self.active_setup}\n"
                    f"Direction: *SHORT*\n"
                    f"Hit @ `${self.tp1_price:.2f}`\n"
                    f"Profit secured: `+${pnl:.2f}`\n"
                    f"Remaining {self.qty_remaining:.2f} units, SL → entry `${self.entry_price:.2f}`, TP2 @ `${self.tp2_price:.2f}`"
                )
            elif exit_price <= self.tp2_price:
                pnl = (self.entry_price - self.tp2_price) * self.qty_remaining
                self.trade_realized_pnl += pnl
                self.capital += pnl
                self.qty_remaining = 0
                self.in_position = False
                send_telegram(
                    f"🏆 *RUNNER TP2 HIT (V26.2)*\n"
                    f"Setup: {self.active_setup}\n"
                    f"Direction: *SHORT*\n"
                    f"Hit @ `${self.tp2_price:.2f}`\n"
                    f"Runner P&L: `+${pnl:.2f}`\n"
                    f"Total Trade P&L: `+${self.trade_realized_pnl:.2f}`\n"
                    f"Equity: `${self.capital:.2f}`"
                )
            elif exit_price >= self.sl_price:
                pnl = (self.entry_price - self.sl_price) * self.qty_remaining
                self.trade_realized_pnl += pnl
                self.capital += pnl
                self.qty_remaining = 0
                self.in_position = False
                reason = "BE Stop" if self.sl_price == self.entry_price else "Hard SL"
                send_telegram(
                    f"❌ *STOP LOSS HIT (V26.2)*\n"
                    f"Setup: {self.active_setup}\n"
                    f"Direction: *SHORT*\n"
                    f"Reason: `{reason}` @ `${self.sl_price:.2f}`\n"
                    f"P&L: `{fmt_usd(pnl, True)}`\n"
                    f"Equity: `${self.capital:.2f}`"
                )

    def check_catchup(self) -> None:
        """Scan today's completed 15M bars for missed V26.2 entries."""
        log.info("Running catch-up scan...")
        df_1d  = self.get_frame(mt5.TIMEFRAME_D1, 5)
        df_15m = self.get_frame(mt5.TIMEFRAME_M15, 200)
        if df_1d.empty or df_15m.empty:
            return

        df_15m = self._compute_indicators(df_15m)
        df_1d  = self._compute_daily_levels(df_1d)
        yesterday = df_1d.iloc[-2]
        is_green = bool(yesterday["Prev_Green"])
        prev_high = float(yesterday["Prev_High"])
        prev_low  = float(yesterday["Prev_Low"])
        fib_1618_bull = float(yesterday["Fib_1618_Bull"])
        fib_0382_bull = float(yesterday["Fib_0382_Bull"])
        fib_0618_bull = float(yesterday["Fib_0618_Bull"])
        fib_1618_bear = float(yesterday["Fib_1618_Bear"])
        fib_0382_bear = float(yesterday["Fib_0382_Bear"])
        fib_0618_bear = float(yesterday["Fib_0618_Bear"])

        today_date = df_15m.index[-1].date()
        missed = []

        for i in range(4, len(df_15m)):
            candle_ts = df_15m.index[i-1]
            if candle_ts.date() != today_date:
                continue

            row = df_15m.iloc[i-1]
            c1 = float(row["Close"])
            c2 = float(df_15m["Close"].iloc[i-2])
            ema_25 = float(row["EMA_25"])
            ema_50 = float(row["EMA_50"])
            ema_200 = float(row["EMA_200"])
            rsi = float(row["RSI_14"])
            atr_ratio = float(row["ATR_ratio"])

            if np.isnan(atr_ratio) or atr_ratio < ATR_MIN_RATIO:
                continue

            session = get_session(candle_ts.hour)
            ema_bull = ema_25 > ema_50
            ema_bear = ema_25 < ema_50
            ema_coil = abs(ema_25 - ema_50) <= 0.50
            bullish_stack = ema_25 > ema_50 and ema_50 > ema_200
            bearish_stack = ema_25 < ema_50 and ema_50 < ema_200

            direction = ""
            setup = ""
            if ema_coil and session not in COILED_EMA_BLOCKED:
                if bullish_stack and (c2 > ema_25) and (c1 > ema_25) and rsi > 50:
                    direction = "LONG"
                    setup = "Coiled_EMA_Breakout"
                elif bearish_stack and (c2 < ema_25) and (c1 < ema_25) and rsi < 50:
                    direction = "SHORT"
                    setup = "Coiled_EMA_Breakout"

            if not direction:
                bull_lvls = [
                    (fib_1618_bull, "1D_Fib_1.618"),
                    (prev_high, "1D_Fib_1.0"),
                    (fib_0382_bull, "1D_Fib_0.382"),
                    (fib_0618_bull, "1D_Fib_0.618")
                ]
                bear_lvls = [
                    (fib_1618_bear, "1D_Fib_1.618"),
                    (prev_low, "1D_Fib_1.0"),
                    (fib_0382_bear, "1D_Fib_0.382"),
                    (fib_0618_bear, "1D_Fib_0.618")
                ]
                if is_green:
                    for lvl, name in bull_lvls:
                        if c2 > lvl and c1 > lvl and ema_bull and rsi > 50:
                            direction = "LONG"
                            setup = name
                            break
                else:
                    for lvl, name in bear_lvls:
                        if c2 < lvl and c1 < lvl and ema_bear and rsi < 50:
                            direction = "SHORT"
                            setup = name
                            break
                if not direction:
                    if is_green:
                        for lvl, name in bull_lvls:
                            if c2 < lvl and c1 < lvl and ema_bear and rsi < 50:
                                direction = "SHORT"
                                setup = name + " (Corr)"
                                break
                    else:
                        for lvl, name in bear_lvls:
                            if c2 > lvl and c1 > lvl and ema_bull and rsi > 50:
                                direction = "LONG"
                                setup = name + " (Corr)"
                                break

            if direction:
                entry = c1
                missed.append(f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} at {candle_ts.strftime('%H:%M')} UTC @ `${entry:.2f}` — {setup}")

        if missed:
            send_telegram("📋 *V26.2 Missed Entries Today:*\n" + "\n".join(missed))
        else:
            send_telegram("✅ *Catch‑up:* No V26.2 entries missed today.")

    def run_engine(self) -> None:
        send_telegram(
            "🚀 *V26.2 Enterprise Alert Engine ONLINE*\n"
            f"Strategy: Coiled EMA + Micro Fib Grid (two‑way)\n"
            f"Risk: {RISK_PERCENTAGE*100:.0f}% per trade, cap ${MAX_RISK_PER_TRADE}\n"
            f"Partial: {PARTIAL_CLOSE_FRACTION*100:.0f}% @1:1, runner @2:1"
        )
        while True:
            try:
                if not is_gold_market_open():
                    time.sleep(300)
                    continue

                tick = mt5.symbol_info_tick(self.symbol)
                if not tick:
                    time.sleep(2)
                    continue
                bid, ask = float(tick.bid), float(tick.ask)

                if self.in_position:
                    self._manage_virtual_position(bid, ask)

                if not self.in_position and self.capital >= HALT_EQUITY:
                    df_15m = self.get_frame(mt5.TIMEFRAME_M15, 5)
                    if df_15m.empty:
                        time.sleep(1)
                        continue
                    latest_ts = df_15m.index[-2]
                    if latest_ts == self.last_processed_15m_ts:
                        time.sleep(POLL_INTERVAL)
                        continue
                    self.last_processed_15m_ts = latest_ts

                    signal = self._compute_signal()
                    if signal:
                        self.entry_price = signal["entry"]
                        self.sl_price = signal["sl"]
                        self.tp1_price = signal["tp1"]
                        self.tp2_price = signal["tp2"]
                        self.qty_total = signal["qty"]
                        self.qty_remaining = self.qty_total
                        self.trade_realized_pnl = 0.0
                        self.risk_amount = signal["risk"]
                        self.in_position = True
                        self.pos_type = signal["direction"]
                        self.tp1_hit = False
                        self.active_setup = signal["setup_name"]   # store the reason

                        display_lots = round(self.qty_total / self.contract_size, 2)
                        direction = signal["direction"]
                        send_telegram(
                            f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT'} *V26.2 ENTRY SIGNAL*\n"
                            f"Setup: {self.active_setup}\n"
                            f"Entry: `${self.entry_price:.2f}`\n"
                            f"SL: `${self.sl_price:.2f}` (risk `${self.risk_amount:.2f}`)\n"
                            f"TP1 (50%): `${self.tp1_price:.2f}` | TP2: `${self.tp2_price:.2f}`\n"
                            f"Units: {self.qty_total:.4f} oz | Lots: {display_lots}\n"
                            f"Virtual Equity: `${self.capital:.2f}`"
                        )

            except Exception as e:
                log.exception(f"Loop error: {e}")
                send_telegram(f"⚠️ *Error:* {e}")
            time.sleep(POLL_INTERVAL)


def fmt_usd(value: float, force_sign: bool = False) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


if __name__ == "__main__":
    bot = V26AlertBot()
    bot.check_catchup()
    bot.run_engine()