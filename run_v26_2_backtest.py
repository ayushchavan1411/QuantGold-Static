#!/usr/bin/env python3
"""
XAU/USD V26.2 Hybrid Prop Firm — Standalone Backtest (yfinance)
================================================================
Same strategy as backtest_backend.py, but data from Yahoo Finance GC=F
(no MT5 required). Free 5m history is capped at ~59 calendar days.

Optional: set TWELVEDATA_API_KEY for longer / more accurate XAU/USD bars.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─── STRATEGY CONSTANTS (match backtest_backend.py) ───────────────────────────
BACKTEST_DAYS    = 59          # Yahoo 5m limit
STARTING_CAP     = 25_000.0
DAILY_LOSS_LIMIT = 500.0

MARGIN_PER_TRADE = 200.0
LEVERAGE         = 50
TRADE_POWER      = MARGIN_PER_TRADE * LEVERAGE

FIB_RISK_PERCENTAGE = 0.004
FIB_MAX_RISK        = 500.0
EMA_RISK_PERCENTAGE = 0.0015
EMA_MAX_RISK        = 200.0

PARTIAL_CLOSE_FRACTION = 0.5
RISK_REWARD_TOTAL      = 2.0

COILED_EMA_BLOCKED_SESSIONS = {
    "London (08-13 UTC)",
    "NY Only (17-22 UTC)",
}
EMA_CROSSOVER_BLOCKED_SESSIONS = {
    "NY Only (17-22 UTC)",
    "Late / Sydney (22-00 UTC)",
}
MIN_DAILY_RANGE_GLOBAL = 25.0

ALL_SETUP_KEYS = [
    "Coiled_EMA_Breakout",
    "1D_Fib_1.618", "1D_Fib_1.0", "1D_Fib_0.382", "1D_Fib_0.618",
    "5M_EMA_Crossover",
]
SESSIONS = [
    "Asian (00-08 UTC)",
    "London (08-13 UTC)",
    "NY Overlap (13-17 UTC)",
    "NY Only (17-22 UTC)",
    "Late / Sydney (22-00 UTC)",
]

OUT_DIR = Path(__file__).resolve().parent / "backtest_results"


def fmt_usd(value: float, force_sign: bool = False) -> str:
    if force_sign:
        sign = "+" if value >= 0 else "-"
    else:
        sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def get_session(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asian (00-08 UTC)"
    if 8 <= hour < 13:
        return "London (08-13 UTC)"
    if 13 <= hour < 17:
        return "NY Overlap (13-17 UTC)"
    if 17 <= hour < 22:
        return "NY Only (17-22 UTC)"
    return "Late / Sydney (22-00 UTC)"


def calc_max_drawdown(equity_series: pd.Series) -> float:
    if len(equity_series) < 2:
        return 0.0
    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max) / roll_max * 100
    val = float(dd.min())
    return val if not np.isnan(val) else 0.0


def calc_sharpe(equity_series: pd.Series, risk_free_annual: float = 0.053) -> float:
    if len(equity_series) < 2:
        return 0.0
    daily_ret = equity_series.pct_change().dropna()
    std = daily_ret.std()
    if std == 0 or np.isnan(std):
        return 0.0
    rf_daily = risk_free_annual / 252
    return float((daily_ret.mean() - rf_daily) / std * np.sqrt(252))


def ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def fetch_yfinance() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Fetching GC=F from Yahoo Finance (period={BACKTEST_DAYS}d)...")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=f"{BACKTEST_DAYS}d", interval="5m")
    df_15m = ticker.history(period=f"{BACKTEST_DAYS}d", interval="15m")
    df_1h = ticker.history(period=f"{BACKTEST_DAYS}d", interval="1h")

    if any(d.empty for d in [df, df_15m, df_1h]):
        raise ValueError("yfinance returned empty data for GC=F.")

    for frame in (df, df_15m, df_1h):
        ensure_utc(frame)
        # Keep only OHLCV columns used by the engine
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
        frame.drop(columns=[c for c in frame.columns if c not in keep], inplace=True, errors="ignore")

    return df, df_15m, df_1h


def fetch_twelvedata(api_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optional longer XAU/USD pull via Twelve Data free tier."""
    import urllib.parse
    import urllib.request

    def pull(interval: str, outputsize: int) -> pd.DataFrame:
        params = urllib.parse.urlencode({
            "symbol": "XAU/USD",
            "interval": interval,
            "outputsize": str(outputsize),
            "apikey": api_key,
            "timezone": "UTC",
            "order": "ASC",
        })
        url = f"https://api.twelvedata.com/time_series?{params}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        if "values" not in payload:
            raise ValueError(f"Twelve Data error ({interval}): {payload}")
        rows = payload["values"]
        frame = pd.DataFrame(rows)
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        frame = frame.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close"]:
            frame[col] = frame[col].astype(float)
        frame.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
        }, inplace=True)
        return frame[["Open", "High", "Low", "Close"]]

    print("Fetching XAU/USD from Twelve Data...")
    # Free tier outputsize caps vary; request generous history
    df = pull("5min", 5000)
    df_15m = pull("15min", 5000)
    df_1h = pull("1h", 5000)
    return df, df_15m, df_1h


def print_report(
    actual_days: int,
    capital: float,
    wins: int,
    losses: int,
    gross_win: float,
    gross_loss: float,
    eq_series: pd.Series,
    matrix_stats: dict,
    data_source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    total_trades = wins + losses
    net_profit = capital - STARTING_CAP
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_win = (gross_win / wins) if wins > 0 else 0.0
    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
    expectancy = (gross_win - gross_loss) / total_trades if total_trades > 0 else 0.0
    roi = (net_profit / STARTING_CAP) * 100
    max_dd = calc_max_drawdown(eq_series)
    sharpe = calc_sharpe(eq_series)

    sep = "=" * 80
    dash = "-" * 80

    print(f"\n{sep}")
    print("    V26.2 HYBRID SAFE PROP FIRM — Daily Range ≥ $25, EMA Session Block")
    print(sep)
    print(f"  Data Source     : {data_source}")
    print(f"  Test Period     : {actual_days} calendar days  ({start.date()} → {end.date()})")
    print(f"  Starting Cap    : ${STARTING_CAP:>10,.2f}")
    print(f"  Final Capital   : ${capital:>10,.2f}")
    print(f"  Net Profit      : ${net_profit:>+10,.2f}   ({roi:+.1f}% ROI)")
    print(dash)
    print(f"  Total Trades    : {total_trades}")
    print(f"  Win Rate        : {win_rate:>6.1f}%   ({wins}W / {losses}L)")
    if pf != float("inf"):
        print(f"  Profit Factor   : {pf:>6.2f}")
    else:
        print("  Profit Factor   :      ∞")
    print(f"  Expectancy      : {fmt_usd(expectancy, True):>12}  per trade")
    print(f"  Avg Win         : {fmt_usd(avg_win, True):>12}")
    print(f"  Avg Loss        : {fmt_usd(-avg_loss):>12}")
    print(f"  Max Drawdown    : {max_dd:>6.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:>6.2f}")
    print(f"\n  Fib Risk/Trade  : {FIB_RISK_PERCENTAGE*100:.1f}% of equity, capped at ${FIB_MAX_RISK}")
    print(f"  EMA Risk/Trade  : {EMA_RISK_PERCENTAGE*100:.2f}% of equity, capped at ${EMA_MAX_RISK}")
    print(f"  Daily Loss Lim  : ${DAILY_LOSS_LIMIT:,.2f}")
    print(f"  Global Filter   : Yesterday's range must be ≥ ${MIN_DAILY_RANGE_GLOBAL}")
    print("  EMA Blocked     : NY Only + Late/Sydney")

    print(f"\n{sep}")
    print("    DEEP MATRIX BREAKDOWN: SESSION vs SETUP")
    print(sep)

    setup_rows = []
    for sess, setups in matrix_stats.items():
        sess_w = sum(v["W"] for v in setups.values())
        sess_l = sum(v["L"] for v in setups.values())
        if (sess_w + sess_l) == 0:
            continue
        print(f"\n  SESSION: {sess}")
        print(dash)
        print(f"  {'Setup':<22} {'W':>4} {'L':>4}  {'WR%':>6}  {'PF':>6}")
        for setup_name, s in setups.items():
            total_s = s["W"] + s["L"]
            if total_s == 0:
                continue
            wr_s = (s["W"] / total_s * 100) if total_s > 0 else 0.0
            pf_s = (s["gross_win"] / s["gross_loss"]) if s["gross_loss"] > 0 else float("inf")
            pf_str = f"{pf_s:>6.2f}" if pf_s != float("inf") else "     ∞"
            print(f"  {setup_name:<22} {s['W']:>4} {s['L']:>4}  {wr_s:>5.1f}%  {pf_str}")
            setup_rows.append({
                "session": sess,
                "setup": setup_name,
                "wins": s["W"],
                "losses": s["L"],
                "win_rate_pct": round(wr_s, 1),
                "profit_factor": None if pf_s == float("inf") else round(pf_s, 2),
                "gross_win": round(s["gross_win"], 2),
                "gross_loss": round(s["gross_loss"], 2),
            })
        print(dash)
    print(f"{sep}\n")

    return {
        "strategy": "V26.2 Hybrid Safe Prop Firm",
        "data_source": data_source,
        "period_start": str(start.date()),
        "period_end": str(end.date()),
        "calendar_days": actual_days,
        "starting_capital": STARTING_CAP,
        "final_capital": round(capital, 2),
        "net_profit": round(net_profit, 2),
        "roi_pct": round(roi, 2),
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": None if pf == float("inf") else round(pf, 2),
        "expectancy": round(expectancy, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "matrix": setup_rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_backtest(df: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame, data_source: str) -> dict:
    actual_days = (df.index[-1] - df.index[0]).days
    print(f"Loaded {len(df):,} × 5M bars ({actual_days} calendar days). Building signals...")

    # ── 15M INDICATORS FOR FIB GRID ───────────────────────────────────────────
    df_15m = df_15m.copy()
    df_15m["EMA_25"] = df_15m["Close"].ewm(span=25, adjust=False).mean()
    df_15m["EMA_50"] = df_15m["Close"].ewm(span=50, adjust=False).mean()
    df_15m["EMA_200"] = df_15m["Close"].ewm(span=200, adjust=False).mean()
    df_15m["C1"] = df_15m["Close"].shift(1)
    df_15m["C2"] = df_15m["Close"].shift(2)
    df_15m["Signal_TS"] = df_15m.index

    # ── 5M INDICATORS FOR EMA CROSSOVER ───────────────────────────────────────
    df = df.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["C1_5m"] = df["Close"].shift(1)
    df["C2_5m"] = df["Close"].shift(2)

    # ── DAILY OHLC FOR FIB GRID ───────────────────────────────────────────────
    # Gold session day ~22:00 UTC. Use 24h buckets (calendar 'D' ignores offset).
    df_1d = pd.DataFrame({
        "open": df_1h["Open"].resample("24h", offset="22h").first(),
        "high": df_1h["High"].resample("24h", offset="22h").max(),
        "low": df_1h["Low"].resample("24h", offset="22h").min(),
        "close": df_1h["Close"].resample("24h", offset="22h").last(),
    }).dropna()

    df_1d["Prev_High"] = df_1d["high"].shift(1)
    df_1d["Prev_Low"] = df_1d["low"].shift(1)
    df_1d["Prev_Range"] = df_1d["Prev_High"] - df_1d["Prev_Low"]
    df_1d["Prev_Green"] = df_1d["close"].shift(1) > df_1d["open"].shift(1)

    df_1d["Fib_1618_Bull"] = df_1d["Prev_High"] + (1.618 * df_1d["Prev_Range"])
    df_1d["Fib_0382_Bull"] = df_1d["Prev_High"] - (0.382 * df_1d["Prev_Range"])
    df_1d["Fib_0618_Bull"] = df_1d["Prev_High"] - (0.618 * df_1d["Prev_Range"])
    df_1d["Fib_1618_Bear"] = df_1d["Prev_Low"] - (1.618 * df_1d["Prev_Range"])
    df_1d["Fib_0382_Bear"] = df_1d["Prev_Low"] + (0.382 * df_1d["Prev_Range"])
    df_1d["Fib_0618_Bear"] = df_1d["Prev_Low"] + (0.618 * df_1d["Prev_Range"])

    cols_15m = ["C1", "C2", "EMA_25", "EMA_50", "EMA_200", "Signal_TS"]
    cols_1d = [
        "Prev_High", "Prev_Low", "Prev_Green",
        "Fib_1618_Bull", "Fib_0382_Bull", "Fib_0618_Bull",
        "Fib_1618_Bear", "Fib_0382_Bear", "Fib_0618_Bear",
    ]
    df = df.join(df_15m[cols_15m], how="left").ffill()
    df = df.join(df_1d[cols_1d], how="left").ffill()

    required = [
        "EMA_25", "EMA_50", "EMA_200", "C1", "C2",
        "Prev_High", "Prev_Low", "EMA9", "EMA21", "C1_5m", "C2_5m",
    ]
    df_clean = df.dropna(subset=required)
    print(f"Usable bars after warm-up: {len(df_clean):,}")

    capital = STARTING_CAP
    daily_pnl = 0.0
    in_position = False
    pos_type = ""
    entry_price = 0.0
    current_sl = 0.0
    current_tp = 0.0
    tp_final = 0.0
    qty_total = 0.0
    qty_partial = 0.0
    partial_taken = False
    trade_risk_amount = 0.0
    accumulated_pnl = 0.0
    active_setup = ""
    active_session = ""
    last_signal_ts = None
    last_signal_ts_5m = None

    wins, losses = 0, 0
    gross_win, gross_loss = 0.0, 0.0
    equity_history: list[float] = [STARTING_CAP]
    equity_curve: list[dict] = []
    trade_log: list[dict] = []
    prev_date = None

    def get_fib_risk(cap: float) -> float:
        return min(FIB_MAX_RISK, cap * FIB_RISK_PERCENTAGE)

    def get_ema_risk(cap: float) -> float:
        return min(EMA_MAX_RISK, cap * EMA_RISK_PERCENTAGE)

    matrix_stats = {
        s: {k: {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0} for k in ALL_SETUP_KEYS}
        for s in SESSIONS
    }

    for idx, row in df_clean.iterrows():
        curr_c = float(row["Close"])
        curr_h = float(row["High"])
        curr_l = float(row["Low"])

        c1_15 = float(row["C1"])
        c2_15 = float(row["C2"])
        ema_25 = float(row["EMA_25"])
        ema_50 = float(row["EMA_50"])
        ema_200 = float(row["EMA_200"])
        ema_bull_15 = ema_25 > ema_50
        ema_bear_15 = ema_25 < ema_50

        ema9 = float(row["EMA9"])
        ema21 = float(row["EMA21"])
        c1_5m = float(row["C1_5m"])
        c2_5m = float(row["C2_5m"])
        ema_bull_5 = ema9 > ema21
        ema_bear_5 = ema9 < ema21

        signal_ts_15 = row["Signal_TS"]
        curr_session = get_session(idx.hour)
        is_green_day = bool(row["Prev_Green"])
        prev_range = float(row["Prev_High"]) - float(row["Prev_Low"])

        today = idx.date()
        if prev_date is not None and today != prev_date:
            equity_history.append(capital)
            equity_curve.append({"date": str(prev_date), "equity": round(capital, 2)})
            daily_pnl = 0.0
        prev_date = today

        if daily_pnl <= -DAILY_LOSS_LIMIT and not in_position:
            continue

        # ── EXIT ──────────────────────────────────────────────────────────────
        if in_position:
            exit_triggered = False
            exit_price = 0.0
            reason = ""

            if pos_type == "LONG":
                if curr_l <= current_sl:
                    pnl_rem = (current_sl - entry_price) * qty_total
                    accumulated_pnl += pnl_rem
                    capital += pnl_rem
                    daily_pnl += pnl_rem
                    exit_price = current_sl
                    reason = (
                        f"Stop Loss Hit (-${trade_risk_amount:.0f})"
                        if not partial_taken else "SL to Entry (Remaining BE)"
                    )
                    exit_triggered = True
                elif not partial_taken and curr_h >= current_tp:
                    tp1_price = current_tp
                    partial_profit = (tp1_price - entry_price) * qty_partial
                    capital += partial_profit
                    daily_pnl += partial_profit
                    accumulated_pnl += partial_profit
                    qty_total -= qty_partial
                    partial_taken = True
                    current_sl = entry_price
                    current_tp = tp_final
                    trade_log.append({
                        "time": idx.isoformat(), "event": "PARTIAL",
                        "side": pos_type, "setup": active_setup,
                        "price": tp1_price, "pnl": round(partial_profit, 2),
                    })
                elif partial_taken and curr_h >= current_tp:
                    pnl_rem = (current_tp - entry_price) * qty_total
                    capital += pnl_rem
                    daily_pnl += pnl_rem
                    accumulated_pnl += pnl_rem
                    exit_price = current_tp
                    reason = "Take Profit 2 Hit (2:1)"
                    exit_triggered = True

            elif pos_type == "SHORT":
                if curr_h >= current_sl:
                    pnl_rem = (entry_price - current_sl) * qty_total
                    accumulated_pnl += pnl_rem
                    capital += pnl_rem
                    daily_pnl += pnl_rem
                    exit_price = current_sl
                    reason = (
                        f"Stop Loss Hit (-${trade_risk_amount:.0f})"
                        if not partial_taken else "SL to Entry (Remaining BE)"
                    )
                    exit_triggered = True
                elif not partial_taken and curr_l <= current_tp:
                    tp1_price = current_tp
                    partial_profit = (entry_price - tp1_price) * qty_partial
                    capital += partial_profit
                    daily_pnl += partial_profit
                    accumulated_pnl += partial_profit
                    qty_total -= qty_partial
                    partial_taken = True
                    current_sl = entry_price
                    current_tp = tp_final
                    trade_log.append({
                        "time": idx.isoformat(), "event": "PARTIAL",
                        "side": pos_type, "setup": active_setup,
                        "price": tp1_price, "pnl": round(partial_profit, 2),
                    })
                elif partial_taken and curr_l <= current_tp:
                    pnl_rem = (entry_price - current_tp) * qty_total
                    capital += pnl_rem
                    daily_pnl += pnl_rem
                    accumulated_pnl += pnl_rem
                    exit_price = current_tp
                    reason = "Take Profit 2 Hit (2:1)"
                    exit_triggered = True

            if exit_triggered:
                trade_realized_pnl = accumulated_pnl
                is_win = trade_realized_pnl > 0
                bucket = "W" if is_win else "L"
                pnl_abs = abs(trade_realized_pnl)

                if is_win:
                    wins += 1
                    gross_win += trade_realized_pnl
                else:
                    losses += 1
                    gross_loss += pnl_abs

                matrix_stats[active_session][active_setup][bucket] += 1
                matrix_stats[active_session][active_setup][
                    "gross_win" if is_win else "gross_loss"
                ] += (trade_realized_pnl if is_win else pnl_abs)

                trade_log.append({
                    "time": idx.isoformat(), "event": "EXIT", "side": pos_type,
                    "setup": active_setup, "session": active_session,
                    "price": exit_price, "reason": reason,
                    "pnl": round(trade_realized_pnl, 2),
                })

                in_position = False
                pos_type = ""
                entry_price = 0.0
                current_sl = 0.0
                current_tp = 0.0
                tp_final = 0.0
                qty_total = 0.0
                qty_partial = 0.0
                partial_taken = False
                trade_risk_amount = 0.0
                accumulated_pnl = 0.0
                active_setup = ""
                active_session = ""
                continue

        # ── ENTRY ─────────────────────────────────────────────────────────────
        if not in_position:
            if prev_range < MIN_DAILY_RANGE_GLOBAL:
                continue

            entered = False
            pos_type = ""
            active_setup = ""

            prev_high = float(row["Prev_High"])
            prev_low = float(row["Prev_Low"])

            # Strategy 1: Coiled EMA Breakout (15M)
            ema_coil_active = abs(ema_25 - ema_50) <= 0.50
            bullish_stack_coiled = (ema_25 > ema_50) and (ema_50 > ema_200)
            bearish_stack_coiled = (ema_25 < ema_50) and (ema_50 < ema_200)

            if (
                not entered and ema_coil_active
                and curr_session not in COILED_EMA_BLOCKED_SESSIONS
                and signal_ts_15 != last_signal_ts
            ):
                if bullish_stack_coiled and (c2_15 > ema_25) and (c1_15 > ema_25):
                    pos_type, active_setup, entered = "LONG", "Coiled_EMA_Breakout", True
                elif bearish_stack_coiled and (c2_15 < ema_25) and (c1_15 < ema_25):
                    pos_type, active_setup, entered = "SHORT", "Coiled_EMA_Breakout", True

            # Strategy 2: Micro Fib Grid (15M)
            if not entered and prev_high > 0 and prev_low > 0 and prev_high > prev_low:
                if signal_ts_15 != last_signal_ts:
                    bull_levels = [
                        (float(row["Fib_1618_Bull"]), "1D_Fib_1.618"),
                        (prev_high, "1D_Fib_1.0"),
                        (float(row["Fib_0382_Bull"]), "1D_Fib_0.382"),
                        (float(row["Fib_0618_Bull"]), "1D_Fib_0.618"),
                    ]
                    bear_levels = [
                        (float(row["Fib_1618_Bear"]), "1D_Fib_1.618"),
                        (prev_low, "1D_Fib_1.0"),
                        (float(row["Fib_0382_Bear"]), "1D_Fib_0.382"),
                        (float(row["Fib_0618_Bear"]), "1D_Fib_0.618"),
                    ]

                    if is_green_day:
                        for fib_val, setup_name in bull_levels:
                            if c2_15 > fib_val and c1_15 > fib_val and ema_bull_15:
                                pos_type, active_setup, entered = "LONG", setup_name, True
                                break
                    else:
                        for fib_val, setup_name in bear_levels:
                            if c2_15 < fib_val and c1_15 < fib_val and ema_bear_15:
                                pos_type, active_setup, entered = "SHORT", setup_name, True
                                break

                    if not entered:
                        if is_green_day:
                            for fib_val, setup_name in bull_levels:
                                if c2_15 < fib_val and c1_15 < fib_val and ema_bear_15:
                                    pos_type, active_setup, entered = "SHORT", setup_name, True
                                    break
                        else:
                            for fib_val, setup_name in bear_levels:
                                if c2_15 > fib_val and c1_15 > fib_val and ema_bull_15:
                                    pos_type, active_setup, entered = "LONG", setup_name, True
                                    break

            # Strategy 3: 5M EMA Crossover (session-blocked)
            if (
                not entered and signal_ts_15 != last_signal_ts_5m
                and curr_session not in EMA_CROSSOVER_BLOCKED_SESSIONS
            ):
                if ema_bull_5 and c2_5m > ema21 and c1_5m > ema21:
                    pos_type, active_setup, entered = "LONG", "5M_EMA_Crossover", True
                elif ema_bear_5 and c2_5m < ema21 and c1_5m < ema21:
                    pos_type, active_setup, entered = "SHORT", "5M_EMA_Crossover", True

            if entered:
                entry_price = curr_c
                qty_total = TRADE_POWER / entry_price

                if active_setup == "5M_EMA_Crossover":
                    trade_risk_amount = get_ema_risk(capital)
                else:
                    trade_risk_amount = get_fib_risk(capital)

                sl_distance = trade_risk_amount / qty_total
                tp1_distance = sl_distance
                tp2_distance = sl_distance * RISK_REWARD_TOTAL

                if pos_type == "LONG":
                    current_sl = round(entry_price - sl_distance, 2)
                    current_tp = round(entry_price + tp1_distance, 2)
                    tp_final = round(entry_price + tp2_distance, 2)
                else:
                    current_sl = round(entry_price + sl_distance, 2)
                    current_tp = round(entry_price - tp1_distance, 2)
                    tp_final = round(entry_price - tp2_distance, 2)

                qty_partial = qty_total * PARTIAL_CLOSE_FRACTION
                partial_taken = False
                accumulated_pnl = 0.0
                in_position = True

                if active_setup == "5M_EMA_Crossover":
                    last_signal_ts_5m = signal_ts_15
                else:
                    last_signal_ts = signal_ts_15

                active_session = curr_session
                trade_log.append({
                    "time": idx.isoformat(), "event": "ENTRY", "side": pos_type,
                    "setup": active_setup, "session": curr_session,
                    "price": entry_price, "sl": current_sl,
                    "tp1": current_tp, "tp2": tp_final,
                    "risk": round(trade_risk_amount, 2),
                })

    equity_history.append(capital)
    if prev_date:
        equity_curve.append({"date": str(prev_date), "equity": round(capital, 2)})

    eq_series = pd.Series(equity_history)
    summary = print_report(
        actual_days, capital, wins, losses,
        gross_win, gross_loss, eq_series, matrix_stats,
        data_source, df.index[0], df.index[-1],
    )
    summary["equity_curve"] = equity_curve
    summary["trade_count_logged"] = len([t for t in trade_log if t["event"] == "ENTRY"])
    summary["trades"] = trade_log
    return summary


def main() -> int:
    api_key = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    try:
        if api_key:
            df, df_15m, df_1h = fetch_twelvedata(api_key)
            data_source = "Twelve Data XAU/USD"
        else:
            df, df_15m, df_1h = fetch_yfinance()
            data_source = "Yahoo Finance GC=F (futures proxy for XAU/USD)"
    except Exception as exc:
        print(f"DATA ERROR: {exc}", file=sys.stderr)
        return 1

    summary = run_backtest(df, df_15m, df_1h, data_source)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"v26_2_results_{stamp}.json"
    # Keep JSON lean for sharing: drop full trade list from printed tip, but save it
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {out_path}")

    # Also write a compact shareable summary without every trade fill
    share = {k: v for k, v in summary.items() if k != "trades"}
    share_path = OUT_DIR / "v26_2_latest_summary.json"
    with share_path.open("w", encoding="utf-8") as f:
        json.dump(share, f, indent=2)
    print(f"Shareable summary → {share_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
