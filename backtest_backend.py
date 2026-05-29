"""
XAU/USD (Gold) V26.2 Hybrid Prop Firm — Daily Range Gate + Session Filters
==========================================================================
Safety filters:
  - Global daily range filter: trades only if yesterday's range >= $25
  - EMA crossover blocked during NY Only & Late/Sydney
  - EMA risk reduced to 0.15% of equity (cap $200)
No RSI, no ATR, no hard DD halt. Daily loss limit $500.
"""

import os
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ═══════════════════════════════════════════════════════════════
# ENGINE 1 — COMMUNICATIONS (FastAPI)
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="XAU/USD V26.2 Hybrid Safe Prop Firm", version="10.5.2HYS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS: dict = {"status": "idle", "stats": None, "equity_curve": [], "logs": []}
data_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════
# STRATEGY CONSTANTS — HYBRID WITH SAFETY GATES
# ═══════════════════════════════════════════════════════════════

MT5_PATH         = r"C:\Program Files\MetaTrader 5\terminal64.exe"
BACKTEST_DAYS    = 36
STARTING_CAP     = 25_000.0
DAILY_LOSS_LIMIT = 500.0

MARGIN_PER_TRADE = 200.0
LEVERAGE         = 50
TRADE_POWER      = MARGIN_PER_TRADE * LEVERAGE   # $10,000 notional

# Fib Grid risk
FIB_RISK_PERCENTAGE = 0.004        # 0.4% of equity
FIB_MAX_RISK        = 500.0

# EMA Crossover risk (reduced for safety)
EMA_RISK_PERCENTAGE = 0.0015       # 0.15% of equity
EMA_MAX_RISK        = 200.0

PARTIAL_CLOSE_FRACTION = 0.5
RISK_REWARD_TOTAL      = 2.0

# Session blocks
COILED_EMA_BLOCKED_SESSIONS = {
    "London (08-13 UTC)",
    "NY Only (17-22 UTC)",
}

EMA_CROSSOVER_BLOCKED_SESSIONS = {
    "NY Only (17-22 UTC)",
    "Late / Sydney (22-00 UTC)",
}

# Global daily range gate
MIN_DAILY_RANGE_GLOBAL = 25.0   # yesterday's range must be at least $25

# No RSI, no ATR, no hard DD halt (only daily loss limit)

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

STAT_KEYS = [
    "paper_equity", "net_profit", "win_rate", "wl_ratio",
    "max_drawdown", "profit_factor", "total_trades",
    "expectancy", "avg_win", "avg_loss", "sharpe",
]

# ═══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════

def fmt_usd(value: float, force_sign: bool = False) -> str:
    if force_sign: sign = "+" if value >= 0 else "-"
    else: sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"

def make_log(log_type: str, action: str, detail: str, meta: str = "", profit: str = "", ts: str = "") -> dict:
    entry: dict = {
        "id":     ts or datetime.now(timezone.utc).isoformat(),
        "type":   log_type,
        "action": action,
        "detail": detail,
        "meta":   meta,
    }
    if profit: entry["profit"] = profit
    return entry

def get_session(hour: int) -> str:
    if 0  <= hour < 8:  return "Asian (00-08 UTC)"
    if 8  <= hour < 13: return "London (08-13 UTC)"
    if 13 <= hour < 17: return "NY Overlap (13-17 UTC)"
    if 17 <= hour < 22: return "NY Only (17-22 UTC)"
    return "Late / Sydney (22-00 UTC)"

def empty_stats(fill: str = "N/A") -> dict:
    return {k: fill for k in STAT_KEYS}

# ═══════════════════════════════════════════════════════════════
# ENGINE 2 — DATA INGESTION (MT5)
# ═══════════════════════════════════════════════════════════════

def discover_gold_symbol() -> str:
    candidates = ["XAUUSD", "GOLD", "XAUUSD.pro", "XAUUSD.", "XAUUSD.m", "XAUUSDm"]
    for name in candidates:
        if mt5.symbol_info(name) is not None:
            mt5.symbol_select(name, True)
            return name
    raise ValueError("Gold symbol not found in MT5 market watch.")

def get_mt5_data(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    return df

# ═══════════════════════════════════════════════════════════════
# ENGINE 5 — ANALYTICS HELPERS
# ═══════════════════════════════════════════════════════════════

def calc_max_drawdown(equity_series: pd.Series) -> float:
    if len(equity_series) < 2: return 0.0
    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max) / roll_max * 100
    val = float(dd.min())
    return val if not np.isnan(val) else 0.0

def calc_sharpe(equity_series: pd.Series, risk_free_annual: float = 0.053) -> float:
    if len(equity_series) < 2: return 0.0
    daily_ret = equity_series.pct_change().dropna()
    std = daily_ret.std()
    if std == 0 or np.isnan(std): return 0.0
    rf_daily = risk_free_annual / 252
    return float((daily_ret.mean() - rf_daily) / std * np.sqrt(252))

def print_report(
    actual_days:  int,
    capital:      float,
    wins:         int,
    losses:       int,
    gross_win:    float,
    gross_loss:   float,
    eq_series:    pd.Series,
    setup_stats:  dict,
    matrix_stats: dict,
) -> None:
    total_trades = wins + losses
    net_profit   = capital - STARTING_CAP
    win_rate     = (wins / total_trades * 100) if total_trades > 0 else 0.0
    pf           = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_win      = (gross_win  / wins)   if wins   > 0 else 0.0
    avg_loss     = (gross_loss / losses) if losses > 0 else 0.0
    expectancy   = (gross_win - gross_loss) / total_trades if total_trades > 0 else 0.0
    roi          = (net_profit / STARTING_CAP) * 100
    max_dd       = calc_max_drawdown(eq_series)
    sharpe       = calc_sharpe(eq_series)

    sep  = "=" * 80
    dash = "-" * 80

    print(f"\n{sep}")
    print(f"    V26.2 HYBRID SAFE PROP FIRM — Daily Range ≥ ${MIN_DAILY_RANGE_GLOBAL}, EMA Session Block")
    print(sep)
    print(f"  Test Period     : {actual_days} calendar days")
    print(f"  Starting Cap    : ${STARTING_CAP:>10,.2f}")
    print(f"  Final Capital   : ${capital:>10,.2f}")
    print(f"  Net Profit      : ${net_profit:>+10,.2f}   ({roi:+.1f}% ROI)")
    print(dash)
    print(f"  Total Trades    : {total_trades} (Unrestricted)")
    print(f"  Win Rate        : {win_rate:>6.1f}%   ({wins}W / {losses}L)")
    print(f"  Profit Factor   : {pf:>6.2f}" if pf != float("inf") else f"  Profit Factor   :      ∞")
    print(f"  Expectancy      : {fmt_usd(expectancy, True):>12}  per trade")
    print(f"  Avg Win         : {fmt_usd(avg_win, True):>12}")
    print(f"  Avg Loss        : {fmt_usd(-avg_loss):>12}")
    print(f"  Max Drawdown    : {max_dd:>6.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:>6.2f}")
    print(f"\n  Fib Risk/Trade  : {FIB_RISK_PERCENTAGE*100:.1f}% of equity, capped at ${FIB_MAX_RISK}")
    print(f"  EMA Risk/Trade  : {EMA_RISK_PERCENTAGE*100:.2f}% of equity, capped at ${EMA_MAX_RISK}")
    print(f"  Daily Loss Lim  : ${DAILY_LOSS_LIMIT:,.2f} (2% of starting capital)")
    print(f"  Hard DD Halt    : DISABLED (daily limit only)")
    print(f"  Global Filter   : Yesterday's range must be ≥ ${MIN_DAILY_RANGE_GLOBAL}")
    print(f"  EMA Blocked     : NY Only + Late/Sydney")

    print(f"\n{sep}")
    print("    DEEP MATRIX BREAKDOWN: SESSION vs SETUP")
    print(sep)

    for sess, setups in matrix_stats.items():
        sess_w = sum(v["W"] for v in setups.values())
        sess_l = sum(v["L"] for v in setups.values())
        if (sess_w + sess_l) == 0: continue

        print(f"\n  SESSION: {sess}")
        print(dash)
        print(f"  {'Setup':<22} {'W':>4} {'L':>4}  {'WR%':>6}  {'PF':>6}")
        for setup_name, s in setups.items():
            total_s = s["W"] + s["L"]
            if total_s == 0: continue
            wr_s   = (s["W"] / total_s * 100) if total_s > 0 else 0.0
            pf_s   = (s["gross_win"] / s["gross_loss"]) if s["gross_loss"] > 0 else float("inf")
            pf_str = f"{pf_s:>6.2f}" if pf_s != float("inf") else "     ∞"
            print(f"  {setup_name:<22} {s['W']:>4} {s['L']:>4}  {wr_s:>5.1f}%  {pf_str}")
        print(dash)

    print(f"{sep}\n")

# ═══════════════════════════════════════════════════════════════
# ENGINES 3 + 4 — STRATEGY PREP + EXECUTION
# ═══════════════════════════════════════════════════════════════

def run_simulation() -> None:
    with data_lock:
        RESULTS.update({
            "status":       "running",
            "logs":         [],
            "equity_curve": [],
            "stats":        empty_stats("—"),
        })

    logs:         list[dict] = []
    equity_curve: list[dict] = []

    try:
        print(f"\nInitializing MT5 — pulling {BACKTEST_DAYS} days of history...")
        initialized = mt5.initialize(path=MT5_PATH) if os.path.exists(MT5_PATH) else mt5.initialize()
        if not initialized:
            raise ValueError("MT5 init failed. Ensure the terminal is open.")

        symbol = discover_gold_symbol()
        df     = get_mt5_data(symbol, mt5.TIMEFRAME_M5,  BACKTEST_DAYS * 24 * 12)
        df_15m = get_mt5_data(symbol, mt5.TIMEFRAME_M15, BACKTEST_DAYS * 24 * 4)
        df_1h  = get_mt5_data(symbol, mt5.TIMEFRAME_H1,  BACKTEST_DAYS * 24)

        if any(d.empty for d in [df, df_15m, df_1h]):
            mt5.shutdown()
            raise ValueError("MT5 returned empty history.")
        mt5.shutdown()

        actual_days = (df.index[-1] - df.index[0]).days

        # ── 15M INDICATORS FOR FIB GRID ────────────────────────────────────
        df_15m["EMA_25"]  = df_15m["Close"].ewm(span=25,  adjust=False).mean()
        df_15m["EMA_50"]  = df_15m["Close"].ewm(span=50,  adjust=False).mean()
        df_15m["EMA_200"] = df_15m["Close"].ewm(span=200, adjust=False).mean()
        df_15m["C1"]      = df_15m["Close"].shift(1)
        df_15m["C2"]      = df_15m["Close"].shift(2)
        df_15m["Signal_TS"] = df_15m.index

        # ── 5M INDICATORS FOR EMA CROSSOVER ────────────────────────────────
        df["EMA9"]  = df["Close"].ewm(span=9,  adjust=False).mean()
        df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
        df["C1_5m"] = df["Close"].shift(1)
        df["C2_5m"] = df["Close"].shift(2)

        # ── DAILY OHLC FOR FIB GRID ───────────────────────────────────────
        df_1d = pd.DataFrame({
            "open":  df_1h["Open"].resample("D", offset="22h").first(),
            "high":  df_1h["High"].resample("D", offset="22h").max(),
            "low":   df_1h["Low"].resample("D", offset="22h").min(),
            "close": df_1h["Close"].resample("D", offset="22h").last(),
        }).dropna()

        df_1d["Prev_High"]  = df_1d["high"].shift(1)
        df_1d["Prev_Low"]   = df_1d["low"].shift(1)
        df_1d["Prev_Range"] = df_1d["Prev_High"] - df_1d["Prev_Low"]
        df_1d["Prev_Green"] = df_1d["close"].shift(1) > df_1d["open"].shift(1)

        df_1d["Fib_1618_Bull"] = df_1d["Prev_High"] + (1.618 * df_1d["Prev_Range"])
        df_1d["Fib_0382_Bull"] = df_1d["Prev_High"] - (0.382 * df_1d["Prev_Range"])
        df_1d["Fib_0618_Bull"] = df_1d["Prev_High"] - (0.618 * df_1d["Prev_Range"])

        df_1d["Fib_1618_Bear"] = df_1d["Prev_Low"]  - (1.618 * df_1d["Prev_Range"])
        df_1d["Fib_0382_Bear"] = df_1d["Prev_Low"]  + (0.382 * df_1d["Prev_Range"])
        df_1d["Fib_0618_Bear"] = df_1d["Prev_Low"]  + (0.618 * df_1d["Prev_Range"])

        # ── JOIN 15M AND DAILY ONTO 5M BASE ───────────────────────────────
        cols_15m = [
            "C1", "C2", "EMA_25", "EMA_50", "EMA_200",
            "Signal_TS",
        ]
        cols_1d = [
            "Prev_High", "Prev_Low", "Prev_Green",
            "Fib_1618_Bull", "Fib_0382_Bull", "Fib_0618_Bull",
            "Fib_1618_Bear", "Fib_0382_Bear", "Fib_0618_Bear",
        ]
        df = df.join(df_15m[cols_15m], how="left").ffill()
        df = df.join(df_1d[cols_1d],   how="left").ffill()

        required = ["EMA_25", "EMA_50", "EMA_200", "C1", "C2",
                    "Prev_High", "Prev_Low",
                    "EMA9", "EMA21", "C1_5m", "C2_5m"]
        df_clean = df.dropna(subset=required)

        # ── SIMULATION STATE ───────────────────────────────────────────────
        capital            = STARTING_CAP
        daily_pnl          = 0.0
        current_date       = None
        in_position        = False
        pos_type           = ""
        entry_price        = 0.0
        current_sl         = 0.0
        current_tp         = 0.0
        tp_final           = 0.0
        qty_total          = 0.0
        qty_partial        = 0.0
        partial_taken      = False
        trade_risk_amount  = 0.0
        accumulated_pnl    = 0.0
        active_setup       = ""
        active_session     = ""
        last_signal_ts     = None
        last_signal_ts_5m  = None

        wins, losses          = 0, 0
        gross_win, gross_loss = 0.0, 0.0
        equity_history: list[float] = [STARTING_CAP]
        prev_date = None

        def get_fib_risk(cap): return min(FIB_MAX_RISK, cap * FIB_RISK_PERCENTAGE)
        def get_ema_risk(cap): return min(EMA_MAX_RISK, cap * EMA_RISK_PERCENTAGE)

        setup_stats   = {k: {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0} for k in ALL_SETUP_KEYS}
        session_stats = {s: {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0} for s in SESSIONS}
        matrix_stats  = {
            s: {k: {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0} for k in ALL_SETUP_KEYS}
            for s in SESSIONS
        }

        STREAM_INTERVAL = 500

        # ──────────────────────────────────────────────────────────────────
        # MAIN LOOP
        # ──────────────────────────────────────────────────────────────────
        for i, (idx, row) in enumerate(df_clean.iterrows()):
            curr_c = float(row["Close"])
            curr_h = float(row["High"])
            curr_l = float(row["Low"])

            # 15M values
            c1_15 = float(row["C1"])
            c2_15 = float(row["C2"])
            ema_25  = float(row["EMA_25"])
            ema_50  = float(row["EMA_50"])
            ema_200 = float(row["EMA_200"])
            ema_bull_15 = ema_25 > ema_50
            ema_bear_15 = ema_25 < ema_50

            # 5M values
            ema9   = float(row["EMA9"])
            ema21  = float(row["EMA21"])
            c1_5m  = float(row["C1_5m"])
            c2_5m  = float(row["C2_5m"])
            ema_bull_5 = ema9 > ema21
            ema_bear_5 = ema9 < ema21

            signal_ts_15 = row["Signal_TS"]
            curr_session = get_session(idx.hour)
            ts           = idx.isoformat()
            is_green_day = bool(row["Prev_Green"])
            prev_range   = float(row["Prev_High"]) - float(row["Prev_Low"])

            today = idx.date()
            if prev_date is not None and today != prev_date:
                equity_history.append(capital)
                equity_curve.append({"date": str(prev_date), "equity": round(capital, 2)})
                daily_pnl = 0.0
                current_date = today
            prev_date = today

            if i % STREAM_INTERVAL == 0:
                with data_lock:
                    RESULTS["equity_curve"] = list(equity_curve)
                    RESULTS["logs"]         = list(reversed(logs))

            # ── DAILY LOSS LIMIT ───────────────────────────────────────────
            if daily_pnl <= -DAILY_LOSS_LIMIT and not in_position:
                continue

            # ── EXIT MODULE ────────────────────────────────────────────────
            if in_position:
                exit_triggered = False
                exit_price     = 0.0
                reason         = ""

                if pos_type == "LONG":
                    if curr_l <= current_sl:
                        pnl_rem = (current_sl - entry_price) * qty_total
                        accumulated_pnl += pnl_rem
                        capital += pnl_rem
                        daily_pnl += pnl_rem
                        exit_price = current_sl
                        reason = f"Stop Loss Hit (-${trade_risk_amount:.0f})" if not partial_taken else "SL to Entry (Remaining BE)"
                        exit_triggered = True
                    elif not partial_taken and curr_h >= current_tp:
                        partial_profit = (current_tp - entry_price) * qty_partial
                        capital += partial_profit
                        daily_pnl += partial_profit
                        accumulated_pnl += partial_profit
                        qty_total -= qty_partial
                        partial_taken = True
                        current_sl = entry_price
                        current_tp = tp_final
                        logs.append(make_log(
                            "PARTIAL", f"PARTIAL CLOSE {pos_type}",
                            f"TP1 hit @ {current_tp:.2f} | Closed {qty_partial:.2f} units",
                            f"Partial P&L: {fmt_usd(partial_profit, True)} | Remaining {qty_total:.2f} units, SL→entry, TP→{tp_final:.2f}",
                            fmt_usd(partial_profit, True),
                            ts + "_partial",
                        ))
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
                        reason = f"Stop Loss Hit (-${trade_risk_amount:.0f})" if not partial_taken else "SL to Entry (Remaining BE)"
                        exit_triggered = True
                    elif not partial_taken and curr_l <= current_tp:
                        partial_profit = (entry_price - current_tp) * qty_partial
                        capital += partial_profit
                        daily_pnl += partial_profit
                        accumulated_pnl += partial_profit
                        qty_total -= qty_partial
                        partial_taken = True
                        current_sl = entry_price
                        current_tp = tp_final
                        logs.append(make_log(
                            "PARTIAL", f"PARTIAL CLOSE {pos_type}",
                            f"TP1 hit @ {current_tp:.2f} | Closed {qty_partial:.2f} units",
                            f"Partial P&L: {fmt_usd(partial_profit, True)} | Remaining {qty_total:.2f} units, SL→entry, TP→{tp_final:.2f}",
                            fmt_usd(partial_profit, True),
                            ts + "_partial",
                        ))
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

                    if is_win: wins += 1; gross_win += trade_realized_pnl
                    else: losses += 1; gross_loss += pnl_abs

                    setup_stats[active_setup][bucket] += 1
                    setup_stats[active_setup]["gross_win" if is_win else "gross_loss"] += (trade_realized_pnl if is_win else pnl_abs)
                    session_stats[active_session][bucket] += 1
                    session_stats[active_session]["gross_win" if is_win else "gross_loss"] += (trade_realized_pnl if is_win else pnl_abs)
                    matrix_stats[active_session][active_setup][bucket] += 1
                    matrix_stats[active_session][active_setup]["gross_win" if is_win else "gross_loss"] += (trade_realized_pnl if is_win else pnl_abs)

                    logs.append(make_log(
                        "EXIT", f"CLOSE {pos_type}",
                        f"{reason} @ {exit_price:.2f}",
                        f"Total Trade P&L: {fmt_usd(trade_realized_pnl, True)}",
                        fmt_usd(trade_realized_pnl, True),
                        ts + "_exit",
                    ))

                    in_position        = False
                    pos_type           = ""
                    entry_price        = 0.0
                    current_sl         = 0.0
                    current_tp         = 0.0
                    tp_final           = 0.0
                    qty_total          = 0.0
                    qty_partial        = 0.0
                    partial_taken      = False
                    trade_risk_amount  = 0.0
                    accumulated_pnl    = 0.0
                    active_setup       = ""
                    active_session     = ""
                    continue

            # ── ENTRY MODULE ───────────────────────────────────────────────
            if not in_position:
                # Global daily range gate – skip all entries if yesterday's range too small
                if prev_range < MIN_DAILY_RANGE_GLOBAL:
                    continue

                entered      = False
                pos_type     = ""
                active_setup = ""

                prev_high = float(row["Prev_High"])
                prev_low  = float(row["Prev_Low"])

                # ── STRATEGY 1: COILED EMA BREAKOUT (15M) ──────────────────
                ema_coil_active = abs(ema_25 - ema_50) <= 0.50
                bullish_stack_coiled = (ema_25 > ema_50) and (ema_50 > ema_200)
                bearish_stack_coiled = (ema_25 < ema_50) and (ema_50 < ema_200)

                if (not entered and ema_coil_active and 
                        curr_session not in COILED_EMA_BLOCKED_SESSIONS and
                        signal_ts_15 != last_signal_ts):
                    if bullish_stack_coiled and (c2_15 > ema_25) and (c1_15 > ema_25):
                        pos_type, active_setup, entered = "LONG",  "Coiled_EMA_Breakout", True
                    elif bearish_stack_coiled and (c2_15 < ema_25) and (c1_15 < ema_25):
                        pos_type, active_setup, entered = "SHORT", "Coiled_EMA_Breakout", True

                # ── STRATEGY 2: MICRO FIB GRID (15M) ───────────────────────
                if not entered and prev_high > 0 and prev_low > 0 and prev_high > prev_low:
                    if signal_ts_15 != last_signal_ts:
                        bull_levels = [
                            (float(row["Fib_1618_Bull"]), "1D_Fib_1.618"),
                            (prev_high,                   "1D_Fib_1.0"),
                            (float(row["Fib_0382_Bull"]), "1D_Fib_0.382"),
                            (float(row["Fib_0618_Bull"]), "1D_Fib_0.618"),
                        ]
                        bear_levels = [
                            (float(row["Fib_1618_Bear"]), "1D_Fib_1.618"),
                            (prev_low,                    "1D_Fib_1.0"),
                            (float(row["Fib_0382_Bear"]), "1D_Fib_0.382"),
                            (float(row["Fib_0618_Bear"]), "1D_Fib_0.618"),
                        ]

                        # Continuation
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

                        # Correction
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

                # ── STRATEGY 3: 5M EMA CROSSOVER (session-blocked) ─────────
                if (not entered and signal_ts_15 != last_signal_ts_5m
                        and curr_session not in EMA_CROSSOVER_BLOCKED_SESSIONS):
                    if ema_bull_5 and c2_5m > ema21 and c1_5m > ema21:
                        pos_type, active_setup, entered = "LONG", "5M_EMA_Crossover", True
                    elif ema_bear_5 and c2_5m < ema21 and c1_5m < ema21:
                        pos_type, active_setup, entered = "SHORT", "5M_EMA_Crossover", True

                # ── EXECUTE TRADE ──────────────────────────────────────────
                if entered:
                    entry_price = curr_c
                    qty_total   = TRADE_POWER / entry_price

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
                        tp_final   = round(entry_price + tp2_distance, 2)
                    else:
                        current_sl = round(entry_price + sl_distance, 2)
                        current_tp = round(entry_price - tp1_distance, 2)
                        tp_final   = round(entry_price - tp2_distance, 2)

                    qty_partial = qty_total * PARTIAL_CLOSE_FRACTION
                    partial_taken = False
                    accumulated_pnl = 0.0
                    in_position    = True

                    if active_setup == "5M_EMA_Crossover":
                        last_signal_ts_5m = signal_ts_15
                    else:
                        last_signal_ts = signal_ts_15

                    active_session = curr_session

                    logs.append(make_log(
                        "ENTRY",
                        f"{pos_type} XAU",
                        f"{active_setup} | 1:2 R:R (50% partial @1:1)",
                        f"Filled @ {entry_price:.2f} | SL: {current_sl:.2f} | TP1: {current_tp:.2f} | TP2: {tp_final:.2f} | Risk: -${trade_risk_amount:.0f}",
                        "",
                        ts + "_entry",
                    ))

        # ── FINALISE ───────────────────────────────────────────────────────
        equity_history.append(capital)
        if prev_date:
            equity_curve.append({"date": str(prev_date), "equity": round(capital, 2)})
        if not equity_curve:
            equity_curve.append({"date": "Start", "equity": STARTING_CAP})

        total_trades = wins + losses
        eq_series    = pd.Series(equity_history)
        print_report(
            actual_days, capital, wins, losses,
            gross_win, gross_loss, eq_series, setup_stats, matrix_stats,
        )

        net_profit = capital - STARTING_CAP
        win_rate   = (wins / total_trades * 100) if total_trades > 0 else 0.0
        pf         = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        avg_win    = (gross_win  / wins)   if wins   > 0 else 0.0
        avg_loss   = (gross_loss / losses) if losses > 0 else 0.0
        expectancy = (gross_win - gross_loss) / total_trades if total_trades > 0 else 0.0

        with data_lock:
            RESULTS["stats"] = {
                "paper_equity":      fmt_usd(capital),
                "net_profit":        fmt_usd(net_profit, True),
                "win_rate":          f"{win_rate:.1f}%",
                "wl_ratio":          f"{wins}W / {losses}L",
                "max_drawdown":      f"{calc_max_drawdown(eq_series):.2f}%",
                "profit_factor":     f"{pf:.2f}" if pf != float("inf") else "∞",
                "total_trades":      str(total_trades),
                "expectancy":        fmt_usd(expectancy, True),
                "avg_win":           fmt_usd(avg_win, True),
                "avg_loss":          fmt_usd(-avg_loss),
                "sharpe":            f"{calc_sharpe(eq_series):.2f}",
                "setup_breakdown":   setup_stats,
                "session_breakdown": session_stats,
                "matrix_breakdown":  matrix_stats,
            }
            RESULTS["equity_curve"] = equity_curve
            RESULTS["logs"]         = list(reversed(logs))
            RESULTS["status"]       = "completed"

    except Exception as exc:
        error_msg = str(exc)
        with data_lock:
            RESULTS["status"] = "error"
            RESULTS["stats"]  = empty_stats("N/A")
            RESULTS["logs"]   = [make_log("EXIT", "SIMULATION ERROR", error_msg)]
        raise

# ═══════════════════════════════════════════════════════════════
# ENGINE 1 — ROUTES
# ═══════════════════════════════════════════════════════════════

@app.post("/trigger")
async def trigger_sim():
    with data_lock:
        if RESULTS["status"] == "running":
            raise HTTPException(status_code=409, detail="Simulation already running.")
    threading.Thread(target=run_simulation, daemon=True).start()
    return {"message": "Simulation started."}

@app.get("/data")
async def get_results():
    with data_lock: return dict(RESULTS)

@app.get("/health")
async def health(): return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")