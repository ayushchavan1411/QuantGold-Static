"""
XAU/USD (Gold) V26 Deep Training Engine — 6-Month Historical Backtest
======================================================================
Strategy Logic:
  1. DAILY BASELINE: Calculate previous day's range.
     · GREEN day  → Fib 1.0 = prev_high | Fib 0.382 = prev_high − 38.2% of range
     · RED day    → Fib 1.0 = prev_low  | Fib 0.382 = prev_low  + 38.2% of range

  2. 15M BREAKOUT TRIGGER (scans both Fib levels simultaneously):
     · LONG  : C3 ≤ level AND C2 > level AND C1 > level AND 5M EMA25 > EMA50
     · SHORT : C3 ≥ level AND C2 < level AND C1 < level AND 5M EMA25 < EMA50

  3. EXIT PHYSICS:
     · SL   : −$20 hard stop (closes 100%)
     · TP1  : +$20 scale-out (closes 50%, moves SL to breakeven)
     · TP2  : +$60 runner    (closes remaining 50%)

Data Source: Native MT5 Broker Cache
Time Horizon: 180 Days (~6 Months)

Setup:
    pip install MetaTrader5 pandas numpy

Run:
    python backtest.py
"""

import os
import sys
from datetime import timezone

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BACKTEST_DAYS    = 180
STARTING_CAP     = 500.0
MARGIN_PER_TRADE = 200.0
LEVERAGE         = 50
TRADE_POWER      = MARGIN_PER_TRADE * LEVERAGE   # $10,000 purchasing power

SL_USD           = 20.0    # Hard stop-loss in Gold price points ($)
TP1_USD          = 20.0    # First target — triggers 50% close + breakeven SL
TP2_USD          = 60.0    # Runner target — closes remaining 50%

MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"


# ─── MT5 HELPERS ─────────────────────────────────────────────────────────────

def discover_gold_symbol() -> str:
    candidates = ["XAUUSD", "GOLD", "XAUUSD.pro", "XAUUSD.", "XAUUSD.m", "XAUUSDm"]
    for name in candidates:
        if mt5.symbol_info(name) is not None:
            mt5.symbol_select(name, True)
            return name
    sys.exit("CRITICAL: Gold symbol not found in MT5 market watch.")


def get_mt5_data(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close"
    }, inplace=True)
    return df


# ─── QUANT METRICS ────────────────────────────────────────────────────────────

def calc_max_drawdown(equity_series: pd.Series) -> float:
    """Peak-to-trough drawdown as a percentage of peak equity."""
    if len(equity_series) < 2:
        return 0.0
    roll_max = equity_series.cummax()
    dd = (equity_series - roll_max) / roll_max * 100
    return float(dd.min())


def calc_sharpe(equity_series: pd.Series, risk_free_annual: float = 0.053) -> float:
    """
    Annualised Sharpe ratio from a daily equity series.
    risk_free_annual: 5.3% (US T-bill proxy for USD accounts)
    """
    if len(equity_series) < 2:
        return 0.0
    daily_ret = equity_series.pct_change().dropna()
    std = daily_ret.std()
    if std == 0 or np.isnan(std):
        return 0.0
    rf_daily = risk_free_annual / 252
    return float((daily_ret.mean() - rf_daily) / std * np.sqrt(252))


def get_session(hour: int) -> str:
    if 0  <= hour < 8:  return "Asian (00-08 UTC)"
    if 8  <= hour < 13: return "London (08-13 UTC)"
    if 13 <= hour < 17: return "NY Overlap (13-17 UTC)"
    if 17 <= hour < 22: return "NY Only (17-22 UTC)"
    return "Late / Sydney (22-00 UTC)"


# ─── MAIN BACKTEST ────────────────────────────────────────────────────────────

def run_training() -> None:
    print(f"\nInitializing MT5 link — pulling {BACKTEST_DAYS} days of history...")

    initialized = (
        mt5.initialize(path=MT5_PATH)
        if os.path.exists(MT5_PATH)
        else mt5.initialize()
    )
    if not initialized:
        sys.exit("MT5 init failed. Ensure the terminal is open.")

    symbol = discover_gold_symbol()
    print(f"✅ Symbol mapped: {symbol}. Fetching data arrays...")

    # Gold trades ~23h/day. Multiply by 24 to ensure full coverage from MT5 cache.
    df      = get_mt5_data(symbol, mt5.TIMEFRAME_M5,  BACKTEST_DAYS * 24 * 12)
    df_15m  = get_mt5_data(symbol, mt5.TIMEFRAME_M15, BACKTEST_DAYS * 24 * 4)
    df_1h   = get_mt5_data(symbol, mt5.TIMEFRAME_H1,  BACKTEST_DAYS * 24)

    if df.empty or df_15m.empty or df_1h.empty:
        mt5.shutdown()
        sys.exit("Data fetch failed. Scroll your MT5 M5 chart back to load history.")

    mt5.shutdown()   # ← Always close the connection after data fetch

    actual_days = (df.index[-1] - df.index[0]).days
    print(f"✅ {len(df):,} × 5M candles fetched ({actual_days} calendar days).")
    print("Crunching V26 strategy logic — please wait...\n")

    # ── 5M EMAs (pre-calculated, no look-ahead) ────────────────────────────────
    df["EMA_25"] = df["Close"].ewm(span=25, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # ── 15M: last three closed candles ────────────────────────────────────────
    # shift(N) gives the Nth previous CLOSED 15M candle at every point in time.
    # C1 = last closed, C2 = two ago, C3 = three ago (oldest of the three).
    df_15m["C1"]          = df_15m["Close"].shift(1)
    df_15m["C2"]          = df_15m["Close"].shift(2)
    df_15m["C3"]          = df_15m["Close"].shift(3)
    df_15m["Signal_TS"]   = df_15m.index   # Stable signal ID for dedup

    # ── 1D OHLC from 1H data ───────────────────────────────────────────────────
    # Gold's trading day resets at ~22:00 UTC (17:00 NY).
    # offset="22h" aligns the daily resample to that boundary.
    df_1d = pd.DataFrame({
        "open":  df_1h["Open"].resample("D", offset="22h").first(),
        "high":  df_1h["High"].resample("D", offset="22h").max(),
        "low":   df_1h["Low"].resample("D", offset="22h").min(),
        "close": df_1h["Close"].resample("D", offset="22h").last(),
    }).dropna()

    # Previous day's data (shift so the signal uses CONFIRMED completed candles)
    df_1d["Prev_High"]  = df_1d["high"].shift(1)
    df_1d["Prev_Low"]   = df_1d["low"].shift(1)
    df_1d["Prev_Range"] = df_1d["Prev_High"] - df_1d["Prev_Low"]
    df_1d["Prev_Green"] = df_1d["close"].shift(1) > df_1d["open"].shift(1)

    # ── Fibonacci levels (verified against stated logic) ──────────────────────
    # GREEN day: Fib 0.382 = prev_high − 38.2% of range  (support level)
    # RED day:   Fib 0.382 = prev_low  + 38.2% of range  (resistance level)
    df_1d["Fib_0382_Bull"] = df_1d["Prev_High"] - (0.382 * df_1d["Prev_Range"])
    df_1d["Fib_0382_Bear"] = df_1d["Prev_Low"]  + (0.382 * df_1d["Prev_Range"])

    # ── Merge higher timeframes onto 5M master frame ──────────────────────────
    cols_15m = ["C1", "C2", "C3", "Signal_TS"]
    cols_1d  = ["Prev_High", "Prev_Low", "Prev_Green", "Fib_0382_Bull", "Fib_0382_Bear"]

    df = df.join(df_15m[cols_15m], how="left").ffill()
    df = df.join(df_1d[cols_1d],   how="left").ffill()

    # Drop rows where any required column is still NaN (start of dataset)
    required = ["EMA_25", "EMA_50", "C1", "C2", "C3", "Prev_High", "Prev_Low"]
    df_clean = df.dropna(subset=required)
    dropped  = len(df) - len(df_clean)
    if dropped:
        print(f"  (Dropped {dropped:,} warm-up rows where history was incomplete.)")

    # ── Portfolio & Trade State ────────────────────────────────────────────────
    capital     = STARTING_CAP
    in_position = False
    tp1_hit     = False
    pos_type    = ""

    entry_price       = 0.0
    current_sl        = 0.0
    qty_total         = 0.0
    qty_remaining     = 0.0
    trade_realized_pnl = 0.0
    active_setup      = ""
    active_session    = ""
    last_signal_ts    = None   # Prevents double-entry on the same 15M signal

    wins, losses   = 0, 0
    gross_win      = 0.0
    gross_loss     = 0.0
    equity_history: list[float] = [STARTING_CAP]  # Daily snapshots for Sharpe/drawdown

    setup_stats: dict[str, dict] = {
        "1D_Fib_1.0":   {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
        "1D_Fib_0.382": {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
    }
    session_stats: dict[str, dict] = {
        "Asian (00-08 UTC)":     {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
        "London (08-13 UTC)":    {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
        "NY Overlap (13-17 UTC)":{"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
        "NY Only (17-22 UTC)":   {"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
        "Late / Sydney (22-00 UTC)":{"W": 0, "L": 0, "gross_win": 0.0, "gross_loss": 0.0},
    }

    prev_date = None

    # ─────────────────────────────────────────────────────────────────────────
    for idx, row in df_clean.iterrows():
        curr_c = float(row["Close"])
        curr_h = float(row["High"])
        curr_l = float(row["Low"])

        # Daily equity snapshot (for Sharpe / drawdown calculation)
        today = idx.date()
        if prev_date is not None and today != prev_date:
            equity_history.append(capital)
        prev_date = today

        # ── Resolved 15M candle values (C1=most recent closed, C3=oldest) ──
        c1 = float(row["C1"])
        c2 = float(row["C2"])
        c3 = float(row["C3"])

        ema_bull = float(row["EMA_25"]) > float(row["EMA_50"])
        ema_bear = float(row["EMA_25"]) < float(row["EMA_50"])

        curr_session = get_session(idx.hour)

        # ── EXIT LOGIC ──────────────────────────────────────────────────────
        if in_position:
            exit_triggered = False

            if pos_type == "LONG":
                tp1_price = entry_price + TP1_USD
                tp2_price = entry_price + TP2_USD

                if not tp1_hit and curr_h >= tp1_price:
                    # TP1: close 50%, move SL to breakeven
                    close_qty  = round(qty_total * 0.5, 10)
                    pnl_tp1    = (tp1_price - entry_price) * close_qty
                    trade_realized_pnl += pnl_tp1
                    capital            += pnl_tp1
                    qty_remaining       = round(qty_total - close_qty, 10)
                    tp1_hit             = True
                    current_sl          = entry_price   # Breakeven

                    # Same-candle TP2 check (candle wicked through both targets)
                    if curr_h >= tp2_price:
                        pnl_tp2             = (tp2_price - entry_price) * qty_remaining
                        trade_realized_pnl += pnl_tp2
                        capital            += pnl_tp2
                        qty_remaining       = 0.0
                        exit_triggered      = True

                    # Same-candle: TP1 hit + wicked back to new breakeven SL
                    # (SL is now at entry_price; check if Low also broke it)
                    elif curr_l <= entry_price:
                        # Runner stopped at breakeven on the same candle
                        pnl_sl              = (entry_price - entry_price) * qty_remaining  # = 0
                        trade_realized_pnl += pnl_sl
                        capital            += pnl_sl
                        qty_remaining       = 0.0
                        exit_triggered      = True

                elif tp1_hit and curr_h >= tp2_price:
                    pnl_tp2             = (tp2_price - entry_price) * qty_remaining
                    trade_realized_pnl += pnl_tp2
                    capital            += pnl_tp2
                    qty_remaining       = 0.0
                    exit_triggered      = True

                elif curr_l <= current_sl:
                    pnl_sl              = (current_sl - entry_price) * qty_remaining
                    trade_realized_pnl += pnl_sl
                    capital            += pnl_sl
                    qty_remaining       = 0.0
                    exit_triggered      = True

            elif pos_type == "SHORT":
                tp1_price = entry_price - TP1_USD
                tp2_price = entry_price - TP2_USD

                if not tp1_hit and curr_l <= tp1_price:
                    # TP1: close 50%, move SL to breakeven
                    close_qty  = round(qty_total * 0.5, 10)
                    pnl_tp1    = (entry_price - tp1_price) * close_qty
                    trade_realized_pnl += pnl_tp1
                    capital            += pnl_tp1
                    qty_remaining       = round(qty_total - close_qty, 10)
                    tp1_hit             = True
                    current_sl          = entry_price   # Breakeven

                    # Same-candle TP2
                    if curr_l <= tp2_price:
                        pnl_tp2             = (entry_price - tp2_price) * qty_remaining
                        trade_realized_pnl += pnl_tp2
                        capital            += pnl_tp2
                        qty_remaining       = 0.0
                        exit_triggered      = True

                    # Same-candle: TP1 + wicked back to breakeven SL
                    elif curr_h >= entry_price:
                        trade_realized_pnl += 0.0   # Breakeven on runner
                        qty_remaining       = 0.0
                        exit_triggered      = True

                elif tp1_hit and curr_l <= tp2_price:
                    pnl_tp2             = (entry_price - tp2_price) * qty_remaining
                    trade_realized_pnl += pnl_tp2
                    capital            += pnl_tp2
                    qty_remaining       = 0.0
                    exit_triggered      = True

                elif curr_h >= current_sl:
                    pnl_sl              = (entry_price - current_sl) * qty_remaining
                    trade_realized_pnl += pnl_sl
                    capital            += pnl_sl
                    qty_remaining       = 0.0
                    exit_triggered      = True

            # ── Record trade outcome ──────────────────────────────────────────
            if exit_triggered:
                is_win = trade_realized_pnl > 0
                if is_win:
                    wins      += 1
                    gross_win += trade_realized_pnl
                    setup_stats[active_setup]["W"]          += 1
                    setup_stats[active_setup]["gross_win"]  += trade_realized_pnl
                    session_stats[active_session]["W"]        += 1
                    session_stats[active_session]["gross_win"]+= trade_realized_pnl
                else:
                    losses      += 1
                    gross_loss  += abs(trade_realized_pnl)
                    setup_stats[active_setup]["L"]           += 1
                    setup_stats[active_setup]["gross_loss"]  += abs(trade_realized_pnl)
                    session_stats[active_session]["L"]         += 1
                    session_stats[active_session]["gross_loss"]+= abs(trade_realized_pnl)

                # ── Reset all trade state on exit ─────────────────────────────
                in_position        = False
                tp1_hit            = False
                last_signal_ts     = None   # ← Allow re-entry after exit
                pos_type           = ""
                entry_price        = current_sl = 0.0
                qty_total          = qty_remaining = trade_realized_pnl = 0.0
                active_setup       = active_session = ""
                continue   # Skip entry logic on same candle as exit

        # ── ENTRY LOGIC ──────────────────────────────────────────────────────
        if not in_position:
            # Dedup: only evaluate entry once per 15M candle window
            signal_ts = row["Signal_TS"]
            if signal_ts == last_signal_ts:
                continue

            prev_high = float(row["Prev_High"])
            prev_low  = float(row["Prev_Low"])

            # Need valid daily range to calculate Fib levels
            if prev_high <= 0 or prev_low <= 0 or prev_high <= prev_low:
                continue

            is_green_day = bool(row["Prev_Green"])

            # ── Build Fibonacci target list ───────────────────────────────────
            # Per stated logic:
            # GREEN day → Fib 1.0 = prev High | Fib 0.382 = High − 38.2% of range
            # RED day   → Fib 1.0 = prev Low  | Fib 0.382 = Low  + 38.2% of range
            if is_green_day:
                targets = [
                    (prev_high,                   "1D_Fib_1.0"),
                    (float(row["Fib_0382_Bull"]), "1D_Fib_0.382"),
                ]
            else:
                targets = [
                    (prev_low,                    "1D_Fib_1.0"),
                    (float(row["Fib_0382_Bear"]), "1D_Fib_0.382"),
                ]

            # ── Scan both Fib levels for entry signal ─────────────────────────
            for fib_val, setup_name in targets:
                entered = False

                # LONG: C3 ≤ level | C2 > level | C1 > level | EMA bullish
                if c3 <= fib_val and c2 > fib_val and c1 > fib_val and ema_bull:
                    entry_price    = curr_c
                    current_sl     = round(entry_price - SL_USD, 2)
                    pos_type       = "LONG"
                    entered        = True

                # SHORT: C3 ≥ level | C2 < level | C1 < level | EMA bearish
                elif c3 >= fib_val and c2 < fib_val and c1 < fib_val and ema_bear:
                    entry_price    = curr_c
                    current_sl     = round(entry_price + SL_USD, 2)
                    pos_type       = "SHORT"
                    entered        = True

                if entered:
                    qty_total          = TRADE_POWER / entry_price
                    qty_remaining      = qty_total
                    trade_realized_pnl = 0.0
                    tp1_hit            = False   # ← Explicit reset at every entry
                    in_position        = True
                    last_signal_ts     = signal_ts
                    active_setup       = setup_name
                    active_session     = curr_session
                    break   # One trade per candle — no double-entry across Fib levels

    # Append final equity point
    equity_history.append(capital)

    # ── COMPUTE REPORT METRICS ────────────────────────────────────────────────
    total_trades = wins + losses
    net_profit   = capital - STARTING_CAP
    win_rate     = (wins / total_trades * 100) if total_trades > 0 else 0.0
    pf           = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    avg_win      = (gross_win  / wins)   if wins   > 0 else 0.0
    avg_loss     = (gross_loss / losses) if losses > 0 else 0.0
    expectancy   = (gross_win - gross_loss) / total_trades if total_trades > 0 else 0.0
    roi          = (net_profit / STARTING_CAP) * 100

    eq_series    = pd.Series(equity_history)
    max_dd       = calc_max_drawdown(eq_series)
    sharpe       = calc_sharpe(eq_series)

    # ── PRINT REPORT ─────────────────────────────────────────────────────────
    sep  = "=" * 58
    dash = "-" * 58

    print(sep)
    print("   V26 INSTITUTIONAL TRAINING REPORT  (6-MONTH HORIZON)  ")
    print(sep)
    print(f"  Test Period     : {actual_days} calendar days")
    print(f"  Starting Cap    : ${STARTING_CAP:>10,.2f}")
    print(f"  Final Capital   : ${capital:>10,.2f}")
    print(f"  Net Profit      : ${net_profit:>+10,.2f}   ({roi:+.1f}% ROI)")
    print(dash)
    print(f"  Total Trades    : {total_trades}")
    print(f"  Win Rate        : {win_rate:>6.1f}%   ({wins}W / {losses}L)")
    print(f"  Profit Factor   : {pf:>6.2f}")
    print(f"  Expectancy      : ${expectancy:>+9,.2f}  per trade")
    print(f"  Avg Win         : ${avg_win:>+9,.2f}")
    print(f"  Avg Loss        : ${avg_loss:>9,.2f}")
    print(f"  Max Drawdown    : {max_dd:>6.2f}%")
    print(f"  Sharpe Ratio    : {sharpe:>6.2f}")
    print(f"  Risk per Trade  : ${SL_USD:.2f} SL | ${TP1_USD:.2f} TP1 | ${TP2_USD:.2f} TP2")

    print(sep)
    print("   BREAKDOWN BY SETUP (FIBONACCI TARGET)")
    print(sep)
    header = f"  {'Setup':<22} {'W':>4} {'L':>4}  {'WR%':>6}  {'PF':>5}  {'Avg Win':>9}  {'Avg Loss':>9}"
    print(header)
    print(dash)
    for setup, s in setup_stats.items():
        total_s = s["W"] + s["L"]
        wr_s    = (s["W"] / total_s * 100) if total_s > 0 else 0.0
        pf_s    = (s["gross_win"] / s["gross_loss"]) if s["gross_loss"] > 0 else float("inf")
        aw_s    = (s["gross_win"]  / s["W"])   if s["W"]   > 0 else 0.0
        al_s    = (s["gross_loss"] / s["L"])   if s["L"]   > 0 else 0.0
        pf_str  = f"{pf_s:>5.2f}" if pf_s != float("inf") else "  ∞  "
        print(f"  {setup:<22} {s['W']:>4} {s['L']:>4}  {wr_s:>5.1f}%  {pf_str}  ${aw_s:>8,.2f}  ${al_s:>8,.2f}")

    print(sep)
    print("   BREAKDOWN BY SESSION TIME ZONE (UTC)")
    print(sep)
    header = f"  {'Session':<30} {'W':>4} {'L':>4}  {'WR%':>6}  {'PF':>5}"
    print(header)
    print(dash)
    for sess, s in session_stats.items():
        total_s = s["W"] + s["L"]
        wr_s    = (s["W"] / total_s * 100) if total_s > 0 else 0.0
        pf_s    = (s["gross_win"] / s["gross_loss"]) if s["gross_loss"] > 0 else float("inf")
        pf_str  = f"{pf_s:>5.2f}" if pf_s != float("inf") else "  ∞  "
        print(f"  {sess:<30} {s['W']:>4} {s['L']:>4}  {wr_s:>5.1f}%  {pf_str}")

    print(sep)
    best_setup   = max(setup_stats,   key=lambda k: setup_stats[k]["W"]   / max(setup_stats[k]["W"]   + setup_stats[k]["L"],   1))
    best_session = max(session_stats, key=lambda k: session_stats[k]["W"] / max(session_stats[k]["W"] + session_stats[k]["L"], 1))
    print(f"  Best Performing Setup   : {best_setup}")
    print(f"  Best Performing Session : {best_session}")
    print(sep)
    print()


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_training()