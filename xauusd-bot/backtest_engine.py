"""
XAU/USD (Gold Futures) Backtest Engine — FastAPI Backend
=========================================================
V26: THE SNIPER MAXED (59 Days - YFinance Limit)
  · Priority: Previous Day High / Low (1.0) Breakouts ONLY
  · Filter: 5m EMA 25/50 Alignment
  · Risk: STRICT -20pt SL / +20pt TP1 / +60pt TP2
  · Note: Maxed out to 59 days (Yahoo Finance free intraday limit)
"""

import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(title="XAU/USD Backtest Engine", version="2.15.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS: dict = {"status": "idle", "stats": None, "equity_curve": [], "logs": []}
data_lock = threading.Lock()

# ─── STRATEGY CONSTANTS ───────────────────────────────────────────────────────
STARTING_CAP     = 500.0     
MARGIN_PER_TRADE = 200.0     
LEVERAGE         = 50        
TRADE_POWER      = MARGIN_PER_TRADE * LEVERAGE  # $10,000

SL_POINTS        = 20.0      # Hard 20-point stop
TP1_POINTS       = 20.0      # 1:1 Scale-out target
TP2_POINTS       = 60.0      # 1:3 Runner target

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def safe_bool(val) -> bool:
    return False if pd.isna(val) else bool(val)

def fmt_usd(value: float, force_sign: bool = False) -> str:
    sign = "+" if (force_sign and value >= 0) else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"

def calc_sharpe(equity_curve: list[dict], risk_free: float = 0.053) -> str:
    if len(equity_curve) < 2: return "N/A"
    series = pd.Series([e["equity"] for e in equity_curve])
    daily_ret = series.pct_change().dropna()
    std = daily_ret.std()
    if std == 0 or np.isnan(std): return "N/A"
    return f"{(daily_ret.mean() - (risk_free / 252)) / std * np.sqrt(252):.2f}"

def calc_max_drawdown(equity_curve: list[dict]) -> str:
    if len(equity_curve) < 2: return "0.00%"
    series = pd.Series([e["equity"] for e in equity_curve])
    dd = ((series - series.cummax()) / series.cummax()).min() * 100
    return f"{dd:.2f}%" if not np.isnan(dd) else "0.00%"

def make_log(log_type: str, action: str, detail: str, meta: str = "", profit: str = "", ts: str = "") -> dict:
    entry = {"id": ts or datetime.now(timezone.utc).isoformat(), "type": log_type, "action": action, "detail": detail, "meta": meta}
    if profit: entry["profit"] = profit
    return entry

# ─── SIMULATION ───────────────────────────────────────────────────────────────
def run_simulation() -> None:
    with data_lock:
        RESULTS.update({"status": "running", "logs": [], "equity_curve": [], "stats": None})

    logs: list[dict] = []
    equity_curve: list[dict] = []
    
    setup_tracker = {
        "1D_Fib_1.0": {"W": 0, "L": 0}
    }

    try:
        # ── 1. Data fetch (MAXED TO 59 DAYS) ───────────────────────────────────
        ticker   = yf.Ticker("GC=F")
        df       = ticker.history(period="59d", interval="5m")
        df_15m   = ticker.history(period="59d", interval="15m")
        df_1h    = ticker.history(period="59d", interval="1h")

        if any(d.empty for d in [df, df_15m, df_1h]):
            raise ValueError("yfinance returned empty data.")

        for frame in [df, df_15m, df_1h]:
            frame.index = frame.index.tz_localize("UTC") if frame.index.tzinfo is None else frame.index.tz_convert("UTC")

        # ── 2. 15m Base Logic ──────────────────────────────────────────────────
        df_15m['Prev_15m_Close_1'] = df_15m['Close'].shift(1)
        df_15m['Prev_15m_Close_2'] = df_15m['Close'].shift(2)
        df_15m['Prev_15m_Close_3'] = df_15m['Close'].shift(3) 
        df_15m['15m_Timestamp_ID'] = df_15m.index

        # ── 3. Daily (1D) Range Logic ──────────────────────────────────────────
        df_1d = pd.DataFrame({
            "open":  df_1h["Open"].resample("D").first(),
            "high":  df_1h["High"].resample("D").max(),
            "low":   df_1h["Low"].resample("D").min(),
            "close": df_1h["Close"].resample("D").last(),
        }).dropna()

        df_1d["Prev_1D_High"]  = df_1d["high"].shift(1)
        df_1d["Prev_1D_Low"]   = df_1d["low"].shift(1)
        df_1d["Prev_1D_Green"] = df_1d["close"].shift(1) > df_1d["open"].shift(1)
        df_1d["Prev_1D_Red"]   = df_1d["close"].shift(1) < df_1d["open"].shift(1)

        # ── 4. Merge to 5M Master Frame ────────────────────────────────────────
        df['EMA_25'] = df['Close'].ewm(span=25, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        cols_15m = ['Prev_15m_Close_1', 'Prev_15m_Close_2', 'Prev_15m_Close_3', '15m_Timestamp_ID']
        df = df.join(df_15m[cols_15m], how="left").ffill()
        df = df.join(df_1d[["Prev_1D_Green", "Prev_1D_Red", "Prev_1D_High", "Prev_1D_Low"]], how="left").ffill().fillna(False)

        # ── 5. Portfolio State ─────────────────────────────────────────────────
        capital, entry_price, current_sl = STARTING_CAP, 0.0, 0.0
        qty_total, qty_remaining = 0.0, 0.0
        trade_realized_pnl = 0.0
        in_position, pos_type, tp1_hit = False, "", False
        active_setup_name = ""
        wins, losses, gross_win, gross_loss = 0, 0, 0.0, 0.0
        last_date, last_signal_id = None, None

        # ── 6. Main Loop ───────────────────────────────────────────────────────
        for idx, row in df.iterrows():
            curr_c, curr_h, curr_l = float(row["Close"]), float(row["High"]), float(row["Low"])
            c1, c2, c3 = float(row['Prev_15m_Close_1']), float(row['Prev_15m_Close_2']), float(row['Prev_15m_Close_3'])
            ema_25, ema_50 = float(row['EMA_25']), float(row['EMA_50'])
            
            ema_bull = ema_25 > ema_50
            ema_bear = ema_25 < ema_50
            ts = idx.isoformat()

            # ==========================================
            # EXIT LOGIC
            # ==========================================
            if in_position:
                exit_price, reason, exit_triggered = 0.0, "", False

                if pos_type == "LONG":
                    tp1_price = entry_price + TP1_POINTS
                    tp2_price = entry_price + TP2_POINTS

                    if not tp1_hit and curr_h >= tp1_price:
                        exit_qty = qty_total * 0.5
                        pnl = (tp1_price - entry_price) * exit_qty
                        trade_realized_pnl += pnl
                        capital += pnl
                        qty_remaining -= exit_qty
                        tp1_hit = True
                        current_sl = entry_price 
                        logs.append(make_log("PARTIAL", "SCALE OUT (50%)", f"Hit 1:1 TP @ {tp1_price:.2f}", f"Secured: {fmt_usd(pnl, True)} | SL moved to BE", fmt_usd(pnl, True), ts + "_partial"))

                        if curr_h >= tp2_price:
                            pnl2 = (tp2_price - entry_price) * qty_remaining
                            trade_realized_pnl += pnl2
                            capital += pnl2
                            qty_remaining = 0
                            reason, exit_price, exit_triggered = "Runner 1:3 Hit!", tp2_price, True

                    elif tp1_hit and curr_h >= tp2_price:
                        pnl2 = (tp2_price - entry_price) * qty_remaining
                        trade_realized_pnl += pnl2
                        capital += pnl2
                        qty_remaining = 0
                        reason, exit_price, exit_triggered = "Runner 1:3 Hit!", tp2_price, True

                    elif curr_l <= current_sl:
                        pnl = (current_sl - entry_price) * qty_remaining
                        trade_realized_pnl += pnl
                        capital += pnl
                        qty_remaining = 0
                        reason = "BE Stop Hit" if current_sl == entry_price else "20pt Hard SL Hit"
                        exit_price, exit_triggered = current_sl, True

                elif pos_type == "SHORT":
                    tp1_price = entry_price - TP1_POINTS
                    tp2_price = entry_price - TP2_POINTS

                    if not tp1_hit and curr_l <= tp1_price:
                        exit_qty = qty_total * 0.5
                        pnl = (entry_price - tp1_price) * exit_qty
                        trade_realized_pnl += pnl
                        capital += pnl
                        qty_remaining -= exit_qty
                        tp1_hit = True
                        current_sl = entry_price 
                        logs.append(make_log("PARTIAL", "SCALE OUT (50%)", f"Hit 1:1 TP @ {tp1_price:.2f}", f"Secured: {fmt_usd(pnl, True)} | SL moved to BE", fmt_usd(pnl, True), ts + "_partial"))

                        if curr_l <= tp2_price:
                            pnl2 = (entry_price - tp2_price) * qty_remaining
                            trade_realized_pnl += pnl2
                            capital += pnl2
                            qty_remaining = 0
                            reason, exit_price, exit_triggered = "Runner 1:3 Hit!", tp2_price, True

                    elif tp1_hit and curr_l <= tp2_price:
                        pnl2 = (entry_price - tp2_price) * qty_remaining
                        trade_realized_pnl += pnl2
                        capital += pnl2
                        qty_remaining = 0
                        reason, exit_price, exit_triggered = "Runner 1:3 Hit!", tp2_price, True

                    elif curr_h >= current_sl:
                        pnl = (entry_price - current_sl) * qty_remaining
                        trade_realized_pnl += pnl
                        capital += pnl
                        qty_remaining = 0
                        reason = "BE Stop Hit" if current_sl == entry_price else "20pt Hard SL Hit"
                        exit_price, exit_triggered = current_sl, True
                        
                if exit_triggered:
                    if trade_realized_pnl > 0: 
                        wins += 1
                        gross_win += trade_realized_pnl
                        setup_tracker[active_setup_name]["W"] += 1
                    else: 
                        losses += 1
                        gross_loss += abs(trade_realized_pnl)
                        setup_tracker[active_setup_name]["L"] += 1

                    logs.append(make_log("EXIT", f"CLOSE {pos_type}", f"{reason} @ {exit_price:.2f}", f"Total Trade P&L: {fmt_usd(trade_realized_pnl, True)}", fmt_usd(trade_realized_pnl, True), ts + "_exit"))
                    in_position, pos_type, tp1_hit = False, "", False
                    active_setup_name = ""
                    continue

            # ==========================================
            # ENTRY LOGIC
            # ==========================================
            if not in_position:
                signal_id = row['15m_Timestamp_ID']
                if signal_id == last_signal_id: continue 

                long_sl  = curr_c - SL_POINTS
                short_sl = curr_c + SL_POINTS
                setup_label = ""

                # HIERARCHY: DAILY (1D) FIB 1.0 ONLY
                prev_day_green = safe_bool(row["Prev_1D_Green"])
                prev_day_red   = safe_bool(row["Prev_1D_Red"])
                day_high, day_low = float(row["Prev_1D_High"]), float(row["Prev_1D_Low"])

                if day_high > 0 and day_low > 0:
                    
                    if prev_day_green:
                        fib_1 = day_high
                    else:
                        fib_1 = day_low
                        
                    daily_fib_targets = [(fib_1, "1D_Fib_1.0")]

                    for fib_val, s_name in daily_fib_targets:
                        if c3 <= fib_val and c2 > fib_val and c1 > fib_val and ema_bull:
                            entry_price, current_sl, pos_type = curr_c, long_sl, "LONG"
                            detail, setup_label = f"1D Fib Breakout UP @ {fib_val:.2f}", s_name
                            break
                        elif c3 >= fib_val and c2 < fib_val and c1 < fib_val and ema_bear:
                            entry_price, current_sl, pos_type = curr_c, short_sl, "SHORT"
                            detail, setup_label = f"1D Fib Breakdown DOWN @ {fib_val:.2f}", s_name
                            break

                # EXECUTE ENTRY
                if pos_type:
                    in_position, last_signal_id = True, signal_id
                    active_setup_name = setup_label
                    qty_total = TRADE_POWER / curr_c
                    qty_remaining = qty_total
                    trade_realized_pnl = 0.0
                    logs.append(make_log("ENTRY", f"{pos_type} XAU", detail, f"Entry: {entry_price:.2f} | Risk: {SL_POINTS}pts | Target: {TP2_POINTS}pts", "", ts + "_entry"))

            # ── Daily equity snapshot
            if (idx.hour == 23 and idx.minute >= 50) or (last_date and idx.date() != last_date):
                equity_curve.append({"date": str(idx.date()), "equity": round(capital, 4)})
            last_date = idx.date()

        if not equity_curve: equity_curve.append({"date": "Start", "equity": STARTING_CAP})
        total_trades = wins + losses
        
        # --- PRINT PERFORMANCE REPORT TO TERMINAL ---
        print("\n" + "="*45)
        print(" V26 SMC: 59-DAY MAX STRESS TEST ")
        print("="*45)
        for setup, stats_dict in setup_tracker.items():
            total_t = stats_dict['W'] + stats_dict['L']
            wr = (stats_dict['W']/total_t*100) if total_t > 0 else 0
            print(f" {setup.ljust(15)} : {stats_dict['W']}W / {stats_dict['L']}L  ({wr:.1f}% Win Rate)")
        print("="*45 + "\n")

        with data_lock:
            RESULTS["stats"] = {
                "paper_equity":  fmt_usd(capital), "net_profit": fmt_usd(capital - STARTING_CAP, True),
                "win_rate":      f"{wins / total_trades * 100:.1f}%" if total_trades > 0 else "0.0%",
                "wl_ratio":      f"{wins}W / {losses}L", "max_drawdown": calc_max_drawdown(equity_curve),
                "profit_factor": f"{gross_win / gross_loss:.2f}" if gross_loss > 0 else "0.00",
                "total_trades":  str(total_trades),
                "expectancy":    fmt_usd((gross_win - gross_loss) / total_trades, True) if total_trades > 0 else "$0.00",
                "avg_win":       fmt_usd(gross_win / wins, True) if wins > 0 else "$0.00",
                "avg_loss":      fmt_usd(-gross_loss / losses, True) if losses > 0 else "$0.00",
                "sharpe":        calc_sharpe(equity_curve),
            }
            RESULTS["equity_curve"], RESULTS["logs"], RESULTS["status"] = equity_curve, list(reversed(logs)), "completed"

    except Exception as exc:
        with data_lock:
            RESULTS["status"] = "error"
            RESULTS["stats"]  = {k: "N/A" for k in RESULTS.get("stats", {})}
            RESULTS["logs"]   = [make_log("ERROR", "API CRASH", str(exc), "YFinance rejected the data pull.")]
        raise  

@app.post("/trigger")
async def trigger_sim():
    with data_lock:
        if RESULTS["status"] == "running": raise HTTPException(status_code=409, detail="Running.")
    threading.Thread(target=run_simulation, daemon=True).start()
    return {"message": "Started."}

@app.get("/data")
async def get_results():
    with data_lock: return dict(RESULTS)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")