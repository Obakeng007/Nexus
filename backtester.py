"""
Backtester v2 — uses features.py + ml_engine signals
Labels: 1=BUY, -1=SELL, 0=HOLD
"""
import numpy as np
import pandas as pd
from features import add_features
from ml_engine import load_model, generate_signals_series


def run_backtest(df: pd.DataFrame, pair: str,
                 initial_capital: float = 10000.0,
                 risk_per_trade: float = 0.02,
                 atr_sl_multiplier: float = 2.0,
                 atr_tp_multiplier: float = 3.0,
                 commission: float = 0.0001,
                 slippage: float = 0.0001,
                 confidence_threshold: float = 55.0) -> dict:
    """Run event-driven backtest. Returns full results dict."""

    df_feat = add_features(df.copy())
    model   = load_model(pair)

    if model:
        signals = generate_signals_series(df_feat, model, confidence_threshold)
    else:
        signals = _rule_signals(df_feat)

    trades, equity, capital = [], [initial_capital], initial_capital
    position = None  # dict when open

    for i in range(1, len(df_feat)):
        row    = df_feat.iloc[i]
        close  = float(row["close"])
        high   = float(row["high"])
        low    = float(row["low"])
        atr    = float(row.get("atr_14", close * 0.001))

        # ── Check open position for SL/TP ──
        if position:
            hit_sl = hit_tp = False
            if position["direction"] == 1:   # BUY
                if low  <= position["sl"]:    hit_sl = True; exit_px = position["sl"]
                elif high >= position["tp"]:  hit_tp = True; exit_px = position["tp"]
            else:                             # SELL
                if high >= position["sl"]:    hit_sl = True; exit_px = position["sl"]
                elif low  <= position["tp"]:  hit_tp = True; exit_px = position["tp"]

            if hit_sl or hit_tp:
                if hit_sl:
                    exit_px *= (1 - slippage) if position["direction"] == 1 else (1 + slippage)
                pnl = position["size"] * (exit_px - position["entry"]) * position["direction"]
                pnl -= commission * position["size"] * exit_px
                capital += pnl
                trades.append({
                    "entry_date": str(df_feat.index[position["ei"]]),
                    "exit_date":  str(df_feat.index[i]),
                    "direction":  "BUY" if position["direction"] == 1 else "SELL",
                    "entry_price": round(position["entry"], 6),
                    "exit_price":  round(exit_px, 6),
                    "sl":          round(position["sl"], 6),
                    "tp":          round(position["tp"], 6),
                    "size":        round(position["size"], 4),
                    "pnl":         round(pnl, 2),
                    "pnl_pct":     round(pnl / initial_capital * 100, 3),
                    "result":      "WIN" if hit_tp else "LOSS",
                    "bars_held":   i - position["ei"],
                })
                position = None

        # ── Open new position from previous bar's signal ──
        if position is None and i - 1 < len(signals):
            sig = int(signals.iloc[i - 1])
            if sig in (1, -1):
                slip_f = (1 + slippage) if sig == 1 else (1 - slippage)
                entry  = close * slip_f
                sl_dist = atr_sl_multiplier * atr
                size   = (capital * risk_per_trade) / sl_dist if sl_dist > 0 else 0
                if size > 0:
                    sl = entry - sl_dist if sig == 1 else entry + sl_dist
                    tp = entry + atr_tp_multiplier * atr if sig == 1 else entry - atr_tp_multiplier * atr
                    capital -= commission * size * entry
                    position = {"direction": sig, "entry": entry, "sl": sl,
                                "tp": tp, "size": size, "ei": i}

        equity.append(capital)

    # Close any open position at end
    if position:
        exit_px = float(df_feat["close"].iloc[-1])
        pnl = position["size"] * (exit_px - position["entry"]) * position["direction"]
        pnl -= commission * position["size"] * exit_px
        capital += pnl
        trades.append({
            "entry_date": str(df_feat.index[position["ei"]]),
            "exit_date":  str(df_feat.index[-1]),
            "direction":  "BUY" if position["direction"] == 1 else "SELL",
            "entry_price": round(position["entry"], 6),
            "exit_price":  round(exit_px, 6),
            "sl": round(position["sl"], 6), "tp": round(position["tp"], 6),
            "size": round(position["size"], 4), "pnl": round(pnl, 2),
            "pnl_pct": round(pnl / initial_capital * 100, 3),
            "result": "WIN" if pnl > 0 else "LOSS",
            "bars_held": len(df_feat) - position["ei"],
        })
        equity[-1] = capital

    metrics = _compute_metrics(equity, trades, initial_capital)

    # Sample equity curve for chart (≤ 600 points)
    step  = max(1, len(equity) // 600)
    eq_s  = [equity[i] for i in range(0, len(equity), step)]
    idx_s = df_feat.index[::step].tolist()
    dates_s = [str(d)[:10] for d in idx_s[:len(eq_s)]]

    return {
        "metrics":       metrics,
        "trades":        trades[-100:],
        "all_trades":    trades,
        "equity_curve":  eq_s,
        "equity_dates":  dates_s,
        "pair":          pair,
        "instrument":    pair,
        "initial_capital": initial_capital,
        "final_capital":   round(capital, 2),
    }


def _rule_signals(df_feat: pd.DataFrame) -> pd.Series:
    rsi   = df_feat.get("rsi_14", pd.Series(50, index=df_feat.index))
    macd  = df_feat.get("macd_hist", pd.Series(0, index=df_feat.index))
    emac  = df_feat.get("ema_cross", pd.Series(0, index=df_feat.index))
    buy   = (rsi < 35) & (macd > 0) & (emac == 1)
    sell  = (rsi > 65) & (macd < 0) & (emac == -1)
    sigs  = pd.Series(0, index=df_feat.index)
    sigs[buy]  = 1
    sigs[sell] = -1
    return sigs


def _compute_metrics(equity: list, trades: list, initial: float) -> dict:
    arr  = np.array(equity)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)

    total_return = (arr[-1] / initial - 1) * 100

    sharpe  = float(np.sqrt(252) * rets.mean() / rets.std()) if rets.std() > 0 else 0
    down    = rets[rets < 0]
    sortino = float(np.sqrt(252) * rets.mean() / down.std()) if len(down) > 1 and down.std() > 0 else 0

    running_max = np.maximum.accumulate(arr)
    dd          = (arr - running_max) / np.maximum(running_max, 1e-9) * 100
    max_dd      = float(dd.min())
    calmar      = float(-total_return / max_dd) if max_dd < -0.01 else 0

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]

    win_rate = len(wins) / max(len(trades), 1) * 100
    avg_win  = float(np.mean([t["pnl"] for t in wins])) if wins else 0
    avg_loss = float(abs(np.mean([t["pnl"] for t in losses]))) if losses else 0
    pf_num   = sum(t["pnl"] for t in wins)
    pf_den   = abs(sum(t["pnl"] for t in losses))
    pf       = pf_num / pf_den if pf_den > 0 else 0
    exp      = avg_win * (win_rate/100) - avg_loss * (1 - win_rate/100)

    return {
        "total_return":  round(total_return, 2),
        "final_capital": round(arr[-1], 2),
        "sharpe_ratio":  round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown":  round(max_dd, 2),
        "calmar_ratio":  round(calmar, 3),
        "total_trades":  len(trades),
        "win_rate":      round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "avg_win":       round(avg_win, 2),
        "avg_loss":      round(avg_loss, 2),
        "avg_bars_held": round(float(np.mean([t["bars_held"] for t in trades])), 1) if trades else 0,
        "buy_trades":    sum(1 for t in trades if t["direction"] == "BUY"),
        "sell_trades":   sum(1 for t in trades if t["direction"] == "SELL"),
        "expectancy":    round(exp, 2),
    }