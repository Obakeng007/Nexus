"""
M15 Signal Engine (Primary)
============================
M15 is the primary signal source.  H1 is used only as a trend filter.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from features import add_features
from data_manager import INSTRUMENTS, load_csv, fetch_from_deriv, save_df


def get_m15_data(pair: str, bars: int = 500) -> pd.DataFrame | None:
    df = load_csv(pair, "M15")
    if df is not None and len(df) >= 80:
        return df.tail(bars).reset_index(drop=True)
    df = fetch_from_deriv(pair, "M15", max(bars, 1000))
    if not df.empty:
        save_df(df, pair, "M15")
        return df.tail(bars).reset_index(drop=True)
    return None


def get_h1_data(pair: str, bars: int = 100) -> pd.DataFrame | None:
    df = load_csv(pair, "H1")
    if df is not None and len(df) >= 50:
        return df.tail(bars).reset_index(drop=True)
    return None


def get_h1_bias(pair: str) -> str:
    df = get_h1_data(pair, 100)
    if df is None or len(df) < 55:
        return "NEUTRAL"
    df_f = add_features(df)
    last = df_f.iloc[-1]
    ema20  = float(last.get("ema_20", last["close"]))
    ema50  = float(last.get("ema_50", last["close"]))
    close  = float(last["close"])
    slope20= float(last.get("sma20_slope", 0))
    bull_pts = sum([close > ema20, close > ema50, ema20 > ema50, slope20 > 0])
    if bull_pts >= 3: return "BULL"
    if bull_pts <= 1: return "BEAR"
    return "NEUTRAL"


def _swing_low(lows, lookback=12):
    return float(np.min(lows[-lookback:]))

def _swing_high(highs, lookback=12):
    return float(np.max(highs[-lookback:]))

def _digits(price):
    if price > 500: return 2
    if price > 10:  return 3
    return 5

def _pip_mult(pair):
    if "XAU" in pair: return 100
    if "JPY" in pair: return 100
    return 10000


def _evaluate_buy(close, open_, high_, low_, atr,
                  rsi, stoch_k, stoch_d, ema10, ema20,
                  macd_h, prev_macd_h, bb_pos, ema_x, body_pct,
                  prev_rsi, prev_stoch_k, h1_bias):
    ok, wait = [], []
    if h1_bias == "BEAR":
        wait.append("H1 trend BEARISH — no BUY")
        return ok, wait
    if close > ema20:
        ok.append(f"Price above EMA20 ({close:.5f})")
    else:
        wait.append("Price below EMA20")
    if 35 <= rsi <= 62:
        ok.append(f"RSI buy zone ({rsi:.1f})")
    elif rsi > 68:
        wait.append(f"RSI overbought ({rsi:.1f})")
    if rsi > prev_rsi and prev_rsi < 50:
        ok.append(f"RSI rising from dip")
    if stoch_k < 65 and stoch_k > prev_stoch_k and stoch_k > stoch_d:
        ok.append(f"Stoch %K crossing up ({stoch_k:.0f})")
    elif stoch_k > 80:
        wait.append(f"Stoch overbought ({stoch_k:.0f})")
    if macd_h > 0 and prev_macd_h <= 0:
        ok.append("MACD just turned bullish ★")
    elif macd_h > 0:
        ok.append("MACD positive")
    else:
        wait.append("MACD negative")
    if close > open_ and body_pct >= 0.35:
        ok.append(f"Bullish candle ({body_pct:.0%} body)")
    else:
        wait.append("No bullish candle")
    if bb_pos < 0.80:
        ok.append(f"BB room to move ({bb_pos:.2f})")
    else:
        wait.append(f"Near upper BB")
    if ema_x == 1:
        ok.append("EMA10 > EMA20 bullish stack")
    return ok, wait


def _evaluate_sell(close, open_, high_, low_, atr,
                   rsi, stoch_k, stoch_d, ema10, ema20,
                   macd_h, prev_macd_h, bb_pos, ema_x, body_pct,
                   prev_rsi, prev_stoch_k, h1_bias):
    ok, wait = [], []
    if h1_bias == "BULL":
        wait.append("H1 trend BULLISH — no SELL")
        return ok, wait
    if close < ema20:
        ok.append(f"Price below EMA20 ({close:.5f})")
    else:
        wait.append("Price above EMA20")
    if 38 <= rsi <= 65:
        ok.append(f"RSI sell zone ({rsi:.1f})")
    elif rsi < 32:
        wait.append(f"RSI oversold ({rsi:.1f})")
    if rsi < prev_rsi and prev_rsi > 50:
        ok.append(f"RSI falling from high")
    if stoch_k > 35 and stoch_k < prev_stoch_k and stoch_k < stoch_d:
        ok.append(f"Stoch %K crossing down ({stoch_k:.0f})")
    elif stoch_k < 20:
        wait.append(f"Stoch oversold ({stoch_k:.0f})")
    if macd_h < 0 and prev_macd_h >= 0:
        ok.append("MACD just turned bearish ★")
    elif macd_h < 0:
        ok.append("MACD negative")
    else:
        wait.append("MACD positive")
    if close < open_ and body_pct >= 0.35:
        ok.append(f"Bearish candle ({body_pct:.0%} body)")
    else:
        wait.append("No bearish candle")
    if bb_pos > 0.20:
        ok.append(f"BB room to move ({bb_pos:.2f})")
    else:
        wait.append(f"Near lower BB")
    if ema_x == -1:
        ok.append("EMA10 < EMA20 bearish stack")
    return ok, wait


def generate_m15_signal(pair: str, df_m15=None, h1_bias=None) -> dict:
    if df_m15 is None:
        df_m15 = get_m15_data(pair, 300)
    if df_m15 is None or len(df_m15) < 60:
        return _no_signal(pair, "Insufficient M15 data")
    if h1_bias is None:
        h1_bias = get_h1_bias(pair)

    df    = add_features(df_m15.copy())
    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) > 1 else last

    close   = float(last["close"])
    open_   = float(last["open"])
    high_   = float(last["high"])
    low_    = float(last["low"])
    atr     = float(last.get("atr_14", close * 0.0005))
    rsi     = float(last.get("rsi_14", 50))
    stoch_k = float(last.get("stoch_k", 50))
    stoch_d = float(last.get("stoch_d", 50))
    ema10   = float(last.get("ema_10", close))
    ema20   = float(last.get("ema_20", close))
    macd_h  = float(last.get("macd_hist", 0))
    bb_pos  = float(last.get("bb_pos", 0.5))
    ema_x   = int(last.get("ema_cross", 0))
    body_pct= float(last.get("body_pct", 0.4))
    prev_rsi     = float(prev.get("rsi_14", rsi))
    prev_stoch_k = float(prev.get("stoch_k", stoch_k))
    prev_macd_h  = float(prev.get("macd_hist", macd_h))

    highs  = df["high"].values
    lows   = df["low"].values
    digits = _digits(close)
    cfg    = INSTRUMENTS.get(pair, {})
    spread = cfg.get("spread", close * 0.00008)

    buy_ok, buy_wait   = _evaluate_buy(close, open_, high_, low_, atr,
                                        rsi, stoch_k, stoch_d, ema10, ema20,
                                        macd_h, prev_macd_h, bb_pos, ema_x, body_pct,
                                        prev_rsi, prev_stoch_k, h1_bias)
    sell_ok, sell_wait = _evaluate_sell(close, open_, high_, low_, atr,
                                         rsi, stoch_k, stoch_d, ema10, ema20,
                                         macd_h, prev_macd_h, bb_pos, ema_x, body_pct,
                                         prev_rsi, prev_stoch_k, h1_bias)

    if len(buy_ok) >= len(sell_ok) and len(buy_ok) >= 3:
        direction  = "BUY"
        conditions = buy_ok
        wait_reasons = buy_wait
    elif len(sell_ok) > len(buy_ok) and len(sell_ok) >= 3:
        direction  = "SELL"
        conditions = sell_ok
        wait_reasons = sell_wait
    else:
        better = buy_wait if len(buy_ok) >= len(sell_ok) else sell_wait
        return _no_signal(pair, " · ".join(better[:2]) or "No M15 setup")

    n     = len(conditions)
    grade = "A" if n >= 5 else "B" if n == 4 else "C"

    if direction == "BUY":
        entry  = round(close + spread, digits)
        sl     = round(min(_swing_low(lows, 10) - atr * 0.3, entry - atr * 1.2), digits)
        risk   = entry - sl
        tp     = round(entry + risk * 2.0, digits)
    else:
        entry  = round(close - spread, digits)
        sl     = round(max(_swing_high(highs, 10) + atr * 0.3, entry + atr * 1.2), digits)
        risk   = sl - entry
        tp     = round(entry - risk * 2.0, digits)

    rr = abs(tp - entry) / risk if risk > 0 else 0

    m15_dict = {
        "available":     True,
        "grade":         grade,
        "entry_type":    "immediate",
        "m15_entry":     entry,
        "m15_sl":        sl,
        "m15_tp":        tp,
        "rr_ratio":      round(rr, 2),
        "conditions":    conditions,
        "wait_reason":   "",
        "m15_rsi":       round(rsi, 1),
        "m15_stoch_k":   round(stoch_k, 1),
        "m15_stoch_d":   round(stoch_d, 1),
        "m15_ema20":     round(ema20, digits),
        "m15_bb_pos":    round(bb_pos, 3),
        "m15_macd_hist": round(macd_h, 6),
        "m15_ema_cross": ema_x,
        "m15_atr":       round(atr, digits),
    }

    return {
        "pair":          pair,
        "instrument":    pair,
        "signal":        direction,
        "grade":         grade,
        "source":        "M15",
        "entry":         entry,
        "stop_loss":     sl,
        "take_profit":   tp,
        "rr_ratio":      round(rr, 2),
        "risk_pips":     round(risk * _pip_mult(pair), 1),
        "conditions":    conditions,
        "wait_reason":   "",
        "h1_bias":       h1_bias,
        "current_price": close,
        "m15_rsi":       round(rsi, 1),
        "m15_stoch_k":   round(stoch_k, 1),
        "m15_stoch_d":   round(stoch_d, 1),
        "m15_ema20":     round(ema20, digits),
        "m15_macd_hist": round(macd_h, 6),
        "m15_bb_pos":    round(bb_pos, 3),
        "m15_atr":       round(atr, digits),
        "m15_ema_cross": ema_x,
        "available":     True,
        "m15":           m15_dict,
        # Keep H1 compatibility fields
        "confidence":    round(min(50 + n * 8, 95), 1),
        "model_trained": False,
        "entry_grade":   grade,
        "entry_type":    "M15 primary",
    }


def analyse_m15(df_m15, direction: str) -> dict:
    """Legacy wrapper."""
    if df_m15 is None or len(df_m15) < 60:
        return _no_m15_legacy(direction)
    df    = add_features(df_m15.copy())
    last  = df.iloc[-1]
    prev  = df.iloc[-2] if len(df) > 1 else last
    close   = float(last["close"]); open_=float(last["open"])
    high_=float(last["high"]); low_=float(last["low"])
    atr     = float(last.get("atr_14", close * 0.0005))
    rsi     = float(last.get("rsi_14", 50))
    stoch_k = float(last.get("stoch_k", 50))
    stoch_d = float(last.get("stoch_d", 50))
    ema10   = float(last.get("ema_10", close))
    ema20   = float(last.get("ema_20", close))
    macd_h  = float(last.get("macd_hist", 0))
    bb_pos  = float(last.get("bb_pos", 0.5))
    ema_x   = int(last.get("ema_cross", 0))
    body_pct= float(last.get("body_pct", 0.4))
    prev_rsi     = float(prev.get("rsi_14", rsi))
    prev_stoch_k = float(prev.get("stoch_k", stoch_k))
    prev_macd_h  = float(prev.get("macd_hist", macd_h))
    highs  = df["high"].values;  lows = df["low"].values
    digits = _digits(close)
    if direction == "BUY":
        ok, wait = _evaluate_buy(close,open_,high_,low_,atr,rsi,stoch_k,stoch_d,ema10,ema20,
                                  macd_h,prev_macd_h,bb_pos,ema_x,body_pct,prev_rsi,prev_stoch_k,"NEUTRAL")
        entry=round(close,digits); sl=round(min(_swing_low(lows,10)-atr*0.3,entry-atr*1.2),digits)
        risk=entry-sl; tp=round(entry+risk*2.0,digits)
    else:
        ok, wait = _evaluate_sell(close,open_,high_,low_,atr,rsi,stoch_k,stoch_d,ema10,ema20,
                                   macd_h,prev_macd_h,bb_pos,ema_x,body_pct,prev_rsi,prev_stoch_k,"NEUTRAL")
        entry=round(close,digits); sl=round(max(_swing_high(highs,10)+atr*0.3,entry+atr*1.2),digits)
        risk=sl-entry; tp=round(entry-risk*2.0,digits)
    n=len(ok); grade="A" if n>=5 else "B" if n==4 else "C" if n==3 else "W"
    rr=abs(tp-entry)/risk if risk>0 else 0
    return {
        "available":True,"direction":direction,"entry_type":"immediate" if grade!="W" else "wait",
        "grade":grade,"m15_entry":entry,"m15_sl":sl,"m15_tp":tp,"m15_atr":round(atr,digits),
        "rr_ratio":round(rr,2),"conditions":ok,"wait_reason":" · ".join(wait[:3]),
        "m15_rsi":round(rsi,1),"m15_stoch_k":round(stoch_k,1),"m15_stoch_d":round(stoch_d,1),
        "m15_ema20":round(ema20,digits),"m15_bb_pos":round(bb_pos,3),"m15_macd_hist":round(macd_h,6),
        "m15_ema_cross":ema_x,"vol_regime":int(last.get("volatility_regime",0)),
        "bb_squeeze":int(last.get("bb_squeeze",0)),
    }


def _no_signal(pair, reason):
    return {"pair":pair,"instrument":pair,"signal":"HOLD","grade":"W","source":"M15",
            "entry":None,"stop_loss":None,"take_profit":None,"rr_ratio":None,"risk_pips":None,
            "conditions":[],"wait_reason":reason,"h1_bias":None,"available":False,
            "confidence":0,"model_trained":False,"entry_grade":"W","entry_type":"Wait",
            "current_price":None,"m15":{"available":False,"wait_reason":reason}}


def _no_m15_legacy(direction):
    return {"available":False,"direction":direction,"entry_type":"h1_only","grade":"N/A",
            "m15_entry":None,"m15_sl":None,"m15_tp":None,"m15_atr":None,"rr_ratio":None,
            "conditions":["M15 data not available"],"wait_reason":"",
            "m15_rsi":None,"m15_stoch_k":None,"m15_stoch_d":None,"m15_ema20":None,
            "m15_bb_pos":None,"m15_macd_hist":None,"m15_ema_cross":None,"vol_regime":None,"bb_squeeze":None}


def get_m15_chart_data(pair: str, bars: int = 120) -> dict:
    df = get_m15_data(pair, bars)
    if df is None or df.empty:
        return {"available": False}
    df_ind = add_features(df.copy())
    def sl(col):
        if col not in df_ind.columns: return []
        return [None if (v is None or (isinstance(v,float) and (np.isnan(v) or np.isinf(v))))
                else round(float(v),6) for v in df_ind[col]]
    dates = [str(d)[:16] for d in df_ind["datetime"]] if "datetime" in df_ind.columns else []
    return {"available":True,"dates":dates,"open":sl("open"),"high":sl("high"),"low":sl("low"),
            "close":sl("close"),"ema10":sl("ema_10"),"ema20":sl("ema_20"),"rsi":sl("rsi_14"),
            "stoch_k":sl("stoch_k"),"stoch_d":sl("stoch_d"),"macd_hist":sl("macd_hist"),"bb_pos":sl("bb_pos")}