"""
Feature Engineering v3 — improved features, proven labeling
- Keeps the richer feature set from v2
- Reverts to original directional labeling (which produced real signal imbalance)
- Removes the "risk-adjusted" label logic that created artificial 50/50 balance
"""

import pandas as pd
import numpy as np

FEATURE_COLS = [
    # Price action
    "body_pct", "upper_wick_pct", "lower_wick_pct", "range_vs_atr",
    "is_bullish", "consecutive_bull", "consecutive_bear",
    # Momentum
    "roc_1", "roc_3", "roc_5", "roc_10", "momentum_alignment",
    # Volatility
    "atr_ratio", "atr_pct", "bb_width", "bb_pos", "bb_squeeze",
    "volatility_regime",
    # Trend
    "close_vs_sma20", "close_vs_sma50", "sma20_slope", "sma50_slope",
    "ema_cross", "trend_strength",
    # Oscillators
    "rsi_7", "rsi_14", "rsi_divergence",
    "stoch_k", "stoch_d", "stoch_signal",
    "macd_hist", "macd_signal_cross",
    # Structure
    "dist_to_high", "dist_to_low", "hl_position", "range_expansion",
    # Session
    "hour", "is_london", "is_newyork", "is_overlap",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]

    # ── 1. Candle structure ───────────────────────────────────────────────────
    body = (close - open_).abs()
    rng  = (high - low).replace(0, np.nan)
    df["body_pct"]       = body / rng
    df["upper_wick_pct"] = (high - df[["close","open"]].max(axis=1)) / rng
    df["lower_wick_pct"] = (df[["close","open"]].min(axis=1) - low)  / rng
    df["is_bullish"]     = (close > open_).astype(int)

    streak_id = (df["is_bullish"] != df["is_bullish"].shift()).cumsum()
    streaks   = df.groupby(streak_id)["is_bullish"].cumcount() + 1
    df["consecutive_bull"] = np.where(df["is_bullish"] == 1, streaks.clip(upper=5), 0)
    df["consecutive_bear"] = np.where(df["is_bullish"] == 0, streaks.clip(upper=5), 0)

    # ── 2. ATR ────────────────────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]          = tr.ewm(span=14, adjust=False).mean()
    df["atr_50"]          = tr.ewm(span=50, adjust=False).mean()
    df["atr_ratio"]       = df["atr_14"] / df["atr_50"].replace(0, np.nan)
    df["atr_pct"]         = df["atr_14"] / close.replace(0, np.nan)
    df["range_vs_atr"]    = rng / df["atr_14"].replace(0, np.nan)
    df["volatility_regime"] = (df["atr_14"] > df["atr_50"]).astype(int)

    # ── 3. Momentum ───────────────────────────────────────────────────────────
    for n in [1, 3, 5, 10]:
        df[f"roc_{n}"] = close.pct_change(n)
    df["momentum_alignment"] = (np.sign(df["roc_3"]) == np.sign(df["roc_5"])).astype(int)

    # ── 4. Moving averages ────────────────────────────────────────────────────
    for p in [10, 20, 50]:
        df[f"sma_{p}"] = close.rolling(p, min_periods=p).mean()
        df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

    df["close_vs_sma20"] = (close - df["sma_20"]) / df["atr_14"].replace(0, np.nan)
    df["close_vs_sma50"] = (close - df["sma_50"]) / df["atr_14"].replace(0, np.nan)
    df["sma20_slope"]    = df["sma_20"].diff(3) / df["atr_14"].replace(0, np.nan)
    df["sma50_slope"]    = df["sma_50"].diff(5) / df["atr_14"].replace(0, np.nan)
    df["ema_cross"]      = np.sign(df["ema_10"] - df["ema_20"]).astype(int)
    df["trend_strength"] = (df["close_vs_sma20"].abs() + df["close_vs_sma50"].abs()) / 2

    # ── 5. Bollinger Bands ────────────────────────────────────────────────────
    std20       = close.rolling(20, min_periods=20).std()
    upper_bb    = df["sma_20"] + 2 * std20
    lower_bb    = df["sma_20"] - 2 * std20
    bb_rng      = (upper_bb - lower_bb).replace(0, np.nan)
    df["bb_pos"]    = (close - lower_bb) / bb_rng
    df["bb_width"]  = bb_rng / close.replace(0, np.nan)
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)).astype(int)

    # ── 6. RSI ────────────────────────────────────────────────────────────────
    def _rsi(s, p):
        d    = s.diff()
        gain = d.clip(lower=0).ewm(span=p, adjust=False).mean()
        loss = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
        return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    df["rsi_7"]          = _rsi(close, 7)
    df["rsi_14"]         = _rsi(close, 14)
    df["rsi_divergence"] = (df["rsi_7"] - df["rsi_14"]) / 10.0

    # ── 7. Stochastic ─────────────────────────────────────────────────────────
    lo14 = low.rolling(14,  min_periods=14).min()
    hi14 = high.rolling(14, min_periods=14).max()
    df["stoch_k"]   = 100 * (close - lo14) / (hi14 - lo14).replace(0, np.nan)
    df["stoch_d"]   = df["stoch_k"].rolling(3).mean()
    k_above         = (df["stoch_k"] > df["stoch_d"]).astype(float)
    df["stoch_signal"] = (k_above - k_above.shift()).fillna(0).astype(int)

    # ── 8. MACD ───────────────────────────────────────────────────────────────
    macd_line   = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_sig    = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd_line - macd_sig) / df["atr_14"].replace(0, np.nan)
    hist_pos        = (macd_line > macd_sig).astype(float)
    df["macd_signal_cross"] = (hist_pos - hist_pos.shift()).fillna(0).astype(int)

    # ── 9. Structure ──────────────────────────────────────────────────────────
    hi20 = high.rolling(20, min_periods=5).max()
    lo20 = low.rolling(20,  min_periods=5).min()
    rng20 = (hi20 - lo20).replace(0, np.nan)
    df["dist_to_high"]    = (hi20 - close) / df["atr_14"].replace(0, np.nan)
    df["dist_to_low"]     = (close - lo20) / df["atr_14"].replace(0, np.nan)
    df["hl_position"]     = (close - lo20) / rng20
    df["range_expansion"] = rng / rng.rolling(20, min_periods=5).mean().replace(0, np.nan)

    # ── 10. Session ───────────────────────────────────────────────────────────
    if "datetime" in df.columns:
        hour = pd.to_datetime(df["datetime"], utc=True).dt.hour
    else:
        hour = pd.Series(0, index=df.index)
    df["hour"]       = hour
    df["is_london"]  = ((hour >= 7)  & (hour < 16)).astype(int)
    df["is_newyork"] = ((hour >= 13) & (hour < 21)).astype(int)
    df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)

    return df


def create_labels(df: pd.DataFrame, forward_bars: int = 5,
                  atr_multiplier: float = 0.5) -> pd.DataFrame:
    """
    Original proven labeling — directional with ATR threshold.

    A bar is labelled BUY(1) if the max close over the next N bars
    exceeds entry by atr_multiplier × ATR.
    A bar is labelled SELL(-1) if the min close drops by the same amount.
    If both or neither, label HOLD(0).

    This preserves the natural market imbalance (more trending bars than
    reversals) which gives the model a learnable signal.
    """
    df    = df.copy()
    close = df["close"].values
    atr   = df["atr_14"].values
    n     = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - forward_bars):
        thresh = atr[i] * atr_multiplier
        if thresh == 0 or np.isnan(thresh):
            continue
        future_close = df["close"].values[i+1 : i+forward_bars+1]
        max_move = np.max(future_close) - close[i]
        min_move = close[i] - np.min(future_close)

        if max_move > thresh and max_move >= min_move:
            labels[i] = 1
        elif min_move > thresh and min_move > max_move:
            labels[i] = -1

    df["label"] = labels
    return df