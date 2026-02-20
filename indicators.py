"""
Technical Indicators - All computed from scratch using pandas/numpy
Covers: Trend, Momentum, Volatility, Volume indicators
"""
import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(series, period)
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid
    pct_b = (series - lower) / (upper - lower)
    return upper, mid, lower, width, pct_b

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return k, d

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical_price = (high + low + close) / 3
    mean_tp = typical_price.rolling(window=period).mean()
    mean_dev = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (typical_price - mean_tp) / (0.015 * mean_dev.replace(0, np.nan))

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Average Directional Index."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_val = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx_val, plus_di, minus_di

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    tpv = typical_price * volume
    return tpv.cumsum() / volume.cumsum()

def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series):
    """Ichimoku Cloud components."""
    conversion = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conversion + base) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    lagging = close.shift(-26)
    return conversion, base, span_a, span_b, lagging

def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20):
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    mid = (upper + lower) / 2
    return upper, mid, lower

def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 20, multiplier: float = 1.5):
    mid = ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = mid + multiplier * atr_val
    lower = mid - multiplier * atr_val
    return upper, mid, lower

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    pos_sum = positive_flow.rolling(period).sum()
    neg_sum = negative_flow.rolling(period).sum()
    mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfi_ratio))

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators and return enriched DataFrame."""
    result = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Trend Indicators
    for p in [9, 21, 50, 100, 200]:
        result[f'sma_{p}'] = sma(close, p)
        result[f'ema_{p}'] = ema(close, p)

    # MACD
    result['macd'], result['macd_signal'], result['macd_hist'] = macd(close)

    # Bollinger Bands
    result['bb_upper'], result['bb_mid'], result['bb_lower'], result['bb_width'], result['bb_pct'] = bollinger_bands(close)

    # Ichimoku
    result['ichi_conv'], result['ichi_base'], result['ichi_span_a'], result['ichi_span_b'], _ = ichimoku(high, low, close)

    # Donchian Channel
    result['dc_upper'], result['dc_mid'], result['dc_lower'] = donchian_channel(high, low)

    # Keltner Channel
    result['kc_upper'], result['kc_mid'], result['kc_lower'] = keltner_channel(high, low, close)

    # Momentum Indicators
    result['rsi_14'] = rsi(close, 14)
    result['rsi_7'] = rsi(close, 7)
    result['rsi_21'] = rsi(close, 21)
    result['stoch_k'], result['stoch_d'] = stochastic(high, low, close)
    result['williams_r'] = williams_r(high, low, close)
    result['cci'] = cci(high, low, close)
    result['mfi'] = mfi(high, low, close, volume)

    # Rate of change
    for p in [5, 10, 20]:
        result[f'roc_{p}'] = close.pct_change(p) * 100

    # Volatility Indicators
    result['atr'] = atr(high, low, close)
    result['atr_pct'] = result['atr'] / close * 100
    result['adx'], result['plus_di'], result['minus_di'] = adx(high, low, close)

    # Historical volatility
    result['hvol_10'] = close.pct_change().rolling(10).std() * np.sqrt(252) * 100
    result['hvol_20'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # Volume Indicators
    result['obv'] = obv(close, volume)
    result['obv_ema'] = ema(result['obv'], 21)
    result['vwap'] = vwap(high, low, close, volume)
    result['vol_sma'] = sma(volume, 20)
    result['vol_ratio'] = volume / result['vol_sma']

    # Price-derived features
    result['returns_1'] = close.pct_change(1)
    result['returns_5'] = close.pct_change(5)
    result['returns_10'] = close.pct_change(10)
    result['log_returns'] = np.log(close / close.shift(1))

    # Candle patterns
    body = close - df['open']
    candle_range = high - low
    result['body_size'] = body.abs() / candle_range.replace(0, np.nan)
    result['upper_shadow'] = (high - pd.concat([close, df['open']], axis=1).max(axis=1)) / candle_range.replace(0, np.nan)
    result['lower_shadow'] = (pd.concat([close, df['open']], axis=1).min(axis=1) - low) / candle_range.replace(0, np.nan)
    result['is_bullish'] = (body > 0).astype(int)

    # Price position relative to recent range
    result['price_vs_52h'] = (close - low.rolling(52).min()) / (high.rolling(52).max() - low.rolling(52).min()).replace(0, np.nan)
    result['price_vs_sma50'] = (close - result['sma_50']) / result['sma_50']
    result['price_vs_sma200'] = (close - result['sma_200']) / result['sma_200']

    # EMA cross signals
    result['ema_9_21_cross'] = np.where(result['ema_9'] > result['ema_21'], 1, -1)
    result['ema_21_50_cross'] = np.where(result['ema_21'] > result['ema_50'], 1, -1)
    result['ema_50_200_cross'] = np.where(result['ema_50'] > result['ema_200'], 1, -1)

    return result
