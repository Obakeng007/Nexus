"""
Technical indicators module for NEXUS Trading System.
Provides 40+ technical indicators for feature engineering.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Series of price data
        period: RSI period (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal line, and Histogram.
    
    Args:
        prices: Series of price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: pd.Series, 
                             period: int = 20, 
                             std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Series of price data
        period: Moving average period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        Series of ATR values
    """
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_stochastic(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series, 
                        k_period: int = 14, 
                        d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
    
    Returns:
        Tuple of (%K, %D)
    """
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    
    # %K
    k = 100 * ((close - low_min) / (high_max - low_min))
    
    # %D (3-period SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_williams_r(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series, 
                        period: int = 14) -> pd.Series:
    """
    Calculate Williams %R.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
    
    Returns:
        Series of Williams %R values
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr

def calculate_cci(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: CCI period
    
    Returns:
        Series of CCI values
    """
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci

def calculate_adx(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
    
    Returns:
        Series of ADX values
    """
    # Calculate +DM and -DM
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
    minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        close: Close prices
        volume: Volume data
    
    Returns:
        Series of OBV values
    """
    obv = np.where(close > close.shift(1), volume, 
                   np.where(close < close.shift(1), -volume, 0))
    return pd.Series(obv, index=close.index).cumsum()

def calculate_ichimoku(high: pd.Series, 
                      low: pd.Series, 
                      close: pd.Series,
                      tenkan_period: int = 9,
                      kijun_period: int = 26,
                      senkou_b_period: int = 52) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate Ichimoku Cloud indicators.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen period
        kijun_period: Kijun-sen period
        senkou_b_period: Senkou Span B period
    
    Returns:
        Tuple of (Tenkan, Kijun, Senkou A, Senkou B, Chikou)
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_b_high = high.rolling(window=senkou_b_period).max()
    senkou_b_low = low.rolling(window=senkou_b_period).min()
    senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    chikou = close.shift(-kijun_period)
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calculate_donchian(high: pd.Series, 
                      low: pd.Series, 
                      period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channel.
    
    Args:
        high: High prices
        low: Low prices
        period: Channel period
    
    Returns:
        Tuple of (Upper channel, Middle channel, Lower channel)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, middle, lower

def calculate_keltner(high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series,
                     ema_period: int = 20,
                     atr_period: int = 10,
                     multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: EMA period
        atr_period: ATR period
        multiplier: ATR multiplier
    
    Returns:
        Tuple of (Upper channel, Middle line, Lower channel)
    """
    ema = close.ewm(span=ema_period, adjust=False).mean()
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    
    return upper, ema, lower

def calculate_mfi(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series, 
                 volume: pd.Series, 
                 period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
        period: MFI period
    
    Returns:
        Series of MFI values
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    
    positive_mf = pd.Series(positive_flow, index=close.index).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow, index=close.index).rolling(window=period).sum()
    
    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        prices: Price series
        period: EMA period
    
    Returns:
        Series of EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        prices: Price series
        period: SMA period
    
    Returns:
        Series of SMA values
    """
    return prices.rolling(window=period).mean()

def calculate_vwap(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series, 
                  volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data
    
    Returns:
        Series of VWAP values
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate historical volatility.
    
    Args:
        close: Close prices
        period: Lookback period
    
    Returns:
        Series of volatility values (annualized)
    """
    returns = close.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)
    return volatility

def calculate_volume_profile(volume: pd.Series, close: pd.Series, bins: int = 10) -> pd.Series:
    """
    Calculate volume profile (price levels with high volume).
    
    Args:
        volume: Volume data
        close: Close prices
        bins: Number of price bins
    
    Returns:
        Series of volume by price level
    """
    price_bins = pd.cut(close, bins=bins)
    volume_profile = volume.groupby(price_bins).sum()
    return volume_profile

def calculate_pivot_points(high: pd.Series, 
                          low: pd.Series, 
                          close: pd.Series) -> Dict[str, pd.Series]:
    """
    Calculate classic pivot points.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
    
    Returns:
        Dictionary of pivot levels
    """
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    return {
        'pivot': pivot,
        'r1': r1,
        's1': s1,
        'r2': r2,
        's2': s2
    }

def calculate_fibonacci_levels(high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
    """
    Calculate Fibonacci retracement levels.
    
    Args:
        high: High prices
        low: Low prices
    
    Returns:
        Dictionary of Fibonacci levels
    """
    diff = high - low
    return {
        'level_0': low,
        'level_236': low + 0.236 * diff,
        'level_382': low + 0.382 * diff,
        'level_5': low + 0.5 * diff,
        'level_618': low + 0.618 * diff,
        'level_786': low + 0.786 * diff,
        'level_1': high
    }

def calculate_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all Ichimoku Cloud components.
    
    Args:
        df: DataFrame with OHLC data
    
    Returns:
        DataFrame with Ichimoku components
    """
    result = df.copy()
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(
        result['high'], result['low'], result['close']
    )
    
    result['ichimoku_tenkan'] = tenkan
    result['ichimoku_kijun'] = kijun
    result['ichimoku_senkou_a'] = senkou_a
    result['ichimoku_senkou_b'] = senkou_b
    result['ichimoku_chikou'] = chikou
    
    # Cloud signals
    result['ichimoku_cloud_top'] = np.maximum(senkou_a, senkou_b)
    result['ichimoku_cloud_bottom'] = np.minimum(senkou_a, senkou_b)
    result['ichimoku_cloud_color'] = np.where(senkou_a > senkou_b, 1, -1)
    result['ichimoku_cloud_signal'] = np.where(
        (result['close'] > result['ichimoku_cloud_top']), 1,
        np.where(result['close'] < result['ichimoku_cloud_bottom'], -1, 0)
    )
    
    return result

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a given dataframe.
    
    Args:
        df: DataFrame with columns: open, high, low, close, volume
    
    Returns:
        DataFrame with all indicators added
    """
    result = df.copy()
    
    # Price-based indicators
    result['rsi'] = calculate_rsi(result['close'], 14)
    result['rsi_14'] = result['rsi']
    result['rsi_21'] = calculate_rsi(result['close'], 21)
    
    # MACD
    macd, signal, hist = calculate_macd(result['close'])
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_hist'] = hist
    result['macd_hist_pct'] = hist / result['close']
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(result['close'])
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_mid
    result['bb_lower'] = bb_lower
    result['bb_width'] = (bb_upper - bb_lower) / bb_mid
    result['bb_position'] = (result['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # ATR
    result['atr'] = calculate_atr(result['high'], result['low'], result['close'], 14)
    result['atr_pct'] = result['atr'] / result['close']
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(result['high'], result['low'], result['close'])
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    
    # Williams %R
    result['williams_r'] = calculate_williams_r(result['high'], result['low'], result['close'])
    
    # CCI
    result['cci'] = calculate_cci(result['high'], result['low'], result['close'])
    result['cci_20'] = result['cci']
    
    # ADX
    result['adx'] = calculate_adx(result['high'], result['low'], result['close'])
    result['adx_14'] = result['adx']
    
    # OBV
    result['obv'] = calculate_obv(result['close'], result['volume'])
    result['obv_ma'] = result['obv'].rolling(20).mean()
    result['obv_ratio'] = result['obv'] / result['obv_ma']
    
    # MFI
    result['mfi'] = calculate_mfi(result['high'], result['low'], result['close'], result['volume'])
    
    # Moving averages
    for period in [9, 20, 50, 100, 200]:
        result[f'ema_{period}'] = calculate_ema(result['close'], period)
        result[f'sma_{period}'] = calculate_sma(result['close'], period)
        
    # EMA crossovers
    result['ema_cross_9_20'] = result['ema_9'] - result['ema_20']
    result['ema_cross_20_50'] = result['ema_20'] - result['ema_50']
    result['ema_cross_50_200'] = result['ema_50'] - result['ema_200']
    
    # SMA crossovers
    result['sma_cross_20_50'] = result['sma_20'] - result['sma_50']
    result['sma_cross_50_200'] = result['sma_50'] - result['sma_200']
    
    # Price relative to MAs
    for period in [20, 50, 200]:
        result[f'price_to_sma_{period}'] = result['close'] / result[f'sma_{period}'] - 1
        result[f'price_to_ema_{period}'] = result['close'] / result[f'ema_{period}'] - 1
    
    # Volume indicators
    result['vwap'] = calculate_vwap(result['high'], result['low'], result['close'], result['volume'])
    result['volume_sma'] = result['volume'].rolling(20).mean()
    result['volume_ratio'] = result['volume'] / result['volume_sma']
    result['volume_std'] = result['volume'].rolling(20).std()
    result['volume_zscore'] = (result['volume'] - result['volume_sma']) / result['volume_std']
    
    # Volatility
    result['volatility'] = calculate_volatility(result['close'])
    result['volatility_20'] = result['volatility']
    result['volatility_ratio'] = result['volatility'] / result['volatility'].rolling(100).mean()
    
    # Ichimoku
    ichimoku_result = calculate_ichimoku_cloud(result)
    for col in ['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 
                'ichimoku_senkou_b', 'ichimoku_cloud_top', 'ichimoku_cloud_bottom',
                'ichimoku_cloud_color', 'ichimoku_cloud_signal']:
        result[col] = ichimoku_result[col]
    
    # Donchian
    donchian_upper, donchian_mid, donchian_lower = calculate_donchian(result['high'], result['low'])
    result['donchian_upper'] = donchian_upper
    result['donchian_middle'] = donchian_mid
    result['donchian_lower'] = donchian_lower
    result['donchian_width'] = (donchian_upper - donchian_lower) / donchian_mid
    result['donchian_position'] = (result['close'] - donchian_lower) / (donchian_upper - donchian_lower)
    
    # Keltner
    keltner_upper, keltner_mid, keltner_lower = calculate_keltner(
        result['high'], result['low'], result['close']
    )
    result['keltner_upper'] = keltner_upper
    result['keltner_middle'] = keltner_mid
    result['keltner_lower'] = keltner_lower
    result['keltner_width'] = (keltner_upper - keltner_lower) / keltner_mid
    result['keltner_position'] = (result['close'] - keltner_lower) / (keltner_upper - keltner_lower)
    
    # Price patterns
    result['high_low_ratio'] = result['high'] / result['low']
    result['close_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
    result['gap'] = result['open'] - result['close'].shift(1)
    result['gap_pct'] = result['gap'] / result['close'].shift(1)
    
    # Candlestick patterns
    result['bullish_engulfing'] = (
        (result['close'] > result['open']) & 
        (result['close'].shift(1) < result['open'].shift(1)) &
        (result['close'] > result['open'].shift(1)) &
        (result['open'] < result['close'].shift(1))
    ).astype(int)
    
    result['bearish_engulfing'] = (
        (result['close'] < result['open']) & 
        (result['close'].shift(1) > result['open'].shift(1)) &
        (result['close'] < result['open'].shift(1)) &
        (result['open'] > result['close'].shift(1))
    ).astype(int)
    
    result['doji'] = (abs(result['close'] - result['open']) <= (result['high'] - result['low']) * 0.1).astype(int)
    result['hammer'] = (
        (result['close'] > result['open']) &
        ((result['high'] - result['close']) <= (result['close'] - result['open']) * 0.3) &
        ((result['open'] - result['low']) > (result['close'] - result['open']) * 2)
    ).astype(int)
    
    result['shooting_star'] = (
        (result['open'] > result['close']) &
        ((result['high'] - result['open']) > (result['open'] - result['close']) * 2) &
        ((result['close'] - result['low']) <= (result['open'] - result['close']) * 0.3)
    ).astype(int)
    
    # Returns
    result['returns'] = result['close'].pct_change()
    result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
    
    for period in [1, 5, 10, 20, 50]:
        result[f'returns_{period}'] = result['returns'].rolling(period).sum()
        result[f'returns_std_{period}'] = result['returns'].rolling(period).std()
        result[f'returns_skew_{period}'] = result['returns'].rolling(period).skew()
        result[f'returns_kurt_{period}'] = result['returns'].rolling(period).kurt()
    
    # Momentum
    for period in [5, 10, 20, 50]:
        result[f'momentum_{period}'] = result['close'] / result['close'].shift(period) - 1
    
    # Rate of Change
    for period in [5, 10, 20]:
        result[f'roc_{period}'] = (result['close'] - result['close'].shift(period)) / result['close'].shift(period) * 100
    
    # Price levels
    result['year_high'] = result['high'].rolling(252).max()
    result['year_low'] = result['low'].rolling(252).min()
    result['year_position'] = (result['close'] - result['year_low']) / (result['year_high'] - result['year_low'])
    
    # Support and resistance distances
    result['dist_to_52w_high'] = (result['year_high'] - result['close']) / result['close']
    result['dist_to_52w_low'] = (result['close'] - result['year_low']) / result['close']
    
    return result