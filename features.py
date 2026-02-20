"""
Advanced feature engineering module for NEXUS Trading System.
Implements market regime detection and advanced feature generation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class AdvancedFeatureEngine:
    """
    Generates advanced features including market regime and cross-asset context.
    """
    
    def __init__(self, lookback_periods: int = 100):
        """
        Initialize feature engine.
        
        Args:
            lookback_periods: Default lookback period for rolling calculations
        """
        self.lookback_periods = lookback_periods
        self.regime_history = {}
        self.feature_importance = {}
        self.pca_components = None
        self.scaler = StandardScaler()
        
    def detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify market regime using multiple indicators.
        
        Regimes:
        0: Ranging / Low Volatility
        1: Trending Up (Strong)
        2: Trending Up (Weak)
        3: Trending Down (Strong)
        4: Trending Down (Weak)
        5: High Volatility Breakout
        6: Mean Reversion Regime
        
        Args:
            df: DataFrame with OHLCV data and indicators
        
        Returns:
            Series with regime classifications
        """
        if len(df) < 50:
            return pd.Series(0, index=df.index)
        
        # Calculate required indicators if not present
        if 'adx' not in df.columns:
            from indicators import calculate_adx
            df['adx'] = calculate_adx(df['high'], df['low'], df['close'], 14)
        
        if 'macd' not in df.columns:
            from indicators import calculate_macd
            macd, _, _ = calculate_macd(df['close'])
            df['macd'] = macd
        
        # Trend strength using ADX
        adx = df['adx'].fillna(0)
        strong_trend = adx > 25
        weak_trend = (adx > 15) & (adx <= 25)
        
        # Trend direction using multiple indicators
        # MACD trend
        macd_trend = df['macd'] > df['macd'].rolling(20).mean()
        
        # EMA alignment
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        ema_200 = df['close'].ewm(span=200).mean() if len(df) > 200 else ema_50
        
        bull_aligned = (ema_20 > ema_50) & (ema_50 > ema_200)
        bear_aligned = (ema_20 < ema_50) & (ema_50 < ema_200)
        
        # Volatility regime
        if 'bb_width' in df.columns:
            bb_width = df['bb_width']
        else:
            from indicators import calculate_bollinger_bands
            bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df['close'])
            bb_width = (bb_upper - bb_lower) / bb_mid
            df['bb_width'] = bb_width
        
        bb_percentile = bb_width.rolling(100).rank(pct=True)
        high_vol = bb_percentile > 0.85
        low_vol = bb_percentile < 0.15
        
        # Momentum
        momentum = df['close'].pct_change(20)
        strong_up = momentum > 0.05
        strong_down = momentum < -0.05
        
        # Mean reversion signals
        if 'rsi' in df.columns:
            rsi = df['rsi']
            oversold = rsi < 30
            overbought = rsi > 70
        else:
            oversold = False
            overbought = False
        
        # Combine into regime classification
        regime = pd.Series(index=df.index, dtype=int)
        
        # Strong up trend
        regime[(strong_trend | strong_up) & bull_aligned & macd_trend & ~high_vol] = 1
        
        # Weak up trend
        regime[weak_trend & bull_aligned & ~high_vol] = 2
        
        # Strong down trend
        regime[(strong_trend | strong_down) & bear_aligned & ~macd_trend & ~high_vol] = 3
        
        # Weak down trend
        regime[weak_trend & bear_aligned & ~high_vol] = 4
        
        # High volatility breakout
        regime[high_vol] = 5
        
        # Mean reversion
        regime[low_vol & (oversold | overbought)] = 6
        
        # Default to ranging
        regime[regime.isna()] = 0
        
        # Store for later use
        self.regime_history[df.index[-1] if len(df) > 0 else None] = regime.iloc[-1] if len(regime) > 0 else 0
        
        # Shift to avoid lookahead bias
        return regime.shift(1).fillna(0).astype(int)
    
    def calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add order flow and market microstructure features.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with microstructure features added
        """
        result = df.copy()
        
        # Price position within bar
        result['bar_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
        result['bar_position'] = result['bar_position'].replace([np.inf, -np.inf], 0.5).fillna(0.5)
        
        # Bar type classification
        result['bullish_bar'] = (result['close'] > result['open']).astype(int)
        result['bearish_bar'] = (result['close'] < result['open']).astype(int)
        result['neutral_bar'] = (abs(result['close'] - result['open']) < (result['high'] - result['low']) * 0.1).astype(int)
        
        # Bar range
        result['bar_range'] = result['high'] - result['low']
        result['bar_range_pct'] = result['bar_range'] / result['close']
        
        # Upper and lower wicks
        result['upper_wick'] = result['high'] - np.maximum(result['open'], result['close'])
        result['lower_wick'] = np.minimum(result['open'], result['close']) - result['low']
        result['wick_ratio'] = (result['upper_wick'] + result['lower_wick']) / (result['bar_range'] + 1e-8)
        result['upper_wick_pct'] = result['upper_wick'] / result['bar_range']
        result['lower_wick_pct'] = result['lower_wick'] / result['bar_range']
        
        # Real body
        result['body'] = abs(result['close'] - result['open'])
        result['body_pct'] = result['body'] / result['bar_range']
        
        # Tick imbalance approximation (using price change as proxy)
        price_change = result['close'].diff()
        result['tick_direction'] = np.sign(price_change)
        result['tick_size'] = price_change.abs()
        
        # Volume-weighted price impact
        result['buying_pressure'] = (result['tick_direction'] * result['volume']).rolling(10).sum()
        result['selling_pressure'] = (-result['tick_direction'] * result['volume']).rolling(10).sum()
        result['pressure_imbalance'] = (result['buying_pressure'] - result['selling_pressure']) / (result['buying_pressure'] + result['selling_pressure'] + 1e-8)
        
        # Volume-weighted average price (VWAP) deviation
        if 'vwap' not in result.columns:
            from indicators import calculate_vwap
            result['vwap'] = calculate_vwap(result['high'], result['low'], result['close'], result['volume'])
        
        result['vwap_deviation'] = (result['close'] - result['vwap']) / result['vwap']
        result['vwap_deviation_ma'] = result['vwap_deviation'].rolling(20).mean()
        
        # Spread estimation (if not provided)
        if 'spread' not in result.columns:
            result['spread_estimate'] = (result['high'] - result['low']) / result['close']
            result['spread_estimate_ma'] = result['spread_estimate'].rolling(20).mean()
            result['spread_estimate_ratio'] = result['spread_estimate'] / result['spread_estimate_ma']
        
        # Liquidity metrics
        result['dollar_volume'] = result['volume'] * result['close']
        result['dollar_volume_ma'] = result['dollar_volume'].rolling(20).mean()
        result['liquidity_ratio'] = result['dollar_volume'] / result['bar_range']
        
        # Order flow imbalance
        result['buy_volume'] = result['volume'] * result['bullish_bar']
        result['sell_volume'] = result['volume'] * result['bearish_bar']
        result['buy_volume_ma'] = result['buy_volume'].rolling(20).mean()
        result['sell_volume_ma'] = result['sell_volume'].rolling(20).mean()
        result['volume_imbalance'] = (result['buy_volume'] - result['sell_volume']) / (result['buy_volume'] + result['sell_volume'] + 1e-8)
        
        return result
    
    def calculate_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        Calculate rolling statistical features.
        
        Args:
            df: DataFrame with price data
            windows: List of window periods
        
        Returns:
            DataFrame with rolling features added
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100]
        
        result = df.copy()
        
        # Ensure returns are calculated
        if 'returns' not in result.columns:
            result['returns'] = result['close'].pct_change()
        
        for window in windows:
            # Rolling statistics for returns
            result[f'returns_mean_{window}'] = result['returns'].rolling(window).mean()
            result[f'returns_std_{window}'] = result['returns'].rolling(window).std()
            result[f'returns_skew_{window}'] = result['returns'].rolling(window).skew()
            result[f'returns_kurt_{window}'] = result['returns'].rolling(window).kurt()
            result[f'returns_min_{window}'] = result['returns'].rolling(window).min()
            result[f'returns_max_{window}'] = result['returns'].rolling(window).max()
            result[f'returns_median_{window}'] = result['returns'].rolling(window).median()
            
            # Rolling quantiles
            result[f'returns_q25_{window}'] = result['returns'].rolling(window).quantile(0.25)
            result[f'returns_q75_{window}'] = result['returns'].rolling(window).quantile(0.75)
            result[f'returns_iqr_{window}'] = result[f'returns_q75_{window}'] - result[f'returns_q25_{window}']
            
            # Rolling autocorrelation
            result[f'returns_autocorr_{window}'] = result['returns'].rolling(window).apply(
                lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
            )
            
            # Rolling correlations
            if 'volume' in result.columns:
                result[f'price_volume_corr_{window}'] = (
                    result['close'].rolling(window).corr(result['volume'])
                )
            
            if 'rsi' in result.columns:
                result[f'price_rsi_corr_{window}'] = (
                    result['close'].rolling(window).corr(result['rsi'])
                )
            
            # Rolling z-score of price
            price_mean = result['close'].rolling(window).mean()
            price_std = result['close'].rolling(window).std()
            result[f'price_zscore_{window}'] = (result['close'] - price_mean) / (price_std + 1e-8)
            
            # Rolling volatility ratio
            if window > 5:
                result[f'volatility_ratio_{window}'] = (
                    result[f'returns_std_{window}'] / result[f'returns_std_{5}']
                )
            
            # Rolling Sharpe ratio
            result[f'sharpe_{window}'] = (
                result[f'returns_mean_{window}'] / (result[f'returns_std_{window}'] + 1e-8) * np.sqrt(252)
            )
        
        return result
    
    def calculate_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based seasonal features.
        
        Args:
            df: DataFrame with datetime index
        
        Returns:
            DataFrame with seasonal features added
        """
        result = df.copy()
        
        # Get datetime index
        if 'date' in result.columns:
            dates = pd.to_datetime(result['date'])
        else:
            dates = result.index
        
        # Basic time features
        result['hour'] = dates.hour
        result['minute'] = dates.minute
        result['day_of_week'] = dates.dayofweek
        result['day_of_month'] = dates.day
        result['week_of_year'] = dates.isocalendar().week.astype(int)
        result['month'] = dates.month
        result['quarter'] = dates.quarter
        result['year'] = dates.year
        
        # Cyclical encoding for time features
        # Hour (0-23) -> sin/cos
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        
        # Day of week (0-6) -> sin/cos
        result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Month (1-12) -> sin/cos
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Day of month (1-31) -> sin/cos
        days_in_month = dates.days_in_month
        result['day_of_month_norm'] = result['day_of_month'] / days_in_month
        result['day_sin'] = np.sin(2 * np.pi * result['day_of_month_norm'])
        result['day_cos'] = np.cos(2 * np.pi * result['day_of_month_norm'])
        
        # Trading sessions (for forex)
        result['is_asia_session'] = ((result['hour'] >= 0) & (result['hour'] < 8)).astype(int)
        result['is_london_session'] = ((result['hour'] >= 8) & (result['hour'] < 16)).astype(int)
        result['is_ny_session'] = ((result['hour'] >= 13) & (result['hour'] < 22)).astype(int)
        result['is_london_ny_overlap'] = ((result['hour'] >= 13) & (result['hour'] < 16)).astype(int)
        result['is_asia_london_overlap'] = ((result['hour'] >= 8) & (result['hour'] < 9)).astype(int)
        
        # Weekend vs weekday
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        result['is_weekday'] = (result['day_of_week'] < 5).astype(int)
        
        # Month start/end
        result['is_month_start'] = (result['day_of_month'] <= 3).astype(int)
        result['is_month_end'] = (result['day_of_month'] >= 28).astype(int)
        
        # Quarter start/end
        result['is_quarter_start'] = ((result['month'] % 3 == 1) & (result['day_of_month'] <= 3)).astype(int)
        result['is_quarter_end'] = ((result['month'] % 3 == 0) & (result['day_of_month'] >= 28)).astype(int)
        
        # Year end/start effects
        result['is_year_end'] = ((result['month'] == 12) & (result['day_of_month'] >= 15)).astype(int)
        result['is_year_start'] = ((result['month'] == 1) & (result['day_of_month'] <= 15)).astype(int)
        
        return result
    
    def calculate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features based on market regime.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with regime features added
        """
        result = df.copy()
        
        # Detect regime
        result['market_regime'] = self.detect_market_regime(result)
        
        # Regime duration
        for regime in range(7):
            regime_mask = (result['market_regime'] == regime).astype(int)
            result[f'regime_{regime}_duration'] = regime_mask.rolling(50, min_periods=1).sum()
        
        # Regime transitions
        result['regime_change'] = (result['market_regime'] != result['market_regime'].shift(1)).astype(int)
        result['regime_streak'] = result.groupby(
            (result['market_regime'] != result['market_regime'].shift(1)).cumsum()
        ).cumcount() + 1
        
        # Performance by regime
        if 'returns' in result.columns:
            for regime in range(7):
                regime_returns = result['returns'] * (result['market_regime'] == regime)
                result[f'regime_{regime}_avg_return'] = regime_returns.rolling(50).mean()
                result[f'regime_{regime}_volatility'] = regime_returns.rolling(50).std()
        
        return result
    
    def calculate_cross_asset_features(self,
                                      main_df: pd.DataFrame,
                                      other_assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add features from correlated assets.
        
        Args:
            main_df: Main instrument DataFrame
            other_assets: Dictionary of other instrument DataFrames
        
        Returns:
            DataFrame with cross-asset features added
        """
        result = main_df.copy()
        
        for asset_name, asset_df in other_assets.items():
            if asset_name == result.attrs.get('instrument', ''):
                continue
            
            # Align indices
            aligned = asset_df['close'].reindex(result.index, method='ffill')
            
            # Calculate returns
            asset_returns = aligned.pct_change()
            
            # Lagged returns (1, 2, 3 periods)
            for lag in [1, 2, 3]:
                result[f'{asset_name}_ret_{lag}'] = asset_returns.shift(lag)
            
            # Rolling correlation
            for window in [20, 50, 100]:
                if window < len(result):
                    result[f'corr_{asset_name}_{window}'] = (
                        result['returns'].rolling(window).corr(asset_returns)
                    )
            
            # Volatility ratio
            asset_vol = asset_returns.rolling(20).std()
            main_vol = result['returns'].rolling(20).std()
            result[f'vol_ratio_{asset_name}'] = main_vol / asset_vol
            
            # Price ratio
            result[f'price_ratio_{asset_name}'] = result['close'] / aligned
            
            # Spread (for highly correlated pairs)
            if f'corr_{asset_name}_50' in result.columns and result[f'corr_{asset_name}_50'].iloc[-1] > 0.7:
                # Normalized spread
                z1 = (result['close'] - result['close'].rolling(50).mean()) / result['close'].rolling(50).std()
                z2 = (aligned - aligned.rolling(50).mean()) / aligned.rolling(50).std()
                result[f'spread_{asset_name}'] = z1 - z2
        
        return result
    
    def reduce_dimensions(self, df: pd.DataFrame, n_components: int = 20) -> pd.DataFrame:
        """
        Apply PCA to reduce feature dimensionality.
        
        Args:
            df: DataFrame with features
            n_components: Number of PCA components
        
        Returns:
            DataFrame with PCA components added
        """
        result = df.copy()
        
        # Select numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) < n_components:
            return result
        
        # Fill NaN values
        X = result[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(feature_cols)))
        components = pca.fit_transform(X_scaled)
        
        # Add components to dataframe
        for i in range(components.shape[1]):
            result[f'pca_{i+1}'] = components[:, i]
            result[f'pca_{i+1}_explained'] = pca.explained_variance_ratio_[i]
        
        self.pca_components = pca
        
        # Store explained variance
        result.attrs['pca_explained_variance'] = pca.explained_variance_ratio_.sum()
        
        return result
    
    def calculate_all_features(self, 
                              df: pd.DataFrame, 
                              other_assets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Calculate all advanced features.
        
        Args:
            df: Main instrument DataFrame
            other_assets: Optional dictionary of other instrument DataFrames
        
        Returns:
            DataFrame with all features added
        """
        logger.info("Calculating all advanced features...")
        
        result = df.copy()
        initial_cols = len(result.columns)
        
        # Add microstructure features
        result = self.calculate_microstructure_features(result)
        logger.debug(f"Added microstructure features: {len(result.columns) - initial_cols}")
        
        # Add rolling features
        result = self.calculate_rolling_features(result)
        logger.debug(f"Added rolling features: {len(result.columns) - initial_cols}")
        
        # Add seasonal features
        result = self.calculate_seasonal_features(result)
        logger.debug(f"Added seasonal features: {len(result.columns) - initial_cols}")
        
        # Add market regime features
        result = self.calculate_market_regime_features(result)
        logger.debug(f"Added regime features: {len(result.columns) - initial_cols}")
        
        # Add cross-asset features if available
        if other_assets:
            result = self.calculate_cross_asset_features(result, other_assets)
            logger.debug(f"Added cross-asset features: {len(result.columns) - initial_cols}")
        
        # Add PCA components (optional, can be memory intensive)
        # result = self.reduce_dimensions(result)
        
        logger.info(f"Total features added: {len(result.columns) - initial_cols}")
        logger.info(f"Final feature count: {len(result.columns)}")
        
        return result
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance from trained model.
        
        Args:
            model: Trained sklearn model
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Store for later use
        self.feature_importance = importance_df
        
        return importance_df