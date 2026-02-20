"""
Enhanced Data Manager for NEXUS Trading System.
Handles data loading, storage, and synthetic data generation.
Now with Deriv API integration.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
import logging
from datetime import datetime, timedelta
import os
import shutil

import config
from indicators import calculate_all_indicators
from deriv_fetcher import DerivDataFetcher

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manages market data for all instruments.
    Now with Deriv API integration for real data.
    """
    
    def __init__(self, use_deriv: bool = True, deriv_token: Optional[str] = None):
        """
        Initialize data manager.
        
        Args:
            use_deriv: Whether to use Deriv for real data
            deriv_token: Your Deriv API token (get from app.deriv.com)
        """
        self.data_cache = {}
        self.data_files = {}
        self.synthetic_data_generated = {}
        self.data_info = {}
        self.data_metadata = {}
        
        # Initialize Deriv fetcher
        self.use_deriv = use_deriv
        self.deriv_fetcher = None
        
        if use_deriv:
            try:
                self.deriv_fetcher = DerivDataFetcher(api_token=deriv_token)
                logger.info("Deriv data fetcher initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Deriv fetcher: {e}")
                self.use_deriv = False
        
        # Default date range for data
        self.default_days_back = 365
        
    def get_data(self, 
                instrument: str, 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                timeframe: str = '1h',
                force_reload: bool = False,
                use_cache: bool = True) -> pd.DataFrame:
        """
        Get data for an instrument. Uses Deriv if available, falls back to synthetic.
        
        Args:
            instrument: Instrument symbol
            start_date: Optional start date
            end_date: Optional end date
            timeframe: Data timeframe ('1m', '5m', '1h', '4h', '1d')
            force_reload: Force reload from source
            use_cache: Use cached data if available
        
        Returns:
            DataFrame with OHLCV data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.default_days_back)
        
        # Create cache key
        cache_key = f"{instrument}_{timeframe}"
        
        # Check cache
        if use_cache and not force_reload and cache_key in self.data_cache:
            data = self.data_cache[cache_key].copy()
            logger.debug(f"Returning cached data for {instrument} ({len(data)} rows)")
            
            # Filter by date if needed
            data = self._filter_by_date(data, start_date, end_date)
            
            if len(data) > 0:
                return data
        
        # Try to get real data from Deriv
        if self.use_deriv and self.deriv_fetcher:
            logger.info(f"Fetching {instrument} data from Deriv...")
            df = self.deriv_fetcher.fetch_historical_data(
                instrument=instrument,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            if df is not None and len(df) > 0:
                logger.info(f"Successfully fetched {len(df)} rows from Deriv for {instrument}")
                
                # Calculate indicators
                df = calculate_all_indicators(df)
                
                # Store metadata
                self.data_metadata[instrument] = {
                    'source': 'deriv',
                    'timeframe': timeframe,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'rows': len(df),
                    'fetched_at': datetime.now().isoformat()
                }
                
                # Cache the data
                self.data_cache[cache_key] = df.copy()
                self.synthetic_data_generated[instrument] = False
                
                # Filter by date again (in case we got extra)
                df = self._filter_by_date(df, start_date, end_date)
                
                return df
            else:
                logger.warning(f"Failed to fetch from Deriv for {instrument}, using synthetic data")
        
        # Fall back to file-based data
        df = self._load_from_file(instrument)
        
        if df is not None and len(df) > 0:
            # Filter by date
            df = self._filter_by_date(df, start_date, end_date)
            
            if len(df) > 0:
                logger.info(f"Loaded {len(df)} rows for {instrument} from file")
                
                # Calculate indicators if not already present
                if 'rsi' not in df.columns:
                    df = calculate_all_indicators(df)
                
                # Cache the data
                self.data_cache[cache_key] = df.copy()
                self.synthetic_data_generated[instrument] = False
                
                return df
        
        # Generate synthetic data as last resort
        logger.info(f"Generating synthetic data for {instrument}")
        df = self._generate_synthetic_data(instrument, start_date, end_date, timeframe)
        
        # Calculate indicators
        df = calculate_all_indicators(df)
        
        # Store metadata
        self.data_metadata[instrument] = {
            'source': 'synthetic',
            'timeframe': timeframe,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'rows': len(df),
            'generated_at': datetime.now().isoformat()
        }
        
        # Cache the data
        self.data_cache[cache_key] = df.copy()
        self.synthetic_data_generated[instrument] = True
        
        # Filter by date
        df = self._filter_by_date(df, start_date, end_date)
        
        return df
    
    def _filter_by_date(self, 
                       df: pd.DataFrame, 
                       start_date: Optional[datetime], 
                       end_date: Optional[datetime]) -> pd.DataFrame:
        """Filter dataframe by date range."""
        if df is None or len(df) == 0:
            return df
        
        result = df.copy()
        
        if start_date:
            result = result[result.index >= start_date]
        
        if end_date:
            result = result[result.index <= end_date]
        
        return result
    
    def _load_from_file(self, instrument: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file if available."""
        data_dir = Path(config.DATA_DIR)
        
        # Look for instrument-specific files
        instrument_safe = instrument.replace('/', '_').replace(' ', '_')
        pattern = f"{instrument_safe}_*.csv"
        files = list(data_dir.glob(pattern))
        
        if not files:
            return None
        
        # Use most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        
        try:
            logger.debug(f"Loading data from {latest_file}")
            df = pd.read_csv(latest_file)
            
            # Parse dates
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            # Sort by index
            df.sort_index(inplace=True)
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns in {latest_file}: {missing_cols}")
                return None
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 1000000
            
            # Store file info
            self.data_files[instrument] = str(latest_file)
            self.data_info[instrument] = {
                'file': str(latest_file),
                'rows': len(df),
                'date_range': [df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')],
                'columns': list(df.columns),
                'synthetic': False
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading file {latest_file}: {e}")
            return None
    
    def _generate_synthetic_data(self, 
                                instrument: str, 
                                start_date: datetime,
                                end_date: datetime,
                                timeframe: str = '1h') -> pd.DataFrame:
        """
        Generate synthetic price data.
        
        Args:
            instrument: Instrument symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
        
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Calculate number of periods
        timeframe_seconds = self._get_timeframe_seconds(timeframe)
        total_seconds = (end_date - start_date).total_seconds()
        n_points = max(100, int(total_seconds / timeframe_seconds) + 1)
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, periods=n_points)
        
        # Get instrument configuration
        if instrument not in config.INSTRUMENTS:
            instr_config = {
                'base_price': 100, 
                'volatility': 0.01, 
                'trend': 0.0001, 
                'spread': 0.001
            }
        else:
            instr_config = config.INSTRUMENTS[instrument]
        
        base_price = instr_config['base_price']
        base_volatility = instr_config['volatility']
        trend = instr_config['trend']
        
        # Generate returns with regime switching
        np.random.seed(int(datetime.now().timestamp()) % 1000)  # Different seed each time
        
        returns = np.zeros(n_points)
        volatility = np.ones(n_points) * base_volatility
        
        # Regime switching parameters
        n_regimes = 3
        regime_probs = [0.7, 0.2, 0.1]  # Normal, High vol, Low vol
        current_regime = 0
        
        for i in range(1, n_points):
            # Regime transition (1% chance per period)
            if np.random.random() < 0.01:
                current_regime = np.random.choice(n_regimes, p=regime_probs)
            
            # Set volatility based on regime
            if current_regime == 0:  # Normal
                vol = base_volatility
            elif current_regime == 1:  # High volatility
                vol = base_volatility * 3
            else:  # Low volatility
                vol = base_volatility * 0.5
            
            volatility[i] = vol
            
            # Generate return with some autocorrelation
            returns[i] = trend + vol * np.random.randn() + 0.1 * returns[i-1]
        
        # Add occasional jumps (1% of periods)
        jump_indices = np.random.choice(n_points, size=int(n_points * 0.01), replace=False)
        returns[jump_indices] += np.random.randn(len(jump_indices)) * base_volatility * 5
        
        # Generate price series
        price = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close price
        daily_vol = volatility * price
        open_price = price * (1 + np.random.randn(n_points) * base_volatility * 0.3)
        
        high = np.maximum(price, open_price) * (1 + np.abs(np.random.randn(n_points) * base_volatility * 0.5))
        low = np.minimum(price, open_price) * (1 - np.abs(np.random.randn(n_points) * base_volatility * 0.5))
        
        # Ensure high >= low
        high = np.maximum(high, low * 1.001)
        
        # Generate volume with some patterns
        base_volume = 1000000
        volume = base_volume * (1 + 0.5 * np.abs(returns)) * (0.5 + 0.5 * np.random.rand(n_points))
        volume = np.maximum(volume, 100).astype(int)
        
        # Create dataframe
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        }, index=dates)
        
        # Add date column
        df['date'] = df.index
        
        logger.info(f"Generated {len(df)} synthetic rows for {instrument}")
        
        return df
    
    def _get_timeframe_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        timeframe_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '1d': 86400,
            '1w': 604800
        }
        return timeframe_map.get(timeframe, 3600)
    
    def store_data(self, instrument: str, df: pd.DataFrame, filepath: str):
        """
        Store uploaded data in cache and record file location.
        
        Args:
            instrument: Instrument symbol
            df: DataFrame with OHLCV data
            filepath: Path to saved file
        """
        # Ensure date is index
        if 'date' in df.columns and df.index.name != 'date':
            df.set_index('date', inplace=True)
        
        self.data_cache[f"{instrument}_uploaded"] = df
        self.data_files[instrument] = filepath
        self.synthetic_data_generated[instrument] = False
        
        self.data_info[instrument] = {
            'file': filepath,
            'rows': len(df),
            'date_range': [df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')],
            'columns': list(df.columns),
            'synthetic': False,
            'source': 'upload'
        }
        
        logger.info(f"Stored uploaded data for {instrument}: {len(df)} rows")
    
    def list_available_data(self) -> List[Dict[str, Any]]:
        """List all available data files and sources."""
        available = []
        
        for instrument in config.INSTRUMENTS.keys():
            info = {
                'instrument': instrument,
                'rows': 0,
                'synthetic': True,
                'date_range': ['', ''],
                'columns': [],
                'file': '',
                'source': 'none'
            }
            
            # Check cache
            cache_key = f"{instrument}_1h"
            if cache_key in self.data_cache:
                data = self.data_cache[cache_key]
                info.update({
                    'rows': len(data),
                    'synthetic': self.synthetic_data_generated.get(instrument, True),
                    'date_range': [
                        data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else '',
                        data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else ''
                    ],
                    'columns': list(data.columns)[:10],
                    'file': self.data_files.get(instrument, ''),
                    'source': self.data_metadata.get(instrument, {}).get('source', 'unknown')
                })
            
            # Check uploaded data
            elif f"{instrument}_uploaded" in self.data_cache:
                data = self.data_cache[f"{instrument}_uploaded"]
                info.update({
                    'rows': len(data),
                    'synthetic': False,
                    'date_range': [
                        data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else '',
                        data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else ''
                    ],
                    'columns': list(data.columns)[:10],
                    'file': self.data_files.get(instrument, ''),
                    'source': 'upload'
                })
            
            available.append(info)
        
        return available
    
    def refresh_data(self, instrument: str, days_back: int = 30) -> bool:
        """
        Force refresh of data for an instrument.
        
        Args:
            instrument: Instrument symbol
            days_back: Number of days to fetch
        
        Returns:
            True if successful
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            df = self.get_data(
                instrument=instrument,
                start_date=start_date,
                end_date=end_date,
                force_reload=True
            )
            
            return df is not None and len(df) > 0
            
        except Exception as e:
            logger.error(f"Error refreshing data for {instrument}: {e}")
            return False
    
    def delete_data(self, instrument: str) -> bool:
        """
        Delete uploaded data for an instrument.
        
        Args:
            instrument: Instrument symbol
        
        Returns:
            True if successful
        """
        # Remove from cache
        cache_keys = [k for k in self.data_cache.keys() if instrument in k]
        for key in cache_keys:
            del self.data_cache[key]
        
        # Delete file if exists
        if instrument in self.data_files:
            filepath = self.data_files[instrument]
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Deleted data file: {filepath}")
            except Exception as e:
                logger.error(f"Error deleting file {filepath}: {e}")
            
            del self.data_files[instrument]
        
        # Reset metadata
        if instrument in self.data_metadata:
            del self.data_metadata[instrument]
        
        # Reset to synthetic
        self.synthetic_data_generated[instrument] = True
        
        return True
    
    def get_data_info(self, instrument: str) -> Dict[str, Any]:
        """
        Get information about data for an instrument.
        
        Args:
            instrument: Instrument symbol
        
        Returns:
            Dictionary with data information
        """
        if instrument in self.data_metadata:
            return self.data_metadata[instrument]
        
        # Try to load data to get info
        try:
            data = self.get_data(instrument, days_back=10)  # Just get a small sample
            if data is not None and len(data) > 0:
                return self.data_metadata.get(instrument, {})
        except:
            pass
        
        return {}
    
    def export_data(self, instrument: str, format: str = 'csv') -> Optional[str]:
        """
        Export data to file.
        
        Args:
            instrument: Instrument symbol
            format: Export format ('csv', 'json', 'parquet')
        
        Returns:
            Path to exported file or None if failed
        """
        cache_key = f"{instrument}_1h"
        if cache_key not in self.data_cache:
            logger.error(f"No data for {instrument}")
            return None
        
        data = self.data_cache[cache_key]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        instrument_safe = instrument.replace('/', '_')
        
        try:
            if format == 'csv':
                filename = f"{instrument_safe}_export_{timestamp}.csv"
                filepath = config.DATA_DIR / filename
                data.to_csv(filepath)
            elif format == 'json':
                filename = f"{instrument_safe}_export_{timestamp}.json"
                filepath = config.DATA_DIR / filename
                data.to_json(filepath)
            elif format == 'parquet':
                filename = f"{instrument_safe}_export_{timestamp}.parquet"
                filepath = config.DATA_DIR / filename
                data.to_parquet(filepath)
            else:
                logger.error(f"Unsupported format: {format}")
                return None
            
            logger.info(f"Exported {instrument} data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None
    
    def subscribe_to_realtime(self, instrument: str, callback: Callable) -> bool:
        """
        Subscribe to real-time price updates from Deriv.
        
        Args:
            instrument: Instrument symbol
            callback: Function to call with price updates
        
        Returns:
            True if successful
        """
        if self.use_deriv and self.deriv_fetcher:
            return self.deriv_fetcher.subscribe_to_price(instrument, callback)
        return False
    
    def get_realtime_price(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get current real-time price from Deriv.
        
        Args:
            instrument: Instrument symbol
        
        Returns:
            Dictionary with price info or None
        """
        if self.use_deriv and self.deriv_fetcher:
            return self.deriv_fetcher.get_realtime_price(instrument)
        return None
    
    def clear_cache(self):
        """Clear all cached data."""
        self.data_cache.clear()
        logger.info("Data cache cleared")