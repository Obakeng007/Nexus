"""
Signal validation module for NEXUS Trading System.
Validates signals using market context and multiple timeframes.
Fixed with proper boolean conversion.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

import config

logger = logging.getLogger(__name__)

class SignalValidator:
    """
    Validates trading signals using market context.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize signal validator.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or config.VALIDATION_CONFIG
        self.validation_history = []
        
    def validate_multi_timeframe(self,
                                signal: Dict[str, Any],
                                data_by_tf: Dict[str, pd.DataFrame],
                                instrument: str) -> Tuple[bool, str, float]:
        """
        Validate signal across multiple timeframes.
        
        Args:
            signal: Signal dictionary
            data_by_tf: Dictionary of timeframe data
            instrument: Instrument symbol
        
        Returns:
            Tuple of (passed, message, score)
        """
        signal_direction = signal['direction']
        signal_time = signal['timestamp']
        
        if not data_by_tf:
            return True, "No higher timeframe data available", 1.0
        
        # Check higher timeframe alignment
        alignments = []
        messages = []
        
        for tf_name, tf_data in data_by_tf.items():
            # Get recent data
            tf_slice = tf_data.loc[:signal_time].tail(50)
            
            if len(tf_slice) < 20:
                continue
            
            # Calculate trend using multiple methods
            # 1. EMA crossover
            ema_fast = tf_slice['close'].ewm(span=10).mean()
            ema_slow = tf_slice['close'].ewm(span=30).mean()
            ema_trend = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
            
            # 2. Price vs SMA
            sma_50 = tf_slice['close'].rolling(50).mean()
            sma_trend = 1 if tf_slice['close'].iloc[-1] > sma_50.iloc[-1] else -1
            
            # 3. MACD trend
            if 'macd' in tf_slice.columns:
                macd_trend = 1 if tf_slice['macd'].iloc[-1] > tf_slice['macd'].iloc[-5:].mean() else -1
            else:
                macd_trend = ema_trend
            
            # Combine trends (majority vote)
            trends = [ema_trend, sma_trend, macd_trend]
            tf_trend = 1 if sum(trends) > 0 else -1
            
            # Check alignment
            is_aligned = (signal_direction == tf_trend)
            alignments.append(is_aligned)
            messages.append(f"{tf_name}: {'aligned' if is_aligned else 'against'}")
        
        if not alignments:
            return True, "No timeframe data available", 1.0
        
        # Calculate alignment score
        alignment_score = np.mean(alignments)
        
        # Check if alignment is required
        require_alignment = self.config['multi_timeframe']['require_alignment']
        
        if require_alignment:
            passed = alignment_score >= 0.5
        else:
            passed = True
        
        message = f"Multi-timeframe: {alignment_score:.0%} aligned | " + " | ".join(messages[:2])
        
        return bool(passed), message, float(alignment_score)
    
    def validate_volume(self,
                       signal: Dict[str, Any],
                       data: pd.DataFrame) -> Tuple[bool, str, float]:
        """
        Validate signal has volume confirmation.
        
        Args:
            signal: Signal dictionary
            data: DataFrame with OHLCV data
        
        Returns:
            Tuple of (passed, message, score)
        """
        signal_time = signal['timestamp']
        
        # Parse timestamp if string
        if isinstance(signal_time, str):
            try:
                signal_time = pd.to_datetime(signal_time)
            except:
                return True, "Could not parse timestamp", 1.0
        
        # Find closest index
        if hasattr(data.index, 'get_indexer'):
            idx = data.index.get_indexer([signal_time], method='nearest')[0]
        else:
            return True, "No timestamp index", 1.0
        
        if idx < 5 or idx >= len(data):
            return True, "Insufficient volume data", 1.0
        
        # Get volume data
        lookback = self.config['volume']['volume_lookback']
        start_idx = max(0, idx - lookback)
        
        recent_volume = data['volume'].iloc[start_idx:idx+1]
        
        if len(recent_volume) < 2:
            return True, "No volume data", 1.0
        
        current_volume = recent_volume.iloc[-1]
        avg_volume = recent_volume.iloc[:-1].mean()
        
        if pd.isna(avg_volume) or avg_volume == 0:
            return True, "No volume data", 1.0
        
        volume_ratio = current_volume / avg_volume
        
        # Check volume confirmation
        min_ratio = self.config['volume']['min_volume_ratio']
        passed = volume_ratio >= min_ratio
        
        # Score is capped at 1.0
        score = min(1.0, volume_ratio / min_ratio)
        
        # Check for volume trend
        volume_trend = recent_volume.iloc[-3:].mean() > recent_volume.iloc[-6:-3].mean()
        
        message = f"Volume: {volume_ratio:.2f}x avg"
        if volume_trend:
            message += " (increasing)"
        
        return bool(passed), message, float(score)
    
    def validate_market_hours(self,
                             signal: Dict[str, Any],
                             instrument: str) -> Tuple[bool, str, float]:
        """
        Validate signal occurs during active market hours.
        
        Args:
            signal: Signal dictionary
            instrument: Instrument symbol
        
        Returns:
            Tuple of (passed, message, score)
        """
        signal_time = signal['timestamp']
        
        # Convert to datetime if string
        if isinstance(signal_time, str):
            try:
                signal_time = pd.to_datetime(signal_time)
            except:
                return True, "Could not parse timestamp", 1.0
        
        # Get market hours from config
        if instrument not in config.INSTRUMENTS:
            return True, "Market hours not defined", 1.0
        
        market_hours = config.INSTRUMENTS[instrument].get('market_hours', (0, 24))
        open_hour, close_hour = market_hours
        
        if hasattr(signal_time, 'hour'):
            hour = signal_time.hour
        else:
            hour = 12
        
        # Handle overnight sessions (e.g., USD/JPY 19:00 - 17:00 next day)
        if open_hour <= close_hour:
            # Same day session
            is_open = open_hour <= hour < close_hour
        else:
            # Overnight session
            is_open = hour >= open_hour or hour < close_hour
        
        # Check if it's weekend
        if hasattr(signal_time, 'dayofweek'):
            day = signal_time.dayofweek
            if day >= 5:  # Saturday or Sunday
                is_open = False
        
        if is_open:
            hour_str = f"{open_hour:02d}:00-{close_hour:02d}:00"
            return True, f"Within market hours ({hour_str})", 1.0
        else:
            hour_str = f"{open_hour:02d}:00-{close_hour:02d}:00"
            return False, f"Outside market hours ({hour_str})", 0.0
    
    def validate_volatility(self,
                          signal: Dict[str, Any],
                          data: pd.DataFrame) -> Tuple[bool, str, float]:
        """
        Validate volatility is within acceptable range.
        
        Args:
            signal: Signal dictionary
            data: DataFrame with OHLCV data
        
        Returns:
            Tuple of (passed, message, score)
        """
        signal_time = signal['timestamp']
        
        # Parse timestamp if string
        if isinstance(signal_time, str):
            try:
                signal_time = pd.to_datetime(signal_time)
            except:
                return True, "Could not parse timestamp", 1.0
        
        if hasattr(data.index, 'get_indexer'):
            idx = data.index.get_indexer([signal_time], method='nearest')[0]
        else:
            return True, "No timestamp index", 1.0
        
        if idx < 20:
            return True, "Insufficient volatility data", 1.0
        
        # Calculate recent volatility
        start_idx = max(0, idx - 20)
        recent_returns = data['close'].iloc[start_idx:idx+1].pct_change().dropna()
        
        if len(recent_returns) < 5:
            return True, "Insufficient returns data", 1.0
            
        current_vol = recent_returns.std()
        
        # Calculate historical volatility
        hist_start = max(0, idx - 100)
        hist_returns = data['close'].iloc[hist_start:idx+1].pct_change().dropna()
        hist_vol = hist_returns.std() if len(hist_returns) > 5 else current_vol
        
        if pd.isna(hist_vol) or hist_vol == 0:
            return True, "No historical volatility", 1.0
        
        vol_ratio = current_vol / hist_vol
        
        # Get thresholds from config
        max_ratio = self.config['volatility']['max_volatility_ratio']
        min_ratio = self.config['volatility']['min_volatility_ratio']
        
        # Volatility should not be too high (avoid panic) or too low (avoid stagnation)
        if vol_ratio > max_ratio:
            score = max(0, 1 - (vol_ratio - max_ratio))
            return False, f"Volatility too high: {vol_ratio:.2f}x normal", float(score)
        elif vol_ratio < min_ratio:
            score = vol_ratio / min_ratio
            return False, f"Volatility too low: {vol_ratio:.2f}x normal", float(score)
        else:
            score = 1.0
            return True, f"Volatility normal: {vol_ratio:.2f}x", float(score)
    
    def validate_trend_strength(self,
                               signal: Dict[str, Any],
                               data: pd.DataFrame) -> Tuple[bool, str, float]:
        """
        Validate trend strength using ADX.
        
        Args:
            signal: Signal dictionary
            data: DataFrame with OHLCV data
        
        Returns:
            Tuple of (passed, message, score)
        """
        signal_time = signal['timestamp']
        
        # Parse timestamp if string
        if isinstance(signal_time, str):
            try:
                signal_time = pd.to_datetime(signal_time)
            except:
                return True, "Could not parse timestamp", 1.0
        
        if 'adx' not in data.columns:
            return True, "ADX not available", 1.0
        
        if hasattr(data.index, 'get_indexer'):
            idx = data.index.get_indexer([signal_time], method='nearest')[0]
        else:
            return True, "No timestamp index", 1.0
        
        if idx < 0 or idx >= len(data):
            return True, "No data", 1.0
        
        current_adx = data['adx'].iloc[idx]
        
        if pd.isna(current_adx):
            return True, "ADX not available", 1.0
        
        # Get thresholds
        min_adx = self.config['trend']['min_adx']
        strong_adx = self.config['trend']['strong_trend_adx']
        
        # ADX above min_adx indicates tradable trend
        if current_adx > strong_adx:
            score = min(1.0, current_adx / 50)
            return True, f"Strong trend (ADX: {current_adx:.1f})", float(score)
        elif current_adx > min_adx:
            score = 0.7
            return True, f"Moderate trend (ADX: {current_adx:.1f})", float(score)
        else:
            score = current_adx / min_adx
            return False, f"Weak trend (ADX: {current_adx:.1f})", float(score)
    
    def validate_correlation(self,
                           signal: Dict[str, Any],
                           data: pd.DataFrame,
                           correlated_pairs: Dict[str, pd.DataFrame]) -> Tuple[bool, str, float]:
        """
        Validate signal against correlated instruments.
        
        Args:
            signal: Signal dictionary
            data: Main instrument data
            correlated_pairs: Dictionary of correlated instrument data
        
        Returns:
            Tuple of (passed, message, score)
        """
        signal_direction = signal['direction']
        signal_time = signal['timestamp']
        
        # Parse timestamp if string
        if isinstance(signal_time, str):
            try:
                signal_time = pd.to_datetime(signal_time)
            except:
                return True, "Could not parse timestamp", 1.0
        
        if not correlated_pairs:
            return True, "No correlated pairs", 1.0
        
        agreements = []
        
        for pair_name, pair_data in correlated_pairs.items():
            # Get recent data
            pair_slice = pair_data.loc[:signal_time].tail(50)
            
            if len(pair_slice) < 20:
                continue
            
            # Determine trend of correlated pair
            ema_fast = pair_slice['close'].ewm(span=10).mean()
            ema_slow = pair_slice['close'].ewm(span=30).mean()
            pair_trend = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
            
            # Expected relationship (most forex pairs are correlated)
            # For simplicity, assume positive correlation
            expected_direction = signal_direction
            
            is_agreeing = (pair_trend == expected_direction)
            agreements.append(is_agreeing)
        
        if not agreements:
            return True, "No correlation data", 1.0
        
        agreement_rate = np.mean(agreements)
        
        if agreement_rate > 0.7:
            return True, f"Strong correlation agreement ({agreement_rate:.0%})", float(agreement_rate)
        elif agreement_rate > 0.5:
            return True, f"Moderate correlation agreement ({agreement_rate:.0%})", float(agreement_rate)
        else:
            return False, f"Correlation disagreement ({agreement_rate:.0%})", float(agreement_rate)
    
    def validate_all(self,
                    signal: Dict[str, Any],
                    data: pd.DataFrame,
                    instrument: str,
                    additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run all validations and return results.
        
        Args:
            signal: Signal dictionary
            data: DataFrame with OHLCV data
            instrument: Instrument symbol
            additional_data: Additional data for validation
        
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        # Market hours (critical)
        passed, msg, score = self.validate_market_hours(signal, instrument)
        validations['market_hours'] = {
            'passed': bool(passed),
            'message': str(msg),
            'score': float(score),
            'critical': True
        }
        
        # Volume (critical)
        passed, msg, score = self.validate_volume(signal, data)
        validations['volume'] = {
            'passed': bool(passed),
            'message': str(msg),
            'score': float(score),
            'critical': True
        }
        
        # Volatility
        passed, msg, score = self.validate_volatility(signal, data)
        validations['volatility'] = {
            'passed': bool(passed),
            'message': str(msg),
            'score': float(score),
            'critical': False
        }
        
        # Trend strength
        passed, msg, score = self.validate_trend_strength(signal, data)
        validations['trend'] = {
            'passed': bool(passed),
            'message': str(msg),
            'score': float(score),
            'critical': False
        }
        
        # Multi-timeframe (if additional data provided)
        if additional_data and 'timeframe_data' in additional_data:
            passed, msg, score = self.validate_multi_timeframe(
                signal, 
                additional_data['timeframe_data'],
                instrument
            )
            validations['timeframe'] = {
                'passed': bool(passed),
                'message': str(msg),
                'score': float(score),
                'critical': True
            }
        
        # Correlation (if additional data provided)
        if additional_data and 'correlated_pairs' in additional_data:
            passed, msg, score = self.validate_correlation(
                signal,
                data,
                additional_data['correlated_pairs']
            )
            validations['correlation'] = {
                'passed': bool(passed),
                'message': str(msg),
                'score': float(score),
                'critical': False
            }
        
        # Log validation
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal.get('id', 'unknown'),
            'instrument': instrument,
            'validations': validations,
            'overall_passed': bool(all(v['passed'] for v in validations.values() if v.get('critical', False)))
        })
        
        return validations
    
    def get_validation_summary(self, signal_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of validation history.
        
        Args:
            signal_id: Optional signal ID to filter
        
        Returns:
            Dictionary of validation statistics
        """
        if signal_id:
            history = [v for v in self.validation_history if v['signal_id'] == signal_id]
        else:
            history = self.validation_history[-100:]  # Last 100 validations
        
        if not history:
            return {}
        
        # Calculate pass rates
        pass_rates = {}
        avg_scores = {}
        
        validation_types = ['market_hours', 'volume', 'volatility', 'trend', 'timeframe', 'correlation']
        
        for vtype in validation_types:
            passes = [h['validations'].get(vtype, {}).get('passed', False) 
                     for h in history if vtype in h['validations']]
            scores = [h['validations'].get(vtype, {}).get('score', 0) 
                     for h in history if vtype in h['validations']]
            
            if passes:
                pass_rates[vtype] = float(sum(passes) / len(passes))
                avg_scores[vtype] = float(np.mean(scores))
        
        # Calculate overall statistics
        overall_passed = [h['overall_passed'] for h in history]
        
        return {
            'total_validations': len(history),
            'overall_pass_rate': float(sum(overall_passed) / len(history)) if overall_passed else 0,
            'pass_rates': {k: float(v) for k, v in pass_rates.items()},
            'average_scores': {k: float(v) for k, v in avg_scores.items()},
            'recent_timestamp': str(history[-1]['timestamp']) if history else None
        }