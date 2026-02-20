"""
Label generation module for NEXUS Trading System.
Implements triple-barrier labeling and meta-labeling.
FIXED: Better barrier calculation based on volatility.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class BarrierResult:
    """Results from triple-barrier method."""
    label: int  # 1: win, 0: loss, -1: neutral
    barrier_hit: Optional[str]  # 'upper', 'lower', or None
    hit_time: Optional[datetime]
    holding_periods: int
    actual_return: float
    max_favorable: float
    max_adverse: float
    exit_price: float
    hit_index: Optional[int]

class TripleBarrierLabeler:
    """
    Implements the triple-barrier labeling method.
    Labels are: 1 (hit upper barrier first), 0 (hit lower barrier first), -1 (neither)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize triple-barrier labeler.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.upper_barrier_pct = self.config.get('upper_barrier_pct', 0.005)    # 0.5% take profit
        self.lower_barrier_pct = self.config.get('lower_barrier_pct', 0.002)    # 0.2% stop loss
        self.max_holding_periods = self.config.get('max_holding_periods', 48)   # Max bars to hold
        self.min_return = self.config.get('min_return', 0.001)                  # Min return to consider a win
        self.use_volatility_adjusted = self.config.get('use_volatility_adjusted', True)
    
    def get_volatility_adjusted_barriers(self, 
                                        prices: pd.Series, 
                                        atr: pd.Series,
                                        idx: int,
                                        atr_multiplier_tp: float = 1.5,
                                        atr_multiplier_sl: float = 1.0) -> Tuple[float, float]:
        """
        Calculate ATR-based barriers for volatility adjustment.
        
        Args:
            prices: Price series
            atr: ATR series
            idx: Current index
            atr_multiplier_tp: ATR multiplier for take profit
            atr_multiplier_sl: ATR multiplier for stop loss
        
        Returns:
            Tuple of (upper_barrier, lower_barrier)
        """
        current_price = prices.iloc[idx]
        current_atr = atr.iloc[idx] if atr is not None else current_price * 0.01
        
        # Use ATR to set dynamic barriers
        upper_barrier = current_price + (current_atr * atr_multiplier_tp)
        lower_barrier = current_price - (current_atr * atr_multiplier_sl)
        
        return upper_barrier, lower_barrier
    
    def get_triple_barrier_labels(self, 
                                  prices: pd.Series,
                                  timestamps,
                                  atr: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate labels using triple-barrier method.
        
        Args:
            prices: Price series
            timestamps: Timestamp index or series
            atr: Optional ATR series for volatility adjustment
        
        Returns:
            DataFrame with labels and barrier information
        """
        labels = []
        barrier_info = []
        
        total_bars = len(prices)
        
        # Convert timestamps to list for easier access
        if isinstance(timestamps, pd.DatetimeIndex):
            timestamp_list = timestamps.tolist()
        elif isinstance(timestamps, pd.Series):
            timestamp_list = timestamps.values
        else:
            timestamp_list = list(timestamps) if timestamps is not None else [None] * total_bars
        
        # Calculate dynamic barriers based on average volatility
        if atr is not None:
            # Use average ATR for the whole series to set consistent barriers
            avg_atr = atr.mean()
            avg_price = prices.mean()
            atr_pct = avg_atr / avg_price if avg_price > 0 else 0.01
            
            # Adjust barriers based on average volatility
            if atr_pct > 0.015:  # High volatility
                upper_pct = self.upper_barrier_pct * 1.5
                lower_pct = self.lower_barrier_pct * 1.5
                logger.info(f"High volatility detected (ATR%={atr_pct:.3%}), widening barriers")
            elif atr_pct < 0.005:  # Low volatility
                upper_pct = self.upper_barrier_pct * 0.7
                lower_pct = self.lower_barrier_pct * 0.7
                logger.info(f"Low volatility detected (ATR%={atr_pct:.3%}), tightening barriers")
            else:  # Normal volatility
                upper_pct = self.upper_barrier_pct
                lower_pct = self.lower_barrier_pct
        else:
            upper_pct = self.upper_barrier_pct
            lower_pct = self.lower_barrier_pct
        
        logger.info(f"Using barriers: TP={upper_pct:.3%}, SL={lower_pct:.3%}")
        
        for i in range(total_bars - 1):
            current_price = prices.iloc[i]
            current_time = timestamp_list[i] if i < len(timestamp_list) else None
            
            # Set barriers - either fixed percentage or ATR-based
            if self.use_volatility_adjusted and atr is not None and i < len(atr):
                # Use ATR-based dynamic barriers
                upper_barrier, lower_barrier = self.get_volatility_adjusted_barriers(
                    prices, atr, i
                )
            else:
                # Use fixed percentage barriers
                upper_barrier = current_price * (1 + upper_pct)
                lower_barrier = current_price * (1 - lower_pct)
            
            # Look forward
            max_lookahead = min(self.max_holding_periods, total_bars - i - 1)
            future_prices = prices.iloc[i+1:i+max_lookahead+1]
            
            # Get future timestamps
            if timestamp_list:
                future_timestamps = timestamp_list[i+1:i+max_lookahead+1]
            else:
                future_timestamps = [None] * len(future_prices)
            
            # Track extremes
            max_price = current_price
            min_price = current_price
            label = -1  # Default: neutral/expiry
            barrier_hit = None
            hit_time = None
            hit_index = None
            
            for j, (future_price, future_time) in enumerate(zip(future_prices, future_timestamps)):
                max_price = max(max_price, future_price)
                min_price = min(min_price, future_price)
                
                if future_price >= upper_barrier:
                    label = 1  # Hit upper barrier first
                    barrier_hit = 'upper'
                    hit_time = future_time
                    hit_index = j
                    break
                elif future_price <= lower_barrier:
                    label = 0  # Hit lower barrier first
                    barrier_hit = 'lower'
                    hit_time = future_time
                    hit_index = j
                    break
            
            # Calculate metrics
            if hit_index is not None:
                holding_periods = hit_index + 1
                exit_price = future_prices.iloc[hit_index]
            else:
                holding_periods = max_lookahead
                exit_price = future_prices.iloc[-1] if len(future_prices) > 0 else current_price
            
            actual_return = (exit_price / current_price) - 1
            max_favorable = (max_price / current_price) - 1
            max_adverse = (min_price / current_price) - 1
            
            # Create barrier result
            barrier_result = BarrierResult(
                label=label,
                barrier_hit=barrier_hit,
                hit_time=hit_time,
                holding_periods=holding_periods,
                actual_return=actual_return,
                max_favorable=max_favorable,
                max_adverse=max_adverse,
                exit_price=exit_price,
                hit_index=hit_index
            )
            
            labels.append(label)
            barrier_info.append(barrier_result)
        
        # Add last row as NaN (no future data)
        labels.append(np.nan)
        barrier_info.append(None)
        
        # Create result dataframe
        result = pd.DataFrame({
            'label': labels,
            'barrier_hit': [b.barrier_hit if b else None for b in barrier_info],
            'hit_time': [b.hit_time if b else None for b in barrier_info],
            'holding_periods': [b.holding_periods if b else np.nan for b in barrier_info],
            'actual_return': [b.actual_return if b else np.nan for b in barrier_info],
            'max_favorable': [b.max_favorable if b else np.nan for b in barrier_info],
            'max_adverse': [b.max_adverse if b else np.nan for b in barrier_info],
            'exit_price': [b.exit_price if b else np.nan for b in barrier_info]
        }, index=prices.index)
        
        # Add derived features
        result['return_to_risk'] = abs(result['actual_return']) / (abs(result['max_adverse']) + 1e-8)
        result['efficiency'] = abs(result['actual_return']) / (abs(result['max_favorable']) + abs(result['max_adverse']) + 1e-8)
        
        # Count labels
        valid_labels = result[result['label'] != -1]
        win_count = len(valid_labels[valid_labels['label'] == 1])
        loss_count = len(valid_labels[valid_labels['label'] == 0])
        neutral_count = len(result[result['label'] == -1])
        
        logger.info(f"Generated labels: Wins={win_count}, Losses={loss_count}, Neutral={neutral_count}")
        
        # Log barrier statistics
        if win_count + loss_count > 0:
            logger.info(f"Label distribution: Wins={win_count/(win_count+loss_count):.1%}, Losses={loss_count/(win_count+loss_count):.1%}")
        
        return result
    
    def get_meta_labels(self, 
                       prices: pd.Series,
                       primary_signals: pd.Series,
                       returns: pd.Series,
                       min_holding_periods: int = 5,
                       profit_threshold: float = 0.005) -> pd.Series:
        """
        Generate meta-labels: 1 if primary signal was profitable, 0 otherwise.
        
        Args:
            prices: Price series
            primary_signals: Series of primary signals (1, -1, 0)
            returns: Returns series
            min_holding_periods: Minimum holding periods to consider
            profit_threshold: Minimum profit to consider a win
        
        Returns:
            Series of meta-labels
        """
        meta_labels = pd.Series(index=primary_signals.index, dtype=float)
        
        for i in range(len(primary_signals) - min_holding_periods):
            if pd.isna(primary_signals.iloc[i]) or primary_signals.iloc[i] == 0:
                continue
            
            # Look forward
            max_lookahead = min(self.max_holding_periods, len(returns) - i - 1)
            future_returns = returns.iloc[i+1:i+max_lookahead+1]
            
            if len(future_returns) < min_holding_periods:
                continue
            
            # Calculate cumulative return
            cum_return = (1 + future_returns).prod() - 1
            
            # Meta-label: 1 if profitable, 0 if not
            # For long signals (1), profit is positive return
            # For short signals (-1), profit is negative return
            signal_type = primary_signals.iloc[i]
            is_profitable = (cum_return * signal_type) > profit_threshold
            
            meta_labels.iloc[i] = 1 if is_profitable else 0
        
        # Fill remaining with neutral
        meta_labels = meta_labels.fillna(0)
        
        positive_count = sum(meta_labels == 1)
        negative_count = sum(meta_labels == 0)
        
        logger.info(f"Generated meta-labels: Positive={positive_count}, Negative={negative_count}")
        
        return meta_labels
    
    def get_sample_weights(self, 
                          labels: pd.Series,
                          method: str = 'balanced',
                          decay_rate: float = 0.99) -> pd.Series:
        """
        Calculate sample weights for imbalanced datasets.
        
        Args:
            labels: Series of labels
            method: Weighting method ('balanced', 'time_decay', 'volatility_adjusted')
            decay_rate: Decay rate for time_decay method
        
        Returns:
            Series of sample weights
        """
        weights = pd.Series(1.0, index=labels.index)
        
        if method == 'balanced':
            # Balance by label frequency
            valid_labels = labels[labels != -1]
            if len(valid_labels) == 0:
                return weights
            
            label_counts = valid_labels.value_counts()
            max_count = label_counts.max()
            
            for label, count in label_counts.items():
                if count > 0:
                    weights[valid_labels[valid_labels == label].index] = max_count / count
        
        elif method == 'time_decay':
            # Weight recent samples more
            for i, idx in enumerate(labels.index):
                weights[idx] = decay_rate ** (len(labels) - i - 1)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def filter_by_holding_period(self, 
                                labels_df: pd.DataFrame,
                                min_holding: int = 1,
                                max_holding: Optional[int] = None) -> pd.DataFrame:
        """
        Filter labels by holding period.
        
        Args:
            labels_df: DataFrame from get_triple_barrier_labels
            min_holding: Minimum holding periods
            max_holding: Maximum holding periods
        
        Returns:
            Filtered DataFrame
        """
        result = labels_df.copy()
        
        # Filter by min holding
        result = result[result['holding_periods'] >= min_holding]
        
        # Filter by max holding if specified
        if max_holding is not None:
            result = result[result['holding_periods'] <= max_holding]
        
        return result
    
    def get_label_statistics(self, labels_df: pd.DataFrame) -> Dict[str, float]:
        """
        Get statistics about generated labels.
        
        Args:
            labels_df: DataFrame from get_triple_barrier_labels
        
        Returns:
            Dictionary of statistics
        """
        valid_labels = labels_df[labels_df['label'] != -1]
        
        if len(valid_labels) == 0:
            return {
                'total_samples': 0,
                'win_rate': 0,
                'loss_rate': 0,
                'avg_holding_period': 0,
                'avg_return': 0,
                'avg_max_favorable': 0,
                'avg_max_adverse': 0
            }
        
        wins = valid_labels[valid_labels['label'] == 1]
        losses = valid_labels[valid_labels['label'] == 0]
        
        stats = {
            'total_samples': len(valid_labels),
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': len(wins) / len(valid_labels) if len(valid_labels) > 0 else 0,
            'loss_rate': len(losses) / len(valid_labels) if len(valid_labels) > 0 else 0,
            'avg_holding_period': valid_labels['holding_periods'].mean(),
            'avg_return': valid_labels['actual_return'].mean(),
            'avg_max_favorable': valid_labels['max_favorable'].mean(),
            'avg_max_adverse': valid_labels['max_adverse'].mean(),
            'avg_return_to_risk': valid_labels['return_to_risk'].mean(),
            'avg_efficiency': valid_labels['efficiency'].mean()
        }
        
        if len(wins) > 0:
            stats['avg_win_return'] = wins['actual_return'].mean()
            stats['avg_win_holding'] = wins['holding_periods'].mean()
        
        if len(losses) > 0:
            stats['avg_loss_return'] = losses['actual_return'].mean()
            stats['avg_loss_holding'] = losses['holding_periods'].mean()
        
        return stats