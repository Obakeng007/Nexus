"""
Signal performance tracking module for NEXUS Trading System.
Tracks actual performance for continuous improvement.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import logging

import config

logger = logging.getLogger(__name__)

class SignalPerformanceTracker:
    """
    Tracks actual performance of signals for continuous improvement.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize performance tracker.
        
        Args:
            db_path: Path to performance database file
        """
        self.db_path = db_path or str(config.DATA_DIR / 'signal_performance.json')
        self.performance_history = self._load_history()
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history from file."""
        if Path(self.db_path).exists():
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
                return []
        return []
    
    def _save_history(self):
        """Save performance history to file."""
        try:
            # Create backup if file exists
            if Path(self.db_path).exists():
                backup_path = str(self.db_path) + '.bak'
                import shutil
                shutil.copy2(self.db_path, backup_path)
            
            with open(self.db_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def log_signal_outcome(self, signal_outcome: Dict[str, Any]):
        """
        Log actual outcome of a signal.
        
        Expected fields:
        - signal_id: unique identifier
        - instrument: trading instrument
        - direction: 1 for long, -1 for short
        - entry_time: timestamp
        - exit_time: timestamp
        - entry_price: float
        - exit_price: float
        - pnl: float (actual profit/loss)
        - pnl_pct: float (percentage return)
        - confidence: float (signal confidence at entry)
        - validations_passed: bool
        - exit_reason: str
        - metadata: dict (optional additional data)
        """
        required_fields = ['signal_id', 'instrument', 'direction', 'pnl']
        missing = [f for f in required_fields if f not in signal_outcome]
        
        if missing:
            logger.error(f"Missing required fields: {missing}")
            return
        
        record = {
            'timestamp': datetime.now().isoformat(),
            **signal_outcome
        }
        
        self.performance_history.append(record)
        
        # Keep only last 10000 records
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-10000:]
        
        self._save_history()
        
        logger.info(f"Logged outcome for signal {signal_outcome.get('signal_id', 'unknown')}")
    
    def analyze_performance(self, 
                           instrument: Optional[str] = None,
                           days_back: int = 30,
                           min_samples: int = 10) -> Dict[str, Any]:
        """
        Analyze signal performance to refine thresholds.
        
        Args:
            instrument: Optional instrument filter
            days_back: Number of days to look back
            min_samples: Minimum samples required for analysis
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.performance_history)
        
        # Convert timestamp strings to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        
        # Filter by instrument
        if instrument:
            df = df[df['instrument'] == instrument]
        
        # Filter by date
        cutoff = datetime.now() - timedelta(days=days_back)
        
        if 'entry_time' in df.columns:
            recent = df[df['entry_time'] > cutoff]
        elif 'timestamp' in df.columns:
            recent = df[df['timestamp'] > cutoff]
        else:
            recent = df
        
        if len(recent) < min_samples:
            return {
                'error': f'Insufficient recent data: {len(recent)} samples (need {min_samples})',
                'total_samples': len(recent)
            }
        
        # Basic statistics
        analysis = {
            'instrument': instrument or 'all',
            'period_days': days_back,
            'total_signals': len(recent),
            'winning_trades': len(recent[recent['pnl'] > 0]),
            'losing_trades': len(recent[recent['pnl'] <= 0]),
            'avg_pnl': float(recent['pnl'].mean()),
            'avg_pnl_pct': float(recent['pnl_pct'].mean()) if 'pnl_pct' in recent.columns else 0,
            'total_pnl': float(recent['pnl'].sum()),
            'std_pnl': float(recent['pnl'].std()) if len(recent) > 1 else 0,
        }
        
        # Win rate
        analysis['win_rate'] = analysis['winning_trades'] / analysis['total_signals'] if analysis['total_signals'] > 0 else 0
        
        # Average win/loss
        wins = recent[recent['pnl'] > 0]
        losses = recent[recent['pnl'] <= 0]
        
        analysis['avg_win'] = float(wins['pnl'].mean()) if len(wins) > 0 else 0
        analysis['avg_loss'] = float(losses['pnl'].mean()) if len(losses) > 0 else 0
        analysis['largest_win'] = float(wins['pnl'].max()) if len(wins) > 0 else 0
        analysis['largest_loss'] = float(losses['pnl'].min()) if len(losses) > 0 else 0
        
        # Profit factor
        gross_profit = float(wins['pnl'].sum()) if len(wins) > 0 else 0
        gross_loss = float(abs(losses['pnl'].sum())) if len(losses) > 0 else 0
        analysis['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        analysis['expectancy'] = (analysis['win_rate'] * analysis['avg_win'] - 
                                 (1 - analysis['win_rate']) * abs(analysis['avg_loss']))
        
        # Confidence analysis
        if 'confidence' in recent.columns:
            # Group by confidence buckets
            recent['confidence_bin'] = pd.cut(
                recent['confidence'], 
                bins=[0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['<30%', '30-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90%+']
            )
            
            confidence_perf = recent.groupby('confidence_bin').agg({
                'pnl': ['count', 'mean', 'std'],
                'signal_id': 'count'
            }).round(3)
            
            analysis['confidence_performance'] = confidence_perf.to_dict()
            
            # Find optimal confidence threshold
            thresholds = np.arange(0.5, 0.95, 0.05)
            best_threshold = 0.5
            best_win_rate = analysis['win_rate']
            best_sharpe = -float('inf')
            
            for thresh in thresholds:
                filtered = recent[recent['confidence'] > thresh]
                if len(filtered) >= 5:
                    win_rate = len(filtered[filtered['pnl'] > 0]) / len(filtered)
                    returns = filtered['pnl_pct'].values if 'pnl_pct' in filtered.columns else filtered['pnl'].values / 10000
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_threshold = thresh
                        best_win_rate = win_rate
            
            analysis['optimal_confidence_threshold'] = float(best_threshold)
            analysis['win_rate_at_optimal'] = float(best_win_rate)
            analysis['sharpe_at_optimal'] = float(best_sharpe)
        
        # Validation analysis
        if 'validations_passed' in recent.columns:
            validated = recent[recent['validations_passed'] == True]
            unvalidated = recent[recent['validations_passed'] == False]
            
            analysis['validation_impact'] = {
                'validated_count': len(validated),
                'validated_win_rate': len(validated[validated['pnl'] > 0]) / len(validated) if len(validated) > 0 else 0,
                'validated_avg_pnl': float(validated['pnl'].mean()) if len(validated) > 0 else 0,
                'unvalidated_count': len(unvalidated),
                'unvalidated_win_rate': len(unvalidated[unvalidated['pnl'] > 0]) / len(unvalidated) if len(unvalidated) > 0 else 0,
                'unvalidated_avg_pnl': float(unvalidated['pnl'].mean()) if len(unvalidated) > 0 else 0,
            }
            
            if analysis['validation_impact']['validated_count'] > 0:
                analysis['validation_improvement'] = (
                    analysis['validation_impact']['validated_win_rate'] - 
                    analysis['validation_impact']['unvalidated_win_rate']
                )
        
        # Exit reason analysis
        if 'exit_reason' in recent.columns:
            exit_analysis = {}
            for reason in recent['exit_reason'].unique():
                reason_trades = recent[recent['exit_reason'] == reason]
                if len(reason_trades) >= 3:
                    exit_analysis[reason] = {
                        'count': len(reason_trades),
                        'win_rate': len(reason_trades[reason_trades['pnl'] > 0]) / len(reason_trades),
                        'avg_pnl': float(reason_trades['pnl'].mean()),
                        'avg_holding': float(reason_trades['holding_periods'].mean()) if 'holding_periods' in reason_trades.columns else 0
                    }
            analysis['exit_reason_analysis'] = exit_analysis
        
        # Time-based analysis
        if 'entry_time' in recent.columns:
            recent['hour'] = pd.to_datetime(recent['entry_time']).dt.hour
            recent['day_of_week'] = pd.to_datetime(recent['entry_time']).dt.dayofweek
            
            hourly = recent.groupby('hour')['pnl'].agg(['mean', 'count', 'std']).round(3)
            daily = recent.groupby('day_of_week')['pnl'].agg(['mean', 'count', 'std']).round(3)
            
            analysis['hourly_performance'] = hourly.to_dict()
            analysis['daily_performance'] = daily.to_dict()
        
        # Sharpe ratio of signals
        if len(recent) > 1:
            if 'pnl_pct' in recent.columns:
                returns = recent['pnl_pct'].values
            else:
                # Approximate returns from pnl
                returns = recent['pnl'].values / 10000
            
            if returns.std() > 0:
                analysis['signal_sharpe'] = float((returns.mean() / returns.std()) * np.sqrt(252))
            else:
                analysis['signal_sharpe'] = 0
        
        # Add timestamp
        analysis['analysis_time'] = datetime.now().isoformat()
        
        logger.info(f"Performance analysis complete: Win rate={analysis['win_rate']:.2%}, Sharpe={analysis.get('signal_sharpe', 0):.2f}")
        
        return analysis
    
    def get_underperforming_instruments(self, min_trades: int = 10) -> List[Dict[str, Any]]:
        """
        Identify instruments that are consistently underperforming.
        
        Args:
            min_trades: Minimum number of trades required
        
        Returns:
            List of underperforming instruments with stats
        """
        if not self.performance_history:
            return []
        
        df = pd.DataFrame(self.performance_history)
        
        if len(df) < min_trades:
            return []
        
        underperforming = []
        
        for instrument in df['instrument'].unique():
            inst_trades = df[df['instrument'] == instrument]
            
            if len(inst_trades) >= min_trades:
                win_rate = len(inst_trades[inst_trades['pnl'] > 0]) / len(inst_trades)
                avg_pnl = inst_trades['pnl'].mean()
                
                # Check if underperforming (win rate < 40% or negative avg pnl)
                if win_rate < 0.4 or avg_pnl < 0:
                    underperforming.append({
                        'instrument': instrument,
                        'win_rate': float(win_rate),
                        'trades': len(inst_trades),
                        'avg_pnl': float(avg_pnl),
                        'total_pnl': float(inst_trades['pnl'].sum())
                    })
        
        return underperforming
    
    def generate_improvement_recommendations(self, analysis: Optional[Dict] = None) -> List[str]:
        """
        Generate recommendations for improving signal quality.
        
        Args:
            analysis: Optional pre-computed analysis dictionary
        
        Returns:
            List of recommendation strings
        """
        if analysis is None:
            analysis = self.analyze_performance(days_back=90)
        
        if 'error' in analysis:
            return ["Insufficient data for recommendations"]
        
        recommendations = []
        
        # Check win rate
        if analysis['win_rate'] < 0.45:
            recommendations.append(
                f"⚠️ Low win rate ({analysis['win_rate']:.1%}). Consider increasing confidence threshold or reviewing signal logic."
            )
        elif analysis['win_rate'] > 0.6:
            recommendations.append(
                f"✅ Good win rate ({analysis['win_rate']:.1%}). You could potentially decrease confidence threshold for more signals."
            )
        
        # Check profit factor
        if analysis['profit_factor'] < 1.5:
            recommendations.append(
                f"⚠️ Low profit factor ({analysis['profit_factor']:.2f}). Review risk management (SL/TP ratios) and exit strategies."
            )
        elif analysis['profit_factor'] > 3:
            recommendations.append(
                f"✅ Excellent profit factor ({analysis['profit_factor']:.2f})"
            )
        
        # Check confidence threshold
        if 'optimal_confidence_threshold' in analysis:
            current_threshold = config.ML_CONFIG['training']['confidence_threshold']
            optimal = analysis['optimal_confidence_threshold']
            
            if abs(current_threshold - optimal) > 0.1:
                direction = "increase" if optimal > current_threshold else "decrease"
                recommendations.append(
                    f"💡 Consider {direction} confidence threshold from {current_threshold:.2f} to {optimal:.2f} (would give {analysis['win_rate_at_optimal']:.1%} win rate)"
                )
        
        # Check validation impact
        if 'validation_impact' in analysis:
            vi = analysis['validation_impact']
            if vi.get('validated_count', 0) > 10 and vi.get('unvalidated_count', 0) > 10:
                improvement = vi['validated_win_rate'] - vi['unvalidated_win_rate']
                if improvement > 0.1:
                    recommendations.append(
                        f"✅ Validation is effective: +{improvement:.1%} win rate with validation"
                    )
                elif improvement < -0.1:
                    recommendations.append(
                        f"⚠️ Validation may be too strict: validated trades have {abs(improvement):.1%} lower win rate. Consider relaxing validation rules."
                    )
                elif abs(improvement) < 0.05:
                    recommendations.append(
                        f"ℹ️ Validation has minimal impact. Consider removing if it's reducing signal count."
                    )
        
        # Check exit reasons
        if 'exit_reason_analysis' in analysis:
            exit_analysis = analysis['exit_reason_analysis']
            
            # Check if too many time exits
            if 'time' in exit_analysis and exit_analysis['time']['count'] > analysis['total_signals'] * 0.3:
                if exit_analysis['time']['win_rate'] < 0.4:
                    recommendations.append(
                        f"⚠️ High rate of time-based exits ({exit_analysis['time']['count']} trades, {exit_analysis['time']['win_rate']:.1%} win rate). Consider adjusting holding period."
                    )
            
            # Check stop loss effectiveness
            if 'sl' in exit_analysis and exit_analysis['sl']['count'] > analysis['total_signals'] * 0.2:
                if exit_analysis['sl']['win_rate'] < 0.2:
                    recommendations.append(
                        f"⚠️ Stop losses are being hit frequently with low win rate. Consider widening stops or improving entry timing."
                    )
        
        # Check underperforming instruments
        underperforming = self.get_underperforming_instruments()
        if underperforming:
            inst_list = ", ".join([u['instrument'] for u in underperforming[:3]])
            recommendations.append(
                f"⚠️ Underperforming instruments: {inst_list}. Consider retraining models or disabling these instruments."
            )
        
        # Check time-based performance
        if 'hourly_performance' in analysis:
            hourly = pd.DataFrame(analysis['hourly_performance']).T
            best_hour = hourly['mean'].idxmax() if len(hourly) > 0 else None
            worst_hour = hourly['mean'].idxmin() if len(hourly) > 0 else None
            
            if best_hour and worst_hour and best_hour != worst_hour:
                recommendations.append(
                    f"⏰ Best trading hour: {best_hour}:00 ({hourly.loc[best_hour, 'mean']:.2%} avg return). "
                    f"Worst: {worst_hour}:00 ({hourly.loc[worst_hour, 'mean']:.2%} avg return)"
                )
        
        # If no recommendations, add a positive message
        if not recommendations:
            recommendations.append("✅ System performing well. Continue monitoring.")
        
        return recommendations
    
    def get_performance_report(self, instrument: Optional[str] = None, days: int = 30) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            instrument: Optional instrument filter
            days: Number of days to analyze
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_performance(instrument, days_back=days)
        
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"📊 SIGNAL PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append(f"Instrument: {analysis.get('instrument', 'All')}")
        lines.append(f"Period: Last {analysis.get('period_days', 30)} days")
        lines.append(f"Analysis Time: {analysis.get('analysis_time', datetime.now().isoformat())}")
        lines.append("-" * 60)
        
        lines.append(f"Total Signals: {analysis.get('total_signals', 0)}")
        lines.append(f"Winning Trades: {analysis.get('winning_trades', 0)}")
        lines.append(f"Losing Trades: {analysis.get('losing_trades', 0)}")
        lines.append(f"Win Rate: {analysis.get('win_rate', 0):.1%}")
        lines.append(f"Profit Factor: {analysis.get('profit_factor', 0):.2f}")
        lines.append(f"Total P&L: ${analysis.get('total_pnl', 0):,.2f}")
        lines.append(f"Avg Trade: ${analysis.get('avg_pnl', 0):,.2f}")
        lines.append(f"Avg Win: ${analysis.get('avg_win', 0):,.2f}")
        lines.append(f"Avg Loss: ${analysis.get('avg_loss', 0):,.2f}")
        lines.append(f"Largest Win: ${analysis.get('largest_win', 0):,.2f}")
        lines.append(f"Largest Loss: ${analysis.get('largest_loss', 0):,.2f}")
        lines.append(f"Signal Sharpe: {analysis.get('signal_sharpe', 0):.2f}")
        lines.append(f"Expectancy: ${analysis.get('expectancy', 0):,.2f}")
        
        if 'optimal_confidence_threshold' in analysis:
            lines.append("-" * 60)
            lines.append(f"Optimal Confidence Threshold: {analysis['optimal_confidence_threshold']:.2f}")
            lines.append(f"Win Rate at Optimal: {analysis['win_rate_at_optimal']:.1%}")
            lines.append(f"Sharpe at Optimal: {analysis['sharpe_at_optimal']:.2f}")
        
        if 'validation_impact' in analysis:
            vi = analysis['validation_impact']
            lines.append("-" * 60)
            lines.append("Validation Impact:")
            lines.append(f"  Validated: {vi.get('validated_count', 0)} trades, {vi.get('validated_win_rate', 0):.1%} win rate")
            lines.append(f"  Unvalidated: {vi.get('unvalidated_count', 0)} trades, {vi.get('unvalidated_win_rate', 0):.1%} win rate")
            if 'validation_improvement' in analysis:
                imp = analysis['validation_improvement']
                lines.append(f"  Improvement: {imp:+.1%}")
        
        lines.append("-" * 60)
        lines.append("RECOMMENDATIONS:")
        recommendations = self.generate_improvement_recommendations(analysis)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)