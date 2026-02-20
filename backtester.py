"""
Enhanced backtesting engine for NEXUS Trading System.
Implements realistic trading simulation with risk management.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import logging
from itertools import product

import config

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    direction: int = 1  # 1 for long, -1 for short
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    position_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""  # 'tp', 'sl', 'signal', 'time', 'end'
    holding_periods: int = 0
    confidence: float = 0.5
    commission: float = 0.0
    slippage: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    initial_capital: float
    final_capital: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_periods: float
    avg_confidence: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)

class Backtester:
    """
    Enhanced backtesting engine with realistic trading simulation.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize backtester.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or config.BACKTEST_CONFIG
        self.results = {}
        
    def run(self,
           data: pd.DataFrame,
           signals: List[Dict[str, Any]],
           capital: float = None,
           risk_per_trade: float = None,
           commission: float = None,
           slippage: float = None,
           max_positions: int = None) -> BacktestResult:
        """
        Run backtest with given data and signals.
        
        Args:
            data: DataFrame with OHLCV data
            signals: List of signal dictionaries
            capital: Initial capital
            risk_per_trade: Risk per trade as decimal
            commission: Commission as decimal
            slippage: Slippage as decimal
            max_positions: Maximum concurrent positions
        
        Returns:
            BacktestResult object
        """
        if capital is None:
            capital = self.config['default_capital']
        if risk_per_trade is None:
            risk_per_trade = self.config['default_risk_per_trade']
        if commission is None:
            commission = self.config['commission']
        if slippage is None:
            slippage = self.config['slippage']
        if max_positions is None:
            max_positions = self.config.get('max_open_positions', 5)
        
        logger.info(f"Starting backtest with capital=${capital:,.2f}, risk={risk_per_trade:.1%}")
        
        # Sort signals by time
        signals = sorted(signals, key=lambda x: x['timestamp'])
        
        trades = []
        equity = [capital]
        open_trades = []  # List of open trades
        
        # Track equity curve
        daily_equity = []
        peak = capital
        
        # Process each bar
        for i in range(len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            
            # Check for new signals (only if under max positions)
            if len(open_trades) < max_positions:
                for signal in signals:
                    signal_time = signal['timestamp']
                    
                    # Convert to comparable type
                    if isinstance(signal_time, str):
                        signal_time = pd.to_datetime(signal_time)
                    
                    # Check if signal time is within current bar
                    if i > 0:
                        prev_time = data.index[i-1]
                        if prev_time <= signal_time <= current_time:
                            # Enter new trade
                            trade = self._enter_trade(
                                signal, current_price, capital + sum(t.pnl for t in open_trades),
                                risk_per_trade, slippage, current_time
                            )
                            open_trades.append(trade)
                            logger.debug(f"Entered {trade.direction} trade at {current_price:.5f}")
            
            # Manage open trades
            trades_to_remove = []
            for trade in open_trades:
                # Update max favorable/adverse
                if trade.direction == 1:  # Long
                    trade.max_favorable = max(trade.max_favorable, (current_high - trade.entry_price) / trade.entry_price)
                    trade.max_adverse = min(trade.max_adverse, (current_low - trade.entry_price) / trade.entry_price)
                else:  # Short
                    trade.max_favorable = max(trade.max_favorable, (trade.entry_price - current_low) / trade.entry_price)
                    trade.max_adverse = min(trade.max_adverse, (trade.entry_price - current_high) / trade.entry_price)
                
                # Check stop loss and take profit
                exit_trade = None
                
                if trade.direction == 1:  # Long
                    if current_low <= trade.stop_loss:
                        exit_trade = self._exit_trade(
                            trade, trade.stop_loss, 'sl', current_time, commission
                        )
                    elif current_high >= trade.take_profit:
                        exit_trade = self._exit_trade(
                            trade, trade.take_profit, 'tp', current_time, commission
                        )
                else:  # Short
                    if current_high >= trade.stop_loss:
                        exit_trade = self._exit_trade(
                            trade, trade.stop_loss, 'sl', current_time, commission
                        )
                    elif current_low <= trade.take_profit:
                        exit_trade = self._exit_trade(
                            trade, trade.take_profit, 'tp', current_time, commission
                        )
                
                # Check maximum holding period (50 bars)
                if exit_trade is None and (i - trade.holding_periods) >= 50:
                    exit_trade = self._exit_trade(
                        trade, current_price, 'time', current_time, commission
                    )
                
                if exit_trade:
                    trades.append(exit_trade)
                    trades_to_remove.append(trade)
            
            # Remove closed trades
            for trade in trades_to_remove:
                open_trades.remove(trade)
            
            # Calculate current equity (mark-to-market)
            current_equity = capital
            for trade in open_trades:
                if trade.direction == 1:
                    unrealized_pnl = (current_price - trade.entry_price) * trade.position_size
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.position_size
                current_equity += unrealized_pnl
            
            equity.append(current_equity)
            
            # Track peak for drawdown
            peak = max(peak, current_equity)
            drawdown = (peak - current_equity) / peak if peak > 0 else 0
            daily_equity.append((current_time, current_equity, drawdown))
        
        # Close any remaining trades at end
        for trade in open_trades:
            final_trade = self._exit_trade(
                trade, data['close'].iloc[-1], 'end', data.index[-1], commission
            )
            trades.append(final_trade)
            capital += final_trade.pnl
        
        # Calculate final capital
        final_capital = capital
        
        # Calculate metrics
        result = self._calculate_metrics(trades, equity, final_capital, daily_equity)
        
        logger.info(f"Backtest complete: {result.total_trades} trades, Return={result.total_return:.2%}, Sharpe={result.sharpe_ratio:.2f}")
        
        return result
    
    def _enter_trade(self,
                     signal: Dict[str, Any],
                     current_price: float,
                     capital: float,
                     risk_per_trade: float,
                     slippage: float,
                     timestamp: datetime) -> Trade:
        """
        Enter a new trade with position sizing.
        
        Args:
            signal: Signal dictionary
            current_price: Current market price
            capital: Current capital
            risk_per_trade: Risk per trade as decimal
            slippage: Slippage as decimal
            timestamp: Entry timestamp
        
        Returns:
            Trade object
        """
        direction = signal['direction']
        
        # Apply slippage to entry
        if direction == 1:
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)
        
        # Calculate position size based on risk
        stop_distance = abs(entry_price - signal['stop_loss'])
        risk_amount = capital * risk_per_trade
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Cap position size to avoid excessive leverage
        max_position_value = capital * self.config.get('max_leverage', 50)
        position_value = position_size * entry_price
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
        
        trade = Trade(
            entry_time=timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            position_size=position_size,
            position_value=position_size * entry_price,
            confidence=signal.get('confidence', 0.5),
            holding_periods=0,
            max_favorable=0.0,
            max_adverse=0.0
        )
        
        return trade
    
    def _exit_trade(self,
                    trade: Trade,
                    exit_price: float,
                    reason: str,
                    timestamp: datetime,
                    commission: float) -> Trade:
        """
        Exit a trade and calculate P&L.
        
        Args:
            trade: Trade object
            exit_price: Exit price
            reason: Exit reason
            timestamp: Exit timestamp
            commission: Commission rate
        
        Returns:
            Updated Trade object
        """
        trade.exit_time = timestamp
        
        # Apply slippage to exit
        if trade.direction == 1:
            trade.exit_price = exit_price * (1 - commission)
        else:
            trade.exit_price = exit_price * (1 + commission)
        
        trade.exit_reason = reason
        trade.holding_periods = len(pd.date_range(trade.entry_time, timestamp)) if hasattr(timestamp, 'date') else 1
        
        # Calculate P&L
        if trade.direction == 1:
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
            trade.pnl_pct = (trade.exit_price / trade.entry_price) - 1
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
            trade.pnl_pct = 1 - (trade.exit_price / trade.entry_price)
        
        # Calculate commission
        trade.commission = commission * trade.position_value
        
        return trade
    
    def _calculate_metrics(self, 
                          trades: List[Trade], 
                          equity: List[float],
                          final_capital: float,
                          daily_equity: List[Tuple]) -> BacktestResult:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            trades: List of completed trades
            equity: Equity curve
            final_capital: Final capital
            daily_equity: Daily equity data
        
        Returns:
            BacktestResult object
        """
        if not trades:
            return BacktestResult(
                initial_capital=equity[0] if equity else 0,
                final_capital=final_capital,
                total_return=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                calmar_ratio=0,
                expectancy=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                avg_holding_periods=0,
                avg_confidence=0
            )
        
        # Basic trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        initial_capital = equity[0] if equity else 0
        
        # Calculate returns for Sharpe/Sortino
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (assuming risk-free rate = 0)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino = 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        max_drawdown_pct = abs(max_drawdown)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calmar ratio
        calmar = (returns.mean() * 252) / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Trade statistics
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        largest_win = max((t.pnl for t in winning_trades), default=0)
        largest_loss = min((t.pnl for t in losing_trades), default=0)
        
        avg_holding = np.mean([t.holding_periods for t in trades]) if trades else 0
        avg_confidence = np.mean([t.confidence for t in trades]) if trades else 0
        
        # Monthly returns
        monthly_returns = {}
        if daily_equity:
            df = pd.DataFrame(daily_equity, columns=['date', 'equity', 'drawdown'])
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M')
            df['returns'] = df['equity'].pct_change()
            monthly_returns = df.groupby('month')['returns'].sum().to_dict()
        
        return BacktestResult(
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=(final_capital / initial_capital) - 1 if initial_capital > 0 else 0,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(trades) if trades else 0,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            calmar_ratio=calmar,
            expectancy=total_pnl / len(trades) if trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_periods=avg_holding,
            avg_confidence=avg_confidence,
            trades=trades,
            equity_curve=equity,
            drawdown_curve=drawdown.tolist() if len(drawdown) > 0 else [],
            monthly_returns={str(k): v for k, v in monthly_returns.items()}
        )
    
    def optimize_parameters(self,
                           data: pd.DataFrame,
                           signals: List[Dict[str, Any]],
                           param_grid: Dict[str, List[Any]],
                           metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize backtest parameters using grid search.
        
        Args:
            data: DataFrame with OHLCV data
            signals: List of signals
            param_grid: Dictionary of parameters to optimize
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor', 'calmar_ratio')
        
        Returns:
            Dictionary with best parameters and results
        """
        best_score = -float('inf')
        best_params = None
        best_result = None
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        total_combinations = np.prod([len(v) for v in values])
        logger.info(f"Testing {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(product(*values)):
            params = dict(zip(keys, combination))
            
            # Run backtest with these parameters
            result = self.run(
                data=data,
                signals=signals,
                capital=params.get('capital', 10000),
                risk_per_trade=params.get('risk_per_trade', 0.02),
                commission=params.get('commission', 0.0001),
                slippage=params.get('slippage', 0.0001)
            )
            
            # Get optimization score
            if metric == 'sharpe_ratio':
                score = result.sharpe_ratio
            elif metric == 'total_return':
                score = result.total_return
            elif metric == 'profit_factor':
                score = result.profit_factor if result.profit_factor != float('inf') else 10
            elif metric == 'calmar_ratio':
                score = result.calmar_ratio
            else:
                # Combined score
                score = (
                    result.sharpe_ratio * 0.3 +
                    result.total_return * 0.3 +
                    min(result.profit_factor, 5) * 0.2 +
                    result.calmar_ratio * 0.2
                )
            
            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
            
            # Log progress every 10%
            if (i + 1) % max(1, total_combinations // 10) == 0:
                progress = (i + 1) / total_combinations * 100
                logger.info(f"Optimization progress: {progress:.1f}%")
        
        # Convert result to serializable dict
        result_dict = None
        if best_result:
            result_dict = {
                'initial_capital': best_result.initial_capital,
                'final_capital': best_result.final_capital,
                'total_return': best_result.total_return,
                'total_trades': best_result.total_trades,
                'win_rate': best_result.win_rate,
                'profit_factor': best_result.profit_factor,
                'sharpe_ratio': best_result.sharpe_ratio,
                'max_drawdown_pct': best_result.max_drawdown_pct
            }
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': result_dict,
            'combinations_tested': total_combinations
        }
    
    def walk_forward_analysis(self,
                             data: pd.DataFrame,
                             signals: List[Dict[str, Any]],
                             train_window: int = 252,
                             test_window: int = 63,
                             step_size: int = 21) -> Dict[str, Any]:
        """
        Perform walk-forward analysis.
        
        Args:
            data: DataFrame with OHLCV data
            signals: List of signals
            train_window: Training window size in bars
            test_window: Testing window size in bars
            step_size: Step size for rolling window
        
        Returns:
            Dictionary with walk-forward results
        """
        if len(data) < train_window + test_window:
            return {'error': 'Insufficient data for walk-forward analysis'}
        
        results = []
        
        for start in range(0, len(data) - train_window - test_window, step_size):
            train_end = start + train_window
            test_end = train_end + test_window
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            # Filter signals for test period
            test_signals = [s for s in signals 
                          if train_end <= data.index.get_loc(s['timestamp']) < test_end]
            
            if not test_signals:
                continue
            
            # Run backtest on test period
            result = self.run(
                data=test_data,
                signals=test_signals,
                capital=self.config['default_capital'],
                risk_per_trade=self.config['default_risk_per_trade']
            )
            
            results.append({
                'period': f"{data.index[train_end]}-{data.index[test_end-1]}",
                'trades': result.total_trades,
                'return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'max_dd': result.max_drawdown_pct
            })
        
        # Aggregate results
        if not results:
            return {'error': 'No valid test periods'}
        
        df_results = pd.DataFrame(results)
        
        return {
            'periods': len(results),
            'avg_return': df_results['return'].mean(),
            'avg_sharpe': df_results['sharpe'].mean(),
            'avg_max_dd': df_results['max_dd'].mean(),
            'consistency': (df_results['return'] > 0).mean(),
            'results': results
        }