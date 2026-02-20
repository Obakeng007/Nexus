# NEXUS Trading System

A comprehensive algorithmic trading system with ML-powered signals, backtesting, and a professional dark trading terminal UI. Now with **real-time data integration from Deriv API**!

![NEXUS Terminal](https://via.placeholder.com/800x400/0a0e1a/00ff88?text=NEXUS+Trading+Terminal)

## 🚀 Features

### 🤖 Machine Learning Engine
- **Ensemble model**: Random Forest + Gradient Boosting with meta-labeling
- **40+ technical features**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX, OBV, Ichimoku, Donchian Channel, Keltner Channel, MFI, and more
- **Time-series cross-validation** with 5-fold split
- **Confidence scoring** on all signals with probability calibration
- **Label engineering**: Triple-barrier method with configurable thresholds
- **Feature importance tracking** and model persistence

### 📊 Instruments
- Major Forex pairs: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, NZD/USD, USD/CHF
- Commodities: XAU/USD (Gold)
- **Extensible** for any instrument with Deriv API support

### 🔌 Real Data Integration
- **Deriv API** for real historical and real-time data
- Automatic fallback to synthetic data when API unavailable
- WebSocket connection for live price updates
- Multiple timeframe support (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- Intelligent caching to minimize API calls
- Rate limiting and error handling

### 📈 Backtest Engine
- Risk-based position sizing (% risk per trade)
- ATR-based stop loss & take profit
- Slippage and commission modeling
- **Comprehensive metrics**: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor, Expectancy
- Full equity curve and trade log
- Walk-forward analysis support
- Parameter optimization with grid search

### 🖥️ Frontend Dashboard
- Live signal scanner with confidence bars and validation badges
- Interactive charts: Price + EMA + Bollinger Bands, RSI, MACD, ADX
- Backtest configuration and results visualization
- Model management & training status with real-time updates
- CSV data upload with validation
- **Performance tracking** with win rate analysis by confidence level
- Multi-timeframe signal validation
- Real-time price display

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Deriv API token for real data - get one free at [app.deriv.com](https://app.deriv.com)

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Obakeng007/Nexus.git
cd Nexus/trading_system