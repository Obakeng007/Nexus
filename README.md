# NEXUS Trading System

A comprehensive algorithmic trading system with ML-powered signals, backtesting, and a professional dark trading terminal UI.

## Features

### 🤖 Machine Learning Engine
- **Ensemble model**: Random Forest + Gradient Boosting
- **40+ features**: RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, CCI, ADX, OBV, Ichimoku, Donchian Channel, Keltner Channel, MFI, and more
- **Time-series cross-validation** with 5-fold split
- **Confidence scoring** on all signals
- **Label engineering**: Forward-looking returns with configurable threshold

### 📊 Instruments
- EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, NZD/USD
- XAU/USD (Gold)

### 📈 Backtest Engine
- Risk-based position sizing (% risk per trade)
- ATR-based stop loss & take profit
- Slippage and commission modeling
- **Metrics**: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor, Expectancy
- Full equity curve and trade log

### 🖥️ Frontend Dashboard
- Live signal scanner with confidence bars
- Interactive charts: Price + EMA + Bollinger Bands, RSI, MACD, ADX
- Backtest configuration and results
- Model management & training status
- CSV data upload

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
cd trading_system
python app.py
```

### 3. Open browser
Navigate to: http://localhost:5000

## Usage

### Getting Signals
1. The system generates rule-based signals immediately on startup
2. Click **"TRAIN ALL"** to train ML models (takes 1-2 min)
3. Click any signal card to see detailed analysis with entry/SL/TP levels

### Uploading Your Data
1. Go to **"DATA UPLOAD"** tab
2. Select instrument and upload a CSV with columns: `date, open, high, low, close, volume`
3. Go to **"MODELS"** tab and train the model for that instrument

### Running a Backtest
1. Go to **"BACKTEST"** tab
2. Configure parameters (capital, risk %, confidence threshold)
3. Click **"RUN BACKTEST"**
4. Review equity curve and trade log

### Chart Analysis
1. Go to **"CHART ANALYSIS"** tab
2. Select instrument and period
3. View price chart with overlaid indicators, RSI, MACD, ADX panels

## File Structure
```
trading_system/
├── app.py              # Flask API server
├── data_manager.py     # Data loading & synthetic generation
├── indicators.py       # 20+ technical indicators
├── ml_engine.py        # ML model training & signal generation
├── backtester.py       # Backtesting engine
├── requirements.txt    
├── models/             # Saved trained models (auto-created)
├── data/               # Uploaded CSV files (auto-created)
└── templates/
    └── index.html      # Frontend dashboard
```

## Adding New Instruments
Edit `data_manager.py` and add to `INSTRUMENTS` dict:
```python
'BTCUSD': {'base_price': 45000.0, 'volatility': 500.0, 'trend': 0.0001, 'spread': 5.0},
```

## Notes
- When no CSV data is uploaded, the system uses synthetic data (Geometric Brownian Motion with regime switching) to demonstrate functionality
- For real trading, always upload real historical data before training
- Signals are for educational purposes only — not financial advice
