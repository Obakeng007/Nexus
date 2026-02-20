"""
Configuration settings for NEXUS Trading System.
"""
import os
from pathlib import Path
import logging.config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Deriv API Configuration
DERIV_CONFIG = {
    'api_token': os.getenv('DERIV_API_TOKEN', ''),
    'app_id': os.getenv('DERIV_APP_ID', '1089'),
    'use_deriv': os.getenv('USE_DERIV', 'true').lower() == 'true',
    'default_timeframe': os.getenv('DERIV_DEFAULT_TIMEFRAME', '1h'),
    'cache_duration': int(os.getenv('DERIV_CACHE_DURATION', '3600'))
}

# Trading instruments configuration
INSTRUMENTS = {
    'EUR/USD': {
        'base_price': 1.1000,
        'volatility': 0.0010,
        'trend': 0.00001,
        'spread': 0.0001,
        'market_hours': (8, 17),
        'pip_size': 0.0001,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxEURUSD'
    },
    'GBP/USD': {
        'base_price': 1.2500,
        'volatility': 0.0012,
        'trend': 0.00002,
        'spread': 0.0001,
        'market_hours': (8, 17),
        'pip_size': 0.0001,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxGBPUSD'
    },
    'USD/JPY': {
        'base_price': 110.00,
        'volatility': 0.10,
        'trend': 0.001,
        'spread': 0.01,
        'market_hours': (19, 17),
        'pip_size': 0.01,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxUSDJPY'
    },
    'AUD/USD': {
        'base_price': 0.7500,
        'volatility': 0.0011,
        'trend': 0.000015,
        'spread': 0.0001,
        'market_hours': (0, 24),
        'pip_size': 0.0001,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxAUDUSD'
    },
    'USD/CAD': {
        'base_price': 1.3500,
        'volatility': 0.0011,
        'trend': 0.000015,
        'spread': 0.0001,
        'market_hours': (0, 24),
        'pip_size': 0.0001,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxUSDCAD'
    },
    'NZD/USD': {
        'base_price': 0.7000,
        'volatility': 0.0011,
        'trend': 0.000015,
        'spread': 0.0001,
        'market_hours': (0, 24),
        'pip_size': 0.0001,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxNZDUSD'
    },
    'USD/CHF': {
        'base_price': 0.9200,
        'volatility': 0.0011,
        'trend': 0.000015,
        'spread': 0.0001,
        'market_hours': (0, 24),
        'pip_size': 0.0001,
        'margin_requirement': 0.02,
        'min_volume': 1000,
        'max_leverage': 50,
        'deriv_symbol': 'frxUSDCHF'
    },
    'XAU/USD': {
        'base_price': 1800.00,
        'volatility': 5.00,
        'trend': 0.05,
        'spread': 0.50,
        'market_hours': (0, 24),
        'pip_size': 0.01,
        'margin_requirement': 0.05,
        'min_volume': 10,
        'max_leverage': 20,
        'deriv_symbol': 'frxXAUUSD'
    }
}

# ML Model Configuration
ML_CONFIG = {
    'ensemble': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 50,
        'random_state': 42,
        'n_jobs': -1
    },
    'training': {
        'test_size': 0.2,
        'validation_size': 0.1,
        'cv_folds': 5,
        'confidence_threshold': 0.6,
        'min_training_samples': 500
    },
    'features': {
        'lookback_periods': 100,
        'min_samples': 500,
        'use_market_regime': True,
        'use_microstructure': True
    }
}

# Label Generation Configuration - CORRECTED BARRIERS
LABEL_CONFIG = {
    'triple_barrier': {
        'upper_barrier_pct': 0.005,    # 0.5% take profit (was 2% - too wide)
        'lower_barrier_pct': 0.002,    # 0.2% stop loss (was 1% - too wide)
        'max_holding_periods': 48,      # Maximum bars to hold (increased)
        'min_return': 0.001,            # Minimum return to consider a win
        'use_volatility_adjusted': True  # Use ATR for dynamic barriers
    }
}

# Signal Validation Configuration
VALIDATION_CONFIG = {
    'multi_timeframe': {
        'higher_tf_multiplier': 4,
        'require_alignment': True
    },
    'volume': {
        'min_volume_ratio': 1.5,
        'volume_lookback': 20,
        'require_confirmation': True
    },
    'news': {
        'cooldown_minutes': 30,
        'high_impact_cooldown': 60,
        'use_news_filter': False
    },
    'volatility': {
        'max_volatility_ratio': 2.0,
        'min_volatility_ratio': 0.3
    },
    'trend': {
        'min_adx': 20,
        'strong_trend_adx': 25
    }
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'default_capital': 10000,
    'default_risk_per_trade': 0.02,
    'commission': 0.0001,
    'slippage': 0.0001,
    'margin_call_threshold': 0.8,
    'max_open_positions': 5,
    'require_confirmation': True
}

# API Configuration
API_CONFIG = {
    'host': os.getenv('API_HOST', 'localhost'),
    'port': int(os.getenv('API_PORT', '5000')),
    'debug': os.getenv('API_DEBUG', 'true').lower() == 'true',
    'secret_key': os.getenv('SECRET_KEY', 'nexus-trading-system-secret-key-2026'),
    'session_timeout': int(os.getenv('SESSION_TIMEOUT', '3600'))
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'trading_system.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG',
            'encoding': 'utf-8'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'error.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'ERROR',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO'
        },
        'werkzeug': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
        'deriv_fetcher': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}