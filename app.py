"""
NEXUS Trading System - Main Flask Application
Complete implementation with all endpoints.
"""
import os
import logging
import logging.config
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import json
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration
import config

# Import enhanced modules
from data_manager import DataManager
from ml_engine import MLEngine
from meta_model_trainer import MetaLabelingSystem
from signal_validator import SignalValidator
from backtester import Backtester
from signal_performance import SignalPerformanceTracker
from indicators import calculate_all_indicators

# Configure logging
logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Get Deriv configuration from environment
DERIV_TOKEN = os.getenv('DERIV_API_TOKEN')
USE_DERIV = os.getenv('USE_DERIV', 'true').lower() == 'true'

if DERIV_TOKEN:
    logger.info(f"Deriv API token found (length: {len(DERIV_TOKEN)})")
else:
    logger.warning("No Deriv API token found - using synthetic data only")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.API_CONFIG['secret_key']
app.config['UPLOAD_FOLDER'] = config.DATA_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=config.API_CONFIG['session_timeout'])

# Custom JSON encoder to handle numpy types
from flask.json.provider import JSONProvider
import json

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.datetime64):
            return pd.Timestamp(obj).isoformat()
        return super().default(obj)

class NumpyJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, cls=NumpyJSONEncoder, **kwargs)
    
    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)

# Set custom JSON provider
app.json = NumpyJSONProvider(app)

# Initialize components with Deriv token
data_manager = DataManager(use_deriv=USE_DERIV, deriv_token=DERIV_TOKEN)
ml_engine = MLEngine()
meta_labeler = MetaLabelingSystem()
signal_validator = SignalValidator(config.VALIDATION_CONFIG)
performance_tracker = SignalPerformanceTracker()
backtester = Backtester(config.BACKTEST_CONFIG)

# Global state
training_status = {}
active_signals = {}
backtest_results = {}

@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html', 
                         instruments=list(config.INSTRUMENTS.keys()),
                         current_year=datetime.now().year)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    try:
        # Get model status
        trained_models = list(ml_engine.models.keys())
        training_in_progress = [k for k, v in training_status.items() if v == 'training']
        
        # Get signal count
        total_signals = sum(len(signals) for signals in active_signals.values())
        
        # Get Deriv status
        deriv_status = {}
        if hasattr(data_manager, 'deriv_fetcher') and data_manager.deriv_fetcher:
            deriv_status = data_manager.deriv_fetcher.get_status()
        
        return jsonify({
            'success': True,
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'instruments': list(config.INSTRUMENTS.keys()),
            'trained_models': trained_models,
            'training_in_progress': training_in_progress,
            'active_signals': total_signals,
            'models_count': len(trained_models),
            'total_instruments': len(config.INSTRUMENTS),
            'system_uptime': 'online',
            'version': '2.0',
            'deriv': deriv_status,
            'using_real_data': bool(deriv_status.get('authorized', False))
        })
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/signals', methods=['GET'])
def get_signals():
    """Get trading signals with advanced validation."""
    instrument = request.args.get('instrument', 'EUR/USD')
    include_validation = request.args.get('validate', 'true').lower() == 'true'
    limit = int(request.args.get('limit', 10))
    
    try:
        # Get base signals from ML engine
        data = data_manager.get_data(instrument)
        signals = ml_engine.generate_signals(instrument, data)
        
        validated_signals = []
        
        for signal in signals[:limit]:
            # Convert numpy types to Python native types
            signal_dict = {
                'id': f"{instrument}_{datetime.now().timestamp()}_{len(validated_signals)}",
                'instrument': str(instrument),
                'direction': int(signal['direction']),
                'timestamp': datetime.now().isoformat(),
                'entry': float(round(signal['entry'], 5)),
                'stop_loss': float(round(signal['stop_loss'], 5)),
                'take_profit': float(round(signal['take_profit'], 5)),
                'confidence': float(round(signal['confidence'], 3)),
                'signal_type': str(signal.get('signal_type', 'ML'))
            }
            
            if include_validation:
                # Run validations
                validation_results = signal_validator.validate_all(
                    signal_dict, data, instrument
                )
                
                # Calculate quality score
                scores = [float(v['score']) for v in validation_results.values()]
                quality_score = float(np.mean(scores)) if scores else 0.5
                
                # Check if critical validations passed
                critical_passed = all(
                    bool(v['passed']) for v in validation_results.values() 
                    if v.get('critical', False)
                )
                
                signal_dict.update({
                    'validated': bool(critical_passed),
                    'validations': validation_results,
                    'quality_score': float(round(quality_score, 3))
                })
            
            validated_signals.append(signal_dict)
        
        # Update global signals
        active_signals[instrument] = validated_signals
        
        return jsonify({
            'success': True,
            'instrument': instrument,
            'signals': validated_signals,
            'count': len(validated_signals),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train ML model with advanced features."""
    data = request.get_json()
    instrument = data.get('instrument', 'EUR/USD')
    use_meta = data.get('use_meta_labeling', True)
    
    try:
        training_status[instrument] = 'training'
        
        # Get data
        df = data_manager.get_data(instrument)
        
        if len(df) < config.ML_CONFIG['training']['min_training_samples']:
            return jsonify({
                'success': False,
                'error': f"Insufficient data: need {config.ML_CONFIG['training']['min_training_samples']} samples"
            }), 400
        
        # Train main model
        ml_result = ml_engine.train(instrument, df)
        
        # Train meta-model if requested
        meta_result = None
        if use_meta and len(df) > 500:
            meta_result = meta_labeler.train(df, ml_engine, instrument)
            ml_engine.meta_models[instrument] = meta_labeler
        
        training_status[instrument] = 'complete'
        
        # Save models
        ml_engine.save_model(instrument)
        if meta_result:
            meta_labeler.save_model(instrument)
        
        # Convert numpy types in result
        clean_result = {}
        for key, value in ml_result.items():
            if isinstance(value, np.floating):
                clean_result[key] = float(value)
            elif isinstance(value, np.integer):
                clean_result[key] = int(value)
            elif isinstance(value, np.ndarray):
                clean_result[key] = value.tolist()
            else:
                clean_result[key] = value
        
        return jsonify({
            'success': True,
            'instrument': instrument,
            'ml_model': clean_result,
            'meta_model': meta_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        training_status[instrument] = 'failed'
        logger.error(f"Training error for {instrument}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with advanced features."""
    data = request.get_json()
    
    try:
        instrument = data.get('instrument', 'EUR/USD')
        capital = float(data.get('capital', 10000))
        risk_per_trade = float(data.get('risk_per_trade', 0.02))
        confidence_threshold = float(data.get('confidence_threshold', 0.7))
        
        # Get data
        df = data_manager.get_data(instrument)
        
        if len(df) < 100:
            return jsonify({
                'success': False,
                'error': "Insufficient data for backtest"
            }), 400
        
        # Generate signals for backtest period
        signals = ml_engine.generate_signals_for_period(
            instrument, df, confidence_threshold
        )
        
        # Run backtest
        result = backtester.run(
            data=df,
            signals=signals,
            capital=capital,
            risk_per_trade=risk_per_trade
        )
        
        # Store results
        result_id = f"{instrument}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backtest_results[result_id] = result
        
        # Convert result to serializable dict
        result_dict = {
            'initial_capital': float(result.initial_capital),
            'final_capital': float(result.final_capital),
            'total_return': float(result.total_return),
            'total_trades': int(result.total_trades),
            'winning_trades': int(result.winning_trades),
            'losing_trades': int(result.losing_trades),
            'win_rate': float(result.win_rate),
            'profit_factor': float(result.profit_factor) if result.profit_factor != float('inf') else 999.99,
            'sharpe_ratio': float(result.sharpe_ratio),
            'sortino_ratio': float(result.sortino_ratio),
            'max_drawdown': float(result.max_drawdown),
            'max_drawdown_pct': float(result.max_drawdown_pct),
            'calmar_ratio': float(result.calmar_ratio),
            'expectancy': float(result.expectancy),
            'avg_win': float(result.avg_win),
            'avg_loss': float(result.avg_loss),
            'largest_win': float(result.largest_win),
            'largest_loss': float(result.largest_loss),
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat() if hasattr(t.entry_time, 'isoformat') else str(t.entry_time),
                    'exit_time': t.exit_time.isoformat() if t.exit_time and hasattr(t.exit_time, 'isoformat') else str(t.exit_time),
                    'direction': int(t.direction),
                    'entry_price': float(t.entry_price),
                    'exit_price': float(t.exit_price),
                    'pnl': float(t.pnl),
                    'pnl_pct': float(t.pnl_pct),
                    'exit_reason': str(t.exit_reason)
                }
                for t in result.trades[-100:]  # Last 100 trades for display
            ],
            'equity_curve': [float(e) for e in result.equity_curve[::max(1, len(result.equity_curve)//100)]]
        }
        
        return jsonify({
            'success': True,
            'instrument': instrument,
            'result_id': result_id,
            'results': result_dict,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload CSV data file."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    instrument = request.form.get('instrument', 'unknown')
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload CSV files only.'}), 400
    
    # Check file size (max 16MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > 16 * 1024 * 1024:
        return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 400
    
    try:
        filename = secure_filename(f"{instrument}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and validate data
        data = pd.read_csv(filepath)
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            os.remove(filepath)
            return jsonify({
                'success': False, 
                'error': f"Missing required columns: {missing_cols}"
            }), 400
        
        # Parse dates
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # Sort by date
        data.sort_index(inplace=True)
        
        # Check for sufficient data
        if len(data) < 100:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f"Insufficient data: {len(data)} rows. Need at least 100 rows."
            }), 400
        
        # Calculate indicators
        data = calculate_all_indicators(data)
        
        # Store in data manager
        data_manager.store_data(instrument, data, filepath)
        
        return jsonify({
            'success': True,
            'instrument': instrument,
            'rows': int(len(data)),
            'date_range': [data.index[0].strftime('%Y-%m-%d'), data.index[-1].strftime('%Y-%m-%d')],
            'filename': filename,
            'columns': list(data.columns)[:20]
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        # Clean up file if error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get signal performance analytics."""
    instrument = request.args.get('instrument', None)
    days = int(request.args.get('days', 30))
    
    try:
        if instrument and instrument.lower() == 'all':
            instrument = None
            
        analysis = performance_tracker.analyze_performance(
            instrument=instrument,
            days_back=days
        )
        
        # Get recommendations
        recommendations = performance_tracker.generate_improvement_recommendations()
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Performance analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train-all', methods=['POST'])
def train_all():
    """Train models for all instruments."""
    results = {}
    
    for instrument in config.INSTRUMENTS.keys():
        try:
            data = data_manager.get_data(instrument)
            if data is not None and len(data) >= config.ML_CONFIG['training']['min_training_samples']:
                training_status[instrument] = 'training'
                result = ml_engine.train(instrument, data)
                training_status[instrument] = 'complete'
                
                # Convert numpy values
                accuracy = float(result.get('accuracy', 0))
                results[instrument] = {
                    'success': True, 
                    'accuracy': accuracy,
                    'samples': int(len(data))
                }
            else:
                results[instrument] = {
                    'success': False, 
                    'error': f'Insufficient data: {len(data) if data is not None else 0} samples'
                }
        except Exception as e:
            training_status[instrument] = 'failed'
            results[instrument] = {'success': False, 'error': str(e)}
    
    return jsonify({
        'success': True,
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/log-outcome', methods=['POST'])
def log_outcome():
    """Log actual signal outcome for performance tracking."""
    signal_data = request.get_json()
    
    try:
        required_fields = ['signal_id', 'instrument', 'direction', 'pnl']
        missing = [f for f in required_fields if f not in signal_data]
        
        if missing:
            return jsonify({
                'success': False, 
                'error': f"Missing required fields: {missing}"
            }), 400
        
        performance_tracker.log_signal_outcome(signal_data)
        
        return jsonify({
            'success': True,
            'message': 'Outcome logged successfully'
        })
        
    except Exception as e:
        logger.error(f"Log outcome error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data-info', methods=['GET'])
def get_data_info():
    """Get information about available data."""
    try:
        available_data = data_manager.list_available_data()
        
        # Convert numpy types
        clean_data = []
        for item in available_data:
            clean_item = {}
            for key, value in item.items():
                if isinstance(value, np.integer):
                    clean_item[key] = int(value)
                elif isinstance(value, np.floating):
                    clean_item[key] = float(value)
                elif isinstance(value, np.bool_):
                    clean_item[key] = bool(value)
                else:
                    clean_item[key] = value
            clean_data.append(clean_item)
        
        return jsonify({
            'success': True,
            'data': clean_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Data info error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_parameters():
    """Optimize backtest parameters."""
    data = request.get_json()
    instrument = data.get('instrument', 'EUR/USD')
    param_grid = data.get('param_grid', {})
    metric = data.get('optimization_metric', 'sharpe_ratio')
    
    try:
        # Get data
        df = data_manager.get_data(instrument)
        
        if len(df) < 200:
            return jsonify({
                'success': False,
                'error': "Insufficient data for optimization. Need at least 200 samples."
            }), 400
        
        # Generate signals
        signals = ml_engine.generate_signals_for_period(
            instrument, df, confidence_threshold=0.5
        )
        
        if len(signals) < 10:
            return jsonify({
                'success': False,
                'error': f"Only {len(signals)} signals generated. Need at least 10 for optimization."
            }), 400
        
        # Run optimization
        result = backtester.optimize_parameters(
            data=df,
            signals=signals,
            param_grid=param_grid,
            metric=metric
        )
        
        # Clean result
        clean_result = {
            'best_params': result['best_params'],
            'best_score': float(result['best_score']),
            'combinations_tested': int(result['combinations_tested'])
        }
        
        return jsonify({
            'success': True,
            'instrument': instrument,
            **clean_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-info/<instrument>', methods=['GET'])
def get_model_info(instrument):
    """Get information about a trained model."""
    try:
        if instrument not in ml_engine.models:
            return jsonify({
                'success': False,
                'error': f"No model trained for {instrument}"
            }), 404
        
        # Get feature importance
        importance = None
        if instrument in ml_engine.feature_importance:
            importance_df = ml_engine.feature_importance[instrument].head(20)
            importance = importance_df.to_dict('records')
        
        # Get training history
        history = ml_engine.training_history.get(instrument, {})
        
        return jsonify({
            'success': True,
            'instrument': instrument,
            'trained': True,
            'training_date': str(history.get('timestamp', 'Unknown')),
            'metrics': history.get('metrics', {}),
            'samples': int(history.get('samples', 0)),
            'feature_importance': importance,
            'feature_count': int(len(ml_engine.feature_cols)) if hasattr(ml_engine, 'feature_cols') else 0
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-data/<instrument>', methods=['DELETE'])
def delete_data(instrument):
    """Delete uploaded data for an instrument."""
    try:
        data_manager.delete_data(instrument)
        return jsonify({
            'success': True,
            'message': f"Data for {instrument} deleted successfully"
        })
    except Exception as e:
        logger.error(f"Delete data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/deriv/status', methods=['GET'])
def deriv_status():
    """Get Deriv connection status."""
    if not hasattr(data_manager, 'deriv_fetcher') or not data_manager.deriv_fetcher:
        return jsonify({'connected': False, 'error': 'Deriv not initialized'})
    
    fetcher = data_manager.deriv_fetcher
    status = fetcher.get_status()
    
    return jsonify({
        'success': True,
        'connected': status.get('connected', False),
        'authorized': status.get('authorized', False),
        'subscriptions': status.get('subscriptions', []),
        'instruments': fetcher.get_available_instruments()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("NEXUS Trading System v2.0 Starting...")
    logger.info("=" * 50)
    logger.info(f"Data directory: {config.DATA_DIR}")
    logger.info(f"Models directory: {config.MODELS_DIR}")
    logger.info(f"Logs directory: {config.LOGS_DIR}")
    logger.info("=" * 50)
    
    # Start the application
    app.run(
        host=config.API_CONFIG['host'],
        port=config.API_CONFIG['port'],
        debug=config.API_CONFIG['debug'],
        threaded=True
    )