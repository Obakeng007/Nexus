"""
Enhanced Machine Learning Engine for NEXUS Trading System.
Implements ensemble models with advanced features.
FIXED: Using VotingClassifier instead of custom EnsembleModel for CalibratedClassifierCV compatibility.
FIXED: Added NaN handling in labels to prevent training errors.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
import joblib
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings
import os

# Import local modules
import config
from features import AdvancedFeatureEngine
from label_generator import TripleBarrierLabeler
from indicators import calculate_all_indicators, calculate_atr

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MLEngine:
    """
    Enhanced machine learning engine for trading signals.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize ML engine.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or config.ML_CONFIG
        self.models = {}  # Trained models by instrument
        self.scalers = {}  # Feature scalers by instrument
        self.feature_importance = {}  # Feature importance by instrument
        self.feature_cols = None  # Feature columns used
        self.meta_models = {}  # Meta-models for validation
        self.training_history = {}  # Training history by instrument
        self.cv_scores = {}  # Cross-validation scores
        
        # Initialize components
        self.feature_engine = AdvancedFeatureEngine(
            lookback_periods=self.config['features']['lookback_periods']
        )
        self.labeler = TripleBarrierLabeler(config.LABEL_CONFIG['triple_barrier'])
        
        # Model parameters
        self.model_params = self.config['ensemble']
        self.cv_folds = self.config['training']['cv_folds']
        self.confidence_threshold = self.config['training']['confidence_threshold']
        self.min_samples = self.config['features']['min_samples']
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for ML.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Tuple of (feature matrix, feature column names)
        """
        # Ensure all indicators are calculated
        if not all(col in df.columns for col in ['rsi', 'macd', 'atr']):
            df = calculate_all_indicators(df)
        
        # Calculate advanced features
        df_with_features = self.feature_engine.calculate_all_features(df)
        
        # Select features (exclude price columns and NaN-heavy columns)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 
                       'label', 'barrier_hit', 'hit_time', 'actual_return']
        
        # Get all numeric columns
        numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols 
                       if c not in exclude_cols 
                       and df_with_features[c].notna().sum() > len(df_with_features) * 0.5]
        
        # Fill remaining NaNs
        X = df_with_features[feature_cols].copy()
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        self.feature_cols = feature_cols
        
        return X, feature_cols
    
    def train(self, 
             instrument: str, 
             df: pd.DataFrame,
             optimize_hyperparams: bool = False,
             save_model: bool = True) -> Dict[str, Any]:
        """
        Train model for a specific instrument.
        
        Args:
            instrument: Instrument symbol
            df: DataFrame with OHLCV data
            optimize_hyperparams: Whether to optimize hyperparameters
            save_model: Whether to save model to disk
        
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training model for {instrument}...")
        
        if len(df) < self.min_samples:
            error_msg = f"Insufficient data: need {self.min_samples} samples, got {len(df)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Prepare features
        X, feature_cols = self.prepare_features(df)
        
        # Calculate ATR for label generation
        if 'atr' in df.columns:
            atr = df['atr']
        else:
            atr = calculate_atr(df['high'], df['low'], df['close'])
        
        # Check data volatility and log it
        avg_atr_pct = (atr / df['close']).mean()
        logger.info(f"Average ATR %: {avg_atr_pct:.4%}")
        
        # Use index as timestamps if date column not present
        timestamps = df.index if 'date' not in df.columns else pd.to_datetime(df['date'])
        
        # Generate labels using triple-barrier method
        labels_df = self.labeler.get_triple_barrier_labels(
            df['close'], 
            timestamps,
            atr
        )
        
        # Use only valid labels (non-neutral)
        valid_idx = labels_df['label'] != -1
        X_filtered = X[valid_idx]
        y = labels_df.loc[valid_idx, 'label']
        
        # Log label statistics
        label_stats = self.labeler.get_label_statistics(labels_df)
        logger.info(f"Label statistics: {label_stats}")
        
        # Check if we have enough valid samples
        if len(X_filtered) < 100:
            error_msg = f"Insufficient valid samples: {len(X_filtered)}"
            logger.error(error_msg)
            
            # Try with even tighter barriers as fallback
            if avg_atr_pct < 0.005:
                logger.info("Attempting with ultra-tight barriers...")
                # Save original barriers
                original_upper = self.labeler.upper_barrier_pct
                original_lower = self.labeler.lower_barrier_pct
                
                # Set ultra-tight barriers
                self.labeler.upper_barrier_pct = 0.002  # 0.2%
                self.labeler.lower_barrier_pct = 0.001  # 0.1%
                
                # Generate new labels
                labels_df = self.labeler.get_triple_barrier_labels(
                    df['close'], 
                    timestamps,
                    atr
                )
                
                # Restore original barriers
                self.labeler.upper_barrier_pct = original_upper
                self.labeler.lower_barrier_pct = original_lower
                
                valid_idx = labels_df['label'] != -1
                X_filtered = X[valid_idx]
                y = labels_df.loc[valid_idx, 'label']
                
                if len(X_filtered) >= 100:
                    logger.info(f"Ultra-tight barriers generated {len(X_filtered)} valid samples")
            
            if len(X_filtered) < 100:
                raise ValueError(f"Insufficient valid samples: {len(X_filtered)}")
        
        X = X_filtered
        logger.info(f"Training with {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Class distribution: Wins={sum(y==1)}, Losses={sum(y==0)}")
        
        # Split data (time-series aware)
        test_size = self.config['training']['test_size']
        split_idx = int(len(X) * (1 - test_size))
        
        # Ensure no NaN values in X or y
        logger.info("Cleaning data before split...")
        X = X.fillna(0)
        y = y.fillna(0).astype(int)  # Convert to int, fill NaN with 0
        
        # Double-check that y has no NaN
        if y.isna().any():
            logger.warning(f"Found NaN in y, filling with 0")
            y = y.fillna(0)
        
        # Split the data
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Verify no NaN in test data
        if y_test.isna().any():
            logger.warning(f"Found {y_test.isna().sum()} NaN in y_test, removing those samples")
            # Find indices where y_test is not NaN
            valid_test_idx = ~y_test.isna()
            X_test = X_test[valid_test_idx]
            y_test = y_test[valid_test_idx]
        
        if y_train.isna().any():
            logger.warning(f"Found {y_train.isna().sum()} NaN in y_train, removing those samples")
            valid_train_idx = ~y_train.isna()
            X_train = X_train[valid_train_idx]
            y_train = y_train[valid_train_idx]
        
        # Final check
        if len(y_test) == 0:
            raise ValueError("Test set is empty after removing NaNs. Try increasing test_size or checking data quality.")
        
        if len(y_train) == 0:
            raise ValueError("Train set is empty after removing NaNs. Check your data.")
        
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        logger.info(f"Train class distribution: Wins={sum(y_train==1)}, Losses={sum(y_train==0)}")
        logger.info(f"Test class distribution: Wins={sum(y_test==1)}, Losses={sum(y_test==0)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble
        if optimize_hyperparams:
            model = self._optimize_hyperparameters(X_train_scaled, y_train)
        else:
            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=self.model_params['n_estimators'],
                max_depth=self.model_params['max_depth'],
                min_samples_split=self.model_params['min_samples_split'],
                random_state=self.model_params['random_state'],
                n_jobs=self.model_params['n_jobs'],
                class_weight='balanced'
            )
            
            # Gradient Boosting
            gb = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                min_samples_split=50,
                random_state=self.model_params['random_state'],
                subsample=0.8
            )
            
            # Train individual models
            logger.info("Training Random Forest...")
            rf.fit(X_train_scaled, y_train)
            
            logger.info("Training Gradient Boosting...")
            gb.fit(X_train_scaled, y_train)
            
            # Create ensemble using VotingClassifier
            from sklearn.ensemble import VotingClassifier
            
            logger.info("Creating ensemble model...")
            model = VotingClassifier(
                estimators=[
                    ('rf', rf),
                    ('gb', gb)
                ],
                voting='soft',
                weights=[1, 1]
            )
            
            model.fit(X_train_scaled, y_train)
        
        # Calibrate probabilities
        logger.info("Calibrating probabilities...")
        calibrated_model = CalibratedClassifierCV(
            estimator=model,
            cv=TimeSeriesSplit(n_splits=3),
            method='sigmoid'
        )
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = calibrated_model.predict(X_test_scaled)
        y_pred_proba = calibrated_model.predict_proba(X_test_scaled)
        
        # Calculate metrics (with safety checks)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC if binary classification
        if len(np.unique(y)) == 2 and len(y_test) > 1:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                roc_auc = 0.5
        else:
            roc_auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'train_size': int(len(X_train)),
            'test_size': int(len(X_test)),
            'feature_count': int(len(feature_cols)),
            'class_distribution': {
                'train_wins': int(sum(y_train==1)),
                'train_losses': int(sum(y_train==0)),
                'test_wins': int(sum(y_test==1)),
                'test_losses': int(sum(y_test==0))
            }
        }
        
        # Cross-validation scores
        cv_scores = self._cross_validate(X_train_scaled, y_train)
        metrics['cv_scores'] = cv_scores
        
        # Store model and metadata
        self.models[instrument] = calibrated_model
        self.scalers[instrument] = scaler
        
        # Calculate feature importance (using RF from the voting classifier)
        if hasattr(model, 'named_estimators_') and 'rf' in model.named_estimators_:
            rf_model = model.named_estimators_['rf']
            if hasattr(rf_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[instrument] = importance
                metrics['top_features'] = importance.head(10).to_dict('records')
        
        # Store training history
        self.training_history[instrument] = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'samples': len(X),
            'features': feature_cols
        }
        
        logger.info(f"Training complete for {instrument}:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        logger.info(f"  ROC AUC: {roc_auc:.3f}")
        
        # Save model if requested
        if save_model:
            self.save_model(instrument)
        
        return metrics
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
        
        Returns:
            Dictionary of CV scores
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train simple model for CV
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred_fold = rf.predict(X_val_fold)
            
            # Calculate scores
            cv_scores['accuracy'].append(accuracy_score(y_val_fold, y_pred_fold))
            cv_scores['precision'].append(precision_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0))
            cv_scores['recall'].append(recall_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0))
            cv_scores['f1'].append(f1_score(y_val_fold, y_pred_fold, average='weighted', zero_division=0))
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = float(np.mean(scores))
            cv_results[f'{metric}_std'] = float(np.std(scores))
        
        self.cv_scores = cv_results
        
        return cv_results
    
    def _optimize_hyperparameters(self, X, y) -> RandomForestClassifier:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target labels
        
        Returns:
            Optimized RandomForestClassifier
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [20, 50, 100],
            'min_samples_leaf': [10, 20, 50],
            'max_features': ['sqrt', 'log2']
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def generate_signals(self, 
                        instrument: str, 
                        df: Optional[pd.DataFrame] = None,
                        confidence_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Generate trading signals using trained model.
        
        Args:
            instrument: Instrument symbol
            df: Optional DataFrame with recent data
            confidence_threshold: Override default confidence threshold
        
        Returns:
            List of signal dictionaries
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        if instrument not in self.models:
            logger.warning(f"No model trained for {instrument}, using rule-based signals")
            return self._generate_rule_based_signals(df)
        
        model = self.models[instrument]
        scaler = self.scalers[instrument]
        
        if df is None:
            from data_manager import DataManager
            data_manager = DataManager()
            df = data_manager.get_data(instrument)
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return []
        
        # Use recent data for signals (last 100 bars)
        recent_X = X.iloc[-100:]
        
        # Scale and predict
        X_scaled = scaler.transform(recent_X)
        proba = model.predict_proba(X_scaled)
        
        # Generate signals
        signals = []
        
        for i in range(len(proba)):
            # Get probability of win (class 1)
            if proba.shape[1] > 1:
                win_probability = proba[i, 1]
                loss_probability = proba[i, 0]
            else:
                win_probability = proba[i, 0]
                loss_probability = 1 - win_probability
            
            # Get corresponding price data
            data_idx = -100 + i
            current_price = float(df['close'].iloc[data_idx])
            current_atr = float(df['atr'].iloc[data_idx]) if 'atr' in df.columns else current_price * 0.01
            
            # Get timestamp
            timestamp = df.index[data_idx]
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            
            # Generate buy signal
            if win_probability > confidence_threshold:
                signal = {
                    'direction': 1,
                    'confidence': float(win_probability),
                    'entry': float(current_price),
                    'stop_loss': float(current_price - (current_atr * 1.5)),
                    'take_profit': float(current_price + (current_atr * 3.0)),
                    'timestamp': timestamp_str,
                    'instrument': str(instrument),
                    'signal_type': 'ML_BUY',
                    'risk_reward': 2.0
                }
                signals.append(signal)
            
            # Generate sell signal
            elif loss_probability > confidence_threshold:
                signal = {
                    'direction': -1,
                    'confidence': float(loss_probability),
                    'entry': float(current_price),
                    'stop_loss': float(current_price + (current_atr * 1.5)),
                    'take_profit': float(current_price - (current_atr * 3.0)),
                    'timestamp': timestamp_str,
                    'instrument': str(instrument),
                    'signal_type': 'ML_SELL',
                    'risk_reward': 2.0
                }
                signals.append(signal)
        
        # Sort by confidence and return most recent
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add meta-model validation if available
        if instrument in self.meta_models:
            meta_model = self.meta_models[instrument]
            filtered_signals = []
            
            for signal in signals:
                success_prob, meta_confidence = meta_model.predict_signal_quality(df, signal)
                if success_prob > 0.5:
                    signal['meta_probability'] = float(success_prob)
                    signal['meta_confidence'] = float(meta_confidence)
                    signal['confidence'] = float((signal.get('confidence', 0.5) + success_prob) / 2)
                    filtered_signals.append(signal)
            
            signals = filtered_signals
        
        return signals[-10:]  # Return most recent 10 signals
    
    def generate_signals_for_period(self,
                                   instrument: str,
                                   df: pd.DataFrame,
                                   confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Generate signals for entire backtest period.
        
        Args:
            instrument: Instrument symbol
            df: DataFrame with historical data
            confidence_threshold: Confidence threshold
        
        Returns:
            List of signals for the entire period
        """
        if instrument not in self.models:
            return self._generate_rule_based_signals(df)
        
        model = self.models[instrument]
        scaler = self.scalers[instrument]
        
        # Prepare features
        X, _ = self.prepare_features(df)
        
        if len(X) < 100:
            return []
        
        # Scale all data
        X_scaled = scaler.transform(X)
        
        # Predict probabilities
        proba = model.predict_proba(X_scaled)
        
        # Generate signals
        signals = []
        
        for i in range(100, len(proba) - 1):  # Skip first 100 bars for feature calculation
            if proba.shape[1] > 1:
                win_probability = proba[i, 1]
                loss_probability = proba[i, 0]
            else:
                win_probability = proba[i, 0]
                loss_probability = 1 - win_probability
            
            current_price = float(df['close'].iloc[i])
            current_atr = float(df['atr'].iloc[i]) if 'atr' in df.columns else current_price * 0.01
            
            # Get timestamp
            timestamp = df.index[i]
            if hasattr(timestamp, 'isoformat'):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp)
            
            if win_probability > confidence_threshold:
                signal = {
                    'direction': 1,
                    'confidence': float(win_probability),
                    'entry': float(current_price),
                    'stop_loss': float(current_price - (current_atr * 1.5)),
                    'take_profit': float(current_price + (current_atr * 3.0)),
                    'timestamp': timestamp_str,
                    'instrument': instrument,
                    'signal_type': 'ML_BUY'
                }
                signals.append(signal)
            elif loss_probability > confidence_threshold:
                signal = {
                    'direction': -1,
                    'confidence': float(loss_probability),
                    'entry': float(current_price),
                    'stop_loss': float(current_price + (current_atr * 1.5)),
                    'take_profit': float(current_price - (current_atr * 3.0)),
                    'timestamp': timestamp_str,
                    'instrument': instrument,
                    'signal_type': 'ML_SELL'
                }
                signals.append(signal)
        
        logger.info(f"Generated {len(signals)} signals for {instrument}")
        
        return signals
    
    def _generate_rule_based_signals(self, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Generate rule-based signals when ML model is not available.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            List of rule-based signals
        """
        if df is None or len(df) < 50:
            return []
        
        from indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands
        
        signals = []
        
        # Calculate indicators
        rsi = calculate_rsi(df['close'], 14)
        macd, signal_line, hist = calculate_macd(df['close'])
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], 20)
        
        # Get latest values
        current_price = float(df['close'].iloc[-1])
        current_atr = float(df['atr'].iloc[-1]) if 'atr' in df.columns else current_price * 0.01
        
        # Get timestamp
        timestamp = df.index[-1]
        if hasattr(timestamp, 'isoformat'):
            timestamp_str = timestamp.isoformat()
        else:
            timestamp_str = str(timestamp)
        
        # RSI oversold bounce
        if rsi.iloc[-1] < 30 and rsi.iloc[-2] < rsi.iloc[-1]:
            signals.append({
                'direction': 1,
                'confidence': 0.65,
                'entry': float(current_price),
                'stop_loss': float(current_price * 0.985),
                'take_profit': float(current_price * 1.03),
                'timestamp': timestamp_str,
                'instrument': 'unknown',
                'signal_type': 'RSI_OVERSOLD'
            })
        
        # RSI overbought reversal
        elif rsi.iloc[-1] > 70 and rsi.iloc[-2] > rsi.iloc[-1]:
            signals.append({
                'direction': -1,
                'confidence': 0.65,
                'entry': float(current_price),
                'stop_loss': float(current_price * 1.015),
                'take_profit': float(current_price * 0.97),
                'timestamp': timestamp_str,
                'instrument': 'unknown',
                'signal_type': 'RSI_OVERBOUGHT'
            })
        
        # MACD crossover
        if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
            signals.append({
                'direction': 1,
                'confidence': 0.6,
                'entry': float(current_price),
                'stop_loss': float(current_price * 0.99),
                'take_profit': float(current_price * 1.02),
                'timestamp': timestamp_str,
                'instrument': 'unknown',
                'signal_type': 'MACD_CROSS'
            })
        
        # Bollinger Band squeeze
        bb_width = (bb_upper - bb_lower) / bb_middle
        if bb_width.iloc[-1] < bb_width.iloc[-20:].quantile(0.2):
            # Squeeze - prepare for breakout
            if df['close'].iloc[-1] > bb_middle.iloc[-1]:
                signals.append({
                    'direction': 1,
                    'confidence': 0.55,
                    'entry': float(current_price),
                    'stop_loss': float(current_price - current_atr),
                    'take_profit': float(current_price + current_atr * 2),
                    'timestamp': timestamp_str,
                    'signal_type': 'BB_SQUEEZE'
                })
        
        return signals[-5:]  # Return most recent 5 signals
    
    def predict_proba(self, instrument: str, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for features.
        
        Args:
            instrument: Instrument symbol
            X: Feature matrix
        
        Returns:
            Probability predictions
        """
        if instrument not in self.models:
            raise ValueError(f"No model trained for {instrument}")
        
        model = self.models[instrument]
        scaler = self.scalers[instrument]
        
        X_scaled = scaler.transform(X)
        return model.predict_proba(X_scaled)
    
    def save_model(self, instrument: str, path: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            instrument: Instrument symbol
            path: Optional file path
        
        Returns:
            Path where model was saved
        """
        if instrument not in self.models:
            raise ValueError(f"No model for {instrument}")
        
        if path is None:
            filename = f"{instrument.replace('/', '_')}_model_{datetime.now().strftime('%Y%m%d')}.pkl"
            path = config.MODELS_DIR / filename
        
        model_data = {
            'model': self.models[instrument],
            'scaler': self.scalers[instrument],
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance.get(instrument),
            'training_history': self.training_history.get(instrument),
            'cv_scores': self.cv_scores,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved for {instrument} at {path}")
        
        return str(path)
    
    def load_model(self, instrument: str, path: Optional[str] = None) -> bool:
        """
        Load trained model from disk.
        
        Args:
            instrument: Instrument symbol
            path: Optional file path
        
        Returns:
            True if successful
        """
        if path is None:
            # Look for most recent model file
            pattern = f"{instrument.replace('/', '_')}_model_*.pkl"
            model_files = list(config.MODELS_DIR.glob(pattern))
            
            if not model_files:
                raise FileNotFoundError(f"No model found for {instrument}")
            
            # Get most recent
            path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        model_data = joblib.load(path)
        
        self.models[instrument] = model_data['model']
        self.scalers[instrument] = model_data['scaler']
        self.feature_cols = model_data.get('feature_cols')
        self.feature_importance[instrument] = model_data.get('feature_importance')
        self.training_history[instrument] = model_data.get('training_history')
        self.cv_scores = model_data.get('cv_scores', {})
        
        logger.info(f"Model loaded for {instrument} from {path}")
        
        return True
    
    def get_model_info(self, instrument: str) -> Dict[str, Any]:
        """
        Get information about trained model.
        
        Args:
            instrument: Instrument symbol
        
        Returns:
            Dictionary of model information
        """
        if instrument not in self.models:
            return {'error': f'No model trained for {instrument}'}
        
        info = {
            'instrument': instrument,
            'trained': True,
            'training_history': self.training_history.get(instrument, {}),
            'feature_count': len(self.feature_cols) if self.feature_cols else 0,
            'cv_scores': self.cv_scores
        }
        
        if instrument in self.feature_importance:
            info['top_features'] = self.feature_importance[instrument].head(10).to_dict('records')
        
        return info
    
    def check_label_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check label quality before training.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with label statistics
        """
        # Calculate ATR if not present
        if 'atr' not in df.columns:
            atr = calculate_atr(df['high'], df['low'], df['close'])
        else:
            atr = df['atr']
        
        # Generate labels
        labels_df = self.labeler.get_triple_barrier_labels(
            df['close'], 
            df.index,
            atr
        )
        
        # Get statistics
        valid_labels = labels_df[labels_df['label'] != -1]
        win_count = len(valid_labels[valid_labels['label'] == 1])
        loss_count = len(valid_labels[valid_labels['label'] == 0])
        neutral_count = len(labels_df[labels_df['label'] == -1])
        
        # Calculate average volatility
        avg_atr_pct = (atr / df['close']).mean()
        
        stats = {
            'total_samples': len(labels_df),
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'neutral_count': int(neutral_count),
            'win_rate': float(win_count / (win_count + loss_count)) if (win_count + loss_count) > 0 else 0,
            'avg_atr_pct': float(avg_atr_pct),
            'avg_holding_period': float(valid_labels['holding_periods'].mean()) if len(valid_labels) > 0 else 0,
            'avg_return': float(valid_labels['actual_return'].mean()) if len(valid_labels) > 0 else 0
        }
        
        logger.info(f"Label quality check: {stats}")
        
        return stats