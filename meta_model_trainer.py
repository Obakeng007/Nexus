"""
Meta-model trainer for NEXUS Trading System.
Implements meta-labeling to filter primary signals.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import os

import config
from label_generator import TripleBarrierLabeler

logger = logging.getLogger(__name__)

class MetaLabelingSystem:
    """
    Implements meta-labeling: ML model predicts success of primary signals.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize meta-labeling system.
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict or config.ML_CONFIG
        self.meta_model = None
        self.calibrated_model = None
        self.scaler = None
        self.feature_cols = None
        self.training_stats = {}
        self.feature_importance = None
        
        self.labeler = TripleBarrierLabeler(config.LABEL_CONFIG['triple_barrier'])
        
    def prepare_meta_features(self, 
                             df: pd.DataFrame,
                             signals: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare features for meta-model at signal points.
        
        Args:
            df: DataFrame with OHLCV data
            signals: List of signal dictionaries
        
        Returns:
            DataFrame of meta-features
        """
        meta_features = []
        
        for signal in signals:
            signal_time = signal['timestamp']
            signal_type = signal['direction']
            signal_confidence = signal.get('confidence', 0.5)
            
            # Find closest index
            if hasattr(df.index, 'get_indexer'):
                idx = df.index.get_indexer([signal_time], method='nearest')[0]
            else:
                # Try to find by date
                date_mask = df['date'] == signal_time if 'date' in df.columns else None
                if date_mask is not None and date_mask.any():
                    idx = date_mask.idxmax()
                else:
                    continue
            
            if idx < 0 or idx >= len(df):
                continue
            
            # Extract features at signal point
            feature_vector = {
                'timestamp': signal_time,
                'signal_type': signal_type,
                'signal_confidence': signal_confidence,
            }
            
            # Add market context features
            context_features = ['rsi', 'adx', 'volume_ratio', 'volatility', 
                              'market_regime', 'bb_width', 'atr_pct', 'macd',
                              'stoch_k', 'williams_r', 'cci', 'mfi']
            
            for col in context_features:
                if col in df.columns:
                    feature_vector[col] = df[col].iloc[idx]
            
            # Add price position features
            lookback = 20
            start_idx = max(0, idx - lookback)
            
            feature_vector['price_position'] = (
                (df['close'].iloc[idx] - df['low'].iloc[start_idx:idx+1].min()) /
                (df['high'].iloc[start_idx:idx+1].max() - df['low'].iloc[start_idx:idx+1].min() + 1e-8)
            )
            
            # Add volatility context
            if 'atr' in df.columns:
                feature_vector['atr_ratio'] = (
                    df['atr'].iloc[idx] / df['atr'].iloc[start_idx:idx+1].mean()
                )
            
            # Add trend context
            if 'adx' in df.columns:
                feature_vector['adx_trend'] = df['adx'].iloc[idx] > 25
            
            # Add volume context
            if 'volume' in df.columns:
                vol_mean = df['volume'].iloc[start_idx:idx+1].mean()
                feature_vector['volume_spike'] = df['volume'].iloc[idx] > (vol_mean * 1.5)
            
            # Add time features
            if hasattr(signal_time, 'hour'):
                feature_vector['hour'] = signal_time.hour
                feature_vector['day_of_week'] = signal_time.dayofweek
            elif isinstance(signal_time, str):
                try:
                    dt = pd.to_datetime(signal_time)
                    feature_vector['hour'] = dt.hour
                    feature_vector['day_of_week'] = dt.dayofweek
                except:
                    feature_vector['hour'] = 12
                    feature_vector['day_of_week'] = 3
            
            meta_features.append(feature_vector)
        
        if not meta_features:
            return pd.DataFrame()
        
        df_features = pd.DataFrame(meta_features)
        
        # Convert boolean to int
        bool_cols = df_features.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_features[col] = df_features[col].astype(int)
        
        return df_features
    
    def train(self, 
             df: pd.DataFrame,
             ml_engine,
             instrument: str = None,
             save_model: bool = True) -> Dict[str, Any]:
        """
        Train meta-model to predict signal success.
        
        Args:
            df: DataFrame with OHLCV data
            ml_engine: Trained ML engine instance
            instrument: Instrument symbol
            save_model: Whether to save model to disk
        
        Returns:
            Dictionary of training statistics
        """
        logger.info("Training meta-model...")
        
        # Generate primary signals
        signals = ml_engine.generate_signals_for_period(
            instrument, df, confidence_threshold=0.5
        )
        
        if len(signals) < 50:
            logger.warning(f"Insufficient signals for meta-training: {len(signals)}")
            return {'error': 'Insufficient signals', 'signal_count': len(signals)}
        
        logger.info(f"Generated {len(signals)} signals for meta-training")
        
        # Prepare meta features
        meta_features = self.prepare_meta_features(df, signals)
        
        if len(meta_features) == 0:
            return {'error': 'No valid features'}
        
        # Generate meta-labels
        meta_labels = []
        valid_indices = []
        
        for idx, signal in enumerate(signals):
            if idx >= len(meta_features):
                break
                
            signal_time = signal['timestamp']
            signal_type = signal['direction']
            
            # Find position in dataframe
            if hasattr(df.index, 'get_indexer'):
                pos = df.index.get_indexer([signal_time], method='nearest')[0]
            else:
                date_mask = df['date'] == signal_time if 'date' in df.columns else None
                if date_mask is not None and date_mask.any():
                    pos = date_mask.idxmax()
                else:
                    continue
            
            if pos < 0 or pos >= len(df) - 24:
                continue
            
            # Look forward 24 periods
            future_prices = df['close'].iloc[pos+1:pos+25]
            if len(future_prices) < 5:
                continue
                
            future_returns = future_prices.pct_change().iloc[1:].values
            cum_return = (1 + future_returns).prod() - 1
            
            # Meta-label: 1 if profitable (0.5% threshold), 0 if not
            is_profitable = (cum_return * signal_type) > 0.005
            meta_labels.append(1 if is_profitable else 0)
            valid_indices.append(idx)
        
        if len(meta_labels) < 30:
            return {'error': f'Insufficient valid samples: {len(meta_labels)}'}
        
        # Prepare feature matrix
        feature_cols = [c for c in meta_features.columns 
                       if c not in ['timestamp']]
        
        X = meta_features.iloc[valid_indices][feature_cols].fillna(0)
        y = np.array(meta_labels)
        
        logger.info(f"Meta-training with {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Class distribution: Positive={sum(y==1)}, Negative={sum(y==0)}")
        
        # Split data (time-series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train meta-model
        self.meta_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Calibrate probabilities
        tscv = TimeSeriesSplit(n_splits=3)
        self.calibrated_model = CalibratedClassifierCV(
            self.meta_model, 
            cv=tscv, 
            method='sigmoid'
        )
        self.calibrated_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.calibrated_model.predict(X_test_scaled)
        y_pred_proba = self.calibrated_model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            roc_auc = 0.5
        
        self.training_stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'n_samples': len(y),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'positive_rate': y.mean(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.feature_cols = feature_cols
        
        # Calculate feature importance
        if hasattr(self.meta_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.meta_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info(f"Meta-model training complete:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        logger.info(f"  ROC AUC: {roc_auc:.3f}")
        
        # Save model if requested
        if save_model and instrument:
            self.save_model(instrument)
        
        return self.training_stats
    
    def predict_signal_quality(self,
                              df: pd.DataFrame,
                              signal: Dict[str, Any]) -> Tuple[float, float]:
        """
        Predict probability that a signal will be profitable.
        
        Args:
            df: DataFrame with OHLCV data
            signal: Signal dictionary
        
        Returns:
            Tuple of (probability_of_success, confidence_in_prediction)
        """
        if self.calibrated_model is None or self.scaler is None:
            return 0.5, 0.0
        
        # Prepare features for this signal
        features_df = self.prepare_meta_features(df, [signal])
        
        if len(features_df) == 0:
            return 0.5, 0.0
        
        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        X = features_df[self.feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability
        proba = self.calibrated_model.predict_proba(X_scaled)[0]
        
        if len(proba) > 1:
            success_prob = proba[1]
        else:
            success_prob = proba[0]
        
        # Confidence based on distance from 0.5
        confidence = min(1.0, abs(success_prob - 0.5) * 2)
        
        return float(success_prob), float(confidence)
    
    def filter_signals(self, 
                      df: pd.DataFrame,
                      signals: List[Dict[str, Any]],
                      threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Filter signals using meta-model predictions.
        
        Args:
            df: DataFrame with OHLCV data
            signals: List of signals to filter
            threshold: Probability threshold for acceptance
        
        Returns:
            Filtered list of signals
        """
        if self.calibrated_model is None or len(signals) == 0:
            return signals
        
        filtered_signals = []
        
        for signal in signals:
            success_prob, meta_confidence = self.predict_signal_quality(df, signal)
            
            if success_prob >= threshold:
                signal['meta_probability'] = success_prob
                signal['meta_confidence'] = meta_confidence
                signal['original_confidence'] = signal.get('confidence', 0.5)
                # Adjust confidence
                signal['confidence'] = (signal.get('confidence', 0.5) + success_prob) / 2
                filtered_signals.append(signal)
        
        logger.info(f"Meta-filter: {len(filtered_signals)}/{len(signals)} signals passed (threshold={threshold})")
        
        return filtered_signals
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from meta-model.
        
        Returns:
            DataFrame of feature importance
        """
        if self.feature_importance is not None:
            return self.feature_importance
        return pd.DataFrame()
    
    def save_model(self, instrument: str, path: Optional[str] = None) -> str:
        """
        Save meta-model to disk.
        
        Args:
            instrument: Instrument symbol
            path: Optional file path
        
        Returns:
            Path where model was saved
        """
        if path is None:
            filename = f"{instrument.replace('/', '_')}_meta_model_{datetime.now().strftime('%Y%m%d')}.pkl"
            path = config.MODELS_DIR / filename
        
        model_data = {
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Meta-model saved to {path}")
        
        return str(path)
    
    def load_model(self, instrument: str, path: Optional[str] = None) -> bool:
        """
        Load meta-model from disk.
        
        Args:
            instrument: Instrument symbol
            path: Optional file path
        
        Returns:
            True if successful
        """
        if path is None:
            pattern = f"{instrument.replace('/', '_')}_meta_model_*.pkl"
            model_files = list(config.MODELS_DIR.glob(pattern))
            
            if not model_files:
                raise FileNotFoundError(f"No meta-model found for {instrument}")
            
            path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"No meta-model found at {path}")
        
        model_data = joblib.load(path)
        
        self.calibrated_model = model_data['calibrated_model']
        self.meta_model = self.calibrated_model.calibrated_classifiers_[0].base_estimator
        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data['feature_cols']
        self.training_stats = model_data.get('training_stats', {})
        
        if model_data.get('feature_importance'):
            self.feature_importance = pd.DataFrame(model_data['feature_importance'])
        
        logger.info(f"Meta-model loaded from {path}")
        
        return True