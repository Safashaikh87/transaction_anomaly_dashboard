"""
Model Trainer - Isolation Forest for anomaly detection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Isolation Forest based anomaly detector for transactions
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.contamination = contamination
        
    def train(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, Any]:
        """
        Train the Isolation Forest model
        """
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert to 0-1 risk scores
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        risk_scores = 1 - (anomaly_scores - min_score) / (max_score - min_score)
        
        results = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': self.feature_names,
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'risk_scores': risk_scores,
            'anomaly_rate': (predictions == -1).mean()
        }
        
        # If labels provided, calculate metrics
        if y is not None:
            y_pred = (predictions == -1).astype(int)
            results['accuracy'] = (y == y_pred).mean()
            results['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
            results['classification_report'] = classification_report(y, y_pred, output_dict=True)
        
        return results
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores for new transactions
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        risk_scores = 1 - (anomaly_scores - min_score) / (max_score - min_score)
        
        return predictions, risk_scores
    
    def get_risk_level(self, risk_scores: np.ndarray) -> np.ndarray:
        """Convert risk scores to risk levels"""
        risk_levels = np.full(len(risk_scores), 'Low', dtype=object)
        risk_levels[risk_scores > 0.7] = 'High'
        risk_levels[(risk_scores > 0.4) & (risk_scores <= 0.7)] = 'Medium'
        return risk_levels
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'contamination': self.contamination
            }, f)
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.contamination = data['contamination']