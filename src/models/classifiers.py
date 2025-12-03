"""
Classifiers for Twitter Bot Detection
Implements Random Forest, SVM, and Neural Network classifiers
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
from pathlib import Path
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotDetectionClassifier:
    """Ensemble classifier for bot detection"""
    
    def __init__(self, classifier_type: str = 'ensemble'):
        """
        Initialize classifier
        
        Args:
            classifier_type: Type of classifier ('rf', 'svm', 'nn', 'ensemble')
        """
        self.classifier_type = classifier_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def create_random_forest(self, **kwargs) -> RandomForestClassifier:
        """Create Random Forest classifier"""
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        logger.info(f"Creating Random Forest with params: {default_params}")
        return RandomForestClassifier(**default_params)
    
    def create_svm(self, **kwargs) -> SVC:
        """Create SVM classifier"""
        default_params = {
            'kernel': 'rbf',
            'C': 10.0,
            'gamma': 'scale',
            'class_weight': 'balanced',
            'probability': True,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        logger.info(f"Creating SVM with params: {default_params}")
        return SVC(**default_params)
    
    def create_neural_network(self, **kwargs) -> MLPClassifier:
        """Create Neural Network classifier"""
        default_params = {
            'hidden_layer_sizes': (256, 128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 32,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42
        }
        default_params.update(kwargs)
        
        logger.info(f"Creating Neural Network with params: {default_params}")
        return MLPClassifier(**default_params)
    
    def create_ensemble(self, **kwargs) -> VotingClassifier:
        """Create ensemble voting classifier"""
        rf = self.create_random_forest()
        svm = self.create_svm()
        nn = self.create_neural_network()
        
        weights = kwargs.get('weights', [2, 1, 1])
        voting = kwargs.get('voting', 'soft')
        
        logger.info(f"Creating Ensemble with voting={voting}, weights={weights}")
        
        return VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm),
                ('nn', nn)
            ],
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
    
    def build_model(self, **kwargs):
        """Build the classifier model"""
        if self.classifier_type == 'rf':
            self.model = self.create_random_forest(**kwargs)
        elif self.classifier_type == 'svm':
            self.model = self.create_svm(**kwargs)
        elif self.classifier_type == 'nn':
            self.model = self.create_neural_network(**kwargs)
        elif self.classifier_type == 'ensemble':
            self.model = self.create_ensemble(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.classifier_type} classifier...")
        logger.info(f"Training set size: {X_train.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Training accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        metrics = {'train_accuracy': train_score}
        
        # Validation accuracy
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")
            metrics['val_accuracy'] = val_score
        
        # Feature importance (for Random Forest)
        if self.classifier_type == 'rf':
            self.feature_importance = self.model.feature_importances_
        elif self.classifier_type == 'ensemble':
            # Get feature importance from RF in ensemble
            self.feature_importance = self.model.named_estimators_['rf'].feature_importances_
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, n_jobs=-1)
        
        metrics = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        logger.info(f"CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
        
        return metrics
    
    def save_model(self, filepath: str = "models/bot_detector.pkl"):
        """
        Save trained model
        
        Args:
            filepath: Path to save model
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'classifier_type': self.classifier_type,
                'feature_importance': self.feature_importance
            }, f)
        
        logger.info(f"Saved model to {output_path}")
    
    def load_model(self, filepath: str = "models/bot_detector.pkl"):
        """
        Load trained model
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.classifier_type = data['classifier_type']
        self.feature_importance = data.get('feature_importance')
        
        logger.info(f"Loaded model from {filepath}")


def main():
    """Main function to train classifiers"""
    print("=" * 60)
    print("Twitter Bot Detection - Model Training")
    print("=" * 60)
    
    # Load features and labels
    X = np.load("data/processed/features.npy")
    y = np.load("data/processed/labels.npy")
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train ensemble classifier
    classifier = BotDetectionClassifier(classifier_type='ensemble')
    classifier.build_model()
    
    metrics = classifier.train(X_train, y_train, X_test, y_test)
    
    # Test accuracy
    test_score = classifier.model.score(
        classifier.scaler.transform(X_test), y_test
    )
    
    print(f"\n✅ Model training complete!")
    print(f"📊 Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"📊 Test accuracy: {test_score:.4f}")
    
    # Save model
    classifier.save_model()
    
    return classifier, metrics


if __name__ == "__main__":
    main()
