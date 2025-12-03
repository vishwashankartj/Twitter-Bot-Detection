"""
Evaluation Metrics for Twitter Bot Detection
Comprehensive evaluation and performance metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate bot detection model performance"""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary')
        }
        
        # ROC-AUC (requires probabilities)
        if y_proba is not None:
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]  # Get probability of positive class
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             filename: str = "confusion_matrix.png"):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Output filename
        """
        logger.info("Plotting confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'Bot'],
                   yticklabels=['Human', 'Bot'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path = self.plots_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      filename: str = "roc_curve.png"):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            filename: Output filename
        """
        logger.info("Plotting ROC curve...")
        
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = self.plots_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curve to {output_path}")
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   filename: str = "precision_recall_curve.png"):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            filename: Output filename
        """
        logger.info("Plotting Precision-Recall curve...")
        
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label='PR Curve')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = self.plots_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PR curve to {output_path}")
    
    def plot_feature_importance(self, feature_importance: np.ndarray,
                               feature_names: list,
                               top_n: int = 20,
                               filename: str = "feature_importance.png"):
        """
        Plot feature importance
        
        Args:
            feature_importance: Feature importance scores
            feature_names: Feature names
            top_n: Number of top features to show
            filename: Output filename
        """
        logger.info("Plotting feature importance...")
        
        # Get top N features
        indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), top_importance, color='steelblue')
        plt.yticks(range(top_n), top_features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.plots_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance to {output_path}")
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      filename: str = "classification_report.txt"):
        """
        Generate and save classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            filename: Output filename
        """
        logger.info("Generating classification report...")
        
        report = classification_report(y_true, y_pred, 
                                      target_names=['Human', 'Bot'],
                                      digits=4)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write("Twitter Bot Detection - Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        
        logger.info(f"Saved classification report to {output_path}")
        print("\n" + report)
    
    def save_metrics(self, metrics: Dict[str, float], 
                    filename: str = "metrics.json"):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_proba: np.ndarray = None,
                      feature_importance: np.ndarray = None,
                      feature_names: list = None) -> Dict[str, float]:
        """
        Complete model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            feature_importance: Feature importance scores (optional)
            feature_names: Feature names (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info("=" * 60)
        logger.info("Starting Model Evaluation")
        logger.info("=" * 60)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred)
        
        if y_proba is not None:
            self.plot_roc_curve(y_true, y_proba)
            self.plot_precision_recall_curve(y_true, y_proba)
        
        if feature_importance is not None and feature_names is not None:
            self.plot_feature_importance(feature_importance, feature_names)
        
        # Generate reports
        self.generate_classification_report(y_true, y_pred)
        self.save_metrics(metrics)
        
        logger.info("=" * 60)
        logger.info("Evaluation Complete")
        logger.info("=" * 60)
        
        return metrics


def main():
    """Main function for evaluation"""
    print("=" * 60)
    print("Twitter Bot Detection - Model Evaluation")
    print("=" * 60)
    
    # This is a standalone evaluation script
    # Load saved predictions and run evaluation
    
    # Example usage:
    # evaluator = ModelEvaluator()
    # metrics = evaluator.evaluate_model(y_true, y_pred, y_proba)
    
    logger.info("Use this module by importing ModelEvaluator")
    logger.info("Or run train.py for complete training and evaluation")


if __name__ == "__main__":
    main()
