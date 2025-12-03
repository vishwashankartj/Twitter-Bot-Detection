"""
Main Training Script for Twitter Bot Detection
Orchestrates the complete training pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve

# Import project modules
from src.data.data_loader import TwitterDataLoader
from src.data.graph_builder import TwitterGraphBuilder
from src.models.graph2vec_embeddings import Graph2VecEmbedder, extract_graph_labels
from src.data.feature_extractor import FeatureExtractor
from src.models.classifiers import BotDetectionClassifier
from src.evaluation.metrics import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def find_optimal_threshold(y_true, y_proba):
    """Find optimal classification threshold for better recall"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba[:, 1])
    
    # Find threshold that gives recall > 0.7 with best precision
    good_recall_idx = np.where(recall >= 0.7)[0]
    
    if len(good_recall_idx) > 0:
        best_idx = good_recall_idx[np.argmax(precision[good_recall_idx])]
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        logger.info(f"At this threshold - Precision: {precision[best_idx]:.3f}, Recall: {recall[best_idx]:.3f}")
        return optimal_threshold
    else:
        return 0.3  # Lower threshold for better recall


def main(args):
    """Main training pipeline"""
    mode_str = "High Recall" if args.mode == 'recall' else "High Precision"
    print("=" * 80)
    print(" " * 20 + "TWITTER BOT DETECTION")
    print(" " * 15 + f"Training Pipeline ({mode_str} Mode)")
    print("=" * 80)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("=" * 80)
    
    loader = TwitterDataLoader(config['data']['raw_dir'])
    df = loader.auto_detect_and_load()
    df_processed, col_map = loader.preprocess_data(df)
    loader.save_processed_data(df_processed)
    
    # Step 2: Build network graphs
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Network Graph Construction")
    logger.info("=" * 80)
    
    builder = TwitterGraphBuilder()
    subgraphs = builder.create_subgraphs_for_graph2vec(
        df_processed, 
        col_map,
        min_nodes=config['graph']['min_nodes'],
        max_graphs=1000
    )
    builder.save_graphs(subgraphs)
    
    # Step 3: Generate Graph2Vec embeddings
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Graph2Vec Embedding Generation")
    logger.info("=" * 80)
    
    embedder = Graph2VecEmbedder(
        dimensions=config['graph2vec']['dimensions'],
        wl_iterations=config['graph2vec']['wl_iterations'],
        epochs=config['graph2vec']['epochs'],
        learning_rate=config['graph2vec']['learning_rate'],
        seed=config['graph2vec']['seed']
    )
    
    embeddings = embedder.fit(subgraphs)
    labels = extract_graph_labels(subgraphs)
    
    # Save embeddings
    np.save(Path(config['data']['processed_dir']) / "graph2vec_embeddings.npy", embeddings)
    np.save(Path(config['data']['processed_dir']) / "graph_labels.npy", labels)
    embedder.save_model()
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    logger.info(f"Label distribution: Bots={labels.sum()}, Humans={len(labels)-labels.sum()}")
    
    # Step 4: Feature extraction
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Feature Extraction")
    logger.info("=" * 80)
    
    extractor = FeatureExtractor()
    
    # Extract profile features for the same users
    profile_features = extractor.extract_profile_features(
        df_processed[:len(embeddings)], 
        col_map
    )
    
    # Combine features
    X, feature_names = extractor.combine_features(
        embeddings, 
        profile_features=profile_features
    )
    y = labels
    
    # Save features
    np.save(Path(config['data']['processed_dir']) / "features.npy", X)
    np.save(Path(config['data']['processed_dir']) / "labels.npy", y)
    
    import json
    with open(Path(config['data']['processed_dir']) / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Step 5: Train classifier
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Model Training")
    logger.info("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train ensemble classifier
    classifier = BotDetectionClassifier(classifier_type='ensemble')
    
    # Configure weights based on mode
    weights = [3, 2, 1] if args.mode == 'recall' else None
    classifier.build_model(weights=weights)
    
    train_metrics = classifier.train(X_train, y_train, X_test, y_test)
    
    # Step 6: Evaluation and Threshold Optimization
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Model Evaluation")
    logger.info("=" * 80)
    
    # Predictions
    y_proba = classifier.predict_proba(X_test)
    
    optimal_threshold = 0.5
    if args.mode == 'recall':
        optimal_threshold = find_optimal_threshold(y_test, y_proba)
        logger.info(f"Using optimized threshold: {optimal_threshold}")
    
    y_pred = (y_proba[:, 1] >= optimal_threshold).astype(int)
    
    # Evaluate
    results_dir = "results_improved" if args.mode == 'recall' else config['output']['results_dir']
    evaluator = ModelEvaluator(output_dir=results_dir)
    metrics = evaluator.evaluate_model(
        y_test, y_pred, y_proba,
        feature_importance=classifier.feature_importance,
        feature_names=feature_names
    )
    
    # Save model
    model_filename = "bot_detector_improved.pkl" if args.mode == 'recall' else "bot_detector.pkl"
    model_path = Path(config['output']['models_dir']) / model_filename
    
    model_data = {
        'classifier': classifier,
        'optimal_threshold': optimal_threshold,
        'mode': args.mode
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Final summary
    print("\n" + "=" * 80)
    print(" " * 25 + "TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Final Test Accuracy: {metrics['accuracy']:.2%}")
    print(f"🎯 Precision: {metrics['precision']:.2%}")
    print(f"🎯 Recall: {metrics['recall']:.2%}")
    print(f"🎯 F1-Score: {metrics['f1_score']:.2%}")
    if 'roc_auc' in metrics:
        print(f"📈 ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"📁 Results saved to: {results_dir}")
    print(f"🎚️  Threshold: {optimal_threshold:.3f}")
    
    print("\n" + "=" * 80)
    
    return classifier, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Twitter Bot Detection Model")
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['precision', 'recall'],
        default='precision',
        help='Training mode: precision (default) or recall (improved bot detection)'
    )
    
    args = parser.parse_args()
    
    try:
        classifier, metrics = main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
