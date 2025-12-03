"""
Prediction Script for Twitter Bot Detection
Use trained model to predict bot accounts
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
import json
import pickle

from src.models.classifiers import BotDetectionClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_accounts():
    """Create some test accounts to predict on (for demo)"""
    
    # Create a mix of bot-like and human-like accounts
    test_data = [
        # Likely BOTS (high following, low followers, many tweets, new account)
        {
            'user_id': 'test_bot_1',
            'username': 'promo_bot_2024',
            'followers_count': 50,
            'friends_count': 5000,
            'statuses_count': 10000,
            'verified': 0,
            'default_profile': 1,
            'default_profile_image': 1,
            'favourites_count': 100,
            'listed_count': 2,
            'account_age_days': 30
        },
        {
            'user_id': 'test_bot_2',
            'username': 'spam_account_x',
            'followers_count': 100,
            'friends_count': 8000,
            'statuses_count': 15000,
            'verified': 0,
            'default_profile': 1,
            'default_profile_image': 0,
            'favourites_count': 50,
            'listed_count': 1,
            'account_age_days': 15
        },
        # Likely HUMANS (balanced ratios, verified, older accounts)
        {
            'user_id': 'test_human_1',
            'username': 'real_person_123',
            'followers_count': 500,
            'friends_count': 400,
            'statuses_count': 2000,
            'verified': 0,
            'default_profile': 0,
            'default_profile_image': 0,
            'favourites_count': 1500,
            'listed_count': 10,
            'account_age_days': 1825  # 5 years
        },
        {
            'user_id': 'test_human_2',
            'username': 'verified_user',
            'followers_count': 10000,
            'friends_count': 500,
            'statuses_count': 5000,
            'verified': 1,
            'default_profile': 0,
            'default_profile_image': 0,
            'favourites_count': 3000,
            'listed_count': 50,
            'account_age_days': 2555  # 7 years
        },
        # Edge cases
        {
            'user_id': 'test_edge_1',
            'username': 'new_but_real',
            'followers_count': 200,
            'friends_count': 150,
            'statuses_count': 50,
            'verified': 0,
            'default_profile': 0,
            'default_profile_image': 0,
            'favourites_count': 100,
            'listed_count': 5,
            'account_age_days': 90
        }
    ]
    
    return pd.DataFrame(test_data)


def load_model_and_predict(df, model_path):
    """Load model and make predictions"""
    logger.info(f"Loading model from {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle different model formats
    if isinstance(model_data, dict) and 'classifier' in model_data:
        # New format
        classifier = model_data['classifier']
        optimal_threshold = model_data.get('optimal_threshold', 0.5)
        logger.info(f"Loaded model with threshold: {optimal_threshold}")
    else:
        # Old format or raw pickle
        classifier = BotDetectionClassifier()
        # Re-open to load with class method if needed, or just assume we can't easily load old format 
        # without the logic from classifiers.py. 
        # Actually, classifiers.py load_model expects the dict.
        # Let's try to use the loaded dict if it looks like the old format
        if isinstance(model_data, dict) and 'model' in model_data:
            classifier.model = model_data['model']
            classifier.scaler = model_data['scaler']
            classifier.classifier_type = model_data.get('classifier_type', 'unknown')
            optimal_threshold = 0.5
        else:
            raise ValueError("Unknown model format")

    # Feature Extraction (Simplified for prediction/demo)
    # In a real production pipeline, you would call the full FeatureExtractor
    # Here we assume we need to construct features to match the model
    
    # Try to load feature names to know expected shape
    try:
        with open("data/processed/feature_names.json", 'r') as f:
            feature_names = json.load(f)
        n_features = len(feature_names)
    except FileNotFoundError:
        logger.warning("Feature names not found, assuming 133 features (default)")
        n_features = 133

    # Generate features
    # 1. Profile features
    if all(col in df.columns for col in ['followers_count', 'friends_count', 'statuses_count']):
        profile_features = df[['followers_count', 'friends_count', 'statuses_count', 
                               'favourites_count', 'listed_count']].values
        
        follower_following_ratio = df['followers_count'] / np.maximum(df['friends_count'], 1)
        tweets_per_follower = df['statuses_count'] / np.maximum(df['followers_count'], 1)
        
        derived_features = np.column_stack([follower_following_ratio, tweets_per_follower])
        
        # 2. Graph embeddings (Dummy for demo/quick predict without graph build)
        # In full pipeline, you'd build graph and run graph2vec
        n_samples = len(df)
        # Assuming 128 dim graph2vec
        dummy_g2v = np.random.randn(n_samples, 128) * 0.1
        
        X = np.hstack([dummy_g2v, profile_features, derived_features])
    else:
        # If input doesn't have raw profile cols, assume it's already processed features?
        # Or just fail. For this script, we assume raw input.
        logger.warning("Input columns missing, creating zero matrix")
        X = np.zeros((len(df), n_features))

    # Adjust dimensions
    if X.shape[1] < n_features:
        padding = np.zeros((len(df), n_features - X.shape[1]))
        X = np.hstack([X, padding])
    elif X.shape[1] > n_features:
        X = X[:, :n_features]
        
    # Predict
    y_proba = classifier.predict_proba(X)
    y_pred = (y_proba[:, 1] >= optimal_threshold).astype(int)
    
    return y_pred, y_proba[:, 1], optimal_threshold


def run_demo(model_path):
    """Run demo predictions"""
    print("=" * 70)
    print(" " * 15 + "TWITTER BOT DETECTION")
    print(" " * 20 + "Prediction Demo")
    print("=" * 70)
    
    print("\n📝 Creating test accounts...")
    df = create_test_accounts()
    
    print("\n🔮 Making predictions...")
    try:
        y_pred, y_proba, threshold = load_model_and_predict(df, model_path)
        
        df['bot_probability'] = y_proba
        df['prediction'] = y_pred
        df['prediction_label'] = df['prediction'].map({0: 'Human', 1: 'Bot'})
        df['confidence'] = np.where(df['prediction'] == 1, 
                                    df['bot_probability'], 
                                    1 - df['bot_probability'])
        
        # Display results
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        
        for idx, row in df.iterrows():
            print(f"\n👤 Account: {row['username']}")
            print(f"   User ID: {row['user_id']}")
            print(f"   Followers: {row['followers_count']:,} | Following: {row['friends_count']:,}")
            print(f"   Tweets: {row['statuses_count']:,} | Account Age: {row['account_age_days']} days")
            print(f"   ⚡ Prediction: {row['prediction_label']}")
            print(f"   📊 Bot Probability: {row['bot_probability']:.1%}")
            print(f"   ✅ Confidence: {row['confidence']:.1%}")
            
        print(f"\n🎚️  Threshold used: {threshold:.3f}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return

    # Save demo results
    output_path = Path("results/demo_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")


def predict_from_csv(input_file, model_path, output_file):
    """Predict from CSV file"""
    print("=" * 60)
    print("Twitter Bot Detection - Prediction")
    print("=" * 60)
    
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} accounts")
    
    y_pred, y_proba, threshold = load_model_and_predict(df, model_path)
    
    df['is_bot'] = y_pred
    df['bot_probability'] = y_proba
    df['prediction'] = df['is_bot'].map({0: 'Human', 1: 'Bot'})
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    bot_count = y_pred.sum()
    print(f"\n✅ Prediction complete!")
    print(f"📊 Results: {bot_count} bots, {len(df) - bot_count} humans")
    print(f"💾 Saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Predict Twitter Bot Accounts")
    parser.add_argument('--input', type=str, help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='results/predictions.csv', help='Path to save predictions')
    parser.add_argument('--model-type', type=str, choices=['precision', 'recall'], default='precision', help='Model type to use')
    parser.add_argument('--model-path', type=str, help='Specific path to model file (overrides model-type)')
    parser.add_argument('--demo', action='store_true', help='Run demo with generated test accounts')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        filename = "bot_detector_improved.pkl" if args.model_type == 'recall' else "bot_detector.pkl"
        model_path = f"models/{filename}"
    
    if args.demo:
        run_demo(model_path)
    elif args.input:
        predict_from_csv(args.input, model_path, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
