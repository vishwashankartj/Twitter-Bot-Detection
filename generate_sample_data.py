"""
Sample Twitter Bot Dataset Generator
Creates a realistic synthetic dataset for testing the bot detection pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)


def generate_sample_dataset(n_samples=5000, bot_ratio=0.3):
    """
    Generate a sample Twitter bot detection dataset
    
    Args:
        n_samples: Number of accounts to generate
        bot_ratio: Proportion of bot accounts
    """
    print(f"Generating {n_samples} sample Twitter accounts...")
    
    n_bots = int(n_samples * bot_ratio)
    n_humans = n_samples - n_bots
    
    data = []
    
    # Generate human accounts
    print(f"  Creating {n_humans} human accounts...")
    for i in range(n_humans):
        account = {
            'user_id': f'human_{i}',
            'username': f'user_{i}',
            'followers_count': np.random.lognormal(5, 2),  # More realistic distribution
            'friends_count': np.random.lognormal(4.5, 1.8),
            'statuses_count': np.random.lognormal(6, 2),
            'verified': np.random.choice([0, 1], p=[0.95, 0.05]),
            'default_profile': np.random.choice([0, 1], p=[0.8, 0.2]),
            'default_profile_image': np.random.choice([0, 1], p=[0.9, 0.1]),
            'favourites_count': np.random.lognormal(4, 2),
            'listed_count': np.random.lognormal(2, 1.5),
            'account_age_days': np.random.randint(365, 3650),
            'is_bot': 0
        }
        data.append(account)
    
    # Generate bot accounts (different patterns)
    print(f"  Creating {n_bots} bot accounts...")
    for i in range(n_bots):
        # Bots typically have:
        # - High following count, low followers
        # - Many tweets in short time
        # - Default profiles
        # - Recent account creation
        
        account = {
            'user_id': f'bot_{i}',
            'username': f'bot_{i}',
            'followers_count': np.random.lognormal(2, 1.5),  # Fewer followers
            'friends_count': np.random.lognormal(6, 1.5),    # Many following
            'statuses_count': np.random.lognormal(7, 1.5),   # Many tweets
            'verified': 0,  # Bots are rarely verified
            'default_profile': np.random.choice([0, 1], p=[0.3, 0.7]),  # Often default
            'default_profile_image': np.random.choice([0, 1], p=[0.4, 0.6]),
            'favourites_count': np.random.lognormal(3, 1),
            'listed_count': np.random.lognormal(1, 1),
            'account_age_days': np.random.randint(1, 365),  # Newer accounts
            'is_bot': 1
        }
        data.append(account)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert to integers
    int_columns = ['followers_count', 'friends_count', 'statuses_count', 
                   'favourites_count', 'listed_count', 'account_age_days']
    for col in int_columns:
        df[col] = df[col].astype(int)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Generate and save sample dataset"""
    print("=" * 60)
    print("Sample Twitter Bot Dataset Generator")
    print("=" * 60)
    
    # Generate dataset
    df = generate_sample_dataset(n_samples=5000, bot_ratio=0.3)
    
    # Save to data/raw
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "twitter_bot_sample.csv"
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n✅ Sample dataset created!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Total accounts: {len(df)}")
    print(f"🤖 Bots: {df['is_bot'].sum()} ({df['is_bot'].sum()/len(df)*100:.1f}%)")
    print(f"👤 Humans: {(1-df['is_bot']).sum()} ({(1-df['is_bot']).sum()/len(df)*100:.1f}%)")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n🎯 You can now run: python train.py")
    print(f"\n💡 Note: This is synthetic data for testing.")
    print(f"   Replace with real Kaggle data later for production use.")
    
    return df


if __name__ == "__main__":
    main()
