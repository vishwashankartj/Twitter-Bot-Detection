"""
Data Loader for Twitter Bot Detection
Loads and preprocesses Twitter bot detection datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitterDataLoader:
    """Load and preprocess Twitter bot detection data"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.df = None
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filename: Name of CSV file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_dir / filename
        logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def auto_detect_and_load(self) -> pd.DataFrame:
        """
        Automatically detect and load Twitter bot dataset
        
        Returns:
            DataFrame with loaded data
        """
        # Look for CSV files in data directory
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                "Please run download_data.py first."
            )
        
        logger.info(f"Found {len(csv_files)} CSV file(s)")
        
        # Load the first CSV file (or combine multiple if needed)
        df = self.load_csv(csv_files[0].name)
        self.df = df
        return df
    
    def identify_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identify important columns in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column types to column names
        """
        columns = df.columns.str.lower()
        column_map = {}
        
        # Identify label column
        label_candidates = ['bot', 'label', 'is_bot', 'account_type', 'class']
        for col in label_candidates:
            matches = [c for c in df.columns if col in c.lower()]
            if matches:
                column_map['label'] = matches[0]
                break
        
        # Identify user ID
        id_candidates = ['id', 'user_id', 'userid', 'screen_name', 'username']
        for col in id_candidates:
            matches = [c for c in df.columns if col in c.lower()]
            if matches:
                column_map['user_id'] = matches[0]
                break
        
        # Identify network features
        network_features = {
            'followers': ['followers', 'followers_count', 'follower_count'],
            'following': ['following', 'friends', 'friends_count', 'following_count'],
            'tweets': ['tweets', 'statuses', 'statuses_count', 'tweet_count'],
            'retweets': ['retweet', 'retweets_count'],
            'mentions': ['mention', 'mentions_count']
        }
        
        for feature_type, candidates in network_features.items():
            for col in candidates:
                matches = [c for c in df.columns if col in c.lower()]
                if matches:
                    column_map[feature_type] = matches[0]
                    break
        
        logger.info(f"Identified columns: {column_map}")
        return column_map
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (preprocessed DataFrame, column mapping)
        """
        logger.info("Preprocessing data...")
        
        # Identify columns
        col_map = self.identify_columns(df)
        
        # Handle missing values
        df = df.dropna(subset=[col_map.get('label', df.columns[0])])
        
        # Convert label to binary if needed
        if 'label' in col_map:
            label_col = col_map['label']
            unique_values = df[label_col].unique()
            logger.info(f"Label values: {unique_values}")
            
            # Convert to binary (1 = bot, 0 = human)
            if df[label_col].dtype == 'object':
                # String labels
                bot_keywords = ['bot', 'automated', '1', 'true', 'yes']
                df['is_bot'] = df[label_col].astype(str).str.lower().apply(
                    lambda x: 1 if any(keyword in x for keyword in bot_keywords) else 0
                )
            else:
                # Numeric labels
                df['is_bot'] = (df[label_col] > 0).astype(int)
            
            col_map['label'] = 'is_bot'
        
        # Fill missing numeric values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Remove duplicates
        initial_len = len(df)
        if 'user_id' in col_map:
            df = df.drop_duplicates(subset=[col_map['user_id']])
        else:
            df = df.drop_duplicates()
        
        logger.info(f"Removed {initial_len - len(df)} duplicate records")
        logger.info(f"Final dataset: {len(df)} records")
        
        # Class distribution
        if 'is_bot' in df.columns:
            bot_count = df['is_bot'].sum()
            human_count = len(df) - bot_count
            logger.info(f"Class distribution: {bot_count} bots, {human_count} humans")
            logger.info(f"Bot ratio: {bot_count/len(df)*100:.2f}%")
        
        return df, col_map
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """
        Save processed data
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Save column mapping
        col_map = self.identify_columns(df)
        map_path = output_dir / "column_mapping.json"
        with open(map_path, 'w') as f:
            json.dump(col_map, f, indent=2)
        logger.info(f"Saved column mapping to {map_path}")


def main():
    """Main function to load and preprocess data"""
    print("=" * 60)
    print("Twitter Bot Detection - Data Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = TwitterDataLoader()
    
    # Load data
    df = loader.auto_detect_and_load()
    
    # Preprocess
    df_processed, col_map = loader.preprocess_data(df)
    
    # Save
    loader.save_processed_data(df_processed)
    
    print("\n✅ Data loading and preprocessing complete!")
    print(f"📊 Dataset shape: {df_processed.shape}")
    print(f"🏷️  Label column: {col_map.get('label', 'N/A')}")
    
    return df_processed, col_map


if __name__ == "__main__":
    main()
