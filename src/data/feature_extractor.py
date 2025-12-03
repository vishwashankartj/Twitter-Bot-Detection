"""
Feature Extractor for Twitter Bot Detection
Extracts network features and combines with Graph2Vec embeddings
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from Twitter data and graphs"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_names = []
        
    def extract_network_metrics(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Extract network metrics from a graph
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary of network metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics['num_nodes'] = graph.number_of_nodes()
            metrics['num_edges'] = graph.number_of_edges()
            metrics['density'] = nx.density(graph)
            
            # Degree metrics
            degrees = dict(graph.degree())
            if degrees:
                metrics['avg_degree'] = np.mean(list(degrees.values()))
                metrics['max_degree'] = np.max(list(degrees.values()))
                metrics['min_degree'] = np.min(list(degrees.values()))
            else:
                metrics['avg_degree'] = 0
                metrics['max_degree'] = 0
                metrics['min_degree'] = 0
            
            # Centrality metrics (for smaller graphs)
            if graph.number_of_nodes() < 1000:
                try:
                    centrality = nx.degree_centrality(graph)
                    metrics['avg_centrality'] = np.mean(list(centrality.values()))
                except:
                    metrics['avg_centrality'] = 0
                
                try:
                    clustering = nx.clustering(graph)
                    metrics['avg_clustering'] = np.mean(list(clustering.values()))
                except:
                    metrics['avg_clustering'] = 0
            else:
                metrics['avg_centrality'] = 0
                metrics['avg_clustering'] = 0
            
            # Connected components
            if nx.is_connected(graph):
                metrics['num_components'] = 1
            else:
                metrics['num_components'] = nx.number_connected_components(graph)
            
        except Exception as e:
            logger.warning(f"Error extracting metrics: {e}")
            # Return default metrics
            metrics = {
                'num_nodes': 0, 'num_edges': 0, 'density': 0,
                'avg_degree': 0, 'max_degree': 0, 'min_degree': 0,
                'avg_centrality': 0, 'avg_clustering': 0, 'num_components': 0
            }
        
        return metrics
    
    def extract_profile_features(self, df: pd.DataFrame, 
                                col_map: Dict[str, str]) -> pd.DataFrame:
        """
        Extract user profile features
        
        Args:
            df: DataFrame with user data
            col_map: Column mapping
            
        Returns:
            DataFrame with profile features
        """
        logger.info("Extracting profile features...")
        
        features = pd.DataFrame()
        
        # Followers count
        if 'followers' in col_map:
            features['followers_count'] = df[col_map['followers']]
        else:
            features['followers_count'] = 0
        
        # Following count
        if 'following' in col_map:
            features['following_count'] = df[col_map['following']]
        else:
            features['following_count'] = 0
        
        # Tweets count
        if 'tweets' in col_map:
            features['tweets_count'] = df[col_map['tweets']]
        else:
            features['tweets_count'] = 0
        
        # Derived features
        features['follower_following_ratio'] = features.apply(
            lambda row: row['followers_count'] / max(row['following_count'], 1),
            axis=1
        )
        
        features['tweets_per_follower'] = features.apply(
            lambda row: row['tweets_count'] / max(row['followers_count'], 1),
            axis=1
        )
        
        # Fill NaN and inf values
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)
        
        logger.info(f"Extracted {len(features.columns)} profile features")
        
        return features
    
    def combine_features(self, 
                        graph2vec_embeddings: np.ndarray,
                        network_metrics: pd.DataFrame = None,
                        profile_features: pd.DataFrame = None) -> Tuple[np.ndarray, List[str]]:
        """
        Combine all features into a single feature matrix
        
        Args:
            graph2vec_embeddings: Graph2Vec embeddings
            network_metrics: Network metrics DataFrame
            profile_features: Profile features DataFrame
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        logger.info("Combining features...")
        
        features_list = [graph2vec_embeddings]
        feature_names = [f'g2v_{i}' for i in range(graph2vec_embeddings.shape[1])]
        
        if network_metrics is not None:
            features_list.append(network_metrics.values)
            feature_names.extend(network_metrics.columns.tolist())
        
        if profile_features is not None:
            features_list.append(profile_features.values)
            feature_names.extend(profile_features.columns.tolist())
        
        # Combine all features
        X = np.hstack(features_list)
        
        logger.info(f"Combined feature matrix shape: {X.shape}")
        logger.info(f"Total features: {len(feature_names)}")
        
        self.feature_names = feature_names
        
        return X, feature_names


def main():
    """Main function to extract features"""
    print("=" * 60)
    print("Twitter Bot Detection - Feature Extraction")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv("data/processed/processed_data.csv")
    
    import json
    with open("data/processed/column_mapping.json", 'r') as f:
        col_map = json.load(f)
    
    # Load Graph2Vec embeddings
    embeddings = np.load("data/processed/graph2vec_embeddings.npy")
    
    # Extract profile features
    extractor = FeatureExtractor()
    profile_features = extractor.extract_profile_features(df[:len(embeddings)], col_map)
    
    # Combine features
    X, feature_names = extractor.combine_features(embeddings, profile_features=profile_features)
    
    # Load labels
    y = np.load("data/processed/graph_labels.npy")
    
    # Save combined features
    np.save("data/processed/features.npy", X)
    np.save("data/processed/labels.npy", y)
    
    with open("data/processed/feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🏷️  Labels shape: {y.shape}")
    
    return X, y, feature_names


if __name__ == "__main__":
    main()
