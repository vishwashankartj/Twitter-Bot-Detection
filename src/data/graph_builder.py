"""
Graph Builder for Twitter Bot Detection
Constructs network graphs from Twitter user interactions
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwitterGraphBuilder:
    """Build network graphs from Twitter data"""
    
    def __init__(self):
        """Initialize graph builder"""
        self.graphs = {}
        
    def build_follower_network(self, df: pd.DataFrame, 
                               user_col: str = 'user_id',
                               followers_col: str = 'followers_count',
                               following_col: str = 'friends_count') -> nx.Graph:
        """
        Build follower network graph
        
        Args:
            df: DataFrame with user data
            user_col: Column name for user IDs
            followers_col: Column name for followers count
            following_col: Column name for following count
            
        Returns:
            NetworkX graph
        """
        logger.info("Building follower network...")
        
        G = nx.Graph()
        
        # Add nodes with attributes
        for idx, row in df.iterrows():
            user_id = row[user_col] if user_col in df.columns else idx
            
            # Node attributes
            attrs = {
                'is_bot': row.get('is_bot', 0),
                'followers': row.get(followers_col, 0),
                'following': row.get(following_col, 0)
            }
            
            G.add_node(user_id, **attrs)
        
        # Create edges based on follower/following patterns
        # For demonstration, we'll create edges between users with similar follower counts
        # In real scenarios, you'd have actual follower relationships
        users = list(G.nodes())
        
        for i, user1 in enumerate(users[:1000]):  # Limit for performance
            for user2 in users[i+1:i+20]:  # Connect to nearby users
                # Create edge if follower counts are similar (proxy for potential connection)
                followers1 = G.nodes[user1].get('followers', 0)
                followers2 = G.nodes[user2].get('followers', 0)
                
                if followers1 > 0 and followers2 > 0:
                    ratio = min(followers1, followers2) / max(followers1, followers2)
                    if ratio > 0.5:  # Similar follower counts
                        G.add_edge(user1, user2, weight=ratio)
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def build_interaction_network(self, df: pd.DataFrame,
                                  user_col: str = 'user_id',
                                  retweet_col: Optional[str] = None,
                                  mention_col: Optional[str] = None) -> nx.Graph:
        """
        Build interaction network based on retweets and mentions
        
        Args:
            df: DataFrame with user data
            user_col: Column name for user IDs
            retweet_col: Column name for retweet counts
            mention_col: Column name for mention counts
            
        Returns:
            NetworkX graph
        """
        logger.info("Building interaction network...")
        
        G = nx.Graph()
        
        # Add nodes
        for idx, row in df.iterrows():
            user_id = row[user_col] if user_col in df.columns else idx
            
            attrs = {
                'is_bot': row.get('is_bot', 0),
                'retweets': row.get(retweet_col, 0) if retweet_col else 0,
                'mentions': row.get(mention_col, 0) if mention_col else 0
            }
            
            G.add_node(user_id, **attrs)
        
        # Create edges based on interaction patterns
        users = list(G.nodes())
        
        for i, user1 in enumerate(users[:1000]):
            for user2 in users[i+1:i+15]:
                # Connect users with similar interaction patterns
                retweets1 = G.nodes[user1].get('retweets', 0)
                retweets2 = G.nodes[user2].get('retweets', 0)
                
                if retweets1 > 0 or retweets2 > 0:
                    # Random interaction probability (in real data, use actual interactions)
                    if np.random.random() < 0.1:
                        weight = (retweets1 + retweets2) / 2
                        G.add_edge(user1, user2, weight=weight)
        
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def build_ego_graphs(self, G: nx.Graph, radius: int = 2) -> List[nx.Graph]:
        """
        Build ego graphs for each node
        
        Args:
            G: Input graph
            radius: Radius of ego graph
            
        Returns:
            List of ego graphs
        """
        logger.info(f"Building ego graphs with radius {radius}...")
        
        ego_graphs = []
        nodes = list(G.nodes())[:500]  # Limit for performance
        
        for node in nodes:
            try:
                ego = nx.ego_graph(G, node, radius=radius)
                if ego.number_of_nodes() >= 3:  # Minimum size
                    ego_graphs.append(ego)
            except:
                continue
        
        logger.info(f"Created {len(ego_graphs)} ego graphs")
        return ego_graphs
    
    def create_subgraphs_for_graph2vec(self, df: pd.DataFrame,
                                       col_map: Dict[str, str],
                                       min_nodes: int = 5,
                                       max_graphs: int = 1000) -> List[nx.Graph]:
        """
        Create subgraphs suitable for Graph2Vec
        
        Args:
            df: DataFrame with user data
            col_map: Column mapping dictionary
            min_nodes: Minimum nodes per subgraph
            max_graphs: Maximum number of subgraphs
            
        Returns:
            List of subgraphs
        """
        logger.info("Creating subgraphs for Graph2Vec...")
        
        # Build main network
        user_col = col_map.get('user_id', df.columns[0])
        followers_col = col_map.get('followers', 'followers_count')
        following_col = col_map.get('following', 'friends_count')
        
        G_main = self.build_follower_network(df, user_col, followers_col, following_col)
        
        # Create ego graphs
        subgraphs = self.build_ego_graphs(G_main, radius=2)
        
        # Filter by size
        subgraphs = [g for g in subgraphs if g.number_of_nodes() >= min_nodes]
        
        # Limit number
        if len(subgraphs) > max_graphs:
            subgraphs = subgraphs[:max_graphs]
        
        logger.info(f"Created {len(subgraphs)} subgraphs for Graph2Vec")
        
        return subgraphs
    
    def save_graphs(self, graphs: List[nx.Graph], filename: str = "graphs.pkl"):
        """
        Save graphs to file
        
        Args:
            graphs: List of graphs
            filename: Output filename
        """
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        with open(output_path, 'wb') as f:
            pickle.dump(graphs, f)
        
        logger.info(f"Saved {len(graphs)} graphs to {output_path}")
    
    def load_graphs(self, filename: str = "graphs.pkl") -> List[nx.Graph]:
        """
        Load graphs from file
        
        Args:
            filename: Input filename
            
        Returns:
            List of graphs
        """
        input_path = Path("data/processed") / filename
        
        with open(input_path, 'rb') as f:
            graphs = pickle.load(f)
        
        logger.info(f"Loaded {len(graphs)} graphs from {input_path}")
        return graphs


def main():
    """Main function to build graphs"""
    print("=" * 60)
    print("Twitter Bot Detection - Graph Builder")
    print("=" * 60)
    
    # Load processed data
    df = pd.read_csv("data/processed/processed_data.csv")
    
    import json
    with open("data/processed/column_mapping.json", 'r') as f:
        col_map = json.load(f)
    
    # Build graphs
    builder = TwitterGraphBuilder()
    subgraphs = builder.create_subgraphs_for_graph2vec(df, col_map)
    
    # Save graphs
    builder.save_graphs(subgraphs)
    
    print(f"\n✅ Graph building complete!")
    print(f"📊 Created {len(subgraphs)} subgraphs")
    
    return subgraphs


if __name__ == "__main__":
    main()
