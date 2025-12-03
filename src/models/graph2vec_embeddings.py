"""
Graph2Vec Embeddings Generator
Generates graph embeddings using a custom Graph2Vec implementation
Compatible with Python 3.13+
"""

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Graph2VecEmbedder:
    """Generate Graph2Vec embeddings for network graphs using gensim"""
    
    def __init__(self, dimensions: int = 128, 
                 wl_iterations: int = 3,
                 epochs: int = 10,
                 learning_rate: float = 0.025,
                 min_count: int = 5,
                 workers: int = 4,
                 seed: int = 42):
        """
        Initialize Graph2Vec embedder
        
        Args:
            dimensions: Embedding dimensions
            wl_iterations: Weisfeiler-Lehman iterations
            epochs: Training epochs
            learning_rate: Learning rate
            min_count: Minimum count for vocabulary
            workers: Number of workers
            seed: Random seed
        """
        self.dimensions = dimensions
        self.wl_iterations = wl_iterations
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.workers = workers
        self.seed = seed
        
        self.model = None
        self.embeddings = None
        self.graph_documents = []
        
    def weisfeiler_lehman_subtree_features(self, graph: nx.Graph, iterations: int) -> List[str]:
        """
        Generate Weisfeiler-Lehman subtree features for a graph
        
        Args:
            graph: NetworkX graph
            iterations: Number of WL iterations
            
        Returns:
            List of WL subtree patterns
        """
        # Initialize node labels
        node_labels = {node: str(graph.degree(node)) for node in graph.nodes()}
        
        all_features = []
        
        for iteration in range(iterations):
            # Collect features from current iteration
            features = list(node_labels.values())
            all_features.extend(features)
            
            # Update labels based on neighborhood
            new_labels = {}
            for node in graph.nodes():
                # Get neighbor labels
                neighbor_labels = sorted([node_labels[neighbor] for neighbor in graph.neighbors(node)])
                # Create new label by concatenating current label with neighbor labels
                new_label = f"{node_labels[node]}_{'_'.join(neighbor_labels)}"
                new_labels[node] = new_label
            
            node_labels = new_labels
        
        return all_features
    
    def fit(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Fit Graph2Vec model and generate embeddings
        
        Args:
            graphs: List of NetworkX graphs
            
        Returns:
            Embedding matrix (n_graphs x dimensions)
        """
        logger.info(f"Fitting Graph2Vec on {len(graphs)} graphs...")
        logger.info(f"Parameters: dim={self.dimensions}, wl_iter={self.wl_iterations}, epochs={self.epochs}")
        
        # Generate WL subtree features for each graph
        self.graph_documents = []
        for i, graph in enumerate(graphs):
            features = self.weisfeiler_lehman_subtree_features(graph, self.wl_iterations)
            self.graph_documents.append(features)
        
        logger.info(f"Generated WL features for {len(self.graph_documents)} graphs")
        
        # Train Word2Vec model on graph documents
        self.model = Word2Vec(
            sentences=self.graph_documents,
            vector_size=self.dimensions,
            window=5,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            sg=1,  # Skip-gram
            seed=self.seed,
            alpha=self.learning_rate
        )
        
        # Generate graph-level embeddings by averaging word vectors
        self.embeddings = np.zeros((len(graphs), self.dimensions))
        
        for i, doc in enumerate(self.graph_documents):
            vectors = []
            for word in doc:
                if word in self.model.wv:
                    vectors.append(self.model.wv[word])
            
            if vectors:
                self.embeddings[i] = np.mean(vectors, axis=0)
            else:
                # If no vectors found, use random initialization
                self.embeddings[i] = np.random.randn(self.dimensions) * 0.01
        
        logger.info(f"Generated embeddings shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def transform(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Transform new graphs to embeddings
        
        Args:
            graphs: List of NetworkX graphs
            
        Returns:
            Embedding matrix
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Transforming {len(graphs)} graphs...")
        
        embeddings = np.zeros((len(graphs), self.dimensions))
        
        for i, graph in enumerate(graphs):
            features = self.weisfeiler_lehman_subtree_features(graph, self.wl_iterations)
            vectors = []
            
            for word in features:
                if word in self.model.wv:
                    vectors.append(self.model.wv[word])
            
            if vectors:
                embeddings[i] = np.mean(vectors, axis=0)
            else:
                embeddings[i] = np.random.randn(self.dimensions) * 0.01
        
        return embeddings
    
    def save_model(self, filepath: str = "models/graph2vec_model.pkl"):
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
                'embeddings': self.embeddings,
                'graph_documents': self.graph_documents,
                'params': {
                    'dimensions': self.dimensions,
                    'wl_iterations': self.wl_iterations,
                    'epochs': self.epochs,
                    'learning_rate': self.learning_rate
                }
            }, f)
        
        logger.info(f"Saved Graph2Vec model to {output_path}")
    
    def load_model(self, filepath: str = "models/graph2vec_model.pkl"):
        """
        Load trained model
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.embeddings = data['embeddings']
        self.graph_documents = data.get('graph_documents', [])
        
        logger.info(f"Loaded Graph2Vec model from {filepath}")
        logger.info(f"Embeddings shape: {self.embeddings.shape}")


def extract_graph_labels(graphs: List[nx.Graph]) -> np.ndarray:
    """
    Extract labels from graphs
    
    Args:
        graphs: List of NetworkX graphs with 'is_bot' node attribute
        
    Returns:
        Array of labels
    """
    labels = []
    
    for graph in graphs:
        # Get majority label from nodes in the graph
        node_labels = [data.get('is_bot', 0) for _, data in graph.nodes(data=True)]
        
        if node_labels:
            # Use majority voting
            majority_label = 1 if sum(node_labels) > len(node_labels) / 2 else 0
        else:
            majority_label = 0
        
        labels.append(majority_label)
    
    return np.array(labels)


def main():
    """Main function to generate Graph2Vec embeddings"""
    print("=" * 60)
    print("Twitter Bot Detection - Graph2Vec Embeddings")
    print("=" * 60)
    
    # Load graphs
    from src.data.graph_builder import TwitterGraphBuilder
    
    builder = TwitterGraphBuilder()
    graphs = builder.load_graphs()
    
    logger.info(f"Loaded {len(graphs)} graphs")
    
    # Extract labels
    labels = extract_graph_labels(graphs)
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Generate embeddings
    embedder = Graph2VecEmbedder(
        dimensions=128,
        wl_iterations=3,
        epochs=10
    )
    
    embeddings = embedder.fit(graphs)
    
    # Save embeddings and labels
    output_dir = Path("data/processed")
    
    np.save(output_dir / "graph2vec_embeddings.npy", embeddings)
    np.save(output_dir / "graph_labels.npy", labels)
    
    embedder.save_model()
    
    print(f"\n✅ Graph2Vec embeddings generated!")
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"🏷️  Labels shape: {labels.shape}")
    print(f"📁 Saved to: {output_dir}")
    
    return embeddings, labels


if __name__ == "__main__":
    main()



def extract_graph_labels(graphs: List[nx.Graph]) -> np.ndarray:
    """
    Extract labels from graphs
    
    Args:
        graphs: List of NetworkX graphs with 'is_bot' node attribute
        
    Returns:
        Array of labels
    """
    labels = []
    
    for graph in graphs:
        # Get majority label from nodes in the graph
        node_labels = [data.get('is_bot', 0) for _, data in graph.nodes(data=True)]
        
        if node_labels:
            # Use majority voting
            majority_label = 1 if sum(node_labels) > len(node_labels) / 2 else 0
        else:
            majority_label = 0
        
        labels.append(majority_label)
    
    return np.array(labels)


def main():
    """Main function to generate Graph2Vec embeddings"""
    print("=" * 60)
    print("Twitter Bot Detection - Graph2Vec Embeddings")
    print("=" * 60)
    
    # Load graphs
    from src.data.graph_builder import TwitterGraphBuilder
    
    builder = TwitterGraphBuilder()
    graphs = builder.load_graphs()
    
    logger.info(f"Loaded {len(graphs)} graphs")
    
    # Extract labels
    labels = extract_graph_labels(graphs)
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Generate embeddings
    embedder = Graph2VecEmbedder(
        dimensions=128,
        wl_iterations=3,
        epochs=10
    )
    
    embeddings = embedder.fit(graphs)
    
    # Save embeddings and labels
    output_dir = Path("data/processed")
    
    np.save(output_dir / "graph2vec_embeddings.npy", embeddings)
    np.save(output_dir / "graph_labels.npy", labels)
    
    embedder.save_model()
    
    print(f"\n✅ Graph2Vec embeddings generated!")
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"🏷️  Labels shape: {labels.shape}")
    print(f"📁 Saved to: {output_dir}")
    
    return embeddings, labels


if __name__ == "__main__":
    main()
