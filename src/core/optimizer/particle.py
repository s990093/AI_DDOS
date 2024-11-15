import numpy as np

class Particle:
    def __init__(self, k, n_features):
        """Initialize a particle for PSO clustering.
        
        Args:
            k (int): Number of clusters
            n_features (int): Number of features in the dataset
        """
        self.k = k
        self.position = None  # Current centroids
        self.velocity = None
        self.best_position = None  # Personal best
        self.best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        self.current_metrics = self.best_metrics.copy()
        self.initialize(n_features)
    
    def initialize(self, n_features):
        """Initialize particle position and velocity."""
        # Initialize centroids randomly
        self.position = np.random.randn(self.k, n_features)
        # Initialize velocity randomly
        self.velocity = np.random.randn(self.k, n_features) * 0.1
        # Initialize personal best
        self.best_position = self.position.copy() 