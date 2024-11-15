class Bee:
    def __init__(self, k, centroids):
        """Initialize a bee with k clusters and centroids."""
        self.k = k
        self.centroids = centroids
        self.trials = 0
        self.metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
    
    def reset(self, k, centroids):
        """Reset the bee with new centroids."""
        self.k = k
        self.centroids = centroids
        self.trials = 0
        self.metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        } 