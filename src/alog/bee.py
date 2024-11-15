import numpy as np

class Bee:
    def __init__(self, k, centroids):
        """初始化蜜蜂
        
        Args:
            k (int): 聚類數量
            centroids (np.ndarray): 中心點座標
        """
        self.k = k
        self.centroids = centroids
        self.metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        self.trials = 0
    
    def reset(self, k, centroids):
        """重置蜜蜂狀態
        
        Args:
            k (int): 新的聚類數量
            centroids (np.ndarray): 新的中心點座標
        """
        self.k = k
        self.centroids = centroids
        self.metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        self.trials = 0 