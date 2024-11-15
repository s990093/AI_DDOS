from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

def calculate_metrics(X, labels):
    """計算聚類評估指標
    
    Args:
        X (np.ndarray): 輸入數據
        labels (np.ndarray): 聚類標籤
        
    Returns:
        dict: 包含各項評估指標的字典
    """
    try:
        metrics = {
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        }
    except ValueError:
        metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
    
    return metrics

def is_better_solution(new_metrics, old_metrics):
    """比較兩個解的優劣
    
    Args:
        new_metrics (dict): 新解的評估指標
        old_metrics (dict): 舊解的評估指標
        
    Returns:
        bool: 如果新解更好則返回True
    """
    # 使用Davies-Bouldin指標作為主要比較標準（越小越好）
    if new_metrics['davies_bouldin'] < old_metrics['davies_bouldin']:
        return True
    
    # 如果Davies-Bouldin指標相同，則比較其他指標
    if new_metrics['davies_bouldin'] == old_metrics['davies_bouldin']:
        if new_metrics['silhouette'] > old_metrics['silhouette']:
            return True
        if (new_metrics['silhouette'] == old_metrics['silhouette'] and 
            new_metrics['calinski_harabasz'] > old_metrics['calinski_harabasz']):
            return True
    
    return False 