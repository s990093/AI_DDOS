import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from numba import jit, prange

@jit(nopython=True, parallel=True)
def calculate_distances_and_labels(X_final, positions_reshaped):
    """Calculate distances and labels using Numba"""
    n_particles, n_clusters, n_samples, n_features = positions_reshaped.shape[0], positions_reshaped.shape[1], X_final.shape[0], X_final.shape[1]
    distances = np.zeros((n_particles, n_clusters, n_samples))
    labels_all = np.zeros((n_particles, n_samples), dtype=np.int64)
    
    for i in prange(n_particles):
        for j in range(n_samples):
            min_dist = np.inf
            min_cluster = 0
            for k in range(n_clusters):
                dist = 0.0
                for f in range(n_features):
                    diff = X_final[j, f] - positions_reshaped[i, k, f]
                    dist += diff * diff
                dist = np.sqrt(dist)
                distances[i, k, j] = dist
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = k
            labels_all[i, j] = min_cluster
    
    return distances, labels_all

def calculate_metrics(X, labels):
    """Calculate clustering evaluation metrics"""
    try:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        return silhouette, calinski, davies
    except Exception:
        return None

def objective_function(positions, shm_name, shape, dtype, clusters, n_processes):
    # Get data from shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    X_final = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    n_particles = positions.shape[0]
    
    # 向量化計算所有粒子的標籤
    positions_reshaped = positions.reshape(n_particles, clusters, -1)
    # 使用廣播計算所有距離
    distances, labels_all = calculate_distances_and_labels(X_final, positions_reshaped)
    
    # 檢查每個粒子的唯一標籤數
    unique_counts = [len(np.unique(labels)) for labels in labels_all]
    valid_indices = [i for i, count in enumerate(unique_counts) if count == clusters]
    
    if not valid_indices:
        return np.full(n_particles, 1e6)
    
    # 只處理有效的標籤
    valid_labels = labels_all[valid_indices]
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(calculate_metrics, X_final, labels) 
                  for labels in valid_labels]
        
        # 初始化分數數組
        scores = np.full(n_particles, float('inf'))
        
        # 收集結果
        for idx, future in zip(valid_indices, futures):
            metrics = future.result()
            if metrics is not None:
                silhouette, calinski, davies = metrics
                scores[idx] = (-silhouette * 0.4 + 
                             -calinski/10000 * 0.3 + 
                             davies * 0.3)
    
    # Clean up shared memory
    existing_shm.close()
    return scores

