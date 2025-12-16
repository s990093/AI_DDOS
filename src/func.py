from pyswarms.single.global_best import GlobalBestPSO
from clusters.objective import objective_function

import numpy as np
from rich.console import Console
from rich.theme import Theme
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import multiprocessing
from sklearn.cluster import KMeans
from multiprocessing import shared_memory



def setup_rich_console():
    """設置Rich控制台主題和實例"""
    custom_theme = Theme({
        "primary": "bold green",
        "secondary": "dim white",
        "info": "bold blue",
        "warning": "bold yellow",
        "error": "bold red",
    })
    return Console(theme=custom_theme)

console = setup_rich_console()

def get_best_centroids(X_final, cpu, k, iterations, population):    

    # 創建共享記憶體
    X_final_shape = X_final.shape
    X_final_dtype = X_final.dtype
    shm = shared_memory.SharedMemory(create=True, size=X_final.nbytes)
    shm_array = np.ndarray(X_final_shape, dtype=X_final_dtype,
                           buffer=shm.buf)
    shm_array[:] = X_final[:]  # 複製數據到共享記憶體

    # 設置進程數
    n_processes = cpu if cpu > 0 else multiprocessing.cpu_count()

    best_k = None
    best_metrics = None
    best_centroids = None
    best_overall_score = float('inf')
    
  
    
    # 為每個k值進行聚類
    console.print(f"\n[bold cyan]Testing k={k}[/bold cyan]")
    
    # 設置PSO參數
    dimensions = k * X_final.shape[1]
    options = {
        'c1': 1.5,    # 降低認知參數，減少個體搜索範圍
        'c2': 1.7,    # 提高社會參數，增加群體影響
        'w': 0.6,     # 降低慣性權重，加強局部搜索
        'k': 7,       # 增加鄰居數量，提高信息共享
        'p': 2        # 保持歐氏距離
    }
    
    # 使用閉包來追踪迭代次數
    iteration_counter = {'count': 0}
    
    def pso_objective(positions):
        # 動態調整慣性權重
        w = 0.9 - (0.9 - 0.4) * (iteration_counter['count'] / iterations)
        iteration_counter['count'] += 1
        optimizer.options['w'] = w
        return objective_function(positions, shm.name, X_final.shape, X_final.dtype, k, n_processes)
    
    # 設置邊界
    X_min = np.min(X_final, axis=0)
    X_max = np.max(X_final, axis=0)
    
    bounds = (
        np.tile(X_min, k),  # 下界
        np.tile(X_max, k)   # 上界
    )
    
    # 初始化PSO優化器
    optimizer = GlobalBestPSO(
        n_particles=population,
        dimensions=dimensions,
        options=options,
        bounds=bounds
    )
    
    # 執優化
    best_cost, best_pos = optimizer.optimize(
        pso_objective,
        iters=iterations,
    )
    
    # 使用最佳位置進行最終的k-means聚類
    best_centroids = best_pos.reshape(k, -1)
    
    final_kmeans = KMeans(
        n_clusters=k,
        init=best_centroids,
        n_init=1
    )
    
    final_labels = final_kmeans.fit_predict(X_final)
    
    # 計算最終的評估指標
    final_metrics = {
        'silhouette': silhouette_score(X_final, final_labels),
        'calinski': calinski_harabasz_score(X_final, final_labels),
        'davies': davies_bouldin_score(X_final, final_labels)
    }
    
    # 計算綜合得分
    calinski_scaled = (
        -final_metrics['calinski'] / 10000 * 0.3 
        if final_metrics['calinski'] <= 10000
        else -np.log10(final_metrics['calinski']) * 0.3
    )
    
    overall_score = (
        -final_metrics['silhouette'] * 0.4 +
        calinski_scaled +
        final_metrics['davies'] * 0.3
    )
    
    console.print(f"[info]k={k} overall score: {overall_score:.3f}")
    
    # 更新最佳結果
    if overall_score < best_overall_score:
        best_overall_score = overall_score
        best_k = k
        best_metrics = final_metrics
        best_centroids = best_centroids

    console.print(f"[info]Silhouette Score: {best_metrics['silhouette']:.3f}", style="primary")
    console.print(f"[info]Calinski-Harabasz Score: {best_metrics['calinski']:.1f}", style="primary")
    console.print(f"[info]Davies-Bouldin Score: {best_metrics['davies']:.3f}", style="primary")
    
    
    return best_centroids
    
