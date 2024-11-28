from sklearn.decomposition import PCA
from pyswarms.single.global_best import GlobalBestPSO
from config.CONFIG import CONFIG
from objective import objective_function
from utils import load_and_preprocess_data
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.theme import Theme
import click
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import multiprocessing
from sklearn.cluster import KMeans
import os
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

# def process_particle(args):
#     particle, shm_name, shape, dtype, clusters = args
#     try:
#         # 從共享記憶體獲取數據
#         existing_shm = shared_memory.SharedMemory(name=shm_name)
#         X_final = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        
#         # 將粒子重塑為中心點
#         centroids = particle.reshape(clusters, -1)
        
#         # 使用這些中心點初始化 k-means
#         kmeans = KMeans(
#             n_clusters=clusters,
#             init=centroids,
#             n_init=1,  # 只運行一次，因為我們提供了初始中心點
#             max_iter=50  # 減少迭代次數，因為PSO會多次嘗試
#         )
        
#         # 執行聚類
#         labels = kmeans.fit_predict(X_final)
        
#         # 計算評估指標
#         if len(np.unique(labels)) < clusters:
#             return float('inf')
            
#         silhouette = silhouette_score(X_final, labels)
#         calinski = calinski_harabasz_score(X_final, labels)
#         davies = davies_bouldin_score(X_final, labels)
        
#         # 組合得分（越小越好）
#         combined_score = (
#             -silhouette * 0.4 +  # 負號因為silhouette越大越好
#             -calinski/10000 * 0.3 +  # 縮放calinski score
#             davies * 0.3
#         )
        
#         return combined_score
#     except Exception as e:
#         print(f"Error in process_particle: {e}")
#         return float('inf')
#     finally:
#         # 確保正確關閉共享記憶體
#         try:
#             existing_shm.close()
#         except Exception:
#             pass

# def objective_function(positions, shm_name, shape, dtype, clusters, n_processes):
#     # 準備參數
#     args = [(pos, shm_name, shape, dtype, clusters) for pos in positions]
    
#     # 使用上下文管理器確保進程池正確關閉
#     with ProcessPoolExecutor(max_workers=n_processes) as executor:
#         try:
#             # 使用 list 強制執行所有任務
#             scores = list(executor.map(process_particle, args, chunksize=max(1, len(positions)//n_processes)))
#         except Exception as e:
#             print(f"Error in parallel processing: {e}")
#             return np.array([float('inf')] * len(positions))
    
#     return np.array(scores)

@click.command()
@click.option('--cpu', '-c', default=-1, help='Number of CPU cores to use. -1 for all cores.')
@click.option('--iterations', '-i', default=300, help='Maximum number of iterations.')
@click.option('--population', '-p', default=40, help='Population size.')
@click.option('--min-clusters', '-min', default=2, help='Minimum number of clusters to try.')
@click.option('--max-clusters', '-max', default=10, help='Maximum number of clusters to try.')
@click.option('--pca-components', '-d', default=0, help='Number of PCA components. 0 for no reduction.')
@click.option('--k', '-k', default=None, type=int, help='Specify a single k value for clustering. Overrides min/max clusters.')
@click.option('--sample-size', '-s', default=2000, help='Number of samples to use for clustering.')
def main(cpu, iterations, population, min_clusters, max_clusters, pca_components, k, sample_size):    
    # 數據處理
    # tuple 需要修改
    X_scaled_tuple = load_and_preprocess_data(console)
    X_scaled = X_scaled_tuple[0]

    if pca_components > 0:
        # 先測試不同維度的解釋方差
        pca_test = PCA(
            n_components=len(CONFIG['numerical_columns']),
            svd_solver='randomized',
            random_state=42
        )
        
        # 採樣用於顯示解釋方差
        sample_size = min(sample_size, len(X_scaled))
        indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[indices]
        
        pca_test.fit(X_sample)
        
        # 顯示累積解釋方差
        cumulative_variance = np.cumsum(pca_test.explained_variance_ratio_) * 100
        
        # 動態生成要顯示的組件數量
        total_components = len(CONFIG['numerical_columns'])
        step = max(1, total_components // 6)
        components_to_show = list(range(step, total_components + 1, step))
        if total_components not in components_to_show:
            components_to_show.append(total_components)
            
        console.print("\n[bold cyan]Cumulative Explained Variance:[/bold cyan]")
        for i in components_to_show:
            console.print(f"{i} components: {cumulative_variance[i-1]:.2f}%")
        
        # 使用指定的組件數進行實際PCA降維
        pca = PCA(n_components=pca_components, random_state=42)
        X_reduced = pca.fit_transform(X_scaled)
        console.print(f"[info]Reduced dimensions to {pca_components} components")
        
        # 在降維後取樣1000筆資料
        sample_size = min(sample_size, len(X_reduced))
        indices = np.random.choice(len(X_reduced), sample_size, replace=False)
        X_final = X_reduced[indices]
        console.print(f"[info]Sampled {sample_size} data points for clustering")
    else:
        console.print("[info]Using original dimensions (no PCA reduction)")
        # 直接在原始數據中取樣
        sample_size = min(sample_size, len(X_scaled))
        indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_final = X_scaled[indices]
        console.print(f"[info]Sampled {sample_size} data points for clustering")

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
    
    # 修改聚類循環邏輯
    if k is not None:
        # 如果指定了k值，只執行一次
        cluster_range = [k]
    else:
        # 否則執行原來的範圍
        cluster_range = range(min_clusters, max_clusters + 1)
    
    # 為每個k值進行聚類
    for k in cluster_range:
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
        overall_score = (
            -final_metrics['silhouette'] * 0.4 +
            -final_metrics['calinski']/10000 * 0.3 +
            final_metrics['davies'] * 0.3
        )
        
        console.print(f"[info]k={k} overall score: {overall_score:.3f}")
        
        # 更新最佳結果
        if overall_score < best_overall_score:
            best_overall_score = overall_score
            best_k = k
            best_metrics = final_metrics
            best_centroids = best_centroids

    # 使用最佳的k值保存結果
    console.print(f"\n[bold green]Best k found: {best_k}[/bold green]")
    console.print("\n[bold cyan]Best Clustering Metrics:[/bold cyan]")
    console.print(f"[info]Silhouette Score: {best_metrics['silhouette']:.3f}", style="primary")
    console.print(f"[info]Calinski-Harabasz Score: {best_metrics['calinski']:.1f}", style="primary")
    console.print(f"[info]Davies-Bouldin Score: {best_metrics['davies']:.3f}", style="primary")
    
    # 確保 models 目錄存在
    os.makedirs('models', exist_ok=True)
    
    # 保存模型
    model_data = {
        'centroids': best_centroids,
        'pca_model': pca if pca_components > 0 else None,
        'n_clusters': best_k,
        'metrics': best_metrics
    }
    
    np.save('models/pso_clustering_model.npy', model_data, allow_pickle=True)
    console.print("[info]Model saved to models/pso_clustering_model.npy", style="primary")
    
    # 修改PCA為3維並繪製3D圖
    if pca_components > 0:
        # 如果已經做過PCA，直接用採樣數據重新做3D PCA
        pca_3d = PCA(n_components=3)
        X_pca = pca_3d.fit_transform(X_scaled[indices])
        centroids_original = best_centroids.dot(pca.components_) + pca.mean_
        centroids_pca = pca_3d.transform(centroids_original)
    else:
        # 如果沒有做過PCA，直接對採樣數據做3D PCA
        pca_3d = PCA(n_components=3)
        X_pca = pca_3d.fit_transform(X_scaled[indices])
        centroids_pca = pca_3d.transform(best_centroids)
    
    # 創建3D圖
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 繪製數據點
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                        c=final_labels, cmap='viridis', alpha=0.6)
    
    # 繪製中心點
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2],
              c='red', marker='*', s=200, label='Centroids',
              edgecolors='black', linewidth=1)
    
    ax.set_title(f"3D Clustering Result (k={best_k})\n" +
                f"Silhouette: {best_metrics['silhouette']:.3f}, " +
                f"Calinski-Harabasz: {best_metrics['calinski']:.1f}, " +
                f"Davies-Bouldin: {best_metrics['davies']:.3f}")
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    
    plt.colorbar(scatter, label='Cluster Label')
    plt.legend()
    plt.savefig('res/clustering_result_3d.png')
    plt.close()

    # 清理共享記憶體
    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass

if __name__ == "__main__":
    main()
