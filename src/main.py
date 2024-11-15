from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.theme import Theme
from random import randint
from math import exp

from alog.abc_optimizer import ABCOptimizer
from load_and_preprocess_data import load_and_preprocess_data



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

def compute_metrics_parallel(X, k):
    """計算KMeans的完整評估指標"""
    # 初始化KMeans
    kmeans = KMeans(n_clusters=k, 
                    random_state=42,
                    n_init=10)  # 增加初始化次數以獲得更穩定的結果
    kmeans.fit(X)
    labels = kmeans.labels_
    
    # 計算各種評估指標
    metrics = {}
    
    # 1. Inertia (Within-cluster sum of squares)
    metrics['inertia'] = kmeans.inertia_
    
    try:
        # 2. Silhouette Score (完整計算，不採樣)
        metrics['silhouette'] = silhouette_score(X, labels)
        
        # 3. Calinski-Harabasz Index (方差比準則)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        
        # 4. Davies-Bouldin Index (集群間相似度，越小越好)
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
        
    except ValueError as e:
        console.print(f"[warning]Error calculating metrics for k={k}: {str(e)}", style="warning")
        metrics['silhouette'] = -1
        metrics['calinski_harabasz'] = -1
        metrics['davies_bouldin'] = float('inf')
    
    return metrics


def main():    
    # 數據處理
    X_scaled = load_and_preprocess_data(console)
    
    K = range(2, 13) 
    
    optimizer = ABCOptimizer(
        X=X_scaled,
        k_range=(5, 13),
        max_iter=300,
        population_size=40,
        limit=20,
        patience=5
    )   


    # 使用ABC算法尋找最佳k值和中心點
    best_k, best_centroids, best_metrics = optimizer.optimize()    
    # plot fitness history
    optimizer.plot_fitness_history()
    
    inertias = []
    silhouette_scores = []
    
    for k in K:
        metrics = compute_metrics_parallel(X_scaled, k)
        inertias.append(metrics['inertia'])
        silhouette_scores.append(metrics['silhouette'])
    
    # 創建包含兩個子圖的圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 繪製 Elbow 圖
    ax1.plot(K, inertias, 'bx-')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method For Optimal k')
    
    # 繪製 Silhouette 圖
    ax2.plot(K, silhouette_scores, 'rx-')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score For Optimal k')
    
    # 調整子圖之間的間距
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig('res/clustering_metrics.png')
    plt.close()
    
    # 使用最佳結果進行最終聚類
    distances = np.array([np.linalg.norm(X_scaled - centroid, axis=1) for centroid in best_centroids])
    final_labels = np.argmin(distances, axis=0)
    
    # 修改PCA為3維並繪製3D圖
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(best_centroids)
    
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
    
    ax.set_title(f"3D Clustering Result (k={best_k}, silhouette score={best_metrics['silhouette']:.3f})")
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    
    plt.colorbar(scatter, label='Cluster Label')
    plt.legend()
    plt.savefig('res/clustering_result_3d.png')
    plt.close()
    
    console.print(f"[info]Best k found: {best_k}", style="primary")
    console.print(f"[info]Best silhouette score: {best_metrics['silhouette']:.3f}", style="primary")

if __name__ == "__main__":
    main()
