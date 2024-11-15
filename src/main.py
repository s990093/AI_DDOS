from sklearn.decomposition import PCA
from core.optimizer import ABCOptimizer
from utils import load_and_preprocess_data
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.theme import Theme
from random import randint
from math import exp
import pyroscope
import click



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

@click.command()
@click.option('--cpu', '-c', default=-1, help='Number of CPU cores to use. -1 for all cores.')
@click.option('--iterations', '-i', default=300, help='Maximum number of iterations.')
@click.option('--population', '-p', default=40, help='Population size.')
@click.option('--clusters', '-k', default=5, help='Number of clusters.')
def main(cpu, iterations, population, clusters):    
    # 數據處理
    X_scaled = load_and_preprocess_data(console)
    
    optimizer = ABCOptimizer(
        X=X_scaled,
        k=clusters,
        max_iter=iterations,
        population_size=population,
        limit=20,
        patience=5,
        n_processes=cpu
    )   

    # 使用ABC算法尋找最佳中心點
    k, best_centroids, best_metrics = optimizer.optimize()    
    
    optimizer.plot_fitness_history()
    
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
    
    ax.set_title(f"3D Clustering Result (k={k}, silhouette score={best_metrics['silhouette']:.3f})")
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    
    plt.colorbar(scatter, label='Cluster Label')
    plt.legend()
    plt.savefig('res/clustering_result_3d.png')
    plt.close()
    
    console.print(f"[info]Best k found: {k}", style="primary")
    console.print(f"[info]Best silhouette score: {best_metrics['silhouette']:.3f}", style="primary")

if __name__ == "__main__":
    main()