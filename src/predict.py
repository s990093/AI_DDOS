import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import click
from rich.console import Console
from rich.theme import Theme
from utils import load_and_preprocess_data  # 假設這個函數可以重用
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA
from rich.table import Table
import os  # 在文件頂部添加這個導入

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

def predict():
    # 載入模型
    try:
        model_data = np.load('models/pso_clustering_model.npy', allow_pickle=True).item()
        centroids = model_data['centroids']
        pca_model = model_data['pca_model']
        n_clusters = model_data['n_clusters']
        best_metrics = model_data['metrics']
        
        console.print("[info]Model loaded successfully", style="primary")
        
        # 打印 best_metrics
        console.print("\n[bold cyan]Best Metrics:[/bold cyan]")
        for metric_name, value in best_metrics.items():
            console.print(f"{metric_name}: {value:.4f}")
        console.print("\n")
        
    except FileNotFoundError:
        console.print("[error]Model file not found. Please train the model first.", style="error")
        return
    
    # 載入並預處理新數據
    X_scaled = load_and_preprocess_data(console)[0]
    
    # 如果模型使用了PCA，對新數據也進行降維
    if pca_model is not None:
        X_transformed = pca_model.transform(X_scaled)
        console.print("[info]Applied PCA transformation", style="primary")
    else:
        X_transformed = X_scaled
    
    # 計算每個數據點到各個中心點的距離
    distances = np.zeros((X_transformed.shape[0], n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.sum((X_transformed - centroids[i]) ** 2, axis=1)
    
    # 分配聚類標籤
    predictions = np.argmin(distances, axis=1)
    
    # 使用PCA降維到3個維度
    pca_visualizer = PCA(n_components=8)
    plot_data = pca_visualizer.fit_transform(X_transformed)
    
    # 繪製3D散點圖
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2],
                        c=predictions, cmap='viridis',
                        alpha=0.6)
    
    # 將聚類中心也轉換到相同的P
    centers = pca_visualizer.transform(centroids)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
              c='red', marker='x', s=200,
              label='Centroids')
        
    ax.set_title('Clustering Results (3D)')
    ax.set_xlabel('Feature 1' if pca_model is None else 'First Principal Component')
    ax.set_ylabel('Feature 2' if pca_model is None else 'Second Principal Component')
    ax.set_zlabel('Feature 3' if pca_model is None else 'Third Principal Component')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    
    # 保存圖片
    plt.savefig('res/clustering_results.png')
    console.print("[info]Clustering visualization saved as res/clustering_results.png", style="primary")
    
    # 保存預測結果
    try:
        # 確保 res 目錄存在
        os.makedirs('res', exist_ok=True)
        
        output_file = f'res/predictions.csv'
        np.savetxt(output_file, predictions, delimiter=',', fmt='%d')
        console.print(f"[info]Predictions saved to {output_file}", style="primary")
        
        # 載入原始標籤
        _, y_full, y_ddos = load_and_preprocess_data(console)
        
        # 獲取唯一的聚類標籤
        unique = np.unique(predictions)
        
        # 計算需要的子圖行數和列數
        n_clusters = len(unique)
        n_cols = 3  # 每行顯示3個��圖
        n_rows = (n_clusters + n_cols - 1) // n_cols  # 向上取整

        # 創建子圖
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('Distribution of Attack Types in Each Cluster', fontsize=16, y=1.02)
        
        # 如果只有一行，需要將axes轉換為2D數組
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # 獲取所有標籤
        labels = np.unique(y_full)
        
        # 為每個cluster創建柱狀圖
        for idx, cluster in enumerate(unique):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            cluster_mask = predictions == cluster
            cluster_samples = np.sum(cluster_mask)
            
            # 計算每個標籤的數量和百分比
            counts = []
            percentages = []
            x_labels = []
            
            # 先添加所有攻擊類型
            for label in labels:
                count = np.sum((y_full[cluster_mask] == label))
                percentage = (count/cluster_samples)*100
                counts.append(count)
                percentages.append(percentage)
                x_labels.append(label)
            
            # 添加DDoS統計
            ddos_count = np.sum(y_ddos[cluster_mask])
            ddos_percentage = (ddos_count/cluster_samples)*100
            counts.append(ddos_count)
            percentages.append(ddos_percentage)
            x_labels.append('DDoS (Total)')
            
            # 繪製柱狀圖
            colors = ['blue'] * (len(x_labels) - 1) + ['red']  # 最後一個（DDoS）設為紅色
            bars = ax.bar(range(len(x_labels)), counts, color=colors)
            
            # 在柱子上添加標籤
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{counts[i]}\n({percentages[i]:.1f}%)',
                       ha='center', va='bottom', fontsize=5)
            
            # 設置標題和標籤
            ax.set_title(f'Cluster {cluster} (n={cluster_samples})')
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.set_ylabel('Count')
            
            # 隱藏空白子圖
            for idx in range(n_clusters, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                axes[row, col].set_visible(False)
            
            # 調整布局
            plt.tight_layout()
            
            # 保存圖片
            plt.savefig('res/cluster_type_distribution.png', bbox_inches='tight', dpi=300)
            console.print("[info]Cluster type distribution visualization saved as res/cluster_type_distribution.png", style="primary")

            # 顯示總體統計
            console.print("\n[bold cyan]Cluster Statistics:[/bold cyan]")
            for cluster in unique:
                cluster_mask = predictions == cluster
                cluster_samples = np.sum(cluster_mask)
                
                # 為每個 cluster 創建一個表格
                table = Table(title=f"Cluster {cluster} Statistics")
                table.add_column("Attack Type", style="cyan")
                table.add_column("Count", justify="right", style="green")
                table.add_column("Percentage", justify="right", style="yellow")
                
                # 添加完整標籤分布到表格
                for label in labels:
                    count = np.sum((y_full[cluster_mask] == label))
                    percentage = (count/cluster_samples)*100
                    table.add_row(
                        str(label),
                        str(count),
                        f"{percentage:.2f}%"
                    )
                
                # 添��總計行
                table.add_row(
                    "Total",
                    str(cluster_samples),
                    "100.00%",
                    style="bold"
                )
                
                # 添加 DDoS 統計
                ddos_count = np.sum(y_ddos[cluster_mask])
                ddos_percentage = (ddos_count/cluster_samples)*100
                table.add_section()
                table.add_row(
                    "DDoS (Total)",
                    str(ddos_count),
                    f"{ddos_percentage:.2f}%",
                    style="bold red"
                )
                
                # 顯示表格
                console.print(table)
                console.print("\n")

            # 顯示總體 DDoS 統計
            total_ddos = np.sum(y_ddos)
            total_samples = len(y_ddos)
            console.print("\n[bold cyan]Overall DDoS Statistics:[/bold cyan]")
            console.print(f"Total DDoS samples: {total_ddos}")
            console.print(f"Total samples: {total_samples}")
            console.print(f"Overall DDoS percentage: {(total_ddos/total_samples)*100:.2f}%\n")
        
    except Exception as e:
        console.print(f"[error]Error saving predictions: {str(e)}", style="error")

if __name__ == "__main__":
    predict() 