import dask.dataframe as dd
import dask_ml.cluster as dkc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dask_ml.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.theme import Theme

# Configuration
CONFIG = {
    'data_path': "data/raw/UNSW_NB15_training-set.csv",
    'blocksize': '64MB',
    'k_range': range(2, 11),
    'numerical_columns': [
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 
        'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 
        'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 
        'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 
        'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
        'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst', 
        'is_sm_ips_ports', 'label'
    ]
}

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

def load_and_preprocess_data(console):
    """加載和預處理數據"""
    console.print("[info]Loading and processing data...", style="primary")
    
    # 加載數據
    ddf = dd.read_csv(CONFIG['data_path'], blocksize=CONFIG['blocksize'])
    X = ddf[CONFIG['numerical_columns']]
    
    # 標準化
    with Progress() as progress:
        task = progress.add_task("[cyan]Scaling the data...", total=1)
        scaler = StandardScaler()
        X_scaled = X.map_partitions(lambda df: scaler.fit_transform(df))
        progress.update(task, advance=1)
        console.print("[info]Data scaling complete.", style="primary")
    
    X_scaled.compute_chunk_sizes()
    return X_scaled

def compute_clustering_metrics(X, k_range):
    """計算聚類指標"""
    def compute_inertia():
        inertia = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Computing inertia for different k...", total=len(k_range))
            for k in k_range:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X)
                inertia.append(kmeans.inertia_)
                progress.update(task, advance=1)
        return inertia

    def compute_silhouette():
        silhouette_scores = []
        with Progress() as progress:
            task = progress.add_task("[cyan]Computing silhouette score for different k...", total=len(k_range))
            for k in k_range:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X)
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
                progress.update(task, advance=1)
        return silhouette_scores

    return compute_inertia(), compute_silhouette()

def plot_metrics(k_range, inertia, silhouette_scores):
    """繪製指標圖表"""
    plt.figure(figsize=(10, 5))
    
    # Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method (Inertia)")
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')

    # Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='orange')
    plt.title("Silhouette Score")
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

def main():
    # 初始化
    console = setup_rich_console()
    
    # 數據處理
    X_scaled = load_and_preprocess_data(console)
    
    # 計算聚類指標
    inertia, silhouette_scores = compute_clustering_metrics(X_scaled, CONFIG['k_range'])
    
    # 繪製圖表
    plot_metrics(CONFIG['k_range'], inertia, silhouette_scores)
    
    # 輸出最佳k值
    best_k_silhouette = CONFIG['k_range'][np.argmax(silhouette_scores)]
    best_k_inertia = CONFIG['k_range'][np.argmin(np.diff(inertia))]
    
    console.print(f"[info]Best k based on Silhouette Score: {best_k_silhouette}", style="primary")
    console.print(f"[info]Best k based on Elbow Method: {best_k_inertia}", style="primary")

if __name__ == "__main__":
    main()
