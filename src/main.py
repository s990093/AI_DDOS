import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pyswarms as ps
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from numba import jit
from multiprocessing import Pool
from functools import partial
from sklearn.ensemble import RandomForestClassifier
import umap  # Add this import at the top

@jit(nopython=True)
def calculate_distances_and_labels(X, centroids):
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]
    distances = np.zeros((n_samples, n_clusters))
    
    for i in range(n_samples):
        for j in range(n_clusters):
            distances[i, j] = np.sum((X[i] - centroids[j]) ** 2)
    
    return np.argmin(distances, axis=1)

def objective_func_single(position, X, n_clusters, n_features):
    centroids = position.reshape(n_clusters, n_features)
    labels = calculate_distances_and_labels(X, centroids)
    
    try:
        score = -silhouette_score(X, labels)
    except:
        score = 0
    
    return score

def get_best_centroids(X, n_clusters, n_particles=30, max_iter=100, n_jobs=4):
    n_features = X.shape[1]
    
    def parallel_objective_func(positions):
        with Pool(n_jobs) as pool:
            scores = pool.map(
                partial(objective_func_single, X=X, n_clusters=n_clusters, n_features=n_features),
                positions
            )
        return np.array(scores)
    
    # 設定 PSO 的邊界
    min_bound = np.min(X, axis=0)
    max_bound = np.max(X, axis=0)
    bounds = (
        np.tile(min_bound, n_clusters),
        np.tile(max_bound, n_clusters)
    )
    
    # 初始化優化器
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=n_clusters * n_features,
        options={
            'c1': 0.7,
            'c2': 0.5,
            'w': 0.9,
            'k': 5,
            'p': 2,
            'ftol': 1e-5,
            'w_decay': 0.95
        },
        bounds=bounds
    )
    
    # 運行優化
    best_cost, best_pos = optimizer.optimize(
        parallel_objective_func,
        iters=max_iter,
        verbose=True
    )
    
    best_centroids = best_pos.reshape(n_clusters, n_features)
    print(f"\nPSO optimization completed with best score: {-best_cost:.4f}")
    return best_centroids

def load_data(base_path, files):
    """
    載入並預處理數據
    """
    # 讀取並合併所有資料
    dataframes = []
    for file in files.values():
        df = pd.read_parquet(base_path + file)
        dataframes.append(df)

    # 合併所有資料框
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def perform_bagging_sampling(df, sample_fraction=0.3, n_bags=1, random_state=42):
    """
    使用分層抽樣的 Bagging 方法
    """
    # 使用分層抽樣
    _, sampled_df = train_test_split(
        df,
        test_size=1-sample_fraction,
        stratify=df['Label'],
        random_state=random_state
    )
    
    final_df = sampled_df
    
    # 如果需要多個 bags，則合併它們
    for i in range(1, n_bags):
        _, bag_df = train_test_split(
            df,
            test_size=1-sample_fraction,
            stratify=df['Label'],
            random_state=random_state+i
        )
        final_df = pd.concat([final_df, bag_df])
    
    # 移除重複的行
    final_df = final_df.drop_duplicates()
    
    print("\nStratified Bagging Sampling Results:")
    print(f"Original dataset size: {len(df)}")
    print(f"Sampled dataset size: {len(final_df)}")
    print(f"Sampling ratio: {len(final_df)/len(df)*100:.2f}%")
    
    return final_df

def preprocess_data(combined_df, sample_fraction=0.3, n_bags=4):
    """
    數據預處理：分層Bagging抽樣、標準化和自適應異常值檢測
    """
    # 先進行分層 Bagging 抽樣
    combined_df = perform_bagging_sampling(combined_df, 
                                         sample_fraction=sample_fraction, 
                                         n_bags=n_bags, 
                                         random_state=42)
    
    X = combined_df.drop('Label', axis=1)
    print_initial_stats(combined_df)

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 基於 IQR 計算自應 contamination
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    contamination = np.mean(IQR) / (np.max(X_scaled) - np.min(X_scaled))
    contamination = min(max(contamination, 0.01), 0.1)  # 限制在 1%-10% 之間

    # 異常值檢測
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    outlier_labels = iso_forest.fit_predict(X_scaled)
    
    # 先將異常值標記添加��� DataFrame
    combined_df['is_outlier'] = (outlier_labels == -1).astype(int)
    
    # 然後再過濾資料
    mask = outlier_labels == 1
    X_filtered = X_scaled[mask]
    combined_df_filtered = combined_df[mask].copy()

    print_preprocessing_stats(X, X_filtered)
    
    # 視覺化異常值分布
    visualize_outliers(X_scaled, outlier_labels)
    
    return X_filtered, combined_df_filtered

def visualize_outliers(X_scaled, outlier_labels):
    """
    使用 PCA 視覺化異常值分布
    """
    plt.figure(figsize=(10, 6))
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.scatter(X_pca[outlier_labels == 1, 0], 
               X_pca[outlier_labels == 1, 1], 
               c='blue', 
               label='Normal',
               alpha=0.5)
    plt.scatter(X_pca[outlier_labels == -1, 0], 
               X_pca[outlier_labels == -1, 1], 
               c='red', 
               label='Outlier',
               alpha=0.7)
    
    plt.title("PCA Visualization of Outliers")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.legend()
    plt.show()

def perform_pca_analysis(X_scaled, target_variance=0.8):
    """
    執行 PCA 分析並返回降維後的數據，保留指定比例的變異性
    
    Args:
        X_scaled: 標準化後的數據
        target_variance: 目標累積變異性比例 (預設 0.8，即 80%)
    """
    # 先用足夠多的組件進行 PCA
    pca = PCA(n_components=min(X_scaled.shape[1], X_scaled.shape[0]))
    pca.fit(X_scaled)
    
    # 計算需要多少組件才能達到目標變異性
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance_ratio >= target_variance) + 1
    
    # 使用確定的組件數重新進行 PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 打印分析結果
    print("\nPCA Analysis Results:")
    print(f"Original dimensions: {X_scaled.shape[1]}")
    print(f"Reduced dimensions: {n_components}")
    print(f"Individual explained variance ratios:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  Component {i+1}: {ratio:.4f}")
    print(f"Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 如果維度超過3，額外進行降至3維用於視覺化
    if n_components > 3:
        print("\nNote: For visualization purposes, further reducing to 3 dimensions")
        pca_viz = PCA(n_components=3)
        X_viz = pca_viz.fit_transform(X_scaled)
        print(f"Visualization explained variance ratio: {np.sum(pca_viz.explained_variance_ratio_):.4f}")
        return X_pca, X_viz
    
    return X_pca, X_pca

def print_preprocessing_stats(X, X_filtered):
    """
    打印預處理統計信息
    """
    print("\nOutlier Detection Results:")
    print(f"Original dataset size: {len(X)}")
    print(f"Dataset size after outlier removal: {len(X_filtered)}")
    print(f"Removed {len(X) - len(X_filtered)} outliers "
          f"({((len(X) - len(X_filtered)) / len(X)) * 100:.2f}%)")

def print_initial_stats(filtered_df):
    """
    打印初始數據統計信息
    """
    print(f"Filtered Dataset Size: {filtered_df.shape}")
    print("\nFiltered Label Distribution:")
    filtered_label_distribution = filtered_df['Label'].value_counts()

    for label, count in filtered_label_distribution.items():
        percentage = (count / len(filtered_df)) * 100
        print(f"{label}: {count} ({percentage:.2f}%)")

def visualize_results(X_scaled, clusters, combined_df_filtered, combined_df):
    plt.figure(figsize=(20, 6))

    # 1. 3D PCA 視覺化
    ax = plt.subplot(131, projection='3d')
    scatter = ax.scatter(X_scaled[:, 0], 
                        X_scaled[:, 1], 
                        X_scaled[:, 2],
                        c=clusters,
                        cmap='viridis',
                        alpha=0.6)
    ax.set_title('3D PCA Clustering')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter)

    # 2. 群集分布柱狀圖
    ax2 = plt.subplot(132)
    cluster_counts = combined_df_filtered['Cluster'].value_counts().sort_index()
    bars = ax2.bar(cluster_counts.index, cluster_counts.values)
    ax2.set_title('Cluster Distribution')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')
    
    for bar in bars:
        height = bar.get_height()
        percentage = (height / len(combined_df_filtered)) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({percentage:.2f}%)',
                ha='center', va='bottom')

    # 3. 每個分群的攻擊類型組成
    ax3 = plt.subplot(133)
    cluster_label_dist = pd.crosstab(
        combined_df_filtered['Cluster'], 
        combined_df_filtered['Label'], 
        normalize='index'
    ) * 100
    
    cluster_label_dist.plot(kind='bar', stacked=True, ax=ax3)
    ax3.set_title('Attack Type Distribution in Each Cluster')
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Percentage')
    ax3.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def print_cluster_metrics(X_scaled, clusters):
    print("\nClustering Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette_score(X_scaled, clusters):.4f}")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, clusters):.4f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, clusters):.4f}")

def print_cluster_distribution(combined_df_filtered):
    print("\nCluster Distribution:")
    cluster_distribution = combined_df_filtered['Cluster'].value_counts()
    for cluster, count in cluster_distribution.items():
        percentage = (count / len(combined_df_filtered)) * 100
        print(f"Cluster {cluster}: {count} ({percentage:.2f}%)")

def print_attack_distribution(combined_df_filtered, desired_labels):
    print("\nCluster Distribution for Each Attack Type:")
    for label in desired_labels:
        print(f"\n{label} Attack Clusters Distribution:")
        label_clusters = combined_df_filtered[combined_df_filtered['Label'] == label]['Cluster'].value_counts()
        total_label = label_clusters.sum()
        for cluster, count in label_clusters.items():
            percentage = (count / total_label) * 100
            print(f"Cluster {cluster}: {count} ({percentage:.2f}%)")

def perform_clustering(X_scaled, n_clusters):
    """
    使用自適應學習率的 K-medoids 聚類
    """
    best_score = float('-inf')
    best_clusters = None
    
    # 嘗試不同的初��化方法
    init_methods = ['k-medoids++', 'random']
    
    for init in init_methods:
        kmedoids = KMedoids(
            n_clusters=n_clusters,
            metric='euclidean',
            method='alternate',
            init=init,
            max_iter=300,
            random_state=42
        )
        clusters = kmedoids.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, clusters)
        
        if score > best_score:
            best_score = score
            best_clusters = clusters
    
    return best_clusters

def validate_clustering(X_scaled, clusters):
    """
    增強的聚類結果驗證
    """
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    from sklearn.metrics import v_measure_score
    
    # 計算穩定性指標
    n_splits = 5
    stability_scores = []
    
    for i in range(n_splits):
        # 隨機分割數據
        mask = np.random.rand(len(X_scaled)) < 0.8
        clusters1 = perform_clustering(X_scaled[mask], len(np.unique(clusters)))
        clusters2 = perform_clustering(X_scaled[mask], len(np.unique(clusters)))
        
        # 計算穩定性分數
        stability_scores.append(adjusted_rand_score(clusters1, clusters2))
    
    print("\nEnhanced Clustering Validation:")
    print(f"Clustering Stability (mean): {np.mean(stability_scores):.4f}")
    print(f"Clustering Stability (std): {np.std(stability_scores):.4f}")

def find_optimal_clusters(X_scaled, max_clusters=10):
    """
    找到最佳的聚類數量
    """
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    
    for k in range(2, max_clusters + 1):
        clusters = perform_clustering(X_scaled, k)
        
        silhouette_scores.append(silhouette_score(X_scaled, clusters))
        ch_scores.append(calinski_harabasz_score(X_scaled, clusters))
        db_scores.append(davies_bouldin_score(X_scaled, clusters))
    
    # 繪製評估指標圖
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    
    plt.subplot(132)
    plt.plot(range(2, max_clusters + 1), ch_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score vs. Number of Clusters')
    
    plt.subplot(133)
    plt.plot(range(2, max_clusters + 1), db_scores, 'go-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Score')
    plt.title('Davies-Bouldin Score vs. Number of Clusters')
    
    plt.tight_layout()
    plt.show()
    
    # 找到最佳聚類數
    best_k = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
    print(f"\nBest number of clusters based on Silhouette Score: {best_k}")
    
    return best_k

def analyze_feature_importance(X_scaled, clusters, feature_names):
    """
    分析特徵對聚類結果的重要性
    """
    # 使用隨機森林來評估特重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, clusters)
    
    # 獲取特徵重要性
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 視覺化特徵重要性
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importances.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features for Clustering')
    plt.show()
    
    return importances

def perform_umap_reduction(X_scaled, n_neighbors=30, min_dist=0.1):
    """
    執行 UMAP 降維分析，加入噪聲以提高穩定性
    """
    # 添加小量噪聲以避免特徵值完全相同
    noise_scale = 1e-4
    X_noisy = X_scaled + np.random.normal(0, noise_scale, X_scaled.shape)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,     
        min_dist=min_dist,
        n_components=3,
        random_state=None,     
        metric='euclidean',
        n_epochs=200,               
        learning_rate=1.0,          
        init='spectral',            
        verbose=True,
        n_jobs=-1                
    )
    
    X_umap = reducer.fit_transform(X_noisy)
    
    print("\nUMAP Reduction Analysis:")
    print(f"Original dimensions: {X_scaled.shape[1]}")
    print(f"Reduced dimensions: {X_umap.shape[1]}")
    
    return X_umap

# if __name__ == "__main__":
#     # 檔案路徑設定
#     base_path = "/Users/hungwei/.cache/kagglehub/datasets/dhoogla/cicddos2019/versions/3/"
    
#     files = {
#         'DNS': 'DNS-testing.parquet',
#         'MSSQL': 'MSSQL-testing.parquet',
#         'LDAP': 'LDAP-testing.parquet'
#     }
    
#     desired_labels = ['DrDoS_DNS', 'Benign', 'DrDoS_MSSQL', 'DrDoS_LDAP']

#     # 載入數據
#     combined_df = load_data(base_path, files)
#     print_initial_stats(combined_df)

#     # 數據預處理
#     X_scaled, combined_df_filtered = preprocess_data(combined_df)
    
#     # Change this part:
#     n_clusters = 6  # Rename variable to be more clear
#     # First perform clustering to get the cluster labels
#     clusters = perform_clustering(X_scaled, n_clusters)
    
#     # Now analyze feature importance with the actual cluster labels
#     feature_names = combined_df_filtered.drop(['Label', 'is_outlier'], axis=1).columns
#     feature_importance = analyze_feature_importance(X_scaled, clusters, feature_names)
    
#     # 選擇重要特徵（例如：前10個最重要的特徵）
#     top_features = feature_importance.head(20)['feature'].tolist()
#     print("\nSelected Important Features:")
#     print(top_features)
        
#     # 標準化重要特徵
#     X_important = X_scaled[:, [list(feature_names).index(feat) for feat in top_features]]
#     scaler = StandardScaler()
#     X_important_scaled = scaler.fit_transform(X_important)
    
#     # 使用 PCA 進行降維，保留80%變異性
#     X_reduced, X_viz = perform_pca_analysis(X_important_scaled, target_variance=0.8)
    
#     # 找到最佳聚類數量 (使用完整的降維結果)
#     n_clusters = find_optimal_clusters(X_reduced, n_clusters)
    
#     # 使用最佳聚類數進行聚類
#     clusters = perform_clustering(X_reduced, n_clusters)
    
#     # 將聚類結果添加到資料框
#     combined_df_filtered.loc[:, 'Cluster'] = clusters
    
#     # 顯示各種統計資訊
#     print_cluster_distribution(combined_df_filtered)
#     print_cluster_metrics(X_important, clusters)  # 使用重要特徵評估
#     print_attack_distribution(combined_df_filtered, desired_labels)
    
#     # 視覺化結果
#     visualize_results(X_important, clusters, combined_df_filtered, combined_df)