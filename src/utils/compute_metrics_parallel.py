from main import console


from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


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