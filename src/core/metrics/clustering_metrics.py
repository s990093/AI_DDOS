import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler

def compute_silhouette(X, labels):
    return silhouette_score(X, labels)

def compute_calinski(X, labels):
    return calinski_harabasz_score(X, labels)

def compute_davies(X, labels):
    return davies_bouldin_score(X, labels)

def calculate_metrics(X, labels):
    """Calculate clustering metrics using optimizations."""
    # Sample 20% of the data
    sample_size = int(X.shape[0] * 0.2)
    indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sampled = X[indices]
    labels_sampled = labels[indices]

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        silhouette_future = executor.submit(compute_silhouette, X_sampled, labels_sampled)
        calinski_future = executor.submit(compute_calinski, X_sampled, labels_sampled)
        davies_future = executor.submit(compute_davies, X_sampled, labels_sampled)

        return {
            'silhouette': silhouette_future.result(),
            'calinski_harabasz': calinski_future.result(),
            'davies_bouldin': davies_future.result(),
            'sample_size': X_sampled.shape[0]
        }

def is_better_solution(new_metrics, old_metrics):
    """Compare two solutions based on their metrics."""
    return (new_metrics['silhouette'] > old_metrics['silhouette'] and
            new_metrics['calinski_harabasz'] > old_metrics['calinski_harabasz'] and
            new_metrics['davies_bouldin'] < old_metrics['davies_bouldin'])

def compute_metrics_parallel(X, k):
    """Compute clustering metrics in parallel."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    return {
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels),
        'davies_bouldin': davies_bouldin_score(X, labels)
    } 