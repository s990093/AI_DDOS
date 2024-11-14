import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import track
from rich.table import Table
import traceback

console = Console()

class ArtificialBeeColony:
    def __init__(self, n_bees, n_clusters, max_iterations):
        self.n_bees = n_bees
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.limit = n_bees * n_clusters
        
    def initialize_solutions(self, data):
        solutions = []
        for _ in range(self.n_bees):
            # 隨機選擇初始群心
            indices = np.random.choice(len(data), self.n_clusters, replace=False)
            solutions.append(data[indices].copy())
        return solutions
    
    def calculate_fitness(self, centers, data):
        kmeans = KMeans(n_clusters=self.n_clusters, init=centers, n_init=1)
        kmeans.fit(data)
        return silhouette_score(data, kmeans.labels_)
    
    def optimize(self, data):
        solutions = self.initialize_solutions(data)
        fitness_values = [self.calculate_fitness(sol, data) for sol in solutions]
        best_solution = solutions[np.argmax(fitness_values)]
        
        
        for _ in track(range(self.max_iterations), description="Clustering in progress..."):
            # 僱用蜂階段
            for i in range(self.n_bees):
                new_solution = solutions[i].copy()
                # 產生新的候選解
                param_to_mod = np.random.randint(0, self.n_clusters)
                new_solution[param_to_mod] = data[np.random.randint(0, len(data))]
                
                new_fitness = self.calculate_fitness(new_solution, data)
                if new_fitness > fitness_values[i]:
                    solutions[i] = new_solution
                    fitness_values[i] = new_fitness
            
            # 更新全局最佳解
            best_idx = np.argmax(fitness_values)
            if fitness_values[best_idx] > self.calculate_fitness(best_solution, data):
                best_solution = solutions[best_idx].copy()
                
        return best_solution

class ClusteringAnalysis:
    def __init__(self, n_clusters=3, n_bees=10, max_iterations=50):
        self.n_clusters = n_clusters
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """資料預處理：清洗與標準化"""
        console.print("[bold blue]開始資料預處理...[/bold blue]")
        
        # 處理遺漏值
        df = df.dropna()
        
        # 處理異常值 (使用 IQR 方法)
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # 標準化
        scaled_data = self.scaler.fit_transform(df)
        
        return scaled_data
    
    def reduce_dimensions(self, data, method='pca', n_components=2):
        """降維處理"""
        console.print(f"[bold blue]使用 {method.upper()} 進行降維...[/bold blue]")
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
        else:  # t-SNE
            reducer = TSNE(n_components=n_components, random_state=42)
            
        reduced_data = reducer.fit_transform(data)
        return reduced_data
    
    def cluster_data(self, data):
        """結合 ABC 和 K-means 進行分群"""
        console.print("[bold blue]開始分群分析...[/bold blue]")
        
        # 使用 ABC 優化初始群心
        abc = ArtificialBeeColony(
            n_bees=self.n_bees,
            n_clusters=self.n_clusters,
            max_iterations=self.max_iterations
        )
        optimized_centers = abc.optimize(data)
        
        # 使用優化後的群心進行 K-means 分群
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=optimized_centers,
            n_init=1
        )
        labels = kmeans.fit_predict(data)
        
        return labels, kmeans.cluster_centers_
    
    def evaluate_clustering(self, data, labels):
        """評估分群結果"""
        metrics = {
            "Silhouette Score": silhouette_score(data, labels),
            "Calinski-Harabasz Score": calinski_harabasz_score(data, labels),
            "Davies-Bouldin Score": davies_bouldin_score(data, labels)
        }
        
        # 使用 Rich 顯示評估結果
        table = Table(title="分群評估結果")
        table.add_column("指標", style="cyan")
        table.add_column("數值", style="green")
        
        for metric, value in metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        
        return metrics
    
    def visualize_clusters(self, data, labels, centers=None):
        """視覺化分群結果"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        
        if centers is not None:
            plt.scatter(
                centers[:, 0],
                centers[:, 1],
                c='red',
                marker='x',
                s=200,
                linewidths=3,
                label='Centroids'
            )
            
        plt.title('分群結果視覺化')
        plt.colorbar(scatter)
        plt.legend()
        plt.show()

def main():
    # 載入資料
    console.print("[bold green]開始載入資料...[/bold green]")
    df = pd.read_csv("data/raw/iot23_combined.csv")
    
    # 選擇數值型特徵
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    
    # 初始化分析器
    analyzer = ClusteringAnalysis(n_clusters=3, n_bees=10, max_iterations=50)
    
    # 資料預處理
    processed_data = analyzer.preprocess_data(df_numeric)
    
    # 降維
    reduced_data = analyzer.reduce_dimensions(processed_data, method='pca')
    
    # 分群
    labels, centers = analyzer.cluster_data(reduced_data)
    
    # 評估
    metrics = analyzer.evaluate_clustering(reduced_data, labels)
    
    # 視覺化
    analyzer.visualize_clusters(reduced_data, labels, centers)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(traceback.format_exc())
    finally:
        console.print("[bold green]分析完成！[/bold green]") 