import numpy as np
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler
from dask_ml.cluster import KMeans
from rich.console import Console
from rich.progress import track
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt 


class ABC:
    def __init__(self, n_bees, n_iterations, n_clusters, data):
        self.console = Console()
        self.n_bees = n_bees
        self.n_iterations = n_iterations
        self.n_clusters = n_clusters
        self.data = data
        self.best_solution = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        return [
            KMeans(n_clusters=self.n_clusters, init='random').fit(self.data).cluster_centers_
            for _ in range(self.n_bees)
        ]

    def evaluate_fitness(self, centroids):
        kmeans = KMeans(n_clusters=self.n_clusters, init=centroids, n_init=1)
        kmeans.fit(self.data)
        return kmeans.inertia_

    def optimize(self):
        population = self.initialize_population()
        self.console.print("[bold green]Starting optimization process...[/]")

        with ThreadPoolExecutor() as executor:
            for iteration in track(range(self.n_iterations), description="Optimizing"):
                futures = []
                for i in range(self.n_bees):
                    futures.append(executor.submit(self.process_bee, population, i))
                for future in futures:
                    future.result()
        return self.best_solution

    def process_bee(self, population, i):
        current_fitness = self.evaluate_fitness(population[i])
        new_solution = self.explore(population[i])
        new_fitness = self.evaluate_fitness(new_solution)
        if new_fitness < current_fitness:
            population[i] = new_solution
            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_solution
                self.console.print(f"[blue]New best fitness: {self.best_fitness:.2f}[/]")

    def explore(self, solution):
        perturbation = np.random.normal(0, 0.1, solution.shape)
        return solution + perturbation

    def write_results(self, filename):
        with open(filename, 'w') as f:
            for centroid in self.best_solution:
                f.write(','.join(map(str, centroid)) + '\n')



# 读取数据（假设数据存储在CSV文件中）
ddf = dd.read_csv("data/raw/iot23_combined.csv", blocksize='64MB')



# 过滤出 DDoS 数据
ddf_ddos = ddf[ddf['Label'] == 'DDoS']

# 定义需要的特征列
numeric_columns = [
    'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes', 'orig_pkts',
    'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'proto_icmp', 'proto_tcp',
    'proto_udp', 'conn_state_OTH', 'conn_state_REJ', 'conn_state_RSTO', 
    'conn_state_RSTOS0', 'conn_state_RSTR', 'conn_state_RSTRH', 'conn_state_S0',
    'conn_state_S1', 'conn_state_S2', 'conn_state_S3', 'conn_state_SF', 
    'conn_state_SH', 'conn_state_SHR'
]

# 选择并确保只包含数值列
ddf_features = ddf_ddos[numeric_columns].select_dtypes(include=[np.number]).dropna().compute()


# 使用 DaskML 对数据进行标准化
scaler = StandardScaler()
ddf_features_scaled = scaler.fit_transform(ddf_features)

# 初始化 ABC 类
n_bees = 20              # 蜜蜂的数量
n_iterations = 10       # 迭代次数
n_clusters = 5           # 聚类的数量

abc = ABC(n_bees, n_iterations, n_clusters, ddf_features_scaled)

# 进行优化
best_solution = abc.optimize()

# 输出最佳聚类中心
print(f"Best clustering centers found: {best_solution}")

# 将最佳聚类结果保存到文件
abc.write_results("best_kmeans_centroids.csv")

# 可视化部分

# 假设数据集是二维的，选择前两列特征进行可视化
# 如果数据集维度更高，可以使用 PCA 或 TSNE 等降维方法将数据降至二维进行可视化
x = ddf_features_scaled.iloc[:, 0]  # 第一列特征
y = ddf_features_scaled.iloc[:, 1]  # 第二列特征

# 使用 KMeans 对数据进行聚类（采用优化后的聚类中心）
kmeans = KMeans(n_clusters=n_clusters, init=best_solution, n_init=1)
kmeans.fit(ddf_features_scaled)

# 获取每个点的标签
labels = kmeans.predict(ddf_features_scaled)

# 获取每个数据点的标签
labels = kmeans.labels_

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 可視化部分
plt.figure(figsize=(12, 8))
plt.scatter(x, y, c=labels, cmap='viridis', alpha=0.5)
plt.scatter(best_solution[:, 0], best_solution[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# 如果要保存圖片
plt.savefig('kmeans_clustering_results.png')
plt.close()