import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from dask import dataframe as dd
from sklearn.preprocessing import StandardScaler
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import track

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
        return [KMeans(n_clusters=self.n_clusters, init='random').fit(self.data).cluster_centers_ for _ in range(self.n_bees)]

    def evaluate_fitness(self, centroids):
        kmeans = KMeans(n_clusters=self.n_clusters, init=centroids, n_init=1)
        kmeans.fit(self.data)
        return kmeans.inertia_

    def optimize(self):
        population = self.initialize_population()
        self.console.print("[bold green]Starting optimization process...[/]")
        
        for iteration in track(range(self.n_iterations), description="Optimizing"):
            for i in range(self.n_bees):
                new_solution = self.explore(population[i])
                new_fitness = self.evaluate_fitness(new_solution)
                if new_fitness < self.evaluate_fitness(population[i]):
                    population[i] = new_solution
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_solution
                        self.console.print(f"[blue]New best fitness: {self.best_fitness:.2f}[/]")
        return self.best_solution

    def explore(self, solution):
        perturbation = np.random.normal(0, 0.1, solution.shape)
        return solution + perturbation

    def write_results(self, filename):
        with open(filename, 'w') as f:
            for centroid in self.best_solution:
                f.write(','.join(map(str, centroid)) + '\n')

# Load data using Dask
ddf = dd.read_csv("data/raw/iot23_combined.csv", blocksize='64MB')
ddf = ddf[(ddf['Label'] == 'Benign') | (ddf['Label'] == 'DDoS')]
df = ddf.compute()

# Preprocess data
df.dropna(inplace=True)
df.columns = df.columns.str.replace(' ', '')
cols = df.drop(columns=['Label']).columns.tolist()
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=['Label']))

# Apply ABC to optimize K-means centroids
abc = ABC(n_bees=10, n_iterations=100, n_clusters=4, data=X)
best_centroids = abc.optimize()
abc.write_results('best_centroids.txt')

# Perform clustering with optimized centroids
kmeans = KMeans(n_clusters=4, init=best_centroids, n_init=1)
kmeans.fit(X)

# Output cluster labels
df['Cluster'] = kmeans.labels_
print(df[['Label', 'Cluster']].head())

# After clustering, create a rich table to display results
console = Console()
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Label")
table.add_column("Cluster")

for label, cluster in df[['Label', 'Cluster']].head().values:
    table.add_row(str(label), str(cluster))

console.print("\n[bold]Clustering Results:[/]")
console.print(table)

# Add cluster distribution summary
console.print("\n[bold]Cluster Distribution:[/]")
cluster_dist = df.groupby(['Label', 'Cluster']).size().unstack(fill_value=0)
print(cluster_dist)