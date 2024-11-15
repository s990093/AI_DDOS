from ..metrics.clustering_metrics import calculate_metrics, is_better_solution
from .bee import Bee
import numpy as np
from .bee import Bee
from rich.progress import Progress
import matplotlib.pyplot as plt
from numba import jit, prange
from multiprocessing import Pool
import multiprocessing

@jit(nopython=True, parallel=True)
def calculate_distances(X, centroids):
    """Calculate the distance from each point to each centroid."""
    n_samples = X.shape[0]
    n_centroids = centroids.shape[0]
    n_features = X.shape[1]
    
    # Initialize output array
    distances = np.zeros((n_samples, n_centroids))
    
    # Calculate distances without broadcasting
    for i in prange(n_samples):
        for j in range(n_centroids):
            diff = 0.0
            for k in range(n_features):
                diff += (X[i, k] - centroids[j, k]) ** 2
            distances[i, j] = np.sqrt(diff)
    
    return distances

class ABCOptimizer:
    def __init__(self, X, k, max_iter=50, population_size=10, 
                 limit=20, patience=5, n_processes=None):
        """Initialize the ABC optimizer."""
        self.X = X
        self.k = k
        self.max_iter = max_iter
        self.population_size = population_size
        self.limit = limit
        self.patience = patience
        self.best_centroids = None
        self.best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        self.fitness_history = []
        self.n_processes = (multiprocessing.cpu_count() if n_processes == -1 
                          else n_processes or multiprocessing.cpu_count())
        
    def initialize_population_fixed_k(self, k):
        """Initialize a population with a fixed k value."""
        population = []
        for _ in range(self.population_size):
            centroids = self.X[np.random.choice(self.X.shape[0], k, replace=False)]
            population.append(Bee(k, centroids))
        return population
    
    def employed_bee_phase(self, bee):
        """Employed bee phase - update centroid positions."""
        new_centroids = bee.centroids.copy()
        
        # Randomly select and update a centroid
        i = np.random.randint(0, bee.k)
        j = np.random.randint(0, bee.k)
        while j == i:
            j = np.random.randint(0, bee.k)
        
        phi = np.random.uniform(-1, 1, new_centroids.shape[1])
        new_centroids[i] = new_centroids[i] + phi * (new_centroids[i] - new_centroids[j])
        
        return new_centroids
    
    def _process_bee(self, bee):
        """Process a single bee's update."""
        new_centroids = self.employed_bee_phase(bee)
        labels = self._assign_labels(new_centroids)
        new_metrics = calculate_metrics(self.X, labels)
        
        if is_better_solution(new_metrics, bee.metrics):
            bee.centroids = new_centroids
            bee.metrics = new_metrics
            bee.trials = 0
        else:
            bee.trials += 1
        return bee
    
    def optimize(self):
        """Optimize centroids for the specified k value."""
        population = self.initialize_population_fixed_k(self.k)
        best_centroids = None
        best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        no_improve_count = 0
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Optimizing...", total=self.max_iter)
            
            with Pool(processes=self.n_processes) as pool:
                for iteration in range(self.max_iter):
                    print(f"[cyan]Iteration {iteration}")
                    # Employed bee phase - parallel processing
                    population = pool.map(self._process_bee, population)
                    
                    # Onlooker bee phase
                    probabilities = self.onlooker_bee_phase(population)
                    selected_bees = [population[np.random.choice(len(population), p=probabilities)]
                                   for _ in range(self.population_size)]
                    
                    # Process selected bees in parallel
                    population = list(pool.map(self._process_bee, selected_bees))
                    
                    # Scout bee phase
                    self._scout_bee_phase(population, self.k)
                    
                    # Update best solution
                    current_best = max(population, key=lambda x: x.metrics['silhouette'])
                    if is_better_solution(current_best.metrics, best_metrics):
                        best_centroids = current_best.centroids.copy()
                        best_metrics = current_best.metrics.copy()
                        no_improve_count = 0
                        print(
                            f"[green]Iteration {iteration}: improved "
                            f"(silhouette={best_metrics['silhouette']:.3f}, "
                            f"davies_bouldin={best_metrics['davies_bouldin']:.3f})"
                        )
                    else:
                        no_improve_count += 1
                    
                    self.fitness_history.append(
                        1.0 / (1.0 + current_best.metrics['davies_bouldin'])
                    )
                    
                    progress.update(task, advance=1)
                    
                    if no_improve_count >= self.patience:
                        print(
                            f"[yellow]Early stopping "
                            f"after {no_improve_count} iterations without improvement"
                        )
                        break
            
        return self.k, best_centroids, best_metrics
    
    def _assign_labels(self, centroids):
        """Assign data points to the nearest centroid."""
        distances = calculate_distances(self.X, centroids)
        return np.argmin(distances, axis=1)
    
    def _scout_bee_phase(self, population, k):
        """Scout bee phase - using a fixed k value."""
        for bee in population:
            if bee.trials >= self.limit:
                centroids = self.X[np.random.choice(self.X.shape[0], k, replace=False)]
                bee.reset(k, centroids)
    
    def onlooker_bee_phase(self, population):
        """Onlooker bee phase."""
        fitness_values = [1.0 / (1.0 + bee.metrics['davies_bouldin']) 
                         for bee in population]
        total_fitness = sum(fitness_values)
        return [fit/total_fitness for fit in fitness_values]
    
    def plot_fitness_history(self, save_path='res/fitness_history.png'):
        """Plot and save the fitness history chart."""
          
        # Plot fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', label='Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title(f'ABC Optimization Fitness History (k={self.k})')
        plt.legend()
        plt.grid(True)
        plt.savefig('res/fitness_history.png')
        plt.close()
        