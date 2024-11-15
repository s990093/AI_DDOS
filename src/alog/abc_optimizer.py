import numpy as np
from .bee import Bee
from .metrics import calculate_metrics, is_better_solution
from rich.progress import Progress
import matplotlib.pyplot as plt
from numba import jit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed

@jit(nopython=True, parallel=True)
def calculate_distances(X, centroids):
    """Calculate the distance from each point to each centroid."""
    num_points, num_features = X.shape
    num_centroids = centroids.shape[0]
    distances = np.empty((num_points, num_centroids), dtype=np.float64)
    
    for i in prange(num_points):
        for j in range(num_centroids):
            dist = 0.0
            for k in range(num_features):
                diff = X[i, k] - centroids[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    return distances

class ABCOptimizer:
    def __init__(self, X, k_range=(5, 10), max_iter=50, population_size=10, 
                 limit=20, patience=5):
        """Initialize the ABC optimizer."""
        self.X = X
        self.k_range = k_range
        self.max_iter = max_iter
        self.population_size = population_size
        self.limit = limit
        self.patience = patience
        self.best_solution = None
        self.best_centroids = None
        self.best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        self.fitness_history = []
    
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
    
    def optimize_centroids(self, k):
        """Optimize centroids for a specific k value."""
        population = self.initialize_population_fixed_k(k)
        best_centroids = None
        best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        no_improve_count = 0
        
        for iteration in range(self.max_iter):
            # Employed bee phase
            for bee in population:
                new_centroids = self.employed_bee_phase(bee)
                labels = self._assign_labels(new_centroids)
                new_metrics = calculate_metrics(self.X, labels)
                
                if is_better_solution(new_metrics, bee.metrics):
                    bee.centroids = new_centroids
                    bee.metrics = new_metrics
                    bee.trials = 0
                else:
                    bee.trials += 1
            
            # Onlooker bee phase
            probabilities = self.onlooker_bee_phase(population)
            for _ in range(self.population_size):
                selected_bee_index = np.random.choice(len(population), p=probabilities)
                selected_bee = population[selected_bee_index]
                new_centroids = self.employed_bee_phase(selected_bee)
                labels = self._assign_labels(new_centroids)
                new_metrics = calculate_metrics(self.X, labels)
                
                if is_better_solution(new_metrics, selected_bee.metrics):
                    selected_bee.centroids = new_centroids
                    selected_bee.metrics = new_metrics
                    selected_bee.trials = 0
                else:
                    selected_bee.trials += 1
            
            # Scout bee phase
            self._scout_bee_phase(population, k)
            
            # Update best solution
            current_best = max(population, key=lambda x: x.metrics['silhouette'])
            if is_better_solution(current_best.metrics, best_metrics):
                best_centroids = current_best.centroids.copy()
                best_metrics = current_best.metrics.copy()
                no_improve_count = 0
                print(
                    f"[green]k={k} improved: "
                    f"silhouette={best_metrics['silhouette']:.3f}, "
                    f"davies_bouldin={best_metrics['davies_bouldin']:.3f}"
                )
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.patience:
                print(
                    f"[yellow]k={k} early stopping "
                    f"after {no_improve_count} iterations without improvement"
                )
                break
        
        return best_centroids, best_metrics
    
    def optimize(self):
        """Execute the full optimization process."""
        overall_best_k = None
        overall_best_centroids = None
        overall_best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }
        
        with Progress() as progress:
            k_task = progress.add_task(
                f"[cyan]Testing k values ({self.k_range[0]}-{self.k_range[1]})...", 
                total=self.k_range[1] - self.k_range[0] + 1
            )
            
            with ProcessPoolExecutor(max_workers=5) as executor:
                future_to_k = {
                    executor.submit(self.optimize_centroids, k): k 
                    for k in range(self.k_range[0], self.k_range[1] + 1)
                }
                
                for future in as_completed(future_to_k):
                    k = future_to_k[future]
                    try:
                        best_centroids, best_metrics = future.result()
                        
                        if is_better_solution(best_metrics, overall_best_metrics):
                            overall_best_k = k
                            overall_best_centroids = best_centroids.copy()
                            overall_best_metrics = best_metrics.copy()
                            progress.print(
                                f"[yellow]New best k found: k={k} "
                                f"(silhouette={best_metrics['silhouette']:.3f}, "
                                f"davies_bouldin={best_metrics['davies_bouldin']:.3f})"
                            )
                        
                        self.fitness_history.append(
                            1.0 / (1.0 + best_metrics['davies_bouldin'])
                        )
                    except Exception as e:
                        progress.print(f"[red]Error optimizing k={k}: {e}")
                    
                    progress.update(k_task, advance=1)
        
        return overall_best_k, overall_best_centroids, overall_best_metrics
    
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
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', label='Best Fitness')
        plt.xlabel('K value')
        plt.ylabel('Fitness Value')
        plt.title('ABC Optimization Fitness History')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close() 