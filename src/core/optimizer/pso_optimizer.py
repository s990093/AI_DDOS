import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..metrics.clustering_metrics import calculate_metrics, is_better_solution
from .particle import Particle
from multiprocessing import Pool
import os
from sklearn.metrics.pairwise import pairwise_distances

class PSOOptimizer:
    def __init__(self, X, k, max_iter=100, n_particles=30, 
                 w_start=0.9, w_end=0.4, c1=1.49, c2=1.49, patience=10, n_processes=-1):
        """Initialize PSO optimizer for clustering.
        
        Args:
            X (np.ndarray): Input data
            k (int): Number of clusters
            max_iter (int): Maximum iterations
            n_particles (int): Number of particles
            w_start (float): Initial inertia weight
            w_end (float): Final inertia weight
            c1 (float): Cognitive weight
            c2 (float): Social weight
            patience (int): Early stopping patience
            n_processes (int): Number of processes for parallel computation. -1 means using all available cores.
        """
        self.X = X
        self.k = k
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.w_start = w_start  # Initial inertia weight
        self.w_end = w_end      # Final inertia weight
        self.w = w_start        # Current inertia weight
        self.c1 = c1  # Cognitive weight
        self.c2 = c2  # Social weight
        self.patience = patience
        self.fitness_history = []
        self.n_processes = os.cpu_count() if n_processes == -1 else n_processes
        
        # Initialize particles
        self.particles = [Particle(k, X.shape[1]) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_metrics = {
            'silhouette': float('-inf'),
            'calinski_harabasz': float('-inf'),
            'davies_bouldin': float('inf')
        }

    def _assign_labels(self, centroids):
        """Assign data points to nearest centroids using vectorized operations."""
        # Reshape for broadcasting
        X_expanded = self.X[:, np.newaxis, :]  # shape: (n_samples, 1, n_features)
        centroids_expanded = centroids[np.newaxis, :, :]  # shape: (1, k, n_features)
        
        # Compute distances using broadcasting
        distances = np.sum((X_expanded - centroids_expanded) ** 2, axis=2)  # Euclidean distance squared
        return np.argmin(distances, axis=1)

    def _update_particle(self, particle):
        """Update and evaluate a single particle."""
        # Add velocity clamping
        v_max = 0.1 * (np.max(self.X) - np.min(self.X))
        particle.velocity = np.clip(particle.velocity, -v_max, v_max)
        
        # Update velocity
        r1, r2 = np.random.rand(2)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        particle.velocity = (self.w * particle.velocity + cognitive + social)
        
        # Update position
        particle.position = particle.position + particle.velocity
        
        # Add position boundary checking
        x_min, x_max = np.min(self.X, axis=0), np.max(self.X, axis=0)
        particle.position = np.clip(particle.position, x_min, x_max)
        
        # Evaluate new position
        labels = self._assign_labels(particle.position)
        particle.current_metrics = calculate_metrics(self.X, labels)
        
        return particle

    def _init_particle(self, particle):
        """Initialize a single particle with metrics."""
        labels = self._assign_labels(particle.position)
        particle.current_metrics = calculate_metrics(self.X, labels)
        return particle

    def optimize(self):
        """Execute PSO optimization process."""
        no_improve_count = 0
        
        # Initialize global best using multiprocessing
        # Initialize particles in parallel with max 5 processes
        with Pool(processes=self.n_processes) as pool:
            initialized_particles = pool.map(self._init_particle, self.particles)
        
        # Update particles and find initial global best
        self.particles = initialized_particles
        for particle in self.particles:
            if is_better_solution(particle.current_metrics, self.global_best_metrics):
                self.global_best_metrics = particle.current_metrics.copy()
                self.global_best_position = particle.position.copy()
        
        # 使用更大的进程池
        n_processes = min(self.n_particles, self.n_processes)
        
        with Pool(processes=n_processes) as pool:
            # 批量处理粒子更新
            chunk_size = max(self.n_particles // n_processes, 1)
            
            with tqdm(total=self.max_iter, desc="PSO Optimization") as pbar:
                for iteration in range(self.max_iter):
                    # Update inertia weight linearly
                    self.w = self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)
                    
                    improved = False
                    
                    # 使用更大的chunk_size进行并行处理
                    updated_particles = pool.map(self._update_particle, self.particles, 
                                              chunksize=chunk_size)
                    
                    # Update particles and check for improvements
                    for particle in updated_particles:
                        # Update personal best
                        if is_better_solution(particle.current_metrics, particle.best_metrics):
                            particle.best_position = particle.position.copy()
                            particle.best_metrics = particle.current_metrics.copy()
                            
                            # Update global best
                            if is_better_solution(particle.current_metrics, self.global_best_metrics):
                                self.global_best_position = particle.position.copy()
                                self.global_best_metrics = particle.current_metrics.copy()
                                improved = True
                                print(
                                    f"[green]Iteration {iteration}: improved "
                                    f"(silhouette={self.global_best_metrics['silhouette']:.3f}, "
                                    f"davies_bouldin={self.global_best_metrics['davies_bouldin']:.3f})"
                                )
                    
                    # Update particles list
                    self.particles = updated_particles
                    
                    # Record fitness history
                    self.fitness_history.append(
                        1.0 / (1.0 + self.global_best_metrics['davies_bouldin'])
                    )
                    
                    # Update progress bar
                    pbar.update(1)
                    if improved:
                        pbar.set_postfix({
                            'silhouette': f"{self.global_best_metrics['silhouette']:.3f}",
                            'davies_bouldin': f"{self.global_best_metrics['davies_bouldin']:.3f}"
                        })
                    
                    # Early stopping check
                    if no_improve_count >= self.patience:
                        pbar.set_description(f"Early stopping after {no_improve_count} iterations")
                        break
        
        # Plot fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', label='Fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.title(f'PSO Optimization Fitness History (k={self.k})')
        plt.legend()
        plt.grid(True)
        plt.savefig('res/fitness_history.png')
        plt.close()
        
        return self.k, self.global_best_position, self.global_best_metrics 