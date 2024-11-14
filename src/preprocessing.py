import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, contamination=0.1, scaling_method='standard', 
                 n_components=0.95, random_state=42):
        """
        初始化資料預處理器
        
        Parameters:
        -----------
        contamination : float, 異常值比例
        scaling_method : str, 'standard' 或 'robust'
        n_components : float 或 int, PCA 保留的變異量比例或組件數
        random_state : int, 隨機種子
        """
        self.contamination = contamination
        self.scaling_method = scaling_method
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        self.feature_names = None
        
    def remove_outliers(self, df):
        """使用 Z-score 和 IQR 方法移除異常值"""
        df_clean = df.select_dtypes(include=[np.number])
        
        # Handle NaN values if necessary
        df_clean = df_clean.fillna(df_clean.mean())
        
        # Z-score method
        z_scores = stats.zscore(df_clean)
        z_filter = (np.abs(z_scores) < 3).all(axis=1)
        df_clean = df_clean[z_filter]
        
        # IQR method
        Q1 = df_clean.quantile(0.25)
        Q3 = df_clean.quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean < (Q1 - 1.5 * IQR)) | 
                             (df_clean > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return df_clean
    
    def scale_features(self, X):
        """特徵縮放"""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()
            
        return self.scaler.fit_transform(X)
    
    def reduce_dimensions(self, X, method='pca'):
        """降維處理"""
        if method == 'pca':
            self.pca = PCA(n_components=self.n_components, 
                          random_state=self.random_state)
            X_reduced = self.pca.fit_transform(X)
            
            if isinstance(self.n_components, float):
                print(f"保留 {self.n_components*100}% 的變異量需要 "
                      f"{X_reduced.shape[1]} 個主成分")
            
            # 計算各特徵對主成分的貢獻
            if self.feature_names is not None:
                components_df = pd.DataFrame(
                    self.pca.components_.T,
                    columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
                    index=self.feature_names
                )
                print("\n特徵對主成分的貢獻:")
                print(components_df)
                
        elif method == 'tsne':
            tsne = TSNE(n_components=2, random_state=self.random_state)
            X_reduced = tsne.fit_transform(X)
            
        return X_reduced
    
    def fit_transform(self, df, feature_names=None):
        """執行完整的預處理流程"""
        self.feature_names = feature_names
        
        # 1. 移除異常值
        print("移除異常值...")
        df_clean = self.remove_outliers(df)
        print(f"移除異常值後的資料量: {len(df_clean)} "
              f"(原始資料量: {len(df)})")
        
        # 2. 特徵縮放
        print("\n進行特徵縮放...")
        X_scaled = self.scale_features(df_clean)
        
        # 3. 降維
        print("\n執行降維...")
        X_reduced = self.reduce_dimensions(X_scaled)
        
        return X_reduced, df_clean.index

class ABCKMeans:
    def __init__(self, n_clusters=3, n_bees=30, max_iterations=100, 
                 limit=20, random_state=42):
        """
        初始化 ABC-KMeans 混合演算法
        
        Parameters:
        -----------
        n_clusters : int, 分群數量
        n_bees : int, 蜜蜂數量
        max_iterations : int, 最大迭代次數
        limit : int, 食物源未改善的最大次數
        random_state : int, 隨機種子
        """
        self.n_clusters = n_clusters
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.limit = limit
        self.random_state = random_state
        self.best_centroids = None
        self.best_score = float('-inf')
        
    def initialize_food_sources(self, X):
        """初始化食物源（即群心）"""
        np.random.seed(self.random_state)
        food_sources = []
        
        for _ in range(self.n_bees):
            # 隨機選擇資料點作為初始群心
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            food_sources.append(X[indices])
            
        return np.array(food_sources)
    
    def calculate_fitness(self, X, centroids):
        """計算適應度（使用 Calinski-Harabasz 指標）"""
        kmeans = KMeans(n_clusters=self.n_clusters, 
                       init=centroids,
                       n_init=1,
                       max_iter=300)
        labels = kmeans.fit_predict(X)
        score = metrics.calinski_harabasz_score(X, labels)
        return score
    
    def fit(self, X):
        """執行 ABC-KMeans 混合演算法"""
        food_sources = self.initialize_food_sources(X)
        trials = np.zeros(self.n_bees)
        
        for iteration in range(self.max_iterations):
            # Employed Bees Phase
            for i in range(self.n_bees):
                # 產生新的食物源
                new_source = food_sources[i].copy()
                k = np.random.randint(self.n_bees)
                j = np.random.randint(self.n_clusters)
                phi = np.random.uniform(-1, 1)
                
                new_source[j] = (food_sources[i][j] + 
                               phi * (food_sources[i][j] - food_sources[k][j]))
                
                # 計算新舊食物源的適應度
                old_fitness = self.calculate_fitness(X, food_sources[i])
                new_fitness = self.calculate_fitness(X, new_source)
                
                # 貪婪選擇
                if new_fitness > old_fitness:
                    food_sources[i] = new_source
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            # Onlooker Bees Phase
            probabilities = self._calculate_probabilities(X, food_sources)
            
            for i in range(self.n_bees):
                if np.random.random() < probabilities[i]:
                    # 同 Employed Bees Phase 的更新過程
                    new_source = food_sources[i].copy()
                    k = np.random.randint(self.n_bees)
                    j = np.random.randint(self.n_clusters)
                    phi = np.random.uniform(-1, 1)
                    
                    new_source[j] = (food_sources[i][j] + 
                                   phi * (food_sources[i][j] - food_sources[k][j]))
                    
                    old_fitness = self.calculate_fitness(X, food_sources[i])
                    new_fitness = self.calculate_fitness(X, new_source)
                    
                    if new_fitness > old_fitness:
                        food_sources[i] = new_source
                        trials[i] = 0
                    else:
                        trials[i] += 1
            
            # Scout Bees Phase
            for i in range(self.n_bees):
                if trials[i] >= self.limit:
                    indices = np.random.choice(X.shape[0], 
                                            self.n_clusters, 
                                            replace=False)
                    food_sources[i] = X[indices]
                    trials[i] = 0
            
            # 更新最佳解
            for source in food_sources:
                score = self.calculate_fitness(X, source)
                if score > self.best_score:
                    self.best_score = score
                    self.best_centroids = source.copy()
        
        return self
    
    def _calculate_probabilities(self, X, food_sources):
        """計算食物源的選擇機率"""
        fitness_values = np.array([self.calculate_fitness(X, source) 
                                 for source in food_sources])
        return fitness_values / np.sum(fitness_values) 