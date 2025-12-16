import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kstest
import numpy as np
import pandas as pd
import numpy as np
import scipy.stats as st


def calculate_feature_entropy(data, feature, window_size):
    """
    Calculate entropy for a given feature over a sliding time window.
    
    Args:
        data (pd.DataFrame): Input data containing the feature.
        feature (str): The column name of the feature to calculate entropy for.
        window_size (int): The size of the sliding time window.
        
    Returns:
        pd.Series: A series of entropy values corresponding to the time window.
    """
    entropies = []
    for i in range(len(data) - window_size + 1):
        window = data[feature].iloc[i:i + window_size]
        probabilities = window.value_counts(normalize=True)
        # Use st.entropy (or scipy.stats.entropy) instead of just entropy
        entropies.append(st.entropy(probabilities, base=2))  
    
    # Fill with NaN for non-computable values at the start
    return pd.Series([np.nan] * (window_size - 1) + entropies)

# Function to calculate the slope (gradient) for a given feature
def calculate_slope(data, feature):
    """
    Calculate the slope of a feature over time.
    
    Args:
        data (pd.DataFrame): Input data containing the feature.
        feature (str): The column name of the feature to calculate the slope for.
        
    Returns:
        pd.Series: The calculated slope for the feature.
    """
    return data[feature].diff() / data.index.to_series().diff()

# Visualization function for entropy comparison
def visualize_entropy_distribution(data, features, label_column, benign_label, attack_label):
    """
    Visualize the distribution of multiple feature entropies for normal and attack traffic.
    
    Args:
        data (pd.DataFrame): The data containing features and labels.
        features (list): List of features to visualize.
        label_column (str): The label column name.
        benign_label (str): Label for normal traffic.
        attack_label (str): Label for attack traffic.
    """
    plt.figure(figsize=(16, 8))
    
    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        sns.kdeplot(
            data[data[label_column] == benign_label][feature],
            color='green', label='Benign', fill=True, alpha=0.5
        )
        sns.kdeplot(
            data[data[label_column] == attack_label][feature],
            color='red', label='Attack', fill=True, alpha=0.5
        )
        
        plt.title(f"Entropy Distribution: {feature}")
        plt.xlabel("Entropy")
        plt.ylabel("Density")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualization function for entropy and slope dynamics
def visualize_entropy_and_slope(data, features):
    """
    Visualize entropy and slope dynamics for multiple features.
    
    Args:
        data (pd.DataFrame): The data containing entropy and slope features.
        features (list): List of entropy features to visualize.
    """
    plt.figure(figsize=(16, 8))
    
    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        plt.plot(data.index, data[feature], label=f"Entropy ({feature})", color='orange')
        
        slope_feature = f"Slope_{feature.split('_')[-1]}"
        if slope_feature in data.columns:
            plt.plot(
                data.index, data[slope_feature],
                color='blue', linestyle='--', alpha=0.7, label=f"Slope ({feature})"
            )
        
        plt.title(f"Entropy and Slope Dynamics: {feature}")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualization function for normality test and distribution fit
def visualize_normality(data, features):
    """
    Visualize normality test and normal distribution fitting for entropy features.
    
    Args:
        data (pd.DataFrame): The data containing entropy features.
        features (list): List of entropy features to test and visualize.
    """
    plt.figure(figsize=(16, 8))
    
    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        entropy_values = data[feature].dropna()
        
        # Histogram
        sns.histplot(entropy_values, kde=False, bins=30, color='purple', label='Entropy Histogram', alpha=0.7)
        
        # Normal distribution fit
        mu, sigma = norm.fit(entropy_values)
        x = np.linspace(entropy_values.min(), entropy_values.max(), 100)
        fitted_pdf = norm.pdf(x, mu, sigma)
        plt.plot(x, fitted_pdf * len(entropy_values) * (x[1] - x[0]), 
                 label=f"Normal Fit ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})")
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = kstest(entropy_values, 'norm', args=(mu, sigma))
        normality_result = "Pass" if ks_p_value > 0.05 else "Fail"
        plt.title(f"Normality Test: {feature} (KS p={ks_p_value:.3f}, {normality_result})")
        plt.xlabel("Entropy")
        plt.ylabel("Frequency")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def classify_entropy_statistically(data, entropy_features):
    """
    Classify entropy values into 'Low', 'Medium', and 'High' categories based on statistical thresholds.

    Args:
        data (pd.DataFrame): The data containing entropy features.
        entropy_features (list): List of entropy features to classify.

    Returns:
        pd.DataFrame: The input DataFrame with added classification columns.
    """
    for feature in entropy_features:
        entropy_values = data[feature].dropna()
        mu, sigma = entropy_values.mean(), entropy_values.std()

        # Define classification boundaries
        low_threshold = mu - sigma
        high_threshold = mu + sigma

        # Classify entropy
        classification = pd.Series(np.nan, index=data.index, dtype='object')
        classification[data[feature] < low_threshold] = 'Low'
        classification[(data[feature] >= low_threshold) & (data[feature] <= high_threshold)] = 'Medium'
        classification[data[feature] > high_threshold] = 'High'

        # Add classification column
        class_col = f"Class_{feature}"
        data[class_col] = classification
        
        print(f"{feature}: μ={mu:.2f}, σ={sigma:.2f}, Low<{low_threshold:.2f}, High>{high_threshold:.2f}")

    return data


def entropy_anomaly_analysis(data, entropy_features):
    """
    Analyze entropy anomalies by calculating Low and High entropy proportions.

    Args:
        data (pd.DataFrame): Data containing entropy classification.
        entropy_features (list): List of entropy features.

    Returns:
        pd.DataFrame: Proportion of Low and High entropy for each feature.
    """
    summary = {}
    for feature in entropy_features:
        class_col = f"Class_{feature}"
        low_count = (data[class_col] == 'Low').sum()
        high_count = (data[class_col] == 'High').sum()
        total = len(data[class_col].dropna())

        summary[feature] = {
            "Low_Proportion": low_count / total,
            "High_Proportion": high_count / total
        }
    
    summary_df = pd.DataFrame(summary).T
    print("Entropy Anomaly Proportions:")
    print(summary_df)
    return 
  

