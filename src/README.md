# Clustering Analysis with Artificial Bee Colony and K-Means

This project implements a clustering analysis using a combination of the Artificial Bee Colony (ABC) algorithm and K-Means clustering. The analysis is performed on a dataset of IoT data, and the results are visualized and evaluated using various clustering metrics.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

The project aims to enhance the clustering process by optimizing the initial cluster centers using the Artificial Bee Colony algorithm before applying the K-Means clustering. This approach is designed to improve the clustering performance and provide better insights into the data.

## Features

- **Data Preprocessing**: Handles missing values and outliers, and standardizes the data.
- **Dimensionality Reduction**: Supports PCA and t-SNE for reducing data dimensions.
- **Clustering**: Combines ABC and K-Means for effective clustering.
- **Evaluation**: Uses Silhouette Score, Calinski-Harabasz Score, and Davies-Bouldin Score to evaluate clustering performance.
- **Visualization**: Visualizes the clustering results with matplotlib.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/clustering-analysis.git
   cd clustering-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset in the `data/raw/` directory. The default dataset is `iot23_combined.csv`.

2. Run the analysis:

   ```bash
   python src/clustering_analysis.py
   ```

3. The script will preprocess the data, perform dimensionality reduction, cluster the data, evaluate the results, and visualize the clusters.

## Dependencies

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- rich

Ensure all dependencies are installed by using the `requirements.txt` file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
