# AI DDoS Detection Project

This project implements an AI-based system for detecting and analyzing DDoS (Distributed Denial of Service) attacks using machine learning and deep learning techniques.

## Project Structure

The project is organized as follows:

- **`src/`**: Contains the core source code for the project, including data processing and model training scripts.
- **`data/`**: Stores data files. Note that large processed files like `processed_output.csv` are excluded from version control.
- **`notebooks/`**: Contains Jupyter notebooks for exploration, visualization, and experimental training.
- **`results/`**: Stores generated results, such as plots, logs, and reports.
- **`legacy/`**: Contains legacy code and unrelated scripts (e.g., an image viewer found in the original directory).

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd AI_DDOS
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Processing

To process the raw data, run the preprocessing script in `src/`:

```bash
python src/preprocessing.py
```

### Training

To train the model, use:

```bash
python src/train_model.py
```

## Features

- **Entropy Calculation**: Analyzes feature entropy in `src/preprocessing.py`.
- **Clustering**: Uses K-Medoids and other clustering techniques to identify attack patterns.
- **Visualization**: Generates entropy distribution and cluster visualization plots.

## Notes

- Large data files are ignored by `.gitignore`. Ensure you have the necessary datasets in `data/`.
