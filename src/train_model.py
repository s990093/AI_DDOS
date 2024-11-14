# Import statements
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                           recall_score, fbeta_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib
from dask import dataframe as dd
from dask.distributed import Client
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich import print as rprint

# Initialize Rich console
console = Console()

# Configure display settings
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
plt.style.use('dark_background')

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    ddf = dd.read_csv("data/raw/iot23_combined.csv", blocksize='64MB')
   
    ddf = ddf[(ddf['Label'] == 'Benign') | (ddf['Label'] == 'DDoS')]
    df = ddf.compute()
    
    # Add index and clean column names
    df.insert(0, column='number', value=list(range(0, len(df))))
    df.set_index(["number"], inplace=True)
    df.columns = df.columns.str.replace(' ', '')
    
    # Clean data and handle inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    
    # Convert features to numeric
    cols = df.drop(columns=['Label']).columns.tolist()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    

    return df

def prepare_features(df, selected_columns):
    """Prepare features and target variables."""
    labels = pd.get_dummies(df['Label'])
    X = df[selected_columns]
    y = labels.Benign
    return X, y

def train_and_evaluate_model(X, y, model_name="Default"):
    """Train and evaluate a random forest model."""
    with console.status(f"[bold green]Training {model_name} model..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25, stratify=y, random_state=10
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        # Train model
        rf = RandomForestClassifier(n_jobs=-1, random_state=3)
        with joblib.parallel_backend("dask"):
            rf.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf.predict(X_test_scaled)
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f2_score": fbeta_score(y_test, y_pred, beta=2.0),
            "accuracy": rf.score(X_test_scaled, y_test) * 100
        }
        
        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(conf_matrix, f'{model_name} Random Forest Confusion Matrix')
        
        return rf, metrics

def plot_confusion_matrix(conf_matrix, title):
    """Plot confusion matrix with labels."""
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = [f"{value:0.0f}" for value in conf_matrix.flatten()]
    group_percentages = [f"{value:0.4%}" for value in conf_matrix.flatten()/np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, cmap='rocket_r', annot=labels, fmt='', 
                square=True, xticklabels=['DDoS', 'Benign'], 
                yticklabels=['DDoS', 'Benign'])
    plt.xlabel('Predicted Traffic Type')
    plt.ylabel('Actual Traffic Type')
    plt.title(title)
    plt.show()

def display_metrics(model_name, metrics):
    """Display model metrics in a beautiful table."""
    table = Table(title=f"{model_name} Model Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    for metric, value in metrics.items():
        table.add_row(
            metric.replace('_', ' ').title(),
            f"{value:.4f}"
        )
    
    console.print(table)

def main():
    # Show welcome message
    console.print(Panel.fit(
        "[bold blue]DDoS Detection Model Training[/bold blue]",
        subtitle="[italic]Using Random Forest Classifier[/italic]"
    ))
    
    # Initialize Dask client
    with console.status("[bold green]Initializing Dask client..."):
        client = Client(processes=False)
    
    # Load and preprocess data
    with console.status("[bold green]Loading and preprocessing data..."):
        df = load_and_preprocess_data()
    
    # Define feature sets
    feature_sets = {
        "Default": ['orig_pkts', 'orig_ip_bytes', 'conn_state_OTH', 'duration',
                   'conn_state_S0', 'missed_bytes', 'orig_bytes', 'proto_tcp',
                   'proto_udp', 'resp_ip_bytes', 'proto_icmp'],
        # "Model_A": ['orig_ip_bytes', 'orig_pkts', 'conn_state_S0', 'conn_state_OTH',
        #            'duration', 'orig_bytes', 'resp_bytes', 'proto_tcp', 'proto_udp'],
        # "Model_B": ['orig_ip_bytes', 'orig_pkts', 'conn_state_S0', 'conn_state_OTH',
        #            'duration']
    }
    
    # Train and evaluate models
    for model_name, features in feature_sets.items():
        console.rule(f"[bold red]{model_name} Model")
        
        with console.status("[bold green]Preparing features..."):
            X, y = prepare_features(df, features)
        
        model, metrics = train_and_evaluate_model(X, y, model_name)
        display_metrics(model_name, metrics)
        console.print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[bold red]Process interrupted by user")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")
    finally:
        console.print("[bold green]Process completed!")