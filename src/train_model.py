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
from datetime import datetime
import json
import os

# Initialize Rich console
console = Console()


def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    ddf = dd.read_csv("data/raw/iot23_combined.csv", blocksize='64MB')
   
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
    
    # 移除原来的过滤条件，保留所有标签类型
    df = ddf.compute()
    
    # 使用表格整理標籤資訊
    table = Table(title="數據集標籤分布", show_header=True, header_style="bold magenta")
    table.add_column("標籤類型", style="cyan")
    table.add_column("樣本數量", justify="right", style="green")
    
    for label, count in df['Label'].value_counts().items():
        table.add_row(str(label), str(count))
    
    console.print(table)
    
    return df

def prepare_features(df, selected_columns):
    """Prepare features and target variables."""
    labels = pd.get_dummies(df['Label'])
    X = df[selected_columns].copy()
    y = labels
    
    # Load cluster predictions
    try:
        cluster_predictions = np.loadtxt('res/predictions.csv', delimiter=',')
        console.print(f"[info]Loaded {len(cluster_predictions)} cluster predictions", style="bold green")
        
        if len(cluster_predictions) != len(df):
            console.print(
                f"[warning]Mismatch in sizes: clusters ({len(cluster_predictions)}) vs data ({len(df)})", 
                style="bold yellow"
            )
            cluster_predictions = cluster_predictions[:len(df)]
            console.print("[info]Truncated cluster predictions to match data size", style="bold blue")
    except FileNotFoundError:
        console.print("[error]Cluster predictions not found. Please run clustering first.", style="bold red")
        return None, None
    
    X['cluster'] = cluster_predictions
    return X, y

def train_and_evaluate_model(X, y, model_name="Default"):
    """Train and evaluate a random forest model."""
    with console.status(f"[bold green]訓練 {model_name} 模型..."):
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
        
        # Save model and scaler
        model_path = f'res/model_{model_name}.joblib'
        scaler_path = f'res/scaler_{model_name}.joblib'
        joblib.dump(rf, model_path)
        joblib.dump(scaler, scaler_path)
        console.print(f"[info]Saved model to {model_path}", style="bold green")
        console.print(f"[info]Saved scaler to {scaler_path}", style="bold green")
        
        # 簡化模型保存訊息
        paths = {
            "model": model_path,
            "scaler": scaler_path
        }
        table = Table(title="模型保存位置", show_header=True, header_style="bold magenta")
        table.add_column("類型", style="cyan")
        table.add_column("路徑", style="green")
        for k, v in paths.items():
            table.add_row(k, v)
        console.print(table)
        
        # Evaluate model
        y_pred = rf.predict(X_test_scaled)
        
        # 計算每個類別的指標
        metrics = {}
        for label in y.columns:
            label_metrics = {
                "precision": precision_score(
                    y_test[label], 
                    y_pred[:, y.columns.get_loc(label)], 
                    average='binary',
                    zero_division=0
                ),
                "recall": recall_score(
                    y_test[label], 
                    y_pred[:, y.columns.get_loc(label)], 
                    average='binary',
                    zero_division=0
                ),
                "f2_score": fbeta_score(
                    y_test[label], 
                    y_pred[:, y.columns.get_loc(label)], 
                    beta=2.0, 
                    average='binary',
                    zero_division=0
                ),
            }
            metrics[label] = label_metrics
        
        # 添加整體準確率
        metrics["overall"] = {
            "accuracy": rf.score(X_test_scaled, y_test) * 100
        }
        
        # 生成多類別混淆矩陣
        conf_matrix = confusion_matrix(y_test.idxmax(axis=1), 
                                     pd.DataFrame(y_pred, columns=y.columns).idxmax(axis=1))
        plot_confusion_matrix(conf_matrix, f'{model_name} Random Forest Confusion Matrix', 
                             labels=y.columns)
        
        return rf, metrics

def plot_confusion_matrix(conf_matrix, title, labels):
    """Plot confusion matrix with labels."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, cmap='rocket_r', annot=True, fmt='d',
                square=True, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Traffic Type')
    plt.ylabel('Actual Traffic Type')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_metrics(model_name, metrics):
    """Display model metrics in a beautiful table."""
    # 創建一個大表格包含所有指標
    table = Table(
        title=f"{model_name} 模型評估結果",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("標籤", style="cyan")
    table.add_column("精確率", justify="right", style="green")
    table.add_column("召回率", justify="right", style="green")
    table.add_column("F2分數", justify="right", style="green")
    
    # 添加每個標籤的指標
    for label, label_metrics in metrics.items():
        if label == "overall":
            continue
        table.add_row(
            label,
            f"{label_metrics['precision']:.4f}",
            f"{label_metrics['recall']:.4f}",
            f"{label_metrics['f2_score']:.4f}"
        )
    
    # 添加整體準確率
    if "overall" in metrics:
        table.add_row(
            "整體準確率",
            f"{metrics['overall']['accuracy']:.4f}",
            "-",
            "-"
        )
    
    console.print(table)

def main():
    # 更新歡迎訊息
    console.print(Panel.fit(
        "[bold blue]DDoS 檢測模型訓練[/bold blue]",
        subtitle="[italic]使用隨機森林分類器[/italic]"
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
    
    # Create a summary dictionary to store all results
    summary = {
        "model_info": {},
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df)
    }
    
    # Train and evaluate models for each cluster
    for model_name, features in feature_sets.items():
        console.rule(f"[bold red]{model_name} Model")
        
        model_summary = {
            "features_used": features,
            "clusters": {}
        }
        
        with console.status("[bold green]Preparing features..."):
            X, y = prepare_features(df, features)
            if X is None:
                return
        
        # Train separate models for each cluster
        unique_clusters = X['cluster'].unique()
        for cluster in unique_clusters:
            cluster_mask = X['cluster'] == cluster
            X_cluster = X[cluster_mask].drop(columns=['cluster'])
            y_cluster = y[cluster_mask]
            
            # Get class distribution
            class_counts = y_cluster.idxmax(axis=1).value_counts()
            
            # Skip if insufficient samples
            if len(class_counts) < 2 or min(class_counts) < 2:
                console.print(
                    f"[yellow]Skipping Cluster {cluster}: Insufficient samples "
                    f"(min samples per class: {min(class_counts) if len(class_counts) > 0 else 0})"
                )
                model_summary["clusters"][f"cluster_{cluster}"] = {
                    "status": "skipped",
                    "reason": "insufficient_samples",
                    "class_distribution": class_counts.to_dict()
                }
                continue
            
            console.rule(f"[bold blue]Cluster {cluster}")
            model, metrics = train_and_evaluate_model(X_cluster, y_cluster, f"{model_name}_Cluster_{cluster}")
            display_metrics(f"{model_name}_Cluster_{cluster}", metrics)
            
            # Add cluster results to summary
            model_summary["clusters"][f"cluster_{cluster}"] = {
                "status": "success",
                "metrics": metrics,
                "samples": len(y_cluster),
                "class_distribution": class_counts.to_dict(),
                "model_path": f'model_{model_name}_Cluster_{cluster}.joblib',
                "scaler_path": f'scaler_{model_name}_Cluster_{cluster}.joblib'
            }
            
            console.print()
        
        summary["model_info"][model_name] = model_summary
    
    # Save summary to JSON
    summary_path = 'res/training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    console.print(f"[bold green]Saved training summary to {summary_path}")

if __name__ == "__main__":  
    main()
