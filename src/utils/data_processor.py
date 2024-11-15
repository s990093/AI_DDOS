import pandas as pd
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress

from config.CONFIG import CONFIG


def load_and_preprocess_data(console):
    """加載和預處理數據"""
    console.print("[info]Loading and processing data...", style="primary")

    # 使用 pandas 替代 dask
    df = pd.read_csv(CONFIG['data_path'])
    X = df[CONFIG['numerical_columns']]

    # 標準化
    with Progress() as progress:
        task = progress.add_task("[cyan]Scaling the data...", total=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        progress.update(task, advance=1)
        console.print("[info]Data scaling complete.", style="primary")

    return X_scaled