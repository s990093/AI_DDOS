from numpy import ndarray
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress

from config.CONFIG import CONFIG


def load_and_preprocess_data(console) -> tuple[ndarray, pd.Series, pd.Series[int]]:
    """加載和預處理數據"""
    console.print("[info]Loading and processing data...", style="primary")

    # 使用 pandas 讀取數據
    df = pd.read_csv(CONFIG['data_path'], index_col=0)

    # 分離特徵和標籤
    X = df.drop('Label', axis=1)
    y_full = df['Label'] 
    y_ddos = (df['Label'] == 'DDoS').astype(int)
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Scaling the data...", total=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        progress.update(task, advance=1)
        console.print("[info]Data scaling complete.", style="primary")

    return X_scaled, y_full, y_ddos  