import pandas as pd
from sklearn.preprocessing import StandardScaler
from rich.progress import Progress

from config.CONFIG import CONFIG


def load_and_preprocess_data(console):
    """加載和預處理數據"""
    console.print("[info]Loading and processing data...", style="primary")

    # 使用 pandas 讀取數據
    df = pd.read_csv(CONFIG['data_path'], index_col=0)
    
    # 顯示完整的標籤分布
    label_counts = df['Label'].value_counts()
    console.print("\n[bold]Full label distribution:[/bold]")
    for label, count in label_counts.items():
        percentage = (count/len(df))*100
        console.print(f"  {label}: {count} ({percentage:.2f}%)")
    
    # 顯示 DDoS vs Non-DDoS 分布
    ddos_count = (df['Label'] == 'DDoS').sum()
    non_ddos_count = len(df) - ddos_count
    console.print("\n[bold]DDoS vs Non-DDoS distribution:[/bold]")
    console.print(f"  DDoS: {ddos_count} ({(ddos_count/len(df))*100:.2f}%)")
    console.print(f"  Non-DDoS: {non_ddos_count} ({(non_ddos_count/len(df))*100:.2f}%)")
    
    # 分離特徵和標籤
    X = df.drop('Label', axis=1)
    y_full = df['Label']  # 完整標籤
    y_ddos = (df['Label'] == 'DDoS').astype(int)  # DDoS 二分類標籤
    
    # 標準化特徵
    with Progress() as progress:
        task = progress.add_task("[cyan]Scaling the data...", total=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        progress.update(task, advance=1)
        console.print("[info]Data scaling complete.", style="primary")

    return X_scaled, y_full, y_ddos  