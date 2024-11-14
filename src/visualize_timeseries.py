import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

console = Console()

def load_and_prepare_data(file_path):
    """載入並準備時間序列資料"""
    with console.status("[bold green]Loading data...") as status:
        # 讀取資料
        df = pd.read_csv(file_path)
        console.print("✓ Data loaded successfully")
        
        # 打印所有唯一的标签
        console.print(f"Available labels: {df['Label'].unique()}")
        
        # 使用索引作為時間戳
        status.update("[bold yellow]Using index as pseudo-timestamp...")
        df['PseudoTimestamp'] = df.index
        
        return df

def create_time_series_plots(df):
    """創建互動式時間序列圖表"""
    with console.status("[bold green]Creating visualizations..."):
        # 創建每分鐘的流量統計
        df_grouped = df.groupby([pd.Grouper(key='PseudoTimestamp', freq='1Min'), 'Label']).size().unstack(fill_value=0)
        
        # 使用 Plotly 創建互動式圖表
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Traffic Flow Over Time', 'Traffic Distribution'),
                           vertical_spacing=0.15)

        # 添加時間序列線圖
        fig.add_trace(
            go.Scatter(x=df_grouped.index, y=df_grouped['Benign'],
                      name='Benign Traffic', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_grouped.index, y=df_grouped['DDoS'],
                      name='DDoS Traffic', line=dict(color='red')),
            row=1, col=1
        )

        # 添加流量分布圖
        fig.add_trace(
            go.Box(y=df[df['Label'] == 'Benign']['orig_ip_bytes'],
                  name='Benign', boxpoints='outliers',
                  marker_color='green'),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=df[df['Label'] == 'DDoS']['orig_ip_bytes'],
                  name='DDoS', boxpoints='outliers',
                  marker_color='red'),
            row=2, col=1
        )

        # 更新布局
        fig.update_layout(
            height=900,
            title_text="Network Traffic Analysis Over Time",
            showlegend=True
        )

        # 保存互動式圖表
        fig.write_html("results/traffic_analysis.html")
        console.print("[bold green]✓ Interactive visualization saved as 'traffic_analysis.html'")

        # 創建靜態圖表
        plt.figure(figsize=(15, 10))
        
        # 繪製時間序列
        plt.subplot(2, 1, 1)
        df_grouped['Benign'].plot(label='Benign', color='green')
        df_grouped['DDoS'].plot(label='DDoS', color='red')
        plt.title('Traffic Flow Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Connections')
        plt.legend()
        plt.grid(True)

        # 繪製每小時流量分布
        plt.subplot(2, 1, 2)
        df['Hour'] = df['PseudoTimestamp'].dt.hour
        hourly_stats = df.groupby(['Hour', 'Label']).size().unstack()
        hourly_stats.plot(kind='bar', stacked=True)
        plt.title('Hourly Traffic Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Connections')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('results/traffic_analysis.png')
        console.print("[bold green]✓ Static visualization saved as 'traffic_analysis.png'")

def analyze_traffic_patterns(df):
    """分析流量模式並顯示統計信息"""
    with console.status("[bold green]Analyzing traffic patterns..."):
        # 計算基本統計信息
        stats_table = Table(title="Traffic Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Benign", style="green")
        stats_table.add_column("DDoS", style="red")

        # 添加統計數據
        benign_stats = df[df['Label'] == 'Benign']['orig_ip_bytes'].describe()
        ddos_stats = df[df['Label'] == 'DDoS']['orig_ip_bytes'].describe()

        for stat in ['count', 'mean', 'std', 'max']:
            stats_table.add_row(
                stat.capitalize(),
                f"{benign_stats[stat]:,.2f}",
                f"{ddos_stats[stat]:,.2f}"
            )

        console.print(stats_table)

def plot_label_distribution(df):
    """繪製標籤分佈圖"""
    with console.status("[bold green]Plotting label distribution..."):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Label', order=df['Label'].value_counts().index)
        plt.title('Label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/label_distribution.png')
        console.print("[bold green]✓ Label distribution plot saved as 'label_distribution.png'")

def main():
    """主函數"""
    console.print(Panel.fit(
        "[bold blue]Network Traffic Visualization",
        subtitle="Time Series Analysis"
    ))

    try:
        # 載入資料
        df = load_and_prepare_data('data/raw/iot23_combined.csv')
        
        if df is not None:
            # 繪製標籤分佈
            plot_label_distribution(df)
            
            # 創建視覺化
            create_time_series_plots(df)
            
            # 分析流量模式
            analyze_traffic_patterns(df)
            
            console.print(Panel.fit(
                "[bold green]Visualization completed successfully!",
                subtitle="Check results folder for outputs"
            ))
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}")

if __name__ == "__main__":
    main() 