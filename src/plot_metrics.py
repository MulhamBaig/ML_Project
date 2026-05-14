import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def main():
    # Setup paths
    csv_path = Path('Results_Comparison/training_metrics_log.csv')
    out_dir = Path('Results_Comparison')
    out_dir.mkdir(exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # Load data
    df = pd.read_csv(csv_path)
    # Filter globally to only show the latest models
    allowed_models = ['resnet50', 'yolov8n-seg-final']
    df = df[df['Model_Name'].isin(allowed_models)].copy()
    # Ensure numeric columns are numeric
    numeric_cols = ['Val_mIoU', 'Inference_Latency_ms', 'Total_Time_Elapsed_sec']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Rename for cleaner display on the charts
    df['Model_Name'] = df['Model_Name'].replace({'resnet50': 'FCN-ResNet50', 'yolov8n-seg-final': 'YOLOv8n-seg'})

    # 1. Accuracy Progression (Epoch 1-5)
    plt.figure(figsize=(10, 6))
    # Filter for numeric epochs only
    epoch_df = df[df['Epoch'].apply(lambda x: str(x).isdigit())].copy()
    epoch_df['Epoch'] = epoch_df['Epoch'].astype(int)
    
    sns.lineplot(data=epoch_df, x='Epoch', y='Val_mIoU', hue='Model_Name', marker='o')
    plt.title('Accuracy Progression over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score (mIoU / Mask mAP50)')
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend(title='Model')
    plt.figtext(0.5, 0.01, "Note: ResNet50 uses mIoU while YOLO uses Mask mAP50", ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_dir / 'accuracy_progression.png', dpi=150)
    print(f"Saved: {out_dir / 'accuracy_progression.png'}")

    # 2. Latency Comparison (Epoch == FINAL)
    plt.figure(figsize=(8, 6))
    final_df = df[df['Epoch'] == 'FINAL'].copy()
    
    # Some models might have multiple FINAL rows if we were messy, let's take the latest for each model
    # Or just keep the ones we have. The user's CSV has resnet50, yolov8n-seg
    # Wait, the user's CSV shown in metadata has:
    # resnet50, FINAL, ..., 39.6
    # yolov8n-seg, FINAL, ..., 13.84
    
    # Filter for specific models if needed, but let's just plot what's there
    final_df = final_df[final_df['Model_Name'].isin(['FCN-ResNet50', 'YOLOv8n-seg'])]
    
    bars = sns.barplot(data=final_df, x='Model_Name', y='Inference_Latency_ms', hue='Model_Name', palette='viridis', legend=False)
    plt.title('Inference Latency Comparison (GPU)')
    plt.ylabel('Latency (ms)')
    plt.xlabel('Model')
    
    # Add labels on top
    for p in bars.patches:
        bars.annotate(f'{p.get_height():.2f} ms', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=12, fontweight='bold')
    
    plt.ylim(0, max(final_df['Inference_Latency_ms']) * 1.2)
    plt.tight_layout()
    plt.savefig(out_dir / 'latency_comparison.png', dpi=150)
    print(f"Saved: {out_dir / 'latency_comparison.png'}")

    # 3. Training Time Comparison (Epoch 5)
    plt.figure(figsize=(8, 6))
    time_df = epoch_df[epoch_df['Epoch'] == 5].copy()
    
    bars = sns.barplot(data=time_df, x='Model_Name', y='Total_Time_Elapsed_sec', hue='Model_Name', palette='magma', legend=False)
    plt.title('Total Training Time Comparison (5 Epochs)')
    plt.ylabel('Total Time (sec)')
    plt.xlabel('Model')
    
    # Add labels on top
    for p in bars.patches:
        bars.annotate(f'{p.get_height():.1f} s', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=12, fontweight='bold')

    plt.ylim(0, max(time_df['Total_Time_Elapsed_sec']) * 1.2)
    plt.tight_layout()
    plt.savefig(out_dir / 'training_time_comparison.png', dpi=150)
    print(f"Saved: {out_dir / 'training_time_comparison.png'}")

if __name__ == "__main__":
    main()
