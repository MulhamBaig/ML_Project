"""Centralized metrics logging for model training comparison."""
import csv
import os
from datetime import datetime


class MetricsLogger:
    def __init__(self, log_dir='Results_Comparison', filename='training_metrics_log.csv'):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            self._init_csv()
    
    def _init_csv(self):
        """Create CSV file with headers."""
        headers = [
            'Model_Name', 'Epoch', 'Train_Loss', 'Val_Loss', 'Val_mIoU', 
            'Epoch_Duration_sec', 'Total_Time_Elapsed_sec', 'Inference_Latency_ms', 'Notes_and_Errors'
        ]
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
    
    def log_epoch(self, model_name, epoch, train_loss, val_loss, val_miou, 
                  epoch_duration_sec, total_time_sec, inference_latency_ms=None, notes=''):
        """Log metrics for a single epoch."""
        row = {
            'Model_Name': model_name,
            'Epoch': epoch,
            'Train_Loss': round(train_loss, 6) if train_loss is not None else '',
            'Val_Loss': round(val_loss, 6) if val_loss is not None else '',
            'Val_mIoU': round(val_miou, 6) if val_miou is not None else '',
            'Epoch_Duration_sec': round(epoch_duration_sec, 2),
            'Total_Time_Elapsed_sec': round(total_time_sec, 2),
            'Inference_Latency_ms': round(inference_latency_ms, 2) if inference_latency_ms is not None else '',
            'Notes_and_Errors': notes
        }
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    def log_inference_final(self, model_name, inference_latency_ms):
        """Log final inference latency as a summary row."""
        row = {
            'Model_Name': model_name,
            'Epoch': 'FINAL',
            'Train_Loss': '',
            'Val_Loss': '',
            'Val_mIoU': '',
            'Epoch_Duration_sec': '',
            'Total_Time_Elapsed_sec': '',
            'Inference_Latency_ms': round(inference_latency_ms, 2),
            'Notes_and_Errors': 'Final inference latency measurement'
        }
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
