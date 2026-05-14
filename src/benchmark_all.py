import torch
import time
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
RESNET_WEIGHTS = Path('runs/phase2_baseline_run3/resnet50_Run3_Epoch5.pth')
YOLO_WEIGHTS = Path('runs/segment/runs/segment/train_trimmed/weights/best.pt')
VAL_DIR = Path('yolo_cityscapes_trimmed/images/val')
IMGSZ = 512
N_WARMUP = 5
N_MEASURE = 100
# ---------------------------------------------------------------------------

def load_resnet(weights_path, device):
    model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=19)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def load_yolo(weights_path, device):
    model = YOLO(str(weights_path))
    # Access the raw PyTorch model inside the Ultralytics wrapper
    return model.model.to(device).eval()

def measure_model(model, inputs, device, name):
    print(f"  [{name}] Running {N_WARMUP} warm-up passes...")
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(inputs[0])
            torch.cuda.synchronize()
    
    print(f"  [{name}] Measuring {N_MEASURE} forward passes...")
    latencies = []
    with torch.no_grad():
        for i in range(min(len(inputs), N_MEASURE)):
            x = inputs[i]
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
            
    avg = np.mean(latencies)
    std = np.std(latencies)
    return avg, std

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Collect validation images
    val_images = sorted(list(VAL_DIR.rglob('*.png')))[:N_MEASURE]
    if not val_images:
        print("No validation images found.")
        return
        
    # Pre-process inputs (OUTSIDE timing window)
    print(f"Pre-processing {len(val_images)} images...")
    transform = transforms.Compose([
        transforms.Resize((IMGSZ, IMGSZ)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inputs = []
    for img_path in val_images:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        inputs.append(tensor)

    # Benchmarking
    results = {}
    
    if RESNET_WEIGHTS.exists():
        resnet_model = load_resnet(RESNET_WEIGHTS, device)
        avg, std = measure_model(resnet_model, inputs, device, "ResNet50")
        results["FCN-ResNet50"] = (avg, std)
    else:
        print(f"ResNet weights not found at {RESNET_WEIGHTS}")

    if YOLO_WEIGHTS.exists():
        yolo_model = load_yolo(YOLO_WEIGHTS, device)
        avg, std = measure_model(yolo_model, inputs, device, "YOLOv8n-seg")
        results["YOLOv8n-seg"] = (avg, std)
    else:
        print(f"YOLO weights not found at {YOLO_WEIGHTS}")

    # Summary
    print("\n" + "="*50)
    print("MASTER LATENCY COMPARISON (GPU Forward-Pass)")
    print("="*50)
    print(f"{'Model':<20} | {'Latency (ms)':<15} | {'Std Dev':<10}")
    print("-" * 50)
    for model, (avg, std) in results.items():
        print(f"{model:<20} | {avg:>12.2f} ms | {std:>8.2f}")
    print("="*50)
    
    if len(results) == 2:
        r_lat = results["FCN-ResNet50"][0]
        y_lat = results["YOLOv8n-seg"][0]
        speedup = r_lat / y_lat
        print(f"YOLO is {speedup:.2f}x faster than ResNet50 baseline.")
    print("="*50)

if __name__ == "__main__":
    main()
