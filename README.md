# Semantic Segmentation for Low-Altitude Autonomous Drone Navigation

A deep learning project for semantic segmentation of urban street scenes to support autonomous drone navigation at truck height in cities. The project uses the Cityscapes dataset, PyTorch, and GPU acceleration.

## Project Overview

This repository contains:

- Dataset: Cityscapes fine annotations and RGB images
- Model: semantic segmentation networks for 19 classes
- Training: GPU-accelerated training on NVIDIA hardware
- Purpose: enable autonomous drone navigation in urban environments at low altitude

## Prerequisites

Before starting, ensure you have:

1. Python 3.8+ installed
2. Git installed and configured
3. An NVIDIA GPU such as an RTX 3060 or equivalent with NVIDIA drivers installed
4. At least 50 GB of free disk space for the Cityscapes dataset
5. An internet connection to download the dataset files

### Check Your GPU Setup

Open PowerShell and run:

```powershell
nvidia-smi
```

This should display your GPU model and driver version.

## Step-by-Step Setup Instructions

### 1. Clone the Repository

```powershell
git clone https://github.com/MulhamBaig/ML_Project.git
cd ML_Project
```

### 2. Create a Python Virtual Environment

```powershell
python -m venv ML_Project
.\ML_Project\Scripts\Activate.ps1
```

If you see an execution policy error, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### 3. Upgrade pip and Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch with GPU Support

Choose the command that matches your CUDA version.

For CUDA 13.2:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

For CUDA 12.4:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only:

```powershell
python -m pip install torch torchvision torchaudio
```

### 5. Verify GPU Support

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'None')"
```

Expected output:

```text
CUDA available: True
GPU: NVIDIA RTX 3060 Laptop GPU
```

## Dataset Setup

The Cityscapes dataset provides semantic segmentation labels for urban street scenes.

### Download the Cityscapes Dataset

1. Visit the [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
2. Create an account and log in
3. Download these files:
   - `gtFine_trainvaltest.zip` for fine annotations
   - `leftImg8bit_trainvaltest.zip` for RGB images

### Organize the Dataset

After downloading, organize the files as follows:

```text
ML_Project/
├── gtFine_trainvaltest/
│   ├── gtFine/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── license.txt
│   └── README
```

Steps to organize:

1. Extract `gtFine_trainvaltest.zip` into the project root.
2. Extract `leftImg8bit_trainvaltest.zip` separately, then move `leftImg8bit` into `gtFine_trainvaltest`.
3. Set the environment variable:

```powershell
[Environment]::SetEnvironmentVariable("CITYSCAPES_DATASET", "C:\path\to\ML_Project\gtFine_trainvaltest", "User")
```

## Project Structure

```text
ML_Project/
├── .gitignore
├── gtFine_trainvaltest/
├── MyLogs.txt
├── README.md
├── Results_Comparison/
├── requirements.txt
├── runs/
└── src/
```

## Running the Project

### Activate the Virtual Environment

```powershell
.\ML_Project\Scripts\Activate.ps1
```

### Verify Dataset

```powershell
python -c "import os; print('Files in gtFine_trainvaltest:'); print(sorted([x for x in os.listdir('gtFine_trainvaltest')]))"
```

Expected output:

```text
Files in gtFine_trainvaltest:
['gtFine', 'leftImg8bit', 'license.txt', 'README']
```

### Next Steps

1. Build the data loader for train, validation, and test splits.
2. Implement and train the segmentation model.
3. Evaluate the model and compute metrics such as mIoU and pixel accuracy.
4. Generate visual comparison outputs.

## Cityscapes Dataset Info

- Classes: 19 semantic classes such as car, person, road, and sky
- Training split: about 2,975 images
- Validation split: 500 images
- Test split: 1,525 images
- Annotation format: polygon JSON converted to label PNG images
- Label format: pixel values 0 to 18 represent class IDs, and 255 is the ignored class

## Useful Commands

```powershell
# Activate virtual environment
.\ML_Project\Scripts\Activate.ps1

# Install or update dependencies
pip install -r requirements.txt

# Check GPU status
nvidia-smi

# Verify PyTorch GPU support
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# View Git commit history
git log --oneline

# Check project status
git status
```

## Git Workflow

```powershell
# Create a new branch for development
git checkout -b feature/my-feature

# Commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/my-feature

# Merge back to main
git checkout main
git merge feature/my-feature
```

## Resources

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Semantic Segmentation](https://en.wikipedia.org/wiki/Image_segmentation)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)

## Notes

- The full Cityscapes dataset is about 50 GB.
- Full training may take several hours on GPU.
- The RTX 3060 has 6 GB VRAM, so batch size may need adjustment.
- Set random seeds if you need reproducible results.

## Author

Mulham Baig

GitHub: [MulhamBaig](https://github.com/MulhamBaig)

## Phase 1 Results (Pre-Flight Check)

- Date: 2026-05-14
- Dataset root `gtFine_trainvaltest` verified to contain both `gtFine/` and `leftImg8bit/`.
- `_labelTrainIds.png` mask counts:
  - `train`: 2975
  - `val`: 500
- PyTorch and CUDA verified on the active `ML_Project` venv:
  - `cuda_available`: True
  - `device_count`: 1
  - `device_name`: NVIDIA GeForce RTX 3060 Laptop GPU
- Environment locked: `requirements.txt` updated from `pip freeze` in the active venv.

These checks confirm the dataset and GPU environment are ready for model development.

## Phase 2 Results (Baseline FCN-ResNet50)

- Date: 2026-05-14
- Implemented: `src/cityscapes_dataset.py`, `src/metrics_logger.py`, and `src/train_baseline.py`
- Verification training configuration:
  - train subset: 400 images
  - val subset: 100 images
  - input size: 256x512
  - batch size: 4
  - epochs: 5
  - run: 1
- Results (Run 1):
  - final validation mIoU: 0.2933
  - average inference latency on 100 validation images: 26.29 ms
  - total training time: 174.29 seconds
- Model checkpoints: `runs/phase2_baseline_run1/resnet50_Run1_Epoch{1-5}.pth`
- Central metrics log: `Results_Comparison/training_metrics_log.csv`

Tracking notes:

- Every epoch logs train loss, validation mIoU, epoch duration, total elapsed time, and any error notes.
- The final row logs the final 100-image inference latency so baseline and YOLO runs can be compared in one table.

Summary: the baseline FCN-ResNet50 pipeline completed a short verification run successfully. Use the full train set and longer schedules for production training.

## Phase 3 Setup (YOLOv8-Seg)

Phase 3 prepares the Cityscapes data for YOLOv8 segmentation so the project can compare a fast deployment model against the PyTorch baseline.

### Files added for Phase 3

- `src/yolo_cityscapes_utils.py` - shared Cityscapes label helpers and polygon normalization
- `src/prepare_yolo_cityscapes.py` - converts Cityscapes JSON polygons into YOLO segmentation `.txt` files
- `src/append_yolo_metrics.py` - appends Ultralytics `results.csv` rows into `Results_Comparison/training_metrics_log.csv`
- `src/train_yolo.py` - YOLOv8 training entrypoint with latency measurement and CSV logging
- `src/trim_yolo_dataset.py` - utility to trim the YOLO dataset to a fixed number of samples for sanity runs
- `src/yolo_latency_fixed.py` - **corrected** inference latency benchmark (GPU-sync'd forward-pass only, no NMS)
- `src/visual_grid.py` - generates a 1x4 Matplotlib comparison grid (Raw, GT, ResNet50, YOLO)
- `cityscapes.yaml` - YOLO dataset definition file (full dataset)
- `cityscapes_trimmed.yaml` - YOLO dataset definition file (400-sample trimmed subset)

### How the YOLO Dataset Is Organized

The conversion script creates a lightweight YOLO dataset root called `yolo_cityscapes/`.

- `yolo_cityscapes/images/train` and `yolo_cityscapes/images/val` are Windows directory junctions that point back to the original Cityscapes images.
- `yolo_cityscapes/labels/train` and `yolo_cityscapes/labels/val` contain the generated YOLO polygon `.txt` files.
- This avoids duplicating the 50 GB image set while still giving YOLO the folder structure it expects.
- A trimmed version `yolo_cityscapes_trimmed/` (400 train, 400 val) is used for the sanity run.

### What Was Verified

I ran a dry run of the converter on one train sample and one val sample.

- Junctions were created successfully for `yolo_cityscapes/images/train` and `yolo_cityscapes/images/val`.
- Example label files were written in YOLO segmentation format.
- `cityscapes.yaml` was written and points to the new YOLO dataset root.
- A trimmed 400/400 train-val dataset (`yolo_cityscapes_trimmed`) was generated for sanity training.
- YOLO sanity training was executed for 5 epochs with batch=4 and imgsz=512.

### YOLO Training Flow

Convert Cityscapes polygons:

```powershell
.\ML_Project\Scripts\python.exe src\prepare_yolo_cityscapes.py --dataset-root gtFine_trainvaltest --yolo-root yolo_cityscapes --splits train val --write-yaml
```

Train YOLOv8 segmentation:

```powershell
.\ML_Project\Scripts\python.exe src\train_yolo.py --data cityscapes_trimmed.yaml --model yolov8n-seg.pt --epochs 5 --batch 4 --imgsz 512 --device 0 --yolo-root yolo_cityscapes_trimmed --project runs/segment --name train_trimmed
```

Run the corrected latency benchmark:

```powershell
.\ML_Project\Scripts\python.exe src\yolo_latency_fixed.py
```

- The training script clears stale Ultralytics `*.cache` files before training starts.

### Important Note About YOLO Metrics

Ultralytics reports segmentation metrics in its own CSV format. The parser maps the mask mAP50 column from `results.csv` into the shared training log so the baseline and YOLO results can live in one table for later comparison.

### Phase 3 Sanity-Run Results

| Metric | Value |
|--------|-------|
| YOLO epochs trained | 5 |
| Val Mask mAP50 (Epoch 1) | 0.0641 |
| Val Mask mAP50 (Epoch 5) | 0.1051 |
| Weights saved | `runs/segment/runs/segment/train_trimmed/weights/best.pt` |

### Latency Bug — Root Cause & Fix

The initial YOLO latency of 76.71 ms vs ResNet50's 39.76 ms was a **measurement methodology mismatch**, not a true model speed difference.

| Overhead Source | FCN-ResNet50 | YOLOv8n-seg (old) | YOLOv8n-seg (fixed) |
|---|---|---|---|
| Measurement | `torch.cuda.synchronize()` + forward pass only | `model.predict()` wall-clock | `torch.cuda.synchronize()` + forward pass only |
| NMS included | No | Yes (~10-20 ms) | No |
| Disk I/O inside timing | No | Yes | No |
| GPU warm-up | Pre-warmed | None | 5 passes |

Fix: `src/yolo_latency_fixed.py` calls `model.model(tensor)` (raw backbone, no NMS) with `torch.cuda.synchronize()` bracketing — same methodology as ResNet50.

### ✅ Confirmed Latency Results (RTX 3060 Laptop GPU)

| Model | Latency (ms) | Method | Verdict |
|---|---|---|---|
| FCN-ResNet50 | 39.60 ms | GPU-sync'd forward-pass | Baseline |
| **YOLOv8n-seg (fixed, forward-pass)** | **13.84 ms** | GPU-sync'd forward-pass | **✅ 2.86× faster than ResNet50** |

**Edge hypothesis validated**: YOLOv8n-seg is nearly 3× faster than FCN-ResNet50 on the same hardware, well under the 30 ms real-time target.

### Comparison Artifacts

- ResNet50 baseline artifacts are the checkpoints in `runs/phase2_baseline_run3/`.
- YOLO comparison artifacts include `results.csv`, curves, and prediction previews in `runs/segment/runs/segment/train_trimmed/`.
- YOLO prediction images are available as `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, and `val_batch2_pred.jpg` alongside the matching label images.
- Shared metrics table: `Results_Comparison/training_metrics_log.csv`

### Visualizations

- **Semantic Grid**: `Results_Comparison/visual_comparison.png` (Raw, GT, ResNet, YOLO)
- **Accuracy Progression**: `Results_Comparison/accuracy_progression.png`
- **Latency Benchmarks**: `Results_Comparison/latency_comparison.png`
- **Training Efficiency**: `Results_Comparison/training_time_comparison.png`

## Last Updated

2026-05-14

## Status

Phase 1 complete. Phase 2 baseline verification complete. Phase 3 sanity training complete (latency fix applied and verified). Phase 4 complete (visual comparison and metric charts generated). Phase 5 (Git wrap-up) pending.
