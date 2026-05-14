# Semantic Segmentation for Low-Altitude Autonomous Drone Navigation

A deep learning project for semantic segmentation of urban street scenes to enable autonomous drone navigation at truck height in cities. This project uses the **Cityscapes dataset** and PyTorch with GPU acceleration.

## Project Overview

This repository contains:
- **Dataset**: Cityscapes fine annotations and RGB images
- **Model**: Semantic segmentation neural network (19 classes)
- **Training**: GPU-accelerated training on NVIDIA hardware
- **Purpose**: Enable autonomous drone navigation in urban environments at truck-level altitude

---

## Prerequisites

Before starting, ensure you have:

1. **Python 3.8+** installed on your system
2. **Git** installed and configured
3. **NVIDIA GPU** (RTX 3060 or equivalent) with:
   - NVIDIA drivers installed (version 596+)
   - CUDA toolkit installed (version 12.1+)
   - cuDNN library (optional but recommended)
4. **At least 50GB free disk space** for the Cityscapes dataset
5. Internet connection to download dataset files (~100GB)

### Check Your GPU Setup

Open PowerShell and run:
```powershell
nvidia-smi
```

This should display your GPU model and driver version.

---

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

> **Note**: If you see an execution policy error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
> ```

### 3. Upgrade pip and Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch with GPU Support

Choose the command that matches your CUDA version:

**For CUDA 13.2:**
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

**For CUDA 12.4:**
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**For CPU-only (no GPU):**
```powershell
python -m pip install torch torchvision torchaudio
```

### 5. Verify GPU Support

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'None')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA RTX 3060 Laptop GPU
```

---

## Dataset Setup

The Cityscapes dataset provides semantic segmentation labels for urban street scenes.

### Download the Cityscapes Dataset

1. Visit [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
2. Create a free account and login
3. Download these files:
   - `gtFine_trainvaltest.zip` (fine annotations - ~240MB)
   - `leftImg8bit_trainvaltest.zip` (RGB images - ~50GB)

### Organize the Dataset

After downloading, organize the files as follows:

```
ML_Project/
├── gtFine_trainvaltest/
│   ├── gtFine/
│   │   ├── train/          (2975 images with labels)
│   │   ├── val/            (500 images with labels)
│   │   └── test/           (1525 images with labels)
│   ├── leftImg8bit/        (RGB images - MUST be in same root as gtFine/)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── license.txt
│   └── README
```

**Steps to organize:**

1. Extract `gtFine_trainvaltest.zip` in the project root:
   ```powershell
   cd ML_Project
   Expand-Archive -Path gtFine_trainvaltest.zip -DestinationPath .
   ```

2. Extract `leftImg8bit_trainvaltest.zip` separately, then move `leftImg8bit` into `gtFine_trainvaltest`:
   ```powershell
   Expand-Archive -Path leftImg8bit_trainvaltest.zip -DestinationPath .\temp
   Move-Item -Path .\temp\leftImg8bit -Destination .\gtFine_trainvaltest\leftImg8bit
   Remove-Item -Path .\temp -Recurse
   ```

3. Set the environment variable:
   ```powershell
   [Environment]::SetEnvironmentVariable("CITYSCAPES_DATASET", "C:\path\to\ML_Project\gtFine_trainvaltest", "User")
   ```

---

## Project Structure

```
ML_Project/
├── .git/                          # Git repository
├── .gitignore                     # Files to exclude from Git
├── ML_Project/                    # Python virtual environment
├── cityscapesScripts/             # Cityscapes tools (not in Git)
├── gtFine_trainvaltest/           # Dataset (not in Git)
│   ├── gtFine/
│   ├── leftImg8bit/
│   ├── license.txt
│   └── README
├── MyLogs.txt                     # Project progress log
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Running the Project

### Activate the Virtual Environment

Every time you start work, activate the virtual environment:

```powershell
.\ML_Project\Scripts\Activate.ps1
```

You should see `(ML_Project)` in your PowerShell prompt.

### Verify Dataset

Check that the dataset is properly organized:

```powershell
python -c "import os; print('Files in gtFine_trainvaltest:'); print([x for x in os.listdir('gtFine_trainvaltest') if os.path.isdir(os.path.join('gtFine_trainvaltest', x)) or x.endswith('.txt')])"
```

Expected output:
```
Files in gtFine_trainvaltest:
['gtFine', 'leftImg8bit', 'license.txt', 'README']
```

### View Dataset with Cityscapes Viewer

```powershell
csViewer
```

A GUI window will open showing sample images and their semantic labels.

### Next Steps: Training

Once the dataset is verified:

1. **Build Data Loader**: Create PyTorch `DataLoader` for train/val/test splits
2. **Implement Model**: Define semantic segmentation architecture
3. **Train Model**: Run training loop with GPU acceleration
4. **Evaluate**: Validate on test set and compute metrics (mIoU, pixel accuracy)
5. **Visualize**: Generate color-coded prediction maps

---

## Troubleshooting

### CUDA Not Available

```
CUDA available: False
```

**Solution**: Reinstall PyTorch with the correct CUDA version matching your system:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
```

### Dataset Files Not Found

```
Error: Data was not found...CITYSCAPES_DATASET
```

**Solution**: 
1. Verify environment variable is set: `$env:CITYSCAPES_DATASET`
2. Check folder structure matches the organization steps above
3. Ensure `gtFine/` and `leftImg8bit/` are in the same root directory

### Virtual Environment Not Activating

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

Then try activating again:
```powershell
.\ML_Project\Scripts\Activate.ps1
```

### Out of Memory (OOM) During Training

Reduce batch size in training configuration:
```python
batch_size = 4  # Reduce from default if needed
```

---

## Cityscapes Dataset Info

- **Classes**: 19 semantic classes (car, person, road, sky, etc.)
- **Training Split**: ~2,975 images (1024×2048 resolution)
- **Validation Split**: 500 images
- **Test Split**: 1,525 images
- **Annotation Format**: Polygon JSON → Converted to label ID PNG images
- **Label Format**: Pixel values 0-18 represent class IDs, 255 = ignored/void class

---

## Useful Commands

```powershell
# Activate virtual environment
.\ML_Project\Scripts\Activate.ps1

# Install/update dependencies
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

---

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

---

## Resources

- **Cityscapes Dataset**: https://www.cityscapes-dataset.com/
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **Semantic Segmentation**: https://en.wikipedia.org/wiki/Image_segmentation
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-toolkit

---

## Notes

- **Dataset Size**: The full Cityscapes dataset is ~50GB. Ensure you have sufficient disk space.
- **Training Time**: Full training may take several hours on GPU. Consider using a smaller subset first.
- **GPU Memory**: RTX 3060 has 6GB VRAM. Adjust batch size if you encounter memory errors.
- **Reproducibility**: Set random seeds for reproducible results during training.

---

## Author

**Mulham Baig**  
GitHub: https://github.com/MulhamBaig

---

## Phase 1 Results (Pre-Flight Check)

- Date: 2026-05-14
- Dataset root `gtFine_trainvaltest` verified to contain both `gtFine/` and `leftImg8bit/`.
- `_labelTrainIds.png` mask counts: `train` = 2975, `val` = 500.
- PyTorch + CUDA verified on the active `ML_Project` venv:
   - `cuda_available`: True
   - GPU: NVIDIA GeForce RTX 3060 Laptop GPU
- Environment locked: `requirements.txt` updated from `pip freeze` in the active venv.

These checks confirm the dataset and GPU environment are ready for model development.

---

## Phase 2 Results (Baseline FCN-ResNet50)

- Date: 2026-05-14
- Implemented: `src/cityscapes_dataset.py`, `src/metrics_logger.py`, `src/train_baseline.py`
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
- Model checkpoints: `runs/phase2_baseline_run1/resnet50_Run1_Epoch{1-5}.pth` (ignored by Git)
- Central metrics log: `Results_Comparison/training_metrics_log.csv`

Tracking notes:
- Every epoch logs train loss, validation mIoU, epoch duration, total elapsed time, and any error notes.
- The final row logs the final 100-image inference latency so baseline and YOLO runs can be compared in one table.

Summary: The baseline FCN-ResNet50 pipeline completed a short verification run successfully. The model shows modest mIoU as expected for a small subset and short schedule. Use the full training set and longer schedules for production training.

---

## Phase 3 Setup (YOLOv8-Seg)

Phase 3 prepares the Cityscapes data for YOLOv8 segmentation so we can compare a fast drone-friendly model against the PyTorch baseline.

### Files added for Phase 3

- `src/yolo_cityscapes_utils.py` - shared Cityscapes label helpers and polygon normalization
- `src/prepare_yolo_cityscapes.py` - converts Cityscapes JSON polygons into YOLO segmentation `.txt` files
- `src/append_yolo_metrics.py` - appends Ultralytics `results.csv` rows into `Results_Comparison/training_metrics_log.csv`
- `src/train_yolo.py` - YOLOv8 training entrypoint with latency measurement and CSV logging
- `cityscapes.yaml` - YOLO dataset definition file

### How the YOLO dataset is organized

The conversion script creates a lightweight YOLO dataset root called `yolo_cityscapes/`.

- `yolo_cityscapes/images/train` and `yolo_cityscapes/images/val` are Windows directory junctions that point back to the original Cityscapes images.
- `yolo_cityscapes/labels/train` and `yolo_cityscapes/labels/val` contain the generated YOLO polygon `.txt` files.
- This avoids duplicating the 50GB image set while still giving YOLO the folder structure it expects.

### What was verified

I ran a dry run of the converter on one train sample and one val sample.

- Junctions were created successfully for `yolo_cityscapes/images/train` and `yolo_cityscapes/images/val`.
- Example label files were written in YOLO segmentation format.
- `cityscapes.yaml` was written and points to the new YOLO dataset root.

### YOLO training flow

1. Convert Cityscapes polygons:
   ```powershell
   .\ML_Project\Scripts\python.exe src\prepare_yolo_cityscapes.py --dataset-root gtFine_trainvaltest --yolo-root yolo_cityscapes --splits train val --write-yaml
   ```
2. Train YOLOv8 segmentation:
   ```powershell
   .\ML_Project\Scripts\python.exe src\train_yolo.py --data cityscapes.yaml --model yolov8n-seg.pt --epochs 20 --imgsz 512 --device 0
   ```
3. Append YOLO metrics into the shared CSV log:
   - The training script already reads `runs/segment/train/results.csv` and appends the values into `Results_Comparison/training_metrics_log.csv`.

### Important note about YOLO metrics

Ultralytics reports segmentation metrics in its own CSV format. The parser maps the mask metric columns from `results.csv` into the shared training log so the baseline and YOLO results can live in one table for later comparison.

---

**Last Updated**: 2026-05-14  
**Status**: Phase 1 complete; Phase 2 baseline verification and metrics logging finished; Phase 3 setup in progress.
