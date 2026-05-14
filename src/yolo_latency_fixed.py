"""
yolo_latency_fixed.py
=====================
Fair GPU-synchronized inference latency measurement for YOLOv8n-seg.

Methodology matches train_baseline.py exactly:
  - Image I/O and preprocessing are OUTSIDE the timed window
  - torch.cuda.synchronize() brackets the raw model forward pass ONLY
  - 5 warm-up passes are run before measurement begins
  - No NMS, no Ultralytics Results object construction inside the timed window

This produces an apples-to-apples comparison against the ResNet50 baseline
whose latency was measured the same way (forward-pass only, GPU-sync'd).

Run from the project root with the ML_Project venv active:
    python src\\yolo_latency_fixed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration — edit these if your weights or dataset path differs
# ---------------------------------------------------------------------------
WEIGHTS_PATH = Path("runs/segment/runs/segment/train_trimmed/weights/best.pt")
YOLO_VAL_ROOT = Path("yolo_cityscapes_trimmed/images/val")
IMGSZ = 512
N_WARMUP = 5
N_MEASURE = 100
MODEL_NAME_FIXED = "yolov8n-seg"
# ---------------------------------------------------------------------------


def load_yolo_backbone(weights: Path, device: torch.device):
    """Load YOLOv8 weights and return the raw nn.Module backbone."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics not installed. Run: pip install ultralytics") from exc

    wrapper = YOLO(str(weights))
    backbone = wrapper.model
    backbone.eval()
    backbone.to(device)
    return backbone


def preprocess_image(img_path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    """Load and preprocess a single image into a normalised (1,3,H,W) tensor."""
    img = Image.open(img_path).convert("RGB").resize((imgsz, imgsz))
    arr = np.array(img, dtype=np.float32) / 255.0          # [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)         # CHW
    tensor = tensor.unsqueeze(0).to(device)                 # BCHW
    return tensor


def collect_val_images(val_root: Path, n: int = 100) -> list[Path]:
    images = sorted(val_root.rglob("*.png"))
    if not images:
        raise FileNotFoundError(f"No .png files found under {val_root}")
    return images[:n]


def run_warmup(backbone, device: torch.device, imgsz: int, n: int = 5) -> None:
    """Run N dummy forward passes to warm up cuDNN and JIT before measurement."""
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        for _ in range(n):
            backbone(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  Completed {n} GPU warm-up passes.")


def measure_latency(
    backbone,
    val_images: list[Path],
    imgsz: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    Measure GPU forward-pass latency (ms) over val_images.

    Returns (mean_ms, std_ms).
    """
    timings: list[float] = []

    with torch.no_grad():
        for img_path in val_images:
            # I/O and preprocessing are OUTSIDE the timed window
            x = preprocess_image(img_path, imgsz, device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            backbone(x)  # raw forward pass — no NMS, no post-processing

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            timings.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    return mean_ms, std_ms


def append_fixed_latency_to_csv(model_name: str, latency_ms: float) -> None:
    """Append a FINAL row into the master metrics CSV."""
    import csv

    csv_path = Path("Results_Comparison/training_metrics_log.csv")
    if not csv_path.exists():
        print(f"  WARNING: CSV not found at {csv_path}. Skipping append.")
        return

    row = {
        "Model_Name": model_name,
        "Epoch": "FINAL",
        "Train_Loss": "",
        "Val_Loss": "",
        "Val_mIoU": "",
        "Epoch_Duration_sec": "",
        "Total_Time_Elapsed_sec": "",
        "Inference_Latency_ms": round(latency_ms, 2),
        "Notes_and_Errors": (
            "Forward-pass only (GPU-sync'd, no NMS, "
            f"{N_WARMUP} warmup passes, {N_MEASURE} images)"
        ),
    }
    fieldnames = [
        "Model_Name", "Epoch", "Train_Loss", "Val_Loss", "Val_mIoU",
        "Epoch_Duration_sec", "Total_Time_Elapsed_sec",
        "Inference_Latency_ms", "Notes_and_Errors",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
    print(f"  Appended fixed latency row to {csv_path}")


def print_comparison_table(yolo_old_ms: float, yolo_fixed_ms: float, resnet_ms: float) -> None:
    """Print a comparison summary to stdout."""
    print("\n" + "=" * 65)
    print("  LATENCY COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<35} {'Latency (ms)':>12}  {'Method'}")
    print(f"  {'-'*35} {'-'*12}  {'-'*20}")
    print(f"  {'FCN-ResNet50 (baseline)':<35} {resnet_ms:>12.2f}  GPU-sync'd forward-pass")
    print(f"  {'YOLOv8n-seg (old wall-clock)':<35} {yolo_old_ms:>12.2f}  model.predict() end-to-end")
    print(f"  {'YOLOv8n-seg (fixed, forward-pass)':<35} {yolo_fixed_ms:>12.2f}  GPU-sync'd forward-pass")
    print("=" * 65)
    ratio = yolo_fixed_ms / resnet_ms if resnet_ms > 0 else float("inf")
    faster = resnet_ms / yolo_fixed_ms if yolo_fixed_ms > 0 else float("inf")
    if yolo_fixed_ms < resnet_ms:
        print(f"  ✅ YOLO is {faster:.1f}x FASTER than ResNet50 (edge hypothesis VALIDATED)")
    else:
        print(f"  ⚠️  YOLO is {ratio:.1f}x slower than ResNet50 — check GPU utilisation")
    print("=" * 65 + "\n")


def main() -> None:
    # ---- sanity checks ----
    if not WEIGHTS_PATH.exists():
        sys.exit(
            f"ERROR: Weights not found at {WEIGHTS_PATH}\n"
            "Adjust WEIGHTS_PATH at the top of this script."
        )
    if not YOLO_VAL_ROOT.exists():
        sys.exit(
            f"ERROR: Val image root not found at {YOLO_VAL_ROOT}\n"
            "Adjust YOLO_VAL_ROOT at the top of this script."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[yolo_latency_fixed] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading weights from: {WEIGHTS_PATH}")
    backbone = load_yolo_backbone(WEIGHTS_PATH, device)

    print(f"\nCollecting up to {N_MEASURE} validation images from: {YOLO_VAL_ROOT}")
    val_images = collect_val_images(YOLO_VAL_ROOT, n=N_MEASURE)
    print(f"  Found {len(val_images)} images.")

    print(f"\nRunning {N_WARMUP} GPU warm-up passes...")
    run_warmup(backbone, device, IMGSZ, n=N_WARMUP)

    print(f"\nMeasuring forward-pass latency over {len(val_images)} images...")
    mean_ms, std_ms = measure_latency(backbone, val_images, IMGSZ, device)
    print(f"  Mean latency : {mean_ms:.2f} ms")
    print(f"  Std dev      : {std_ms:.2f} ms")

    append_fixed_latency_to_csv(MODEL_NAME_FIXED, mean_ms)

    # Reference numbers from training_metrics_log.csv
    RESNET_MS = 39.76
    YOLO_OLD_MS = 76.71
    print_comparison_table(YOLO_OLD_MS, mean_ms, RESNET_MS)


if __name__ == "__main__":
    main()
