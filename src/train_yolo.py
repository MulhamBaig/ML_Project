from __future__ import annotations

import argparse
import time
from pathlib import Path

from append_yolo_metrics import append_yolo_results
from metrics_logger import MetricsLogger

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Ultralytics is not installed. Install it with: python -m pip install ultralytics"
    ) from exc


def measure_inference_latency(model: YOLO, image_paths: list[Path], device: int | str, imgsz: int, count: int = 100) -> float:
    timings: list[float] = []
    for image_path in image_paths[:count]:
        start = time.time()
        model.predict(source=str(image_path), imgsz=imgsz, device=device, verbose=False)
        end = time.time()
        timings.append((end - start) * 1000.0)
    return sum(timings) / len(timings) if timings else 0.0


def collect_validation_images(yolo_root: Path, split: str = "val") -> list[Path]:
    image_root = yolo_root / "images" / split
    return sorted(image_root.rglob("*.png"))


def clear_yolo_cache_files(dataset_root: Path) -> int:
    removed = 0
    for cache_file in dataset_root.rglob("*.cache"):
        try:
            cache_file.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on Cityscapes and log metrics into the master CSV.")
    parser.add_argument("--data", default="cityscapes.yaml", help="Path to the YOLO data YAML file")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics segmentation checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default=0)
    parser.add_argument("--batch", type=int, default=4, help="batch size for training")
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="train")
    parser.add_argument("--yolo-root", default="yolo_cityscapes")
    parser.add_argument("--run-name", default="yolov8n-seg")
    parser.add_argument("--skip-train", action="store_true", help="Skip the actual training call for a configuration-only dry run")
    parser.add_argument("--skip-latency", action="store_true", help="Skip the post-training latency measurement (use yolo_latency_fixed.py for a fair GPU-sync'd measurement instead)")
    args = parser.parse_args()

    model = YOLO(args.model)

    yolo_root = Path(args.yolo_root)
    removed_caches = clear_yolo_cache_files(yolo_root)
    if removed_caches:
        print(f"Removed {removed_caches} stale YOLO cache file(s) from {yolo_root}")

    train_results = None
    if not args.skip_train:
        train_results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            batch=args.batch,
            exist_ok=True,
        )

    results_csv = Path(args.project) / args.name / "results.csv"
    if train_results is not None and hasattr(train_results, "save_dir"):
        results_csv = Path(train_results.save_dir) / "results.csv"
    if results_csv.exists():
        append_yolo_results(results_csv, model_name=args.run_name, notes="Imported from YOLO results.csv")

    if args.skip_latency:
        print("Skipping wall-clock latency measurement (--skip-latency set).")
        print("Run src/yolo_latency_fixed.py for a GPU-sync'd forward-pass latency measurement.")
    else:
        val_images = collect_validation_images(yolo_root, split="val")
        if val_images:
            avg_latency_ms = measure_inference_latency(model, val_images, device=args.device, imgsz=args.imgsz, count=100)
            print(f"Average inference latency on 100 validation images: {avg_latency_ms:.2f} ms")
            MetricsLogger().log_inference_final(args.run_name, avg_latency_ms)
        else:
            print("No validation images were found for latency measurement.")


if __name__ == "__main__":
    main()
