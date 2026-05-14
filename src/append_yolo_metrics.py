from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable

from metrics_logger import MetricsLogger


def _get_float(row: Dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def append_yolo_results(results_csv: Path, model_name: str, notes: str = "Imported from YOLO results.csv") -> int:
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find YOLO results file: {results_csv}")

    logger = MetricsLogger()
    rows_written = 0

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epoch = int(float(row.get("epoch", rows_written))) + 1
            train_loss = sum(
                value or 0.0
                for value in [
                    _get_float(row, "train/box_loss"),
                    _get_float(row, "train/seg_loss"),
                ]
            )
            val_loss = sum(
                value or 0.0
                for value in [
                    _get_float(row, "val/box_loss"),
                    _get_float(row, "val/seg_loss"),
                ]
            )
            yolo_mask_metric = _get_float(row, "metrics/mAP50-95(M)", "metrics/mAP50(M)")
            logger.log_epoch(
                model_name=model_name,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_miou=yolo_mask_metric,
                epoch_duration_sec=0.0,
                total_time_sec=0.0,
                inference_latency_ms=None,
                notes=notes,
            )
            rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Append YOLO results.csv rows to the master metrics log.")
    parser.add_argument("results_csv", help="Path to Ultralytics results.csv")
    parser.add_argument("--model-name", default="yolov8n-seg", help="Model name stored in the master log")
    parser.add_argument("--notes", default="Imported from YOLO results.csv", help="Notes to store in the master log")
    args = parser.parse_args()

    rows_written = append_yolo_results(Path(args.results_csv), args.model_name, notes=args.notes)
    print(f"Appended {rows_written} YOLO rows to Results_Comparison/training_metrics_log.csv")


if __name__ == "__main__":
    main()
