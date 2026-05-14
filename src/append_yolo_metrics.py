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
            # Normalize header keys by stripping whitespace
            normalized = {k.strip(): (v if v is not None else "") for k, v in row.items()}

            # determine epoch
            epoch = int(float(normalized.get("epoch", rows_written))) + 1

            # aggregate train/val loss if present
            train_loss = sum(
                value or 0.0
                for value in [
                    _get_float(normalized, "train/box_loss", "train/seg_loss", "train/obj_loss", "train/cls_loss"),
                ]
            )
            val_loss = sum(
                value or 0.0
                for value in [
                    _get_float(normalized, "val/box_loss", "val/seg_loss", "val/obj_loss", "val/cls_loss"),
                ]
            )

            # Find YOLO mask metric robustly: look for keys with 'mask' and 'map' or keys that include '(M)'
            yolo_mask_metric = None
            for k, v in normalized.items():
                k_low = k.lower().replace(" ", "")
                if v == "":
                    continue
                try:
                    val = float(v)
                except Exception:
                    continue
                if "(m)" in k_low or "mask" in k_low or "(mask)" in k_low:
                    yolo_mask_metric = val
                    break
                # fallback: columns like metrics/mAP50(M) or metrics/mAP50-95(M)
                if "map" in k_low and "(m)" in k_low:
                    yolo_mask_metric = val
                    break

            # If no explicit mask mAP found, try common map keys without (M)
            if yolo_mask_metric is None:
                for k, v in normalized.items():
                    k_low = k.lower().replace(" ", "")
                    if v == "":
                        continue
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if "map50" in k_low and "(m)" not in k_low:
                        # prefer mask-specific earlier; this is a fallback
                        yolo_mask_metric = val
                        break

            # map YOLO mask mAP into Val_mIoU column and add explanatory note
            note_text = "YOLO metric is Mask mAP50, Baseline is mIoU. " + (notes or "")
            logger.log_epoch(
                model_name=model_name,
                epoch=epoch,
                train_loss=train_loss if train_loss != 0.0 else None,
                val_loss=val_loss if val_loss != 0.0 else None,
                val_miou=yolo_mask_metric,
                epoch_duration_sec=0.0,
                total_time_sec=0.0,
                inference_latency_ms=None,
                notes=note_text,
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
