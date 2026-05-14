from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

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


def _sum_floats(row: Dict[str, Any], *keys: str) -> float | None:
    values = []
    for key in keys:
        value = _get_float(row, key)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values)


def _read_existing_rows(log_file: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not log_file.exists():
        return [], []

    with log_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return rows, fieldnames


def _write_rows(log_file: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with log_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_yolo_results(results_csv: Path, model_name: str, notes: str = "Imported from YOLO results.csv") -> int:
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find YOLO results file: {results_csv}")

    logger = MetricsLogger()
    existing_rows, fieldnames = _read_existing_rows(Path(logger.log_file))
    existing_rows = [row for row in existing_rows if row.get("Model_Name") != model_name]

    rows_to_write: list[dict[str, str]] = []
    rows_written = 0
    previous_time = 0.0

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Normalize header keys by stripping whitespace
            normalized = {k.strip(): (v if v is not None else "") for k, v in row.items()}

            # determine epoch
            epoch_raw = int(float(normalized.get("epoch", rows_written)))
            epoch = epoch_raw if epoch_raw > 0 else (rows_written + 1)

            current_time = _get_float(normalized, "time") or previous_time
            epoch_duration = max(current_time - previous_time, 0.0)
            total_elapsed = current_time
            previous_time = current_time

            # aggregate train/val loss if present
            train_loss = _sum_floats(
                normalized,
                "train/box_loss",
                "train/seg_loss",
                "train/obj_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "train/sem_loss",
            )
            val_loss = _sum_floats(
                normalized,
                "val/box_loss",
                "val/seg_loss",
                "val/obj_loss",
                "val/cls_loss",
                "val/dfl_loss",
                "val/sem_loss",
            )

            # Prefer mask mAP50, then mask mAP50-95, and avoid precision/recall columns.
            yolo_mask_metric = None
            for key in ("metrics/mAP50(M)", "metrics/mAP50-95(M)"):
                value = _get_float(normalized, key)
                if value is not None:
                    yolo_mask_metric = value
                    break

            if yolo_mask_metric is None:
                for k, v in normalized.items():
                    k_low = k.lower().replace(" ", "")
                    if v == "":
                        continue
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    if "mask" in k_low and "map" in k_low:
                        yolo_mask_metric = val
                        break

            # map YOLO mask mAP into Val_mIoU column and add explanatory note
            note_text = "YOLO metric is Mask mAP50, Baseline is mIoU. " + (notes or "")
            row_data = {
                "Model_Name": model_name,
                "Epoch": str(epoch),
                "Train_Loss": f"{train_loss:.6f}" if train_loss is not None else "",
                "Val_Loss": f"{val_loss:.6f}" if val_loss is not None else "",
                "Val_mIoU": f"{yolo_mask_metric:.6f}" if yolo_mask_metric is not None else "",
                "Epoch_Duration_sec": f"{epoch_duration:.2f}",
                "Total_Time_Elapsed_sec": f"{total_elapsed:.2f}",
                "Inference_Latency_ms": "",
                "Notes_and_Errors": note_text,
            }
            rows_to_write.append(row_data)
            rows_written += 1

    rows_to_write = existing_rows + rows_to_write
    _write_rows(Path(logger.log_file), [
        "Model_Name",
        "Epoch",
        "Train_Loss",
        "Val_Loss",
        "Val_mIoU",
        "Epoch_Duration_sec",
        "Total_Time_Elapsed_sec",
        "Inference_Latency_ms",
        "Notes_and_Errors",
    ], rows_to_write)

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
