from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from yolo_cityscapes_utils import build_label_to_class_index, get_dataset_paths, get_yolo_class_names, normalize_polygon


def create_directory_junction(source: Path, link_path: Path) -> None:
    """Create a Windows directory junction to avoid duplicating Cityscapes images."""
    if link_path.exists() or link_path.is_symlink():
        return

    link_path.parent.mkdir(parents=True, exist_ok=True)
    command = f'mklink /J "{link_path}" "{source}"'
    result = os.system(f'cmd /c {command}')
    if result != 0:
        raise RuntimeError(f"Failed to create junction: {link_path} -> {source}")


def iter_polygon_files(polygons_dir: Path, limit: int | None = None) -> Iterable[Path]:
    polygon_files = sorted(polygons_dir.rglob("*_gtFine_polygons.json"))
    if limit is not None and limit > 0:
        polygon_files = polygon_files[:limit]
    return polygon_files


def convert_polygon_file(polygon_file: Path, output_labels_root: Path, label_to_class: dict[str, int]) -> int:
    with polygon_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    img_height = int(payload["imgHeight"])
    img_width = int(payload["imgWidth"])

    city = polygon_file.parent.name
    label_path = output_labels_root / city / polygon_file.name.replace("_gtFine_polygons.json", ".txt")
    label_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for obj in payload.get("objects", []):
        label_name = obj.get("label")
        polygon = obj.get("polygon", [])
        if not label_name or label_name not in label_to_class:
            continue
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue

        flattened = normalize_polygon(polygon, img_width, img_height)
        if len(flattened) < 6:
            continue

        class_index = label_to_class[label_name]
        coords = " ".join(f"{value:.6f}" for value in flattened)
        lines.append(f"{class_index} {coords}")

    with label_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
        if lines:
            handle.write("\n")

    return len(lines)


def prepare_split(dataset_root: Path, yolo_root: Path, split: str, limit: int | None = None) -> tuple[int, int]:
    images_dir, polygons_dir = get_dataset_paths(dataset_root, split)
    output_images_dir = yolo_root / "images" / split
    output_labels_dir = yolo_root / "labels" / split
    output_images_dir.parent.mkdir(parents=True, exist_ok=True)
    output_labels_dir.parent.mkdir(parents=True, exist_ok=True)

    create_directory_junction(images_dir, output_images_dir)

    label_to_class = build_label_to_class_index()
    converted_files = 0
    converted_objects = 0

    for polygon_file in iter_polygon_files(polygons_dir, limit=limit):
        converted_objects += convert_polygon_file(polygon_file, output_labels_dir, label_to_class)
        converted_files += 1

    return converted_files, converted_objects


def write_cityscapes_yaml(yolo_root: Path, yaml_path: Path) -> None:
    names = get_yolo_class_names()
    yaml_text = [
        f"path: {yolo_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"names: {names}",
    ]
    yaml_path.write_text("\n".join(yaml_text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Cityscapes polygons for YOLOv8 segmentation.")
    parser.add_argument("--dataset-root", default="gtFine_trainvaltest", help="Cityscapes dataset root")
    parser.add_argument("--yolo-root", default="yolo_cityscapes", help="Output YOLO dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of polygon files per split for a quick test")
    parser.add_argument("--write-yaml", action="store_true", help="Write cityscapes.yaml next to the repo root")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    yolo_root = Path(args.yolo_root)
    yolo_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_objects = 0
    for split in args.splits:
        files, objects = prepare_split(dataset_root, yolo_root, split, limit=args.limit)
        total_files += files
        total_objects += objects
        print(f"Prepared split={split}: files={files}, polygons={objects}")

    if args.write_yaml:
        write_cityscapes_yaml(yolo_root, Path("cityscapes.yaml"))
        print("Wrote cityscapes.yaml")

    print(f"Done. Total polygon files: {total_files}, total polygons converted: {total_objects}")


if __name__ == "__main__":
    main()
