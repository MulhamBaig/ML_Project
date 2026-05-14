from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from yolo_cityscapes_utils import build_label_to_class_index, get_dataset_paths, get_yolo_class_names, normalize_polygon



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
    # YOLO requires label basename to match image basename.
    # Cityscapes image: <id>_leftImg8bit.png -> label: <id>_leftImg8bit.txt
    label_name = polygon_file.name.replace("_gtFine_polygons.json", "_leftImg8bit.txt")
    label_path = output_labels_root / city / label_name
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
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    if split == 'val':
        limit = 250  # Enforce exactly 250 images for the val split

    label_to_class = build_label_to_class_index()
    converted_files = 0
    converted_objects = 0

    for polygon_file in iter_polygon_files(polygons_dir, limit=limit):
        # Convert label
        converted_objects += convert_polygon_file(polygon_file, output_labels_dir, label_to_class)
        
        # Copy image explicitly instead of junction to fix YOLOv8 Windows path resolution bug
        city = polygon_file.parent.name
        img_name = polygon_file.name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
        src_img = images_dir / city / img_name
        dest_img_dir = output_images_dir / city
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        if src_img.exists():
            shutil.copy2(src_img, dest_img_dir / img_name)
            
        converted_files += 1

    return converted_files, converted_objects


def write_cityscapes_yaml(yolo_root: Path, yaml_path: Path) -> None:
    names = get_yolo_class_names()
    yaml_text = [
        f"path: {yolo_root.resolve().as_posix()}",
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
