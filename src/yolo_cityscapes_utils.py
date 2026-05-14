from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from cityscapesscripts.helpers.labels import labels


@dataclass(frozen=True)
class YoloCityscapesClassInfo:
    class_index: int
    train_id: int
    name: str


def get_cityscapes_yolo_classes() -> List[YoloCityscapesClassInfo]:
    """Return the 19 Cityscapes training classes in trainId order."""
    valid_labels = sorted(
        [label for label in labels if label.trainId not in (-1, 255)],
        key=lambda label: label.trainId,
    )

    classes: List[YoloCityscapesClassInfo] = []
    for class_index, label in enumerate(valid_labels):
        classes.append(
            YoloCityscapesClassInfo(
                class_index=class_index,
                train_id=label.trainId,
                name=label.name,
            )
        )
    return classes


def build_label_to_class_index() -> Dict[str, int]:
    """Map Cityscapes polygon labels to YOLO class indices."""
    mapping: Dict[str, int] = {}
    for class_info in get_cityscapes_yolo_classes():
        mapping[class_info.name] = class_info.class_index
    return mapping


def get_yolo_class_names() -> List[str]:
    return [class_info.name for class_info in get_cityscapes_yolo_classes()]


def normalize_polygon(points: List[List[float]], width: int, height: int) -> List[float]:
    """Flatten and normalize polygon points into YOLO's 0-1 coordinate space."""
    normalized: List[float] = []
    for x, y in points:
        x_norm = max(0.0, min(1.0, float(x) / float(width)))
        y_norm = max(0.0, min(1.0, float(y) / float(height)))
        normalized.extend([x_norm, y_norm])
    return normalized


def get_dataset_paths(dataset_root: str | Path, split: str) -> Tuple[Path, Path]:
    root = Path(dataset_root)
    images_dir = root / "leftImg8bit" / split
    polygons_dir = root / "gtFine" / split
    return images_dir, polygons_dir
