from pathlib import Path
import shutil


def trim_split(yolo_root: Path, split: str, out_root: Path):
    labels_dir = yolo_root / 'labels' / split
    images_dir = yolo_root / 'images' / split
    out_images_dir = out_root / 'images' / split
    out_labels_dir = out_root / 'labels' / split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    for label_file in labels_dir.rglob('*.txt'):
        rel = label_file.relative_to(labels_dir)
        # label path like city/name.txt -> image path mirrors images_dir / city / name.png
        city = rel.parts[0]
        if rel.name.endswith('_leftImg8bit.txt'):
            name = rel.name.replace('.txt', '.png')
        else:
            name = rel.name.replace('.txt', '_leftImg8bit.png')
        src_img = images_dir / city / name
        if not src_img.exists():
            # Sometimes images are stored with different casing; try alternative
            possible = list((images_dir / city).glob(name.replace('_leftImg8bit.png', '*leftImg8bit.png')))
            if possible:
                src_img = possible[0]
            else:
                continue

        dest_img_dir = out_images_dir / city
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dest_img_dir / src_img.name)

        dest_label_dir = out_labels_dir / city
        dest_label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_file, dest_label_dir / label_file.name)


def main():
    yolo_root = Path('yolo_cityscapes')
    out_root = Path('yolo_cityscapes_trimmed')
    for split in ('train', 'val'):
        trim_split(yolo_root, split, out_root)
    # write a simple YAML next to repo root
    yaml_path = Path('cityscapes_trimmed.yaml')
    yaml_text = """path: yolo_cityscapes_trimmed
train: images/train
val: images/val
test: images/test
names:
  - road
  - sidewalk
  - building
  - wall
  - fence
  - pole
  - traffic light
  - traffic sign
  - vegetation
  - terrain
  - sky
  - person
  - rider
  - car
  - truck
  - bus
  - train
  - motorcycle
  - bicycle
"""
    yaml_path.write_text(yaml_text, encoding='utf-8')
    print('Trimmed YOLO dataset written to', out_root)


if __name__ == '__main__':
    main()
