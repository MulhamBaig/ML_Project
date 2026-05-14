from pathlib import Path
import time
from ultralytics import YOLO
from metrics_logger import MetricsLogger


def collect_validation_images(yolo_root: Path, split: str = "val"):
    image_root = yolo_root / "images" / split
    return sorted(image_root.rglob("*.png"))


def main():
    yolo_root = Path('yolo_cityscapes')
    weights = Path('runs/segment/train/weights/best.pt')
    if not weights.exists():
        print('weights not found:', weights)
        return

    model = YOLO(str(weights))
    val_images = collect_validation_images(yolo_root, 'val')
    if not val_images:
        print('No validation images found under', yolo_root)
        return

    timings = []
    print('Measuring inference on', min(100, len(val_images)), 'images...')
    for img in val_images[:100]:
        if model.device.type == 'cuda':
            model.model.to('cuda')
        t0 = time.time()
        model.predict(source=str(img), imgsz=512, device=0, verbose=False)
        if model.device.type == 'cuda':
            import torch
            torch.cuda.synchronize()
        t1 = time.time()
        timings.append((t1 - t0) * 1000.0)

    avg_ms = sum(timings) / len(timings)
    print(f'Average inference latency (ms) over {len(timings)} images: {avg_ms:.2f}')

    logger = MetricsLogger()
    logger.log_inference_final(model_name='yolov8n-seg', inference_latency_ms=avg_ms)


if __name__ == '__main__':
    main()
