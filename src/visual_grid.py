from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from ultralytics import YOLO


def load_baseline_model(weights_path: Path, device):
    model = fcn_resnet50(pretrained=False, num_classes=19)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_baseline(model, img_pil, size=(256, 512), device='cpu'):
    t = transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    x = t(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)['out']
        pred = out.argmax(1).squeeze(0).cpu().numpy()
    # resize back to original image size
    pred_img = Image.fromarray(pred.astype(np.uint8)).resize(img_pil.size, resample=Image.NEAREST)
    return np.array(pred_img)


def predict_yolo(weights_path: Path, img_path: Path):
    model = YOLO(str(weights_path))
    res = model.predict(source=str(img_path), imgsz=512, device=0, verbose=False)
    # use result.plot() to get colorized overlay
    vis = res[0].plot()
    return vis


def colorize_label(mask):
    # simple color map for visualization
    cmap = plt.get_cmap('tab20')
    colored = cmap(mask % 20)
    return (colored[..., :3] * 255).astype(np.uint8)


def main():
    # pick a random validation image
    yolo_root = Path('yolo_cityscapes')
    val_images = sorted((yolo_root / 'images' / 'val').rglob('*.png'))
    if not val_images:
        print('No val images found')
        return
    img_path = random.choice(val_images)

    # raw image
    img = Image.open(img_path).convert('RGB')

    # ground truth mask path from original dataset
    # infer filename base
    base = img_path.name.replace('_leftImg8bit.png', '')
    # city is parent dir name
    city = img_path.parent.name
    gt_path = Path('gtFine_trainvaltest') / 'gtFine' / 'val' / city / f"{base}_gtFine_labelTrainIds.png"
    if not gt_path.exists():
        # sometimes val images are in different folders; attempt alternative
        gt_path = Path('gtFine_trainvaltest') / 'gtFine' / 'val' / city / f"{base}_gtFine_labelTrainIds.png"

    gt = Image.open(gt_path).resize(img.size, resample=Image.NEAREST)
    gt_arr = np.array(gt)
    gt_col = colorize_label(gt_arr)

    # baseline prediction
    baseline_weights = Path('runs/phase2_baseline_run3/resnet50_Run3_Epoch5.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_model = load_baseline_model(baseline_weights, device)
    baseline_mask = predict_baseline(baseline_model, img, size=(512,512), device=device)
    baseline_col = colorize_label(baseline_mask)

    # YOLO prediction (visual overlay)
    yolo_weights = Path('runs/segment/runs/segment/train_trimmed/weights/best.pt')
    if yolo_weights.exists():
        yolo_vis = predict_yolo(yolo_weights, img_path)
    else:
        yolo_vis = np.array(img)

    # plot grid: raw, gt, baseline, yolo
    fig, axes = plt.subplots(1,4, figsize=(20,6))
    axes[0].imshow(img); axes[0].set_title('Raw Image'); axes[0].axis('off')
    axes[1].imshow(gt_col); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(baseline_col); axes[2].set_title('Baseline (FCN-ResNet50)'); axes[2].axis('off')
    axes[3].imshow(yolo_vis); axes[3].set_title('YOLOv8-Seg'); axes[3].axis('off')

    out_dir = Path('Results_Comparison')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'visual_comparison.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print('Saved visual grid to', out_path)


if __name__ == '__main__':
    main()
