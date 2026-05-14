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
    # use result.plot() to get colorized overlay (disable boxes for fair comparison)
    vis = res[0].plot(boxes=False)
    return vis


def colorize_label(mask):
    # simple color map for visualization
    cmap = plt.get_cmap('tab20')
    colored = cmap(mask % 20)
    return (colored[..., :3] * 255).astype(np.uint8)


def main():
    # Setup directories
    yolo_root = Path('yolo_cityscapes')
    out_dir = Path('Results_Comparison')
    out_dir.mkdir(exist_ok=True)
    
    val_images = sorted((yolo_root / 'images' / 'val').rglob('*.png'))
    if not val_images:
        print('No val images found')
        return

    # Load models once outside the loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_weights = Path('runs/phase2_baseline_run5/resnet50_Run5_Epoch5.pth')
    yolo_weights = Path('runs/segment/runs/segment/train_final/weights/best.pt')
    
    print("Loading models...")
    baseline_model = load_baseline_model(baseline_weights, device)
    
    # Generate 5 comparisons
    for i in range(1, 6):
        print(f"Generating visual comparison {i}/5...")
        img_path = random.choice(val_images)
        
        # raw image
        img = Image.open(img_path).convert('RGB')
        
        # ground truth
        base = img_path.name.replace('_leftImg8bit.png', '')
        city = img_path.parent.name
        gt_path = Path('gtFine_trainvaltest') / 'gtFine' / 'val' / city / f"{base}_gtFine_labelTrainIds.png"
        
        if not gt_path.exists():
             continue # Skip if GT is missing for this random pick
             
        gt = Image.open(gt_path).resize(img.size, resample=Image.NEAREST)
        gt_arr = np.array(gt)
        gt_col = colorize_label(gt_arr)

        # baseline prediction
        baseline_mask = predict_baseline(baseline_model, img, size=(512,512), device=device)
        baseline_col = colorize_label(baseline_mask)

        # YOLO prediction
        if yolo_weights.exists():
            yolo_vis = predict_yolo(yolo_weights, img_path)
        else:
            yolo_vis = np.array(img)

        # plot 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].imshow(img); axes[0, 0].set_title('Raw Image'); axes[0, 0].axis('off')
        axes[0, 1].imshow(gt_col); axes[0, 1].set_title('Ground Truth'); axes[0, 1].axis('off')
        axes[1, 0].imshow(baseline_col); axes[1, 0].set_title('FCN-ResNet50'); axes[1, 0].axis('off')
        axes[1, 1].imshow(yolo_vis); axes[1, 1].set_title('YOLOv8n-seg'); axes[1, 1].axis('off')

        plt.tight_layout()
        out_path = out_dir / f'visual_comparison_{i}.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
