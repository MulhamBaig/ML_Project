import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
from PIL import Image

from cityscapes_dataset import CityscapesDataset
from metrics_logger import MetricsLogger


def compute_iou(pred, target, num_classes=19, ignore_index=255):
    """Compute mIoU from predictions and targets."""
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return 0.0
    
    k = (target.cpu().numpy() * num_classes + pred.cpu().numpy())
    valid = target.cpu().numpy() != 255
    k = k[valid]
    if k.size == 0:
        return 0.0
    
    bins = np.bincount(k, minlength=num_classes**2)
    conf = bins.reshape((num_classes, num_classes)).astype(np.int64)
    
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iu = inter / (union + 1e-9)
    miou = np.mean(iu)
    return miou


def evaluate(model, dataloader, device):
    """Evaluate model on validation set and return mIoU."""
    model.eval()
    num_classes = 19
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)['out']
            preds = out.argmax(1)
            
            for p, t in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                mask = (t != 255)
                k = (t[mask] * num_classes + p[mask])
                bins = np.bincount(k, minlength=num_classes**2)
                conf += bins.reshape((num_classes, num_classes)).astype(np.int64)
    
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iu = inter / (union + 1e-9)
    miou = np.mean(iu)
    return miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='gtFine_trainvaltest', help='dataset root')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--size', type=int, nargs=2, default=(256, 512))
    parser.add_argument('--subset', type=int, default=400)
    parser.add_argument('--run', type=int, default=1, help='Run number for checkpoint naming')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize logger
    logger = MetricsLogger()
    model_name = 'resnet50'
    
    # Initialize dataset and loaders
    train_ds = CityscapesDataset(args.root, split='train', size=tuple(args.size), subset=args.subset)
    val_ds = CityscapesDataset(args.root, split='val', size=tuple(args.size), subset=100)
    
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
    
    # Initialize model
    model = fcn_resnet50(pretrained=False, num_classes=19)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    
    runs_dir = os.path.join('runs', f'phase2_baseline_run{args.run}')
    os.makedirs(runs_dir, exist_ok=True)
    
    total_start = time.time()
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            running_loss = 0.0
            
            try:
                for i, (imgs, masks) in enumerate(train_loader, 1):
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    optimizer.zero_grad()
                    out = model(imgs)['out']
                    loss = criterion(out, masks)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                    if i % 10 == 0:
                        print(f"Epoch {epoch} iter {i} loss {running_loss / i:.4f}")
                
                train_loss = running_loss / len(train_loader)
                epoch_duration = time.time() - epoch_start
                total_elapsed = time.time() - total_start
                
                print(f"Epoch {epoch} finished. avg loss: {train_loss:.4f}. duration: {epoch_duration:.1f}s")
                
                # Validate
                model.eval()
                val_miou = evaluate(model, val_loader, device)
                print(f"Epoch {epoch} val mIoU: {val_miou:.4f}")
                
                # Save checkpoint
                checkpoint_name = f'{model_name}_Run{args.run}_Epoch{epoch}.pth'
                checkpoint_path = os.path.join(runs_dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved: {checkpoint_path}")
                
                # Log epoch metrics (val_loss is not easily computed, set to empty)
                logger.log_epoch(
                    model_name=model_name,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=None,
                    val_miou=val_miou,
                    epoch_duration_sec=epoch_duration,
                    total_time_sec=total_elapsed,
                    inference_latency_ms=None,
                    notes=''
                )
                
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    error_msg = f"OOM Error at epoch {epoch}: {str(e)[:100]}"
                    print(f"ERROR: {error_msg}")
                    logger.log_epoch(
                        model_name=model_name,
                        epoch=epoch,
                        train_loss=None,
                        val_loss=None,
                        val_miou=None,
                        epoch_duration_sec=time.time() - epoch_start,
                        total_time_sec=time.time() - total_start,
                        inference_latency_ms=None,
                        notes=error_msg
                    )
                    # Save emergency checkpoint
                    emergency_path = os.path.join(runs_dir, f'{model_name}_Run{args.run}_EmergencyEpoch{epoch}.pth')
                    torch.save(model.state_dict(), emergency_path)
                    break
                else:
                    raise
        
        # Measure end-to-end inference latency on 100 validation images.
        # This includes image decode, resize/normalize, host-to-device transfer, and forward pass.
        print("\nMeasuring inference latency on 100 validation images...")
        model.eval()
        timings = []
        with torch.no_grad():
            for img_path in val_ds.images[:100]:
                img = Image.open(img_path).convert('RGB')
                img = val_ds.img_transform(img).unsqueeze(0).to(device)

                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()

                out = model(img)['out']
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.time()
                
                timings.append((t1 - t0) * 1000.0)  # Convert to ms
                
                if len(timings) >= 100:
                    break
        
        avg_latency_ms = np.mean(timings)
        print(f"Average inference latency (ms): {avg_latency_ms:.2f}")
        
        # Log final inference latency
        logger.log_inference_final(model_name=model_name, inference_latency_ms=avg_latency_ms)
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)[:200]}"
        print(f"ERROR: {error_msg}")
        logger.log_epoch(
            model_name=model_name,
            epoch=0,
            train_loss=None,
            val_loss=None,
            val_miou=None,
            epoch_duration_sec=time.time() - total_start,
            total_time_sec=time.time() - total_start,
            inference_latency_ms=None,
            notes=error_msg
        )
        raise


if __name__ == '__main__':
    main()
