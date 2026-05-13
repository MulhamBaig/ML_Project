import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50

from cityscapes_dataset import CityscapesDataset


def compute_confusion(pred, target, num_classes=19, ignore_index=255):
    mask = (target != ignore_index)
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    k = (target * num_classes + pred).numpy()
    bins = np.bincount(k, minlength=num_classes**2)
    conf = bins.reshape((num_classes, num_classes)).astype(np.int64)
    return torch.from_numpy(conf)


def evaluate(model, dataloader, device):
    model.eval()
    num_classes = 19
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    import numpy as np
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)['out']
            preds = out.argmax(1)
            for p, t in zip(preds.cpu(), masks.cpu()):
                mask = (t != 255)
                tp = (p[mask] == t[mask]).sum().item()
            # Accumulate confusion matrix per batch
            for p, t in zip(preds.cpu(), masks.cpu()):
                k = (t.numpy().flatten() * num_classes + p.numpy().flatten())
                valid = t.numpy().flatten() != 255
                k = k[valid]
                if k.size > 0:
                    bins = np.bincount(k, minlength=num_classes**2)
                    conf += torch.from_numpy(bins.reshape((num_classes, num_classes)).astype(np.int64))

    # compute mIoU
    inter = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - inter
    iu = inter / (union + 1e-9)
    miou = iu.mean().item()
    return miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='gtFine_trainvaltest', help='dataset root')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--size', type=int, nargs=2, default=(256, 512))
    parser.add_argument('--subset', type=int, default=400)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CityscapesDataset(args.root, split='train', size=tuple(args.size), subset=args.subset)
    val_ds = CityscapesDataset(args.root, split='val', size=tuple(args.size), subset=100)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    model = fcn_resnet50(pretrained=False, num_classes=19)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    runs_dir = os.path.join('runs', 'phase2_baseline')
    os.makedirs(runs_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start = time.time()
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
        duration = time.time() - start
        print(f"Epoch {epoch} finished. avg loss: {running_loss / len(train_loader):.4f}. time: {duration:.1f}s")

        # validate
        miou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} val mIoU: {miou:.4f}")

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(runs_dir, f'model_epoch{epoch}.pth'))

    # measure inference latency on a single val image
    model.eval()
    import numpy as np
    timings = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            t0 = time.time()
            out = model(imgs)['out']
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t1 = time.time()
            timings.append((t1 - t0) * 1000.0)
            if len(timings) >= 20:
                break
    avg_ms = sum(timings) / len(timings)
    print(f"Avg inference latency (ms) on val images: {avg_ms:.2f}")


if __name__ == '__main__':
    main()
