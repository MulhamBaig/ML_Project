import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', size=(256, 512), subset=None):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, 'leftImg8bit', split)
        self.mask_dir = os.path.join(root, 'gtFine', split)
        self.size = size

        imgs = []
        for city in sorted(os.listdir(self.img_dir)):
            city_dir = os.path.join(self.img_dir, city)
            if not os.path.isdir(city_dir):
                continue
            for fn in sorted(os.listdir(city_dir)):
                if fn.endswith('_leftImg8bit.png'):
                    imgs.append(os.path.join(city_dir, fn))

        if subset is not None and subset > 0:
            imgs = imgs[:subset]

        self.images = imgs

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def _mask_path(self, img_path):
        # img_path: .../leftImg8bit/<split>/<city>/<name>_leftImg8bit.png
        city = os.path.basename(os.path.dirname(img_path))
        name = os.path.basename(img_path)
        mask_name = name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
        return os.path.join(self.root, 'gtFine', self.split, city, mask_name)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self._mask_path(img_path)

        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)

        mask = Image.open(mask_path)
        mask = mask.resize(self.size, resample=Image.NEAREST)
        mask = np.array(mask, dtype=np.int64)
        mask = torch.from_numpy(mask).long()

        return img, mask
