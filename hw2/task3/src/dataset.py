import torch
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
import config

class JointTransform:
    def __init__(self, size=(256, 256), train=True):
        self.size = size
        self.train = train
    def __call__(self, image, mask):
        # 统一Resize
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        if self.train:
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask[mask == 2] = 0 
        mask[mask == 3] = 0 
        mask[mask == 1] = 1  
        
        # 图像正常转为Tensor
        image = TF.to_tensor(image)
        
        return image, mask

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

def get_dataloaders():
    train_transform = JointTransform(size=(256, 256), train=True)
    val_transform = JointTransform(size=(256, 256), train=False)

    trainval = datasets.OxfordIIITPet(
        root=config.DATA_ROOT,
        split="trainval",
        target_types="segmentation",
        transform=None,
        target_transform=None,
        download=False
    )

    train_size = int(0.8 * len(trainval))
    val_size = len(trainval) - train_size
    train_subset, val_subset = random_split(trainval, [train_size, val_size], generator=torch.Generator().manual_seed(config.SEED))

    train_ds = DatasetWrapper(train_subset, train_transform)
    val_ds = DatasetWrapper(val_subset, val_transform)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader