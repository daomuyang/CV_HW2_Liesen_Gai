import torch
import numpy as np
import random
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import config

# 种子设置
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(config.SEED)
# 固定生成器，保证数据分割一致
generator = torch.Generator().manual_seed(config.SEED)

def get_dataloaders(batch_size=None):
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    trainval = datasets.OxfordIIITPet(
        root=config.DATA_ROOT,
        split="trainval",
        transform=train_transform,
        download=False
    )
    
    test = datasets.OxfordIIITPet(
        root=config.DATA_ROOT,
        split="test",
        transform=val_test_transform,
        download=False
    )

    train_size = int(0.8 * len(trainval))
    val_size = len(trainval) - train_size
    train_ds, val_ds = random_split(trainval, [train_size, val_size], generator=generator)

    # 构建DataLoader
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False)

    return train_loader, val_loader, test_loader