import torch
from tqdm import tqdm
import config

# 计算IoU
def calculate_iou(pred, target, smooth=1e-6):
    # pred: [B, H, W] 预测结果（0/1）
    # target: [B, H, W] 真实标签（0/1）
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

# def train_one_epoch(model, loader, criterion, optimizer):
#     model.train()
#     total_loss = 0.0
#     total_iou = 0.0

#     for img, mask in tqdm(loader):
#         img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
#         optimizer.zero_grad()
        
#         # 前向传播
#         logits = model(img)
#         loss = criterion(logits, mask)
        
#         # 反向传播
#         loss.backward()
#         optimizer.step()

#         # 计算IoU
#         pred = torch.sigmoid(logits).squeeze(1) > 0.5
#         iou = calculate_iou(pred, mask)

#         total_loss += loss.item()
#         total_iou += iou.item()

#     return total_loss / len(loader), total_iou / len(loader)

# def validate(model, loader, criterion):
#     model.eval()
#     total_loss = 0.0
#     total_iou = 0.0

#     with torch.no_grad():
#         for img, mask in tqdm(loader):
#             img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
            
#             # 前向传播
#             logits = model(img)
#             loss = criterion(logits, mask)

#             # 计算IoU
#             pred = torch.sigmoid(logits).squeeze(1) > 0.5
#             iou = calculate_iou(pred, mask)

#             total_loss += loss.item()
#             total_iou += iou.item()

#     return total_loss / len(loader), total_iou / len(loader)



def calculate_soft_iou(pred, target, smooth=1e-6):
    # pred: [B, H, W] sigmoid后的概率（不用>0.5）
    # target: [B, H, W] 真实标签
    intersection = (pred * target).sum((1, 2))
    union = pred.sum((1, 2)) + target.sum((1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    for img, mask in tqdm(loader):
        img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
        optimizer.zero_grad()
        
        logits = model(img)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        
        probs = torch.sigmoid(logits).squeeze(1)
        iou = calculate_soft_iou(probs, mask)  # 训练用软IoU
        
        total_loss += loss.item()
        total_iou += iou.item()
    return total_loss / len(loader), total_iou / len(loader)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for img, mask in tqdm(loader):
            img, mask = img.to(config.DEVICE), mask.to(config.DEVICE)
            
            logits = model(img)
            loss = criterion(logits, mask)
            
            # 验证时保持硬IoU不变
            pred = torch.sigmoid(logits).squeeze(1) > 0.5
            iou = calculate_iou(pred, mask)
            
            total_loss += loss.item()
            total_iou += iou.item()
    return total_loss / len(loader), total_iou / len(loader)