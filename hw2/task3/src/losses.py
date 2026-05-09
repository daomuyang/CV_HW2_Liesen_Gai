import torch
import torch.nn as nn
import torch.nn.functional as F

# 手动实现Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        # logits: [B, 1, H, W] -> squeeze to [B, H, W]
        probs = torch.sigmoid(logits).squeeze(1)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# 组合损失
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.6, dice_weight=0.4):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    def forward(self, logits, targets):
        ce_loss = self.ce(logits.squeeze(1), targets.float())
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

# 根据配置获取损失函数
def get_loss(loss_type):
    if loss_type == "ce":
        class BCEWithSqueeze(nn.Module):
            def __init__(self):
                super().__init__()
                self.ce = nn.BCEWithLogitsLoss()
            def forward(self, logits, targets):
                return self.ce(logits.squeeze(1), targets.float())
        return BCEWithSqueeze()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "combined":
        return CombinedLoss()
    else:
        raise ValueError(f"不支持的损失类型：{loss_type}")