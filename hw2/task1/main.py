import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import config
from src.dataset import get_dataloaders
from src.models import build_model
from src.trainer import train_one_epoch, validate
import wandb

# 种子设置
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(config.SEED)

def main():
    wandb.init(
        project="CV_HW2_TASK1_Pet_Recognition",  # 项目名
        name=config.EXP_NAME,  # 实验名（baseline/se_attention/no_pretrain）
        config={  # 记录超参
            "batch_size": config.BATCH_SIZE,
            "base_lr": config.BASE_LR,
            "head_lr": config.HEAD_LR,
            "epochs": config.EPOCHS,
            "pretrained": config.USE_PRETRAINED,
            "attention": config.USE_ATTENTION
        }
    )

    # 加载数据
    train_loader, val_loader, test_loader = get_dataloaders()
    
    # 构建模型
    model, params = build_model()
    
    # 损失函数+优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #baseline与注意力机制
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #注意力机制优化版

    # 训练记录
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 训练主循环
    for epoch in range(config.EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{config.EPOCHS} ===")
        

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best.pth"))
            print(f"保存最优模型 (Val Acc: {best_acc:.4f})")

        scheduler.step()
    # 测试集评估
    print("\n=== 测试集评估 ===")
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, "best.pth")))
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    wandb.finish()  # 结束wandb

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Val")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "curve.png"))
    plt.close()

if __name__ == "__main__":
    main()

