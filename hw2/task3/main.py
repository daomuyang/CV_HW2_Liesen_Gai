import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import config
from src.dataset import get_dataloaders
from src.unet import UNet
from src.losses import get_loss
from src.trainer import train_one_epoch, validate
import wandb 

def main():
    wandb.init(
        project="CV_HW2_TASK3_Pet_Segmentation",  # 项目名
        name=config.EXP_NAME,  # 实验名（ce_loss/dice_loss/combined_loss）
        config={  # 记录超参
            "batch_size": config.BATCH_SIZE,
            "lr": config.LR,
            "epochs": config.EPOCHS,
            "loss_type": config.LOSS_TYPE
        }
    )
    
    # 加载数据
    train_loader, val_loader = get_dataloaders()
    
    # 从零初始化U-Net
    model = UNet(n_channels=3, n_classes=1).to(config.DEVICE)
    
    # 损失函数+优化器
    criterion = get_loss(config.LOSS_TYPE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练记录
    best_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    # 训练主循环
    for epoch in range(config.EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{config.EPOCHS} ===")
        
        # 训练
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer)
        # 验证
        val_loss, val_iou = validate(model, val_loader, criterion)
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mIoU": train_iou,
            "val_mIoU": val_iou
        })

        print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   mIoU: {val_iou:.4f}")

        # 保存最优模型
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best.pth"))
            print(f"保存最优模型 (Val mIoU: {best_iou:.4f})")

        scheduler.step()

    # 打印最终结果
    print(f"\n=== 训练完成 ===")
    print(f"最优验证mIoU: {best_iou:.4f}")

    wandb.log({"best_val_mIoU": best_iou})
    wandb.finish()

    # 绘制训练曲线
    plt.figure(figsize=(14, 5))
    
    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2, marker='.')
    plt.plot(val_losses, label="Val Loss", linewidth=2, marker='.')
    plt.title(f"Loss Curve ({config.LOSS_TYPE})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # mIoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label="Train mIoU", linewidth=2, marker='.')
    plt.plot(val_ious, label="Val mIoU", linewidth=2, marker='.')
    plt.title(f"mIoU Curve ({config.LOSS_TYPE})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("mIoU", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(config.OUTPUT_DIR, "curve.png")
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    print(f"📈 训练曲线已保存至：{curve_path}")
    plt.close()

if __name__ == "__main__":
    main()