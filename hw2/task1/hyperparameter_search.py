import os
import torch
import numpy as np
import random
import config
import src.dataset as dataset
import src.models as models
import src.trainer as trainer

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

def hyperparameter_search():
    """超参搜索：保存所有组合日志 + 单独保存最优结果"""
    search_space = {
        "base_lrs": [1e-5, 1e-4, 5e-4],
        "head_lrs": [1e-4, 1e-3, 5e-3],
        "weight_decays": [1e-4, 5e-4],
        "batch_sizes": [32, 64]
    }  #baseline实验搜索空间

    best_hparams = None
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    # 记录所有超参组合的结果
    all_results = []

    # 遍历超参组合
    for bs in search_space["batch_sizes"]:
        train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=bs)
        for blr in search_space["base_lrs"]:
            for hlr in search_space["head_lrs"]:
                for wd in search_space["weight_decays"]:
                    print(f"\n===== 超参：bs={bs}, blr={blr}, hlr={hlr}, wd={wd} =====")
                    
                    # 临时覆盖配置
                    ori_blr, ori_hlr, ori_wd = config.BASE_LR, config.HEAD_LR, config.WEIGHT_DECAY
                    config.BASE_LR, config.HEAD_LR, config.WEIGHT_DECAY = blr, hlr, wd

                    # 构建模型
                    model, params = models.build_model()
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(
                        params, 
                        momentum=config.MOMENTUM, 
                        weight_decay=wd
                    )
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

                    # 短周期训练
                    current_best_val = 0.0
                    for epoch in range(10):
                        train_loss, train_acc = trainer.train_one_epoch(model, train_loader, criterion, optimizer)
                        val_loss, val_acc = trainer.validate(model, val_loader, criterion)
                        current_best_val = max(current_best_val, val_acc)
                        scheduler.step()

                    # 测试集评估
                    test_loss, test_acc = trainer.validate(model, test_loader, criterion)
                    
                    # 记录当前组合的结果
                    current_result = {
                        "batch_size": bs,
                        "base_lr": blr,
                        "head_lr": hlr,
                        "weight_decay": wd,
                        "best_val_acc": current_best_val,
                        "test_acc": test_acc
                    }
                    all_results.append(current_result)
                    
                    # 恢复配置
                    config.BASE_LR, config.HEAD_LR, config.WEIGHT_DECAY = ori_blr, ori_hlr, ori_wd

                    # 清理显存
                    del model, optimizer, scheduler
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                    # 更新最优结果
                    if current_best_val > best_val_acc:
                        best_val_acc = current_best_val
                        best_test_acc = test_acc
                        best_hparams = {
                            "batch_size": bs, 
                            "base_lr": blr, 
                            "head_lr": hlr, 
                            "weight_decay": wd
                        }
                        print(f"🔄 更新最优超参：{best_hparams} | 最优Val Acc: {best_val_acc:.4f}")

    # ====================== 保存结果 ======================
    print(f"\n===== 超参搜索完成 =====")
    print(f"最优超参：{best_hparams}")
    print(f"最优验证精度：{best_val_acc:.4f}")
    print(f"对应测试精度：{best_test_acc:.4f}")
    
    # 保存所有超参组合的完整日志
    all_log_path = os.path.join(config.OUTPUT_DIR, "hyperparam_all_logs.txt")
    with open(all_log_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("所有超参组合的完整日志\n")
        f.write("="*80 + "\n\n")
        for i, res in enumerate(all_results, 1):
            f.write(f"--- 组合 {i} ---\n")
            f.write(f"  Batch Size: {res['batch_size']}\n")
            f.write(f"  Base LR: {res['base_lr']}\n")
            f.write(f"  Head LR: {res['head_lr']}\n")
            f.write(f"  Weight Decay: {res['weight_decay']}\n")
            f.write(f"  Best Val Acc: {res['best_val_acc']:.4f}\n")
            f.write(f"  Test Acc: {res['test_acc']:.4f}\n\n")
    print(f"📝 所有超参组合日志已保存至：{all_log_path}")
    
    # 单独保存最优超参的结果
    best_result_path = os.path.join(config.OUTPUT_DIR, "hyperparam_best_result.txt")
    with open(best_result_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("最优超参结果\n")
        f.write("="*80 + "\n\n")
        f.write(f"实验名：{config.EXP_NAME}\n")
        f.write(f"最优超参：{best_hparams}\n")
        f.write(f"最优验证精度：{best_val_acc:.4f}\n")
        f.write(f"对应测试精度：{best_test_acc:.4f}\n")
    print(f"📝 最优超参结果已保存至：{best_result_path}")

if __name__ == "__main__":
    hyperparameter_search()