import os
import torch
import numpy as np
import random
import config
from src.dataset import get_dataloaders
from src.models import build_model
from src.trainer import validate

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

def main():
    # 设置种子
    set_seed(config.SEED)
    
    # 加载数据
    _, _, test_loader = get_dataloaders()
    
    # 构建模型
    model, _ = build_model()
    
    # 加载最优权重
    best_model_path = os.path.join(config.OUTPUT_DIR, "best.pth")
    print(f"正在加载模型权重：{best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
    
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 测试
    print("\n=== 开始测试 ===")
    test_loss, test_acc = validate(model, test_loader, criterion)
    
    print(f"\n=== 测试结果 ===")
    print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}")

if __name__ == "__main__":
    main()