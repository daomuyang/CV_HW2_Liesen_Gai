import os
import torch
import yaml
import random
import numpy as np

# 实验名
EXP_NAME = os.environ.get("T3_EXP_NAME", "ce_loss")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 读取配置
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", f"{EXP_NAME}.yaml")
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# 路径配置
DATA_ROOT = SCRIPT_DIR  # 数据集自动下载到task3根目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", EXP_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 超参数
BATCH_SIZE = int(cfg["batch_size"])
EPOCHS = int(cfg["epochs"])
LR = float(cfg["lr"])
LOSS_TYPE = cfg["loss_type"]  # ce / dice / combined
SEED = 42

# 固定随机种子
def set_seed(seed=SEED):
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

set_seed()