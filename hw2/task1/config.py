import os
import torch
import yaml

# 实验名
EXP_NAME = os.environ.get("T1_EXP_NAME", "baseline")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 读取YAML
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", f"{EXP_NAME}.yaml")
with open(CONFIG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

# 路径
DATA_ROOT = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", EXP_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设备
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 超参
SEED = 42
BATCH_SIZE = int(cfg["batch_size"])
EPOCHS = int(cfg["epochs"])
BASE_LR = float(cfg["base_lr"])
HEAD_LR = float(cfg["head_lr"])
WEIGHT_DECAY = float(cfg["weight_decay"])
MOMENTUM = float(cfg["momentum"])
USE_PRETRAINED = cfg["use_pretrained"]
USE_ATTENTION = cfg["use_attention"]