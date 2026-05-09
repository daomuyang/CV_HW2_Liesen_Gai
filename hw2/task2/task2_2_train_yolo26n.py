import os
import sys

MODEL_NAME = "yolo26n"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
import wandb
import pandas as pd
from ultralytics import YOLO

TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(TASK2_DIR, "visdrone.yaml")
MODEL_SAVE_DIR = os.path.join(TASK2_DIR, "train_output", "yolo26n")
LOCAL_MODEL_PATH = os.path.join(TASK2_DIR, "yolo26n.pt")

def get_hyperparams(model_name):
    params = {
        "epochs": 50,
        "imgsz": 800,
        "multi_scale": True,
        
        "optimizer": "AdamW",
        "lr0": 0.0015, 
        "lrf": 0.01,
        "weight_decay": 0.0001,
        
        "cls": 2.0,
        "box": 7.5,
        "dfl": 1.5,
        
        "warmup_epochs": 3,
        "patience": 5,
        "label_smoothing": 0.1,
        
        "hsv_h": 0.02,
        "hsv_s": 0.6,
        "hsv_v": 0.5,
        "degrees": 5.0,
        "translate": 0.15,
        "scale": 0.2,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.1,
        "copy_paste": 0.1,
    }
    
    if model_name == "yolo26n":
        params["batch"] = 24  
        params["desc"] = "YOLO26n-800imgsz"
    else:
        raise ValueError(f"不支持的模型：{model_name}")
    
    return params

def sync_wandb_metrics(results_csv_path):
    if not os.path.exists(results_csv_path):
        return
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()
    for idx, row in df.iterrows():
        epoch = int(row['epoch'])
        wandb.log({
            "Train/Box_Loss": row['train/box_loss'],
            "Train/Cls_Loss": row['train/cls_loss'],
            "Val/mAP50": row['metrics/mAP50(B)'],
            "Val/mAP50-95": row['metrics/mAP50-95(B)'],
        }, step=epoch)


def train_yolo_aliyun_gpu():
    params = get_hyperparams(MODEL_NAME)
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ 找不到本地模型：{LOCAL_MODEL_PATH}\n"
            "请先运行：wget -c https://hf-mirror.com/ultralytics/assets/resolve/main/yolo26n.pt -O yolo26n.pt"
        )
    
    print(f"加载本地模型：{LOCAL_MODEL_PATH}")
    print(f"Batch Size：{params['batch']}")

    try:
        wandb.init(
            project="CV-HW2-Task2",
            name="YOLO26n-800imgsz-Final",
            mode="offline",
            dir=TASK2_DIR,
            config={
                "model": params['desc'],
                "dataset": "VisDrone2019-DET",
                **params
            }
        )
    except:
        pass

    model = YOLO(LOCAL_MODEL_PATH)
    
    results = model.train(
        data=YAML_PATH,
        epochs=params["epochs"],
        batch=params["batch"],
        imgsz=params["imgsz"],
        multi_scale=params["multi_scale"],
        
        lr0=params["lr0"],
        lrf=params["lrf"],
        optimizer=params["optimizer"],
        weight_decay=params["weight_decay"],
        
        project=MODEL_SAVE_DIR,
        name="yolo26n_visdrone_final",
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        device="cuda",
        workers=8,
        amp=True,
        cache="disk",
        patience=params["patience"],
        label_smoothing=params["label_smoothing"],
        
        cls=params["cls"],
        box=params["box"],
        dfl=params["dfl"],
        
        warmup_epochs=params["warmup_epochs"],
        hsv_h=params["hsv_h"],
        hsv_s=params["hsv_s"],
        hsv_v=params["hsv_v"],
        degrees=params["degrees"],
        translate=params["translate"],
        scale=params["scale"],
        fliplr=params["fliplr"],
        mosaic=params["mosaic"],
        mixup=params["mixup"],
        copy_paste=params["copy_paste"],
    )

    csv_path = os.path.join(MODEL_SAVE_DIR, "yolo26n_visdrone_final", "results.csv")
    sync_wandb_metrics(csv_path)
    
    try:
        wandb.finish()
    except:
        pass
    
    print("\n" + "="*60)
    print(f"✅ YOLO26n 训练完成！")
    print("="*60)
    print(f"最优模型：{MODEL_SAVE_DIR}/yolo26n_visdrone_final/weights/best.pt")
    print("="*60)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到 CUDA！")
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ 检测到 CUDA：{torch.cuda.get_device_name(0)}，显存：{gpu_mem:.1f} GB")
    
    train_yolo_aliyun_gpu()