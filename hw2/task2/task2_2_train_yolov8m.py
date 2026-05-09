## 优化版yolov8m模型训练脚本，调整了部分超参数
import os
import sys

MODEL_NAME = "yolov8m"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_H_TRANSFER"] = "1"

import torch
import wandb
import pandas as pd
from ultralytics import YOLO

TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(TASK2_DIR, "visdrone.yaml")
MODEL_SAVE_DIR = os.path.join(TASK2_DIR, "train_output", "优化版yolov8m")

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

    if model_name == "yolov8m":
        params["batch"] = 12
        params["desc"] = "优化版YOLOv8m-800imgsz"
    elif model_name == "yolov8n":
        params["batch"] = 24
        params["desc"] = "YOLOv8n-800imgsz"
    elif model_name == "yolo11n":
        params["batch"] = 24
        params["desc"] = "YOLO11n-800imgsz"
    else:
        raise ValueError(f"不支持的模型：{model_name}")

    return params

def sync_wandb_metrics(results_csv_path):
    if not os.path.exists(results_csv_path):
        print(f"⚠️ 找不到 results.csv")
        return

    print(f"📊 正在同步 WandB 曲线...")
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()

    for idx, row in df.iterrows():
        epoch = int(row['epoch'])
        metrics = {
            "Train/Box_Loss": row['train/box_loss'],
            "Train/Cls_Loss": row['train/cls_loss'],
            "Val/Box_Loss": row['val/box_loss'],
            "Val/Cls_Loss": row['val/cls_loss'],
            "Val/mAP50": row['metrics/mAP50(B)'],
            "Val/mAP50-95": row['metrics/mAP50-95(B)'],
            "Val/Precision": row['metrics/precision(B)'],
            "Val/Recall": row['metrics/recall(B)'],
            "Learning_Rate": row['lr/pg0'],
        }
        wandb.log(metrics, step=epoch)

    print("✅ WandB 曲线同步完成！")

def train_yolo_aliyun_gpu():
    params = get_hyperparams(MODEL_NAME)
    print(f"\n🚀 训练：优化版 YOLOv8m 800imgsz")
    print(f"💾 保存到：{MODEL_SAVE_DIR}\n")

    # 路径检查
    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(f"❌ 找不到 {YAML_PATH}")

    # WandB
    try:
        wandb.init(
            project="CV-HW2-Task2",
            name="优化版YOLOv8m-800imgsz-Final",
            dir=TASK2_DIR,
            resume=False,
            mode="offline",
            config={
                "model": params['desc'],
                "dataset": "VisDrone",
                **params
            }
        )
        print("✅ WandB 离线模式已启动")
    except:
        print("⚠️ WandB 启动失败，继续训练")

    # 模型
    model = YOLO(f"{MODEL_NAME}.pt")

    # 训练
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
        name="优化版yolov8m_final",
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
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

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

    csv_path = os.path.join(MODEL_SAVE_DIR, "优化版yolov8m_final", "results.csv")
    sync_wandb_metrics(csv_path)

    try:
        wandb.finish()
    except:
        pass

    print("\n✅ 训练完成！")
    print(f"📂 新模型：{MODEL_SAVE_DIR}/优化版yolov8m_final/weights/best.pt")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 无 GPU")
    print(f"✅ GPU 已就绪")
    train_yolo_aliyun_gpu()



###  未优化版yolov8m训练脚本（之前的版本，超参数较基础，未针对 VisDrone 小目标进行优化）
# import os
# import sys

# MODEL_NAME = "yolov8m"

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# import torch
# import wandb
# import pandas as pd
# from ultralytics import YOLO

# TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
# YAML_PATH = os.path.join(TASK2_DIR, "visdrone.yaml")
# MODEL_SAVE_DIR = os.path.join(TASK2_DIR, "train_output", MODEL_NAME)

# def get_hyperparams(model_name):
#     params = {
#         "epochs": 50,
#         "imgsz": 640,
#         "lr0": 0.001,
#         "lrf": 0.01,
#         "optimizer": "SGD",
#         "momentum": 0.937,
#         "weight_decay": 0.0005,
#         "cls": 1.5,
#         "box": 7.5,
#         "dfl": 1.5,
#         "warmup_epochs": 2,
#         "patience": 0,
#         "hsv_h": 0.015,
#         "hsv_s": 0.5,
#         "hsv_v": 0.4,
#         "degrees": 0.0,
#         "translate": 0.1,
#         "scale": 0.0,
#         "fliplr": 0.5,
#         "mosaic": 0.5 if "v8" in model_name or "11" in model_name else 0.0,
#         "mixup": 0.0,
#         "copy_paste": 0.0,
#     }
    
#     if model_name == "yolov8m":
#         params["batch"] = 16
#         params["desc"] = "YOLOv8m"
#     elif model_name == "yolov8n":
#         params["batch"] = 32
#         params["desc"] = "YOLOv8n"
#     elif model_name == "yolo11n":
#         params["batch"] = 32
#         params["desc"] = "YOLO11n"
#     elif model_name == "yolov5nu":
#         params["batch"] = 32
#         params["desc"] = "YOLOv5nu"
#     else:
#         raise ValueError(f"不支持的模型：{model_name}")
    
#     return params

# def sync_wandb_metrics(results_csv_path):
#     """
#     手动读取 results.csv 并同步到 WandB，确保作业要求的所有曲线都显示
#     """
#     if not os.path.exists(results_csv_path):
#         print(f"⚠️ 找不到 results.csv：{results_csv_path}，跳过手动同步")
#         return
    
#     print(f"📊 正在手动同步指标到 WandB...")
#     df = pd.read_csv(results_csv_path)
    
#     # 去除列名中的空格
#     df.columns = df.columns.str.strip()
    
#     # 遍历每一行，同步到 WandB
#     for idx, row in df.iterrows():
#         epoch = int(row['epoch'])
#         metrics = {
#             "Train/Box_Loss": row['train/box_loss'],
#             "Train/Cls_Loss": row['train/cls_loss'],
#             "Train/DFL_Loss": row['train/dfl_loss'],
#             "Val/Box_Loss": row['val/box_loss'],
#             "Val/Cls_Loss": row['val/cls_loss'],
#             "Val/DFL_Loss": row['val/dfl_loss'],
#             "Val/mAP50": row['metrics/mAP50(B)'],
#             "Val/mAP50-95": row['metrics/mAP50-95(B)'],
#             "Val/Precision": row['metrics/precision(B)'],
#             "Val/Recall": row['metrics/recall(B)'],
#             "Learning_Rate": row['lr/pg0'],
#         }
        
#         # 同步到 WandB
#         wandb.log(metrics, step=epoch)
    
#     print("✅ WandB 指标同步完成！")

# def train_yolo_aliyun_gpu():
#     params = get_hyperparams(MODEL_NAME)
#     print(f"\n🚀 开始训练模型：{MODEL_NAME}")
#     print(f"📝 模型描述：{params['desc']}")
#     print(f"📦 Batch Size：{params['batch']}")
#     print(f"💾 保存路径：{MODEL_SAVE_DIR}\n")

#     if not os.path.exists(YAML_PATH):
#         raise FileNotFoundError(f"❌ 找不到 {YAML_PATH}！请先运行 task2_1_data_prep.py")
    
#     with open(YAML_PATH, 'r') as f:
#         if '/Users/gailiesen' in f.read():
#             raise RuntimeError(
#                 "❌ visdrone.yaml 里还是 Mac 路径！\n"
#                 "请运行：sed -i 's|/Users/gailiesen/Documents/学/大三下/计算机视觉/hw2/task2|/mnt/workspace/hw2/task2|g' visdrone.yaml"
#             )

#     try:
#         wandb.init(
#             project="CV-HW2-Task2",
#             name=f"{MODEL_NAME}-VisDrone-Final",
#             dir=TASK2_DIR,
#             resume=False,
#             mode="offline",
#             config={
#                 "model": params['desc'],
#                 "dataset": "VisDrone2019-DET",
#                 "device": "NVIDIA A10",
#                 **params
#             }
#         )
#         print("✅ WandB 在线模式初始化成功！")
#     except Exception as e:
#         print(f"⚠️ WandB 在线模式失败，尝试离线模式：{e}")
#         wandb.init(mode="offline", project="CV-HW2-Task2", dir=TASK2_DIR)

#     model_file = f"{MODEL_NAME}.pt"
#     print(f"📥 正在加载模型：{model_file}...")
#     model = YOLO(model_file)

#     results = model.train(
#         data=YAML_PATH,
#         epochs=params["epochs"],
#         batch=params["batch"],
#         imgsz=params["imgsz"],
#         lr0=params["lr0"],
#         lrf=params["lrf"],
#         optimizer=params["optimizer"],
#         momentum=params["momentum"],
#         weight_decay=params["weight_decay"],
#         project=MODEL_SAVE_DIR,
#         name=f"{MODEL_NAME}_visdrone_final",
#         exist_ok=True,
#         save=True,
#         val=True,
#         plots=True,  
#         device="cuda",
#         workers=8,
#         multi_scale=False,
#         amp=True,
#         cache="disk",
#         patience=params["patience"],

#         # 损失权重
#         cls=params["cls"],
#         box=params["box"],
#         dfl=params["dfl"],

#         # Warmup
#         warmup_epochs=params["warmup_epochs"],
#         warmup_momentum=0.8,
#         warmup_bias_lr=0.1,

#         # 数据增强
#         hsv_h=params["hsv_h"],
#         hsv_s=params["hsv_s"],
#         hsv_v=params["hsv_v"],
#         degrees=params["degrees"],
#         translate=params["translate"],
#         scale=params["scale"],
#         fliplr=params["fliplr"],
#         mosaic=params["mosaic"],
#         mixup=params["mixup"],
#         copy_paste=params["copy_paste"],
#     )

#     results_csv_path = os.path.join(MODEL_SAVE_DIR, f"{MODEL_NAME}_visdrone_final", "results.csv")
#     sync_wandb_metrics(results_csv_path)

#     try:
#         wandb.finish()
#     except:
#         pass

#     print("\n" + "="*60)
#     print(f"✅ {MODEL_NAME} 训练完成！")
#     print("="*60)
#     print(f"1. 最优模型：{MODEL_SAVE_DIR}/{MODEL_NAME}_visdrone_final/weights/best.pt")
#     print(f"2. 本地曲线：{MODEL_SAVE_DIR}/{MODEL_NAME}_visdrone_final/results.png")
#     print(f"3. WandB 报告：https://wandb.ai/{wandb.api.default_entity}/CV-HW2-Task2")
#     print("="*60)
#     print("\n📌 WandB 曲线：")
#     print("   - Train/Box_Loss + Val/Box_Loss")
#     print("   - Train/Cls_Loss + Val/Cls_Loss")
#     print("   - Val/mAP50")
#     print("   - Val/mAP50-95")
#     print("="*60)

# if __name__ == "__main__":
#     if not torch.cuda.is_available():
#         raise RuntimeError("❌ 未检测到 CUDA！")
#     print(f"✅ 检测到 CUDA：{torch.cuda.get_device_name(0)}，显存：{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    
#     train_yolo_aliyun_gpu()


