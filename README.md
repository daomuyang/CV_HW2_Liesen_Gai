# CV-HW2 计算机视觉课程作业    
## 盖烈森23307130013@m.fudan.edu.cn（独立完成）
本仓库包含计算机视觉课程第二次作业的全部代码，涵盖图像分类、目标检测与跟踪、语义分割三个任务。

## 目录结构
```
hw2/
├── task1/                          # 任务1
│   ├── config.py                   # 全局配置文件
│   ├── hyperparameter_search.py    # 超参数搜索脚本
│   ├── main.py                     # 训练主脚本
│   ├── test.py                     # 模型测试脚本
│   ├── configs/                    # 各实验配置文件
│   │   ├── baseline.yaml
│   │   ├── no_pretrain.yaml
│   │   ├── se_attention.yaml
│   │   └── se_attention_opt.yaml
│   ├── src/                        # 核心代码模块
│   │   ├── __init__.py
│   │   ├── dataset.py              # 数据集加载与预处理
│   │   ├── models.py               # 模型定义（含注意力机制）
│   │   ├── trainer.py              # 训练器实现
│   │   └── utils.py                # 工具函数
│   └── outputs/                    # 自动生成：训练/测试结果
│       ├── baseline/
│       ├── no_pretrain/
│       ├── se_attention/
│       └── se_attention_opt/
├── task2/                          # 任务2
│   ├── task2_1_data_prep.py        # 数据集自动下载与格式转换
│   ├── task2_2_train_yolo26n.py    # YOLO26n模型训练脚本
│   ├── task2_2_train_yolov8m.py    # 优化版YOLOv8m模型训练脚本与未优化版YOLOv8m模型训练脚本（已注释）
│   ├── task2_3_tracking.py         # 视频多目标跟踪与越线计数
│   ├── task2_4_analysis.py         # 跟踪结果自动分析与可视化
│   ├── video.mp4                   # 待处理测试视频（为满足GitHub上传要求已进行适当压缩）
│   ├── visdrone.yaml               # 自动生成：YOLO数据集配置
│   ├── train_output/               # 自动生成：所有模型训练结果
│   │   ├── yolo11n/                 # YOLO11n模型训练输出
│   │   ├── yolo26n/                 # YOLO26n模型训练输出
│   │   ├── yolov8m/                 # 原版YOLOv8m模型训练输出
│   │   └── 优化版yolov8m/           # 优化版YOLOv8m模型训练输出（本次实验最优）
│   ├── tracking_output/            # 自动生成：跟踪与计数结果
│   │   ├── tracking_result.mp4     # 标注后的结果视频（为满足GitHub上传要求已进行适当压缩）
│   │   └── tracking_log.csv        # 跟踪日志文件
│   └── analysis_results/           # 自动生成：跟踪实验基本描述统计
│       ├── tracking_analysis_report.txt
│       ├── frame_object_count.png
│       ├── class_distribution.png
│       ├── cross_event_distribution.png
│       └── track_lifetime_distribution.png
└── task3/                          # 任务3
    ├── config.py                   # 全局配置文件
    ├── main.py                     # 训练主脚本
    ├── configs/                    # 各损失函数配置文件
    │   ├── ce_loss.yaml
    │   ├── dice_loss.yaml
    │   ├── dice_loss_opt.yaml
    │   └── combined_loss.yaml
    ├── src/                        # 核心代码模块
    │   ├── __init__.py
    │   ├── dataset.py              # 数据集加载
    │   ├── losses.py               # 损失函数实现
    │   ├── unet.py                 # UNet模型定义
    │   ├── trainer.py              # 训练器实现
    │   └── utils.py                # 工具函数
    └── outputs/                    # 自动生成：训练结果
        ├── ce_loss/
        ├── dice_loss/
        ├── dice_loss_opt/
        └── combined_loss/
```

## 未上传文件说明
以下文件因体积过大未上传至GitHub，各模型已上传至：https://drive.google.com/drive/folders/1L4abv-IdNAn4ldqruwdPQzMai51aKVxH?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto
1. **预训练模型权重**：
   - YOLO26n预训练权重：`task2/yolo26n.pt`
   - YOLOv8m预训练权重：`task2/yolov8m.pt`
2. **训练好的模型权重**：
   - YOLO26n最优权重：`task2/train_output/yolo26n/yolo26n_visdrone_final/weights/best.pt`
   - YOLOv8m最优权重：`task2/train_output/优化版yolov8m/优化版yolov8m_final/weights/best.pt`
3. **VisDrone数据集**：将通过`task2_1_data_prep.py`自动下载生成

## 环境配置
```bash
# 安装PyTorch
# CPU/MPS版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1 GPU版本
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装YOLOv8及任务依赖
pip install tqdm matplotlib pyyaml wandb ultralytics opencv-python
```

## 运行步骤
所有脚本均已内置完整配置，无需传入额外命令行参数，**必须在对应任务目录下运行**。

### 任务1
```bash
cd task1

# 超参数分析
export T1_EXP_NAME=baseline
python hyperparameter_search.py 

# 所有训练板块用main.py
# Baseline训练
export T1_EXP_NAME=baseline
python main.py 
# 预训练消融实验
export T1_EXP_NAME=no_pretrain
python main.py 
# 注意力机制
export T1_EXP_NAME=se_attention
python main.py 
# 优化版注意力机制（先在main中将gamma改为0.5）
export T1_EXP_NAME=se_attention_opt
python main.py 

# 所有测试板块用test.py
# Baseline测试
T1_EXP_NAME=baseline python test.py 
# 预训练消融
T1_EXP_NAME=se_attention_opt python test.py
# 注意力机制
T1_EXP_NAME=se_attention_opt python test.py
python main.py 
# 优化版注意力机制
T1_EXP_NAME=se_attention_opt python test.py
```

### 任务2
```bash
cd task2

# 数据准备：自动下载VisDrone-DET 2019数据集并转换为YOLO格式
# 输出：生成VisDrone/、VisDrone_YOLO/文件夹和visdrone.yaml配置文件
python task2_1_data_prep.py

# 模型训练
# 训练YOLO26n轻量级模型
python task2_2_train_yolo26n.py
# 训练优化版YOLOv8m中等规模模型（本次实验最优模型）
# 如需训练未优化版YOLOv8m请将优化版代码注释，将未优化代码解除注释后运行，未优化版在优化版下方
python task2_2_train_yolov8m.py

# 视频跟踪与越线计数
# 测试视频命名为video.mp4，放在task2根目录下
# 训练好的最优模型已存在于：task2/train_output/优化版yolov8m/优化版yolov8m_final/weights/best.pt
# 输出：跟踪结果视频和日志保存在task2/tracking_output/文件夹下
python task2_3_tracking.py

# 跟踪结果基本描述统计结果生成
# 需要步骤3已生成tracking_log.csv日志文件
# 输出：分析报告和4张可视化图表保存在task2/analysis_results/文件夹下
python task2_4_analysis.py
```

### 任务3
```bash
cd task3
# 交叉熵损失
T3_EXP_NAME=ce_loss 
python main.py
# Dice损失
T3_EXP_NAME=dice_loss 
python main.py
# Dice损失优化
T3_EXP_NAME=dice_loss_opt 
python main.py
# 组合损失
T3_EXP_NAME=combined_loss 
python main.py
```

## 注意事项
1. 所有路径均为相对路径，必须在对应任务目录下执行脚本，否则会出现文件找不到错误
2. 任务2的跟踪脚本默认加载优化版YOLOv8m的最优权重
3. 训练过程中生成的所有结果文件均保存在对应任务的`outputs/`、`train_output/`、`tracking_output/`等文件夹中
4. 如需修改超参数，请编辑对应任务的`config.py`或`configs/`目录下的yaml配置文件
