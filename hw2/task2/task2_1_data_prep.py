import os
import shutil
import zipfile
import urllib.request
from tqdm import tqdm
import cv2

TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
VISDRONE_ROOT = os.path.join(TASK2_DIR, "VisDrone")
YOLO_DATA_ROOT = os.path.join(TASK2_DIR, "VisDrone_YOLO")
YAML_PATH = os.path.join(TASK2_DIR, "visdrone.yaml")

# VisDrone类别
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# 下载函数
def download_with_progress(url, save_path):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(save_path)) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

def download_visdrone():
    urls = {
        "train": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
        "val": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
        "test": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"
    }
    os.makedirs(VISDRONE_ROOT, exist_ok=True)
    
    for split, url in urls.items():
        zip_path = os.path.join(VISDRONE_ROOT, f"{split}.zip")
        if not os.path.exists(zip_path):
            print(f"\n开始下载 {split} 集...")
            download_with_progress(url, zip_path)
        
        out_dir = os.path.join(VISDRONE_ROOT, f"VisDrone2019-DET-{split}")
        if not os.path.exists(out_dir):
            print(f"开始解压 {split} 集...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(VISDRONE_ROOT)
            print(f"{split} 集解压完成！")

def convert_visdrone_to_yolo():
    # 创建YOLO格式文件夹
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_DATA_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATA_ROOT, "labels", split), exist_ok=True)
    
    # 转换标注
    for split in ["train", "val", "test"]:
        src_folder_suffix = "test-dev" if split == "test" else split
        src_img_dir = os.path.join(VISDRONE_ROOT, f"VisDrone2019-DET-{src_folder_suffix}", "images")
        src_ann_dir = os.path.join(VISDRONE_ROOT, f"VisDrone2019-DET-{src_folder_suffix}", "annotations")
        dst_img_dir = os.path.join(YOLO_DATA_ROOT, "images", split)
        dst_lab_dir = os.path.join(YOLO_DATA_ROOT, "labels", split)
        
        if not os.path.exists(src_img_dir):
            continue
        
        for img_file in tqdm(os.listdir(src_img_dir), desc=f"转换 {split} 集"):
            if not img_file.endswith(".jpg"):
                continue
            
            img_path = os.path.join(src_img_dir, img_file)
            cap = cv2.VideoCapture(img_path)
            if not cap.isOpened():
                continue
            w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            shutil.copy(img_path, os.path.join(dst_img_dir, img_file))
            
            txt_name = img_file.replace(".jpg", ".txt")
            src_txt = os.path.join(src_ann_dir, txt_name)
            dst_txt = os.path.join(dst_lab_dir, txt_name)
            lines = []
            
            if os.path.exists(src_txt):
                with open(src_txt, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) < 6:
                            continue
                        x1, y1, w, h, score, cls = map(int, parts[:6])
                        
                        if cls < 1 or cls > 10 or score < 0:
                            continue
                        cls -= 1
                        
                        cx = (x1 + w/2) / w_img
                        cy = (y1 + h/2) / h_img
                        nw = w / w_img
                        nh = h / h_img
                        
                        if 0 <= cx <= 1 and 0 <= cy <= 1 and nw > 0 and nh > 0:
                            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            if lines:
                with open(dst_txt, "w", encoding="utf-8") as f_out:
                    f_out.write("\n".join(lines))
    
    # 生成yaml
    names_str = "\n  ".join([f"{i}: {name}" for i, name in enumerate(VISDRONE_CLASSES)])
    yaml_content = f"""path: {os.path.abspath(YOLO_DATA_ROOT)}
train: images/train
val: images/val
test: images/test
nc: {len(VISDRONE_CLASSES)}
names:
  {names_str}
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"\n✅ 数据集转换完成！")
    print(f"- 数据路径: {YOLO_DATA_ROOT}")
    print(f"- 配置文件: {YAML_PATH}")

if __name__ == "__main__":
    download_visdrone()
    convert_visdrone_to_yolo()