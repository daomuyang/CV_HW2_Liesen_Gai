##  跟踪与越线计数优化版
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import random
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

TASK2_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    TASK2_DIR, 
    "train_output", 
    "优化版yolov8m", 
    "优化版yolov8m_final", 
    "weights", 
    "best.pt"
)
VIDEO_PATH = os.path.join(TASK2_DIR, "video.mp4")      
OUTPUT_FOLDER_NAME = "tracking_output_optimized" 
OUTPUT_DIR = os.path.join(TASK2_DIR, OUTPUT_FOLDER_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)  
OUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "tracking_result.mp4")
OUT_LOG_PATH = os.path.join(OUTPUT_DIR, "tracking_log.csv")

LINE_POINTS = [(750, 0), (750, 1440)]  # 垂直直线，x=750
COUNT_DIRECTION = "both"  

DETECTION_CONF = 0.05  
NMS_IOU = 0.6         

count_total = 0
prev_positions = defaultdict(lambda: deque(maxlen=10))  
last_counted_frame = defaultdict(int)
MIN_COUNT_INTERVAL = 30  

# 优化后的越线判断逻辑
def is_cross_line(box, prev_boxes, line):
    (x1_line, y1_line), (x2_line, y2_line) = line
    line_x = x1_line  
    
    # 当前框的左右边界
    x1_curr, y1_curr, x2_curr, y2_curr = box
    
    # 检查历史框，看是否有穿过线的情况
    for prev_box in prev_boxes:
        x1_prev, y1_prev, x2_prev, y2_prev = prev_box
        
        # 不只是看中心，看整个框是否穿过线
        # 上一帧框完全在左边，当前框完全在右边
        if x2_prev < line_x and x1_curr > line_x:
            return "left2right"
        # 上一帧框完全在右边，当前框完全在左边
        if x1_prev > line_x and x2_curr < line_x:
            return "right2left"
        # 上一帧框和当前框都跨线，但中心移动方向明确
        center_prev = (x1_prev + x2_prev) / 2
        center_curr = (x1_curr + x2_curr) / 2
        if center_prev < line_x and center_curr > line_x:
            return "left2right"
        if center_prev > line_x and center_curr < line_x:
            return "right2left"
    
    return None

def main():
    global count_total
    
    # 加载模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 找不到模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✅ 成功加载模型：{MODEL_PATH}")
    
    # 打开视频
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"❌ 找不到视频：{VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"无法打开视频：{VIDEO_PATH}"
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    # 初始化日志
    log_file = open(OUT_LOG_PATH, "w")
    log_file.write("frame_id,track_id,class_id,class_name,cx,cy,x1,y1,x2,y2,confidence,crossed\n")
    
    print(f"\n✅ 开始处理视频：{width}x{height} @ {fps}fps")
    print(f"✅ 虚拟线位置：x={LINE_POINTS[0][0]}")
    print(f"✅ 计数方向：{COUNT_DIRECTION}")
    print(f"✅ 检测置信度：{DETECTION_CONF}")
    print("="*60)
    
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        if frame_id % 10 == 0:
            print(f"\n处理进度：{frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%) | 当前总数：{count_total}")
        
        # 跟踪推理
        results = model.track(
            frame, 
            persist=True,
            imgsz=800,
            conf=DETECTION_CONF,  
            iou=NMS_IOU,         
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        # 绘制计数线
        cv2.line(frame, LINE_POINTS[0], LINE_POINTS[1], (0, 0, 255), 6)
        cv2.line(frame, (LINE_POINTS[0][0]-50, 0), (LINE_POINTS[0][0]-50, 1440), (0, 255, 255), 2)
        cv2.line(frame, (LINE_POINTS[0][0]+50, 0), (LINE_POINTS[0][0]+50, 1440), (0, 255, 255), 2)
        
        crossed_this_frame = []
        
        if results[0].boxes.id is not None:
            for box, track_id_tensor, cls_tensor, conf in zip(
                results[0].boxes.xyxy,  
                results[0].boxes.id.int(), 
                results[0].boxes.cls.int(),
                results[0].boxes.conf
            ):
                # 类型转换
                track_id = int(track_id_tensor)
                cls_int = int(cls_tensor)
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cls_name = model.names[cls_int]
                conf_val = float(conf)
                
                # 绘制检测框和ID
                color = (255, 0, 0) if track_id not in crossed_this_frame else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, 
                    f"ID:{track_id} {cls_name} {conf_val:.2f}", 
                    (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 255), 
                    2
                )
                
                prev_positions[track_id].append((x1, y1, x2, y2))
                if len(prev_positions[track_id]) > 1:
                    pts = np.array([( (p[0]+p[2])//2, (p[1]+p[3])//2 ) for p in prev_positions[track_id]], np.int32)
                    cv2.polylines(frame, [pts], False, (255, 255, 0), 2)
                
                # 越线判断
                crossed = False
                if len(prev_positions[track_id]) >= 2:
                    direction = is_cross_line(
                        (x1, y1, x2, y2), 
                        list(prev_positions[track_id])[:-1],  # 除了当前帧的所有历史帧
                        LINE_POINTS
                    )
                    
                    # 检查是否应该计数
                    if direction:
                        # 检查时间间隔，避免重复计数
                        if frame_id - last_counted_frame[track_id] > MIN_COUNT_INTERVAL:
                            # 检查方向是否符合要求
                            if COUNT_DIRECTION == "both" or COUNT_DIRECTION == direction:
                                count_total += 1
                                last_counted_frame[track_id] = frame_id
                                crossed = True
                                crossed_this_frame.append(track_id)
                                # 越线成功时画绿色大圈和闪烁效果
                                cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                                cv2.putText(
                                    frame, 
                                    f"+1 ({direction})", 
                                    (cx-50, cy-30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1.5, 
                                    (0, 255, 0), 
                                    4
                                )
                                print(f"\n✅ 第{frame_id}帧：ID={track_id} ({cls_name}) 从{direction}越线！当前总数：{count_total}")
                
                # 写入日志
                log_file.write(
                    f"{frame_id},{track_id},{cls_int},{cls_name},{cx},{cy},{x1},{y1},{x2},{y2},{conf_val:.4f},{crossed}\n"
                )
        
        # 绘制总计数（右上角+黑色背景）
        count_text = f"Total Count: {count_total}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        thickness = 5
        (text_width, text_height), baseline = cv2.getTextSize(count_text, font, font_scale, thickness)
        text_x = width - text_width - 50
        text_y = 100 + text_height
        
        # 黑色半透明背景
        overlay = frame.copy()
        cv2.rectangle(
            overlay, 
            (text_x - 20, text_y - text_height - 20), 
            (text_x + text_width + 20, text_y + baseline + 20), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 白色文字
        cv2.putText(
            frame, 
            count_text, 
            (text_x, text_y), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness
        )
        
        out_writer.write(frame)
        
        # 实时显示
        cv2.imshow("Tracking + Counting (Optimized)", frame)
        if cv2.waitKey(1) == 27:
            break
    
    # 释放资源
    cap.release()
    out_writer.release()
    log_file.close()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"✅ 处理完成！")
    print(f"- 输出视频：{OUT_VIDEO_PATH}")
    print(f"- 跟踪日志：{OUT_LOG_PATH}")
    print(f"- 最终总越线数：{count_total}")
    print("="*60)

if __name__ == "__main__":
    main()