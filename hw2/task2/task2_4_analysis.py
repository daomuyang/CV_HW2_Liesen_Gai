import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_PATH = "./tracking_output/tracking_log.csv"
OUTPUT_DIR = "./analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(LOG_PATH)

total_frames = df["frame_id"].max()
total_detections = len(df)
unique_track_ids = df["track_id"].nunique()
class_counts = df["class_name"].value_counts().to_dict()

crossed_df = df[df["crossed"] == True].copy()
total_crossed = len(crossed_df)
crossed_by_class = crossed_df["class_name"].value_counts().to_dict()

crossed_directions = defaultdict(int)
unclassified_crosses = 0
line_x = 750

for idx, cross_event in crossed_df.iterrows():
    track_id = cross_event["track_id"]
    cross_frame = cross_event["frame_id"]
    curr_cx = cross_event["cx"]
    
    prev_records = df[(df["track_id"] == track_id) & (df["frame_id"] < cross_frame)]
    
    if len(prev_records) > 0:
        prev_record = prev_records.iloc[-1]
        prev_cx = prev_record["cx"]
        
        if prev_cx < line_x and curr_cx > line_x:
            crossed_directions["left2right"] += 1
        elif prev_cx > line_x and curr_cx < line_x:
            crossed_directions["right2left"] += 1
        else:
            unclassified_crosses += 1
    else:
        next_records = df[(df["track_id"] == track_id) & (df["frame_id"] > cross_frame)]
        if len(next_records) > 0:
            next_record = next_records.iloc[0]
            next_cx = next_record["cx"]
            
            if curr_cx < line_x and next_cx > line_x:
                crossed_directions["left2right"] += 1
            elif curr_cx > line_x and next_cx < line_x:
                crossed_directions["right2left"] += 1
            else:
                unclassified_crosses += 1
        else:
            unclassified_crosses += 1

track_lifetimes = df.groupby("track_id")["frame_id"].agg(["min", "max", "count"])
track_lifetimes["lifetime"] = track_lifetimes["max"] - track_lifetimes["min"] + 1
track_lifetimes["frame_coverage"] = track_lifetimes["count"] / track_lifetimes["lifetime"]

id_switches = 0
frame_ids = sorted(df["frame_id"].unique())
prev_ids = set(df[df["frame_id"] == frame_ids[0]]["track_id"].unique())

for frame in frame_ids[1:]:
    curr_ids = set(df[df["frame_id"] == frame]["track_id"].unique())
    disappeared = prev_ids - curr_ids
    reappeared = curr_ids & disappeared
    
    for track_id in reappeared:
        last_seen = df[(df["track_id"] == track_id) & (df["frame_id"] < frame)]["frame_id"].max()
        if frame - last_seen > 5:
            id_switches += 1
    
    prev_ids = curr_ids

avg_lifetime = track_lifetimes["lifetime"].mean()
avg_coverage = track_lifetimes["frame_coverage"].mean()
short_lived_ids = len(track_lifetimes[track_lifetimes["lifetime"] < 10])

frame_object_counts = df.groupby("frame_id").size()
avg_objects_per_frame = frame_object_counts.mean()
std_objects_per_frame = frame_object_counts.std()
dense_threshold = avg_objects_per_frame + std_objects_per_frame

dense_frames = frame_object_counts[frame_object_counts > dense_threshold].index.tolist()
dense_segments = []
if dense_frames:
    current_segment = [dense_frames[0]]
    for frame in dense_frames[1:]:
        if frame == current_segment[-1] + 1:
            current_segment.append(frame)
        else:
            if len(current_segment) >= 3:
                dense_segments.append((current_segment[0], current_segment[-1]))
            current_segment = [frame]
    if len(current_segment) >= 3:
        dense_segments.append((current_segment[0], current_segment[-1]))

top5_dense_frames = frame_object_counts.nlargest(5).to_dict()

report_text = f"""
多目标跟踪与越线计数实验基本信息

一、基本统计信息
- 总处理帧数：{total_frames}
- 总检测次数：{total_detections}
- 唯一跟踪ID数：{unique_track_ids}
- 平均每帧目标数：{avg_objects_per_frame:.2f} ± {std_objects_per_frame:.2f}

    各类别检测数量分布
"""
for cls, count in class_counts.items():
    report_text += f"- {cls}：{count} 次\n"

report_text += f"""
二、越线计数结果
- 总越线物体数：{total_crossed}
- 各方向越线数：
  - 从左到右（left2right）：{crossed_directions.get('left2right', 0)}
  - 从右到左（right2left）：{crossed_directions.get('right2left', 0)}
  - 无法判断方向：{unclassified_crosses}

各类别越线数量分布
"""
for cls, count in crossed_by_class.items():
    report_text += f"- {cls}：{count} 个\n"

report_text += f"""
三、跟踪稳定性与ID跳变
- 平均跟踪ID生命周期：{avg_lifetime:.2f} 帧
- 平均帧覆盖率：{avg_coverage:.2%}
- 短生命周期ID（<10帧）数量：{short_lived_ids} 个（占比 {short_lived_ids/unique_track_ids:.2%}）
- 检测到的ID跳变次数：{id_switches} 次

四、密集/遮挡片段
- 密集片段阈值：每帧超过 {dense_threshold:.2f} 个目标
- 检测到的连续密集片段（≥3帧）：
"""
for start, end in dense_segments:
    report_text += f"- 帧 {start} 至 帧 {end}（共 {end-start+1} 帧）\n"

if not dense_segments:
    report_text += "- 未检测到连续3帧以上的密集片段\n"

report_text += f"""
 最密集的5帧
"""
for frame, count in top5_dense_frames.items():
    report_text += f"- 第 {frame} 帧：{count} 个目标\n"

with open(os.path.join(OUTPUT_DIR, "tracking_analysis_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

plt.figure(figsize=(12, 6))
plt.plot(frame_object_counts.index, frame_object_counts.values, linewidth=1, label="每帧目标数")
plt.axhline(y=dense_threshold, color="r", linestyle="--", label=f"密集阈值 ({dense_threshold:.2f})")

for start, end in dense_segments:
    plt.axvspan(start, end, color="yellow", alpha=0.3, label="密集片段" if start == dense_segments[0][0] else "")

plt.xlabel("帧号")
plt.ylabel("目标数量")
plt.title("视频中每帧目标数量变化")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "frame_object_count.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
classes = list(class_counts.keys())
counts = list(class_counts.values())
plt.bar(classes, counts, color="skyblue")
plt.xlabel("类别")
plt.ylabel("检测次数")
plt.title("各类别检测数量分布")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
cross_frames = crossed_df["frame_id"].values
plt.hist(cross_frames, bins=20, color="orange", edgecolor="black")
plt.xlabel("帧号")
plt.ylabel("越线事件数")
plt.title("越线事件时间分布")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cross_event_distribution.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(track_lifetimes["lifetime"], bins=30, color="green", edgecolor="black")
plt.xlabel("ID生命周期（帧）")
plt.ylabel("ID数量")
plt.title("跟踪ID生命周期分布")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "track_lifetime_distribution.png"), dpi=300)
plt.close()