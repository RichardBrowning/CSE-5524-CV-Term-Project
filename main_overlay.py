#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import math
import argparse
from pathlib import Path

OUTPUT_DIR  = Path("output")
OUTPUT_PATH = OUTPUT_DIR / "overlay_sample.mp4"

# ============ 可调参数 ============
DISPLAY_PREVIEW = True          # 若不需要边看边写，改为 False
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_COLOR = (255, 255, 255)
THICKNESS = 1
LINE_TYPE = cv2.LINE_AA
TARGET_FPS  = None              # None => 使用原视频 FPS；否则重设
FOURCC = "mp4v"                 # 可尝试 "avc1" "mp4v" "H264"


BG_HISTORY       = 500        # MOG2 history
BG_THRESHOLD     = 50         # MOG2 varThreshold
MIN_AREA         = 800        # minimum contour area
AREA_THRESHOLD   = 5000       # > this → “CAR”
RATIO_PEDESTRIAN = (0.3, 0.8) # aspect ratio range → “PED”
# =================================

def parse_args():
    parser = argparse.ArgumentParser(description='Add overlay to video with object tracking simulation')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video file (.mov)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (default: output/overlay_sample.mp4)')
    return parser.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    return cap

def prepare_writer(out_path: Path, fps: float, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    return cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)

def draw_text(img, text, org, color=(255,255,255)):
    cv2.putText(img, text, org, FONT, FONT_SCALE, color, THICKNESS, LINE_TYPE)

def draw_boxes(frame, boxes):
    for b in boxes:
        x,y,w,h = b["bbox"]
        cls = b["cls"]
        # 颜色区分
        color = (0,215,255) if cls=="PED" else (0,165,255)
        # 框
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2, LINE_TYPE)
        # 标签背景
        label = f'{cls} #{b["id"]}'
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame, (x, y-th-6), (x+tw+4, y), color, -1)
        cv2.putText(frame, label, (x+2, y-5), FONT, FONT_SCALE, (0,0,0), THICKNESS, LINE_TYPE)

def detect_moving_objects(frame, bg_subtractor):
    """
    对一帧做前景减法，连通域提框，并简单分类。
    返回：[{ 'bbox':(x,y,w,h), 'id':i, 'cls':'PED'/'CAR' }, ...]
    """
    fgmask = bg_subtractor.apply(frame)  # 前景 mask
    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    idx = 0
    h_img, w_img = frame.shape[:2]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        x,y,w,h = cv2.boundingRect(cnt)
        # 防止超画面
        if w> w_img*0.9 or h> h_img*0.9:
            continue

        # 简单分类
        ratio = w/float(h)
        if area > AREA_THRESHOLD and ratio>1.0:
            cls = "CAR"
        elif RATIO_PEDESTRIAN[0] < ratio < RATIO_PEDESTRIAN[1]:
            cls = "PED"
        else:
            cls = "OBJ"  # 未知，可当“杂”
        boxes.append({
            "bbox":(x,y,w,h),
            "id": idx,     # 暂用检测索引
            "cls": cls
        })
        idx += 1

    return boxes

def main():
    args = parse_args()
    
    INPUT_PATH = Path(args.input)
    if args.output:
        OUTPUT_PATH = Path(args.output)
    else:
        OUTPUT_PATH = OUTPUT_DIR / "overlay_sample.mp4"
    
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input video not found: {INPUT_PATH}")

    ensure_dir(OUTPUT_DIR)
    cap = open_video(INPUT_PATH)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = prepare_writer(OUTPUT_PATH, src_fps, (width, height))

    # 初始化背景减法器
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=BG_HISTORY,
                                                varThreshold=BG_THRESHOLD,
                                                detectShadows=False)

    print(f"[INFO] 开始处理: {INPUT_PATH.name}  共{total}帧")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ======= 示例 Overlay: 帧号 + 时间 =======
        t = frame_idx / src_fps
        draw_text(frame, f"Frame: {frame_idx}", (10, 25))
        draw_text(frame, f"Time : {t:.2f}s",    (10, 50))

        # ======= 核心：前景检测 + 候选框 =======
        boxes = detect_moving_objects(frame, bg_sub)
        draw_boxes(frame, boxes)

        writer.write(frame)
        if DISPLAY_PREVIEW:
            cv2.imshow("Detection Demo", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if DISPLAY_PREVIEW:
        cv2.destroyAllWindows()

    print(f"[INFO] 完成 → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()