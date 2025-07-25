import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random

class TargetTracker:
    """简化版目标跟踪器 - 仅使用Mean-Shift"""
    def __init__(self, target_id, bbox, frame):
        self.id = target_id
        self.bbox = bbox  # [x, y, w, h]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.lost_count = 0  # 目标丢失计数
        self.original_size = (bbox[2], bbox[3])  # 保存原始尺寸
        
        # 初始化Mean-Shift跟踪器
        self._init_meanshift_tracker(frame)
    
    def _init_meanshift_tracker(self, frame):
        """初始化Mean-Shift跟踪器"""
        x, y, w, h = self.bbox
        roi = frame[y:y+h, x:x+w]
        
        # 设置ROI的直方图 (H和S通道)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        # 保存初始位置
        self.track_window = (x, y, w, h)

    def update(self, frame):
        """使用Mean-Shift更新目标位置"""
        try:
            x, y, w, h = self.bbox
            track_window = (x, y, w, h)
            
            # 设置终止条件
            termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1)
            
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 计算反向投影 (使用H和S通道)
            dst = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
            
            # 应用Mean-Shift获取新位置
            _, track_window = cv2.meanShift(dst, track_window, termination_criteria)
            
            # 更新边界框位置但保持原始尺寸
            new_x, new_y, new_w, new_h = track_window
            self.bbox = [new_x, new_y, self.original_size[0], self.original_size[1]]
            
            # 计算反向投影的响应值
            response = np.mean(dst[new_y:new_y+self.original_size[1], new_x:new_x+self.original_size[0]])
            
            # 根据响应值判断是否丢失目标
            if response < 5:  # 低响应表示目标可能丢失
                self.lost_count += 1
                return False
            
            self.lost_count = 0
            return True
            
        except Exception as e:
            print(f"Mean-Shift更新错误: {str(e)}")
            self.lost_count += 1
            return False
    
    def draw(self, frame):
        """在帧上绘制目标"""
        x, y, w, h = self.bbox
        
        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)
        
        # 绘制目标ID
        cv2.putText(frame, f"ID:{self.id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        
        return frame

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=False):
        # 视频输入/输出设置
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not self.out.isOpened():
            raise ValueError(f"无法创建输出视频: {output_path}")
        
        # 预览设置
        self.display_preview = display_preview
        
        # 运动检测相关变量
        self.prev_gray = None
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 10  # MHI衰减时间（秒）
        
        # 目标跟踪相关
        self.trackers = []  # 活动跟踪器列表
        self.next_id = 1    # 下一个目标ID
        self.tracker_max_lost = 5  # 目标丢失阈值
        
        # 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        
    def process_video(self):
        """主处理循环"""
        progress = tqdm(total=self.total_frames, desc="Processing Video")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 预处理 - 降噪
            processed_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 运动目标检测
            detected_objects = self._detect_motion(frame, gray)
            
            # 更新现有跟踪器
            self._update_trackers(frame)
            
            # 为未跟踪的目标创建新跟踪器
            self._create_new_trackers(detected_objects, frame)
            
            # 清理丢失的跟踪器
            self._cleanup_lost_trackers()
            
            # 绘制所有活动跟踪器
            for tracker in self.trackers:
                processed_frame = tracker.draw(processed_frame)
            
            # 写入输出视频
            self.out.write(processed_frame)
            
            # 显示预览
            if self.display_preview:
                preview = cv2.resize(processed_frame, (1280, 720))
                cv2.imshow('Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            progress.update(1)
            
            # 更新前一帧
            self.prev_gray = gray
        
        progress.close()
        self.cap.release()
        self.out.release()
        if self.display_preview:
            cv2.destroyAllWindows()
    
    def _detect_motion(self, frame, gray):
        """运动目标检测 - 简化版"""
        try:
            # 使用背景减除获取前景掩码
            fg_mask = self.bg_subtractor.apply(frame)
            
            # 形态学操作去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # 更新运动历史图像(MHI)
            if self.prev_gray is not None:
                frame_diff = cv2.absdiff(gray, self.prev_gray)
                _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                self.mhi = np.where(motion_mask == 255, self.tau, np.maximum(self.mhi - 1, 0))
            
            # 结合背景减除和MHI
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(fg_mask, mhi_mask)
            
            # 提取轮廓
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤小轮廓和无效检测
            detected_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # 过滤过大或过小的区域
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # 忽略边缘的小物体
                if w < 40 or h < 40:
                    continue
                    
                # 忽略图像顶部区域
                if y < self.height * 0.2:
                    continue
                
                # 扩展边界框以确保包含完整目标
                expansion = 5
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = min(self.width - x, w + 2*expansion)
                h = min(self.height - y, h + 2*expansion)
                
                detected_objects.append((x, y, w, h))
            
            return detected_objects
            
        except Exception as e:
            print(f"运动检测错误: {str(e)}")
            return []
    
    def _update_trackers(self, frame):
        """更新现有跟踪器"""
        for tracker in self.trackers:
            try:
                # 更新跟踪器
                success = tracker.update(frame)
                
                # 如果跟踪失败，增加丢失计数
                if not success:
                    tracker.lost_count += 1
            except Exception as e:
                print(f"跟踪器更新错误: {str(e)}")
                tracker.lost_count += 1
    
    def _create_new_trackers(self, detected_objects, frame):
        """为未跟踪的目标创建新跟踪器"""
        for obj in detected_objects:
            x, y, w, h = obj
            
            # 检查是否已有跟踪器覆盖此区域
            already_tracked = False
            for tracker in self.trackers:
                try:
                    tx, ty, tw, th = tracker.bbox
                    
                    # 计算重叠面积
                    x_overlap = max(0, min(x+w, tx+tw) - max(x, tx))
                    y_overlap = max(0, min(y+h, ty+th) - max(y, ty))
                    overlap_area = x_overlap * y_overlap
                    
                    # 如果重叠面积超过小矩形面积的40%，视为已跟踪
                    min_area = min(w*h, tw*th)
                    if min_area > 0 and overlap_area > min_area * 0.4:
                        already_tracked = True
                        break
                except:
                    continue
            
            if not already_tracked:
                try:
                    # 创建新跟踪器
                    new_tracker = TargetTracker(self.next_id, [x, y, w, h], frame)
                    self.trackers.append(new_tracker)
                    self.next_id += 1
                except Exception as e:
                    print(f"创建新跟踪器错误: {str(e)}")
    
    def _cleanup_lost_trackers(self):
        """清理丢失的跟踪器"""
        self.trackers = [t for t in self.trackers if t.lost_count < self.tracker_max_lost]

def parse_args():
    parser = argparse.ArgumentParser(description='Video motion detection and tracking processor')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output video file')
    parser.add_argument('--display_preview', action='store_true', default=True,
                       help='Display preview of the processed video')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 使用命令行参数设置输入输出路径
    input_video = args.input
    output_video = args.output
    if not input_video or not output_video:
        raise ValueError("Input and output video paths must be specified.")
    display_preview = args.display_preview
    
    # 创建处理器并运行
    processor = VideoProcessor(input_video, output_video, display_preview=display_preview)
    processor.process_video()
    print("Processing completed! Output saved to", output_video)