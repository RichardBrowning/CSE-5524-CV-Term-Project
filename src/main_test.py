import cv2
import numpy as np
from tqdm import tqdm
import random
import argparse

class TargetTracker:
    """单个目标的跟踪器"""
    def __init__(self, target_id, bbox, frame):
        self.id = target_id
        self.bbox = bbox  # [x, y, w, h]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.history = []  # 跟踪历史
        self.lost_count = 0  # 目标丢失计数
        
        # 初始化目标特征
        x, y, w, h = bbox
        self.roi = frame[y:y+h, x:x+w]
        
        # 提取KLT特征点
        self._init_klt_features(frame)
        
        # 初始化Mean-Shift跟踪器
        self._init_meanshift_tracker(frame)
    
    def _init_klt_features(self, frame):
        """初始化KLT特征点 - 修复版本"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.bbox
            
            # 确保ROI区域有效
            if w <= 0 or h <= 0 or y+h > gray.shape[0] or x+w > gray.shape[1]:
                self.features = None
                return
                
            roi_gray = gray[y:y+h, x:x+w]
            
            # 使用GoodFeaturesToTrack检测角点
            features = cv2.goodFeaturesToTrack(
                roi_gray, maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7
            )
            
            if features is not None and len(features) > 0:
                # 将局部坐标转换为全局坐标
                features = features.reshape(-1, 2)
                features[:, 0] += x
                features[:, 1] += y
                self.features = features
            else:
                self.features = np.empty((0, 2))
        except Exception as e:
            print(f"特征点初始化错误: {str(e)}")
            self.features = np.empty((0, 2))
    
    def _init_meanshift_tracker(self, frame):
        """初始化Mean-Shift跟踪器"""
        x, y, w, h = self.bbox
        roi = frame[y:y+h, x:x+w]
        
        # 设置ROI的直方图
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        # 保存初始位置
        self.track_window = (x, y, w, h)

    def update(self, prev_gray, current_gray, current_frame):
        """更新目标位置 - 修复版本"""
        # 如果特征点太少，重新初始化
        if self.features is None or len(self.features) < 5:
            self._init_klt_features(cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR))
            if self.features is None or len(self.features) < 2:
                self.lost_count += 1
                return False
        
        # 确保有前一帧特征点
        if not hasattr(self, 'features_prev') or self.features_prev is None:
            self.features_prev = self.features.copy()
        
        # 使用KLT光流跟踪特征点 - 添加错误处理
        prev_pts = self.features_prev.reshape(-1, 1, 2).astype(np.float32)
        
        # 检查特征点是否有效
        if prev_pts.size == 0:
            self.lost_count += 1
            return False
        
        try:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, current_gray, 
                prev_pts, 
                None, winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # 确保status不为None
            if status is None:
                self.lost_count += 1
                return False
                
            # 保留跟踪成功的点
            success_indices = status.ravel() == 1
            if np.sum(success_indices) < 3:  # 确保有足够的点
                self.lost_count += 1
                return False
            
            # 更新当前特征点和前一帧特征点
            self.features = new_features[success_indices].reshape(-1, 2)
            prev_success = prev_pts[success_indices].reshape(-1, 2)
            
            # 计算特征点的中值位移 (确保相同数量的点)
            displacement = self.features - prev_success
            mean_shift = np.median(displacement, axis=0)
            
            # 更新边界框位置
            self.bbox[0] += int(mean_shift[0])
            self.bbox[1] += int(mean_shift[1])
            
            # 使用Mean-Shift优化位置
            self._update_meanshift(current_frame)
            
            # 记录历史位置
            center_x = int(self.bbox[0] + self.bbox[2]//2)
            center_y = int(self.bbox[1] + self.bbox[3]//2)
            self.history.append((center_x, center_y))
            
            # 限制历史长度
            if len(self.history) > 100:
                self.history.pop(0)
            
            # 保存当前特征点用于下一帧
            self.features_prev = self.features.copy()
            self.lost_count = 0
            return True
            
        except Exception as e:
            print(f"光流计算错误: {str(e)}")
            self.lost_count += 1
            return False
    
    def _update_meanshift(self, frame):
        """使用Mean-Shift优化目标位置"""
        x, y, w, h = self.bbox
        track_window = (x, y, w, h)
        
        # 设置终止条件
        termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 计算反向投影
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # 应用Mean-Shift获取新位置
        _, track_window = cv2.meanShift(dst, track_window, termination_criteria)
        
        # 更新边界框
        self.bbox = list(track_window)
        self.track_window = track_window
    
    def classify_target(self):
        """简单分类目标（行人/车辆）"""
        _, _, w, h = self.bbox
        aspect_ratio = w / float(h)
        area = w * h
        
        # 分类规则（可根据实际场景调整）
        if area < 2000:  # 小目标
            return "Pedestrian"
        else:
            if aspect_ratio > 1.5:  # 宽大于高
                return "Vehicle"
            else:
                return "Pedestrian"
    
    def draw(self, frame, draw_history=True):
        """在帧上绘制目标和轨迹"""
        x, y, w, h = self.bbox
        
        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)
        
        # 绘制目标ID
        cv2.putText(frame, f"ID:{self.id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        
        # 绘制分类
        target_type = self.classify_target()
        cv2.putText(frame, target_type, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        
        # 绘制特征点
        for point in self.features:
            cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 255), -1)
        
        # 绘制运动轨迹
        if draw_history and len(self.history) > 1:
            for i in range(1, len(self.history)):
                cv2.line(frame, self.history[i-1], self.history[i], self.color, 2)
        
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
        self.tau = 15  # MHI衰减时间（秒）
        
        # 目标跟踪相关
        self.trackers = []  # 活动跟踪器列表
        self.next_id = 1    # 下一个目标ID
        self.tracker_max_lost = 10  # 目标丢失阈值
        
        # 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
    def process_video(self):
        """主处理循环 - 修复版本"""
        progress = tqdm(total=self.total_frames, desc="Processing Video")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 预处理 - 降噪和灰度转换
            processed_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 运动目标检测
            detected_objects = self._detect_motion(frame, gray)
            
            # 更新现有跟踪器
            self._update_trackers(detected_objects, gray, processed_frame)
            
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
        """改进的运动目标检测 - 修复版本"""
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
                _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
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
                if area < 100 or area > 10000:  # 根据场景调整
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # 忽略边缘的小物体
                if w < 20 or h < 20:
                    continue
                    
                # 忽略图像顶部区域
                if y < self.height * 0.2:
                    continue
                    
                detected_objects.append((x, y, w, h))
            
            return detected_objects
            
        except Exception as e:
            print(f"运动检测错误: {str(e)}")
            return []
    
    def _update_trackers(self, detected_objects, current_gray, current_frame):
        """更新现有跟踪器 - 修复版本"""
        if self.prev_gray is None:
            return
            
        for tracker in self.trackers:
            try:
                # 更新跟踪器
                success = tracker.update(self.prev_gray, current_gray, current_frame)
                
                # 如果跟踪失败，增加丢失计数
                if not success:
                    tracker.lost_count += 1
            except Exception as e:
                print(f"跟踪器更新错误: {str(e)}")
                tracker.lost_count += 1
    
    def _create_new_trackers(self, detected_objects, frame):
        """为未跟踪的目标创建新跟踪器 - 修复版本"""
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
                    
                    # 如果重叠面积超过小矩形面积的30%，视为已跟踪
                    min_area = min(w*h, tw*th)
                    if min_area > 0 and overlap_area > min_area * 0.3:
                        already_tracked = True
                        break
                except:
                    continue
            
            if not already_tracked:
                try:
                    # 创建新跟踪器
                    new_tracker = TargetTracker(self.next_id, [x, y, w, h], frame)
                    if hasattr(new_tracker, 'features'):
                        new_tracker.features_prev = new_tracker.features.copy() if new_tracker.features is not None else None
                    self.trackers.append(new_tracker)
                    self.next_id += 1
                except Exception as e:
                    print(f"创建新跟踪器错误: {str(e)}")
    
    def _cleanup_lost_trackers(self):
        """清理丢失的跟踪器"""
        self.trackers = [t for t in self.trackers if t.lost_count < self.tracker_max_lost]

        
def parse_args():
    parser = argparse.ArgumentParser(description='Video motion detection processor')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output video file')
    parser.add_argument('--display_preview', action='store_true', default=True,
                       help='Display preview of the processed video')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Use command line arguments for input and output paths
    input_video = args.input
    output_video = args.output
    if not input_video or not output_video:
        raise ValueError("Input and output video paths must be specified.")
    display_preview = args.display_preview
    
    # 创建处理器并运行
    processor = VideoProcessor(input_video, output_video, display_preview=display_preview)
    processor.process_video()
    print("Processing completed! Output saved to", output_video)