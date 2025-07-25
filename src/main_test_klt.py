import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random

class TargetTracker:
    """基于KLT光流的目标跟踪器"""
    def __init__(self, target_id, bbox, frame):
        self.id = target_id
        self.bbox = bbox  # [x, y, w, h]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.lost_count = 0  # 目标丢失计数
        self.stable_count = 0  # 目标稳定计数
        self.last_position = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)  # 中心点
        
        # 初始化特征点
        self._init_klt_features(frame)
    
    def _init_klt_features(self, frame):
        """初始化KLT特征点"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.bbox
            
            # 确保ROI区域有效
            if w <= 10 or h <= 10 or y+h > gray.shape[0] or x+w > gray.shape[1]:
                self.features = None
                return
                
            # 创建中心区域掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w//2, h//2
            cv2.circle(mask, (center_x, center_y), min(w, h)//3, 255, -1)
            
            roi_gray = gray[y:y+h, x:x+w]
            
            # 使用GoodFeaturesToTrack检测角点
            features = cv2.goodFeaturesToTrack(
                roi_gray, 
                maxCorners=50, 
                qualityLevel=0.05,  # 更高的质量阈值
                minDistance=min(w, h)//5,  # 动态距离
                mask=mask,
                blockSize=min(w, h)//10
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

    def update(self, prev_gray, current_gray, frame):
        """使用KLT光流更新目标位置"""
        # 如果特征点太少，重新初始化
        if self.features is None or len(self.features) < 5:
            self._init_klt_features(frame)
            if self.features is None or len(self.features) < 5:
                self.lost_count += 1
                return False
        
        # 确保有前一帧特征点
        if not hasattr(self, 'features_prev') or self.features_prev is None:
            self.features_prev = self.features.copy()
        
        # 使用KLT光流跟踪特征点
        prev_pts = self.features_prev.reshape(-1, 1, 2).astype(np.float32)
        
        try:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, current_gray, 
                prev_pts, 
                None, 
                winSize=(25, 25),  # 增大窗口大小
                maxLevel=3,  # 增加金字塔层数
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS  # 使用更稳定的特征点
            )
            
            if status is None:
                self.lost_count += 1
                return False
                
            # 保留跟踪成功的点
            success_indices = status.ravel() == 1
            if np.sum(success_indices) < 5:  # 需要足够多的点
                self.lost_count += 1
                return False
            
            # 更新当前特征点
            self.features = new_features[success_indices].reshape(-1, 2)
            prev_success = prev_pts[success_indices].reshape(-1, 2)
            
            # 计算位移并过滤异常点
            displacement = self.features - prev_success
            displacement_magnitude = np.linalg.norm(displacement, axis=1)
            median_mag = np.median(displacement_magnitude)
            std_mag = np.std(displacement_magnitude)
            
            # 过滤位移异常点 (使用更宽松的范围)
            valid_indices = (displacement_magnitude > median_mag - 2.0 * std_mag) & \
                           (displacement_magnitude < median_mag + 2.0 * std_mag)
            
            if np.sum(valid_indices) < 5:
                self.lost_count += 1
                return False
                
            valid_displacement = displacement[valid_indices]
            mean_shift = np.median(valid_displacement, axis=0)
            
            # 更新边界框位置
            self.bbox[0] += int(mean_shift[0])
            self.bbox[1] += int(mean_shift[1])
            
            # 更新中心点
            current_center = (self.bbox[0] + self.bbox[2]//2, self.bbox[1] + self.bbox[3]//2)
            
            # 检查目标是否稳定移动
            dx = current_center[0] - self.last_position[0]
            dy = current_center[1] - self.last_position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 2:  # 有显著移动
                self.stable_count = 0
            else:
                self.stable_count += 1
                
            self.last_position = current_center
            
            # 保存当前特征点用于下一帧
            self.features_prev = self.features.copy()
            self.lost_count = 0
            return True
            
        except Exception as e:
            print(f"光流计算错误: {str(e)}")
            self.lost_count += 1
            return False
    
    def draw(self, frame):
        """在帧上绘制目标"""
        x, y, w, h = self.bbox
        
        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)
        
        # 绘制目标ID
        cv2.putText(frame, f"ID:{self.id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        
        # 绘制特征点
        for point in self.features:
            cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 0), -1)
        
        # 绘制中心点
        center_x = self.bbox[0] + self.bbox[2]//2
        center_y = self.bbox[1] + self.bbox[3]//2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
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
        self.tau = 8  # MHI衰减时间（秒）
        
        # 目标跟踪相关
        self.trackers = []  # 活动跟踪器列表
        self.next_id = 1    # 下一个目标ID
        self.tracker_max_lost = 10  # 目标丢失阈值
        
        # 背景减除器
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        
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
            gray = cv2.GaussianBlur(gray, (7, 7), 0)  # 更强的降噪
            
            # 运动目标检测
            detected_objects = self._detect_motion(frame, gray)
            
            # 更新现有跟踪器
            if self.prev_gray is not None:
                self._update_trackers(self.prev_gray, gray, frame)
            
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
                cv2.imshow('KLT Tracking Preview', preview)
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
        """运动目标检测"""
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
                _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
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
                if area < 800 or area > 50000:  # 过滤过大或过小的区域
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # 忽略边缘的小物体
                if w < 50 or h < 50:
                    continue
                    
                # 忽略图像顶部区域
                if y < self.height * 0.2:
                    continue
                
                # 确保边界框在图像范围内
                x = max(0, x)
                y = max(0, y)
                w = min(self.width - x, w)
                h = min(self.height - y, h)
                
                detected_objects.append((x, y, w, h))
            
            return detected_objects
            
        except Exception as e:
            print(f"运动检测错误: {str(e)}")
            return []
    
    def _update_trackers(self, prev_gray, current_gray, frame):
        """更新现有跟踪器"""
        for tracker in self.trackers:
            try:
                # 更新跟踪器
                success = tracker.update(prev_gray, current_gray, frame)
                
                # 如果跟踪失败，增加丢失计数
                if not success:
                    tracker.lost_count += 1
                else:
                    # 减少丢失计数如果成功
                    tracker.lost_count = max(0, tracker.lost_count - 1)
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
                    center_x = tx + tw//2
                    center_y = ty + th//2
                    
                    # 检查新目标中心是否在现有跟踪器内
                    if (x <= center_x <= x+w) and (y <= center_y <= y+h):
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
    parser = argparse.ArgumentParser(description='Video motion detection and KLT tracking processor')
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