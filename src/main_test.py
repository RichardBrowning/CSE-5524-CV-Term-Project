"""
生成的版本：先使用背景减除法检测运动，再使用KLT光流法跟踪目标。
然后使用Mean-Shift优化目标位置，最后绘制目标和轨迹。
"""

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
        self.history = []  # track history
        self.lost_count = 0  # target lost count

        # Initialize target features
        x, y, w, h = bbox
        # Region of Interest
        self.roi = frame[y:y+h, x:x+w]

        # Extract KLT features
        self._init_klt_features(frame)

        # Initialize Mean-Shift tracker
        self._init_meanshift_tracker(frame)
    
    def _init_klt_features(self, frame):
        """Initialize KLT features - fixed version"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.bbox

            # Ensure ROI is valid
            if w <= 0 or h <= 0 or y+h > gray.shape[0] or x+w > gray.shape[1]:
                self.features = None
                return
                
            roi_gray = gray[y:y+h, x:x+w]
            
            # cv function to find good features
            """goodFeaturesToTrack (
                roi_gray -> region of interest in grayscale
                maxCorners -> maximum number of corners to return
                qualityLevel -> quality level for corner detection
                minDistance -> minimum distance between corners
                blockSize -> size of the neighborhood considered for corner detection
            )"""
            features = cv2.goodFeaturesToTrack(
                roi_gray, maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7
            )
            
            if features is not None and len(features) > 0:
                # treat local features as absolute coordinates
                features = features.reshape(-1, 2)
                features[:, 0] += x
                features[:, 1] += y
                self.features = features
            else:
                self.features = np.empty((0, 2))
        except Exception as e:
            print(f"Feature initialization error: {str(e)}")
            self.features = np.empty((0, 2))
    
    def _init_meanshift_tracker(self, frame):
        """Initialize Mean-Shift tracker"""
        x, y, w, h = self.bbox
        roi = frame[y:y+h, x:x+w]

        # Set up ROI histogram
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Save initial position
        self.track_window = (x, y, w, h)

    def update(self, prev_gray, current_gray, current_frame):
        """Update target position - fixed version"""
        # If feature points are too few, reinitialize
        if self.features is None or len(self.features) < 5:
            self._init_klt_features(cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR))
            if self.features is None or len(self.features) < 2:
                self.lost_count += 1
                return False

        # Ensure we have previous frame features
        if not hasattr(self, 'features_prev') or self.features_prev is None:
            self.features_prev = self.features.copy()

        # Use KLT optical flow to track features - add error handling
        prev_pts = self.features_prev.reshape(-1, 1, 2).astype(np.float32)

        # Check if feature points are valid
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
            
            # NOTE: status is None if no features are found
            if status is None:
                self.lost_count += 1
                return False
                
            success_indices = status.ravel() == 1
            if np.sum(success_indices) < 3:  # 确保有足够的点
                self.lost_count += 1
                return False
            
            # renew features and previous points
            self.features = new_features[success_indices].reshape(-1, 2)
            prev_success = prev_pts[success_indices].reshape(-1, 2)
            
            # Calculate mean shift
            displacement = self.features - prev_success
            mean_shift = np.median(displacement, axis=0)

            # Update bounding box position
            self.bbox[0] += int(mean_shift[0])
            self.bbox[1] += int(mean_shift[1])

            # Use Mean-Shift to optimize position
            self._update_meanshift(current_frame)
            
            center_x = int(self.bbox[0] + self.bbox[2]//2)
            center_y = int(self.bbox[1] + self.bbox[3]//2)
            self.history.append((center_x, center_y))
            
            if len(self.history) > 100:
                self.history.pop(0)
            
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
        
        # tuple: ([if either terminating epsilon and maxCount (enabled together) are met], maxCount, epsilon) 
        termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # to HSVspace (hue and satur separated, it may not be necessary)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # back projection
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # get new position using Mean-Shift
        _, track_window = cv2.meanShift(dst, track_window, termination_criteria)

        # Update bounding box
        self.bbox = list(track_window)
        self.track_window = track_window
    
    def classify_target(self):
        """
        Simple target classification (Pedestrian/Vehicle)
        """
        _, _, w, h = self.bbox
        aspect_ratio = w / float(h)
        area = w * h

        # Classification rules (can be adjusted based on actual scenarios)
        if area < 2000:  # neglect object
            return "Pedestrian"
        else:
            if aspect_ratio > 1.5:  # width greater than height
                return "Vehicle"
            else:
                return "Pedestrian"
    
    def draw(self, frame, draw_history=True):
        """在帧上绘制目标和轨迹"""
        x, y, w, h = self.bbox
        
        # draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)
        
        # on top of hte bounding box, print ID
        cv2.putText(frame, f"ID:{self.id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        
        # draw category
        target_type = self.classify_target()
        cv2.putText(frame, target_type, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        # draw features
        for point in self.features:
            cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 255), -1)

        # draw motion trajectory
        if draw_history and len(self.history) > 1:
            for i in range(1, len(self.history)):
                cv2.line(frame, self.history[i-1], self.history[i], self.color, 2)
        
        return frame

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=True):
        # Initialize video capture and output 初始化视频输入
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error Opening the file: {input_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.display_preview = display_preview
        
        # video output setup 初始化视频输出
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not self.out.isOpened():
            raise ValueError(f"Error creating the output video: {output_path}")

        
        # Initialize motion detection variables 初始化运动检测变量
        self.prev_gray = None
        # Initialize MHI (Motion History Image) and tau
        # MHI是一个浮点型的图像，记录运动历史
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 15  # MHI衰减时间（秒）
        
        # tracker list, next target ID, and max lost count 追踪实例list，下一个目标id，最大丢失计数
        self.trackers = [] 
        self.next_id = 1
        self.tracker_max_lost = 10 
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
    def process_video(self):
        # 用tqdm显示进度条
        progress = tqdm(total=self.total_frames, desc="Processing Video")

        # 循环每一帧
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # pre-processing - convert to grayscale and apply Gaussian blur
            # 预处理 - 转换为灰度图并应用高斯模糊
            processed_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # motion detection
            # 调用自身的_detect_motion，使用背景减除法和MHI检测运动
            detected_objects = self._detect_motion(frame, gray)

            # update existing trackers 更新现有跟踪器
            self._update_trackers(detected_objects, gray, processed_frame)

            # create new trackers for untracked objects 创建新的追踪器
            self._create_new_trackers(detected_objects, frame)
            
            # clean up lost trackers 清除脱锁的跟踪器
            self._cleanup_lost_trackers()
            
            # draw results 绘制结果
            for tracker in self.trackers:
                processed_frame = tracker.draw(processed_frame)
            
            # write to output video 写入输出视频
            self.out.write(processed_frame)

            # preview display 预览显示 （如果勾选了选项）
            if self.display_preview:
                preview = cv2.resize(processed_frame, (1280, 720))
                cv2.imshow('Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # update progress bar 更新进度条
            progress.update(1)
            
            # 更新前一帧，目前帧已成为过去
            self.prev_gray = gray

        # 处理完成，关闭进度条和释放资源
        progress.close()
        self.cap.release()
        self.out.release()
        # 关闭预览窗口
        if self.display_preview:
            cv2.destroyAllWindows()

    def _detect_motion(self, frame, gray):
        """fixed: Detect motion using background subtraction and MHI"""
        try:
            # Apply background subtraction 应用背景减除（调用OpenCV的背景减除器BackgroundSubtractorMOG2）
            fg_mask = self.bg_subtractor.apply(frame)

            """ Morphological operations to remove noise 变形学操作去除噪声 """
            # getStructuringElement用于创建一个椭圆形的结构元素
            # MORPH_ELLIPSE -> 形态学操作的类型，这里使用
            # (5, 5) -> 结构元素的大小
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # MORPH_OPEN -> 开运算，去除小噪点
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            # MORPH_CLOSE -> 闭运算，填充小孔洞
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Update motion history image (MHI) 更新运动历史图像
            if self.prev_gray is None:
                self.prev_gray = gray
                return []

            # frame difference 计算当前帧与前一帧的差异
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            # create motion mask based on the difference 创建运动掩码
            _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            # 结合运动掩码和背景减除掩码
            # np.where() -> 根据条件选择元素
            # motion_mask == 255 -> 如果运动掩码为255（表示有运动
            # self.tau -> 设置MHI的值为tau（运动持续时间）
            # np.maximum() -> 保持MHI的值不小于0
            # self.mhi - 1 -> MHI衰减1
            # 所以这一行的作用是：mhi = 如果运动掩码为255，则设置MHI为tau，否则将MHI衰减1，但不小于0
            self.mhi = np.where(motion_mask == 255, self.tau, np.maximum(self.mhi - 1, 0))

            # background subtraction mask 背景减除掩码，由mhi（如果mhi>0，则为255，否则为0）
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            # combine motion mask and MHI mask 结合运动掩码和MHI掩码
            combined_mask = cv2.bitwise_and(fg_mask, mhi_mask)
            
            # find contours in the combined mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # filter contours based on area and size
            detected_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > 10000:  # 根据场景调整
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                if w < 20 or h < 20:
                    continue
                    
                if y < self.height * 0.2:
                    continue
                    
                detected_objects.append((x, y, w, h))
            
            return detected_objects
            
        except Exception as e:
            print(f"Error detecting motion: {str(e)}")
            return []
    
    def _update_trackers(self, detected_objects, current_gray, current_frame):
        """更新现有跟踪器 - 修复版本"""
        if self.prev_gray is None:
            return
            
        for tracker in self.trackers:
            try:
                success = tracker.update(self.prev_gray, current_gray, current_frame)
                
                if not success:
                    tracker.lost_count += 1
            except Exception as e:
                print(f"跟踪器更新错误: {str(e)}")
                tracker.lost_count += 1
    
    def _create_new_trackers(self, detected_objects, frame):
        """为未跟踪的目标创建新跟踪器 - 修复版本"""
        for obj in detected_objects:
            x, y, w, h = obj
            
            already_tracked = False
            for tracker in self.trackers:
                try:
                    tx, ty, tw, th = tracker.bbox
                    
                    x_overlap = max(0, min(x+w, tx+tw) - max(x, tx))
                    y_overlap = max(0, min(y+h, ty+th) - max(y, ty))
                    overlap_area = x_overlap * y_overlap
                    
                    min_area = min(w*h, tw*th)
                    if min_area > 0 and overlap_area > min_area * 0.3:
                        already_tracked = True
                        break
                except:
                    continue
            
            if not already_tracked:
                try:
                    # create new tracker
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

    # Create processor and run
    processor = VideoProcessor(input_video, output_video, display_preview=display_preview)
    processor.process_video()
    print("Processing completed! Output saved to", output_video)