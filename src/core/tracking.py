import cv2
import numpy as np
import random
from config import Config

class TargetTracker:
    def __init__(self, target_id, bbox, frame):
        self.id = target_id
        self.bbox = list(bbox)  # [x, y, w, h]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.history = []  # 轨迹历史
        self.lost_count = 0  # 丢失计数
        
        # 初始化目标特征
        self._init_klt_features(frame)
        self._init_meanshift_tracker(frame)
        
        # 保存前一帧特征点
        self.features_prev = self.features.copy() if self.features is not None else None
    
    def _init_klt_features(self, frame):
        """初始化KLT特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox
        roi_gray = gray[y:y+h, x:x+w]
        
        # 检测角点
        features = cv2.goodFeaturesToTrack(
            roi_gray, maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7
        )
        
        if features is not None:
            # 转换到全局坐标
            features = features.reshape(-1, 2)
            features[:, 0] += x
            features[:, 1] += y
            self.features = features
        else:
            self.features = np.empty((0, 2))
    
    def _init_meanshift_tracker(self, frame):
        """初始化Mean-Shift跟踪器"""
        x, y, w, h = self.bbox
        roi = frame[y:y+h, x:x+w]
        
        # 设置ROI的直方图
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        self.track_window = (x, y, w, h)
    
    def update(self, prev_gray, current_gray, current_frame):
        """更新目标位置"""
        # 特征点不足时重新初始化
        if self.features is None or len(self.features) < Config.MIN_FEATURES:
            self._init_klt_features(cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR))
            if self.features is None or len(self.features) < 2:
                self.lost_count += 1
                return False
        
        # KLT光流跟踪
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, current_gray, 
            self.features.astype(np.float32), 
            None, winSize=Config.KLT_WIN_SIZE, 
            maxLevel=Config.KLT_MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 保留成功跟踪的点
        success_indices = status.ravel() == 1
        self.features = new_features[success_indices]
        
        if len(self.features) < 3:
            self.lost_count += 1
            return False
        
        # 计算中值位移
        mean_shift = np.median(self.features - self.features_prev, axis=0)
        self.bbox[0] += int(mean_shift[0])
        self.bbox[1] += int(mean_shift[1])
        
        # Mean-Shift优化
        self._update_meanshift(current_frame)
        
        # 更新轨迹
        center = (int(self.bbox[0] + self.bbox[2]//2), 
                 int(self.bbox[1] + self.bbox[3]//2))
        self.history.append(center)
        if len(self.history) > Config.HISTORY_LENGTH:
            self.history.pop(0)
        
        # 保存当前特征点
        self.features_prev = self.features.copy()
        self.lost_count = 0
        return True
    
    def _update_meanshift(self, frame):
        """使用Mean-Shift优化位置"""
        x, y, w, h = self.bbox
        track_window = (x, y, w, h)
        
        # 设置终止条件
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        
        # 计算反向投影
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # 应用Mean-Shift
        _, track_window = cv2.meanShift(dst, track_window, criteria)
        self.bbox = list(track_window)