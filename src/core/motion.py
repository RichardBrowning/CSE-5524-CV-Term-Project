import cv2
import numpy as np
from config import Config

class MotionDetector:
    def __init__(self, frame_shape):
        self.height, self.width = frame_shape[:2]
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=Config.BG_HISTORY, 
            varThreshold=16, 
            detectShadows=False
        )
        self.prev_gray = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    def detect(self, frame):
        """检测运动目标"""
        # 预处理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 背景减除
        fg_mask = self.bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        
        # 更新运动历史图像(MHI)
        if self.prev_gray is not None:
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
            self.mhi = np.where(motion_mask == 1, Config.MHI_TAU, np.maximum(self.mhi - 1, 0))
        
        # 结合检测方法
        combined_mask = cv2.bitwise_and(fg_mask, (self.mhi > 0).astype(np.uint8) * 255)
        
        # 提取轮廓
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < Config.MIN_CONTOUR_AREA or area > Config.MAX_CONTOUR_AREA:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            if w < 20 or h < 20 or y < self.height * 0.2:
                continue
                
            detected_objects.append((x, y, w, h))
        
        # 更新前一帧
        self.prev_gray = gray
        
        return detected_objects, combined_mask