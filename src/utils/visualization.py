import cv2
import numpy as np
from config import Config

class Visualizer:
    @staticmethod
    def draw_tracker(frame, tracker):
        """绘制单个跟踪器"""
        x, y, w, h = tracker.bbox
        
        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x+w, y+h), tracker.color, 2)
        
        # 绘制ID
        cv2.putText(frame, f"ID:{tracker.id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker.color, 2)
        
        # 绘制分类
        target_type = FeatureClassifier.classify(tracker.bbox)
        cv2.putText(frame, target_type, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker.color, 2)
        
        # 绘制特征点
        if Config.DRAW_FEATURES and tracker.features is not None:
            for point in tracker.features:
                cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 255), -1)
        
        # 绘制轨迹
        if Config.DRAW_HISTORY and len(tracker.history) > 1:
            for i in range(1, len(tracker.history)):
                cv2.line(frame, tracker.history[i-1], tracker.history[i], tracker.color, 2)
        
        return frame
    
    @staticmethod
    def visualize_mhi(mhi):
        """可视化MHI"""
        mhi_norm = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(mhi_norm, cv2.COLOR_GRAY2BGR)
    
    @staticmethod
    def draw_detections(frame, detections):
        """绘制检测结果"""
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Motion', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame