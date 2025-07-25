import cv2
import numpy as np
from core.motion import MotionDetector
from core.tracking import TargetTracker
from utils.video_io import VideoProcessorIO
from utils.visualization import Visualizer
from config import Config

class VideoProcessor:
    def __init__(self, input_path, output_path):
        # 初始化视频I/O
        self.video_io = VideoProcessorIO(input_path, output_path)
        
        # 初始化运动检测器
        self.motion_detector = MotionDetector(
            (self.video_io.height, self.video_io.width)
        )
        
        # 目标跟踪相关
        self.trackers = []  # 活动跟踪器列表
        self.next_id = 1    # 下一个目标ID
        self.prev_gray = None  # 前一帧灰度图
    
    def process_video(self):
        """主处理流程"""
        progress = self.video_io.get_progress()
        
        while True:
            ret, frame = self.video_io.read_frame()
            if not ret:
                break
            
            # 预处理 - 转换为灰度图
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
            
            # 运动目标检测
            detected_objects, _ = self.motion_detector.detect(frame)
            
            # 更新现有跟踪器
            self._update_trackers(current_gray, frame)
            
            # 为未跟踪的目标创建新跟踪器
            self._create_new_trackers(detected_objects, frame)
            
            # 清理丢失的跟踪器
            self._cleanup_lost_trackers()
            
            # 绘制结果
            output_frame = frame.copy()
            for tracker in self.trackers:
                output_frame = Visualizer.draw_tracker(output_frame, tracker)
            
            # 写入输出视频
            self.video_io.write_frame(output_frame)
            progress.update(1)
            
            # 更新前一帧
            self.prev_gray = current_gray
        
        progress.close()
        self.video_io.release()
    
    def _update_trackers(self, current_gray, current_frame):
        """更新现有跟踪器"""
        if self.prev_gray is None:
            return
            
        for tracker in self.trackers:
            success = tracker.update(
                self.prev_gray, current_gray, current_frame
            )
            if not success:
                tracker.lost_count += 1
    
    def _create_new_trackers(self, detected_objects, frame):
        """创建新跟踪器"""
        for obj in detected_objects:
            if not self._is_tracked(obj):
                new_tracker = TargetTracker(self.next_id, obj, frame)
                self.trackers.append(new_tracker)
                self.next_id += 1
    
    def _is_tracked(self, new_obj):
        """检查目标是否已被跟踪"""
        x, y, w, h = new_obj
        new_area = w * h
        
        for tracker in self.trackers:
            tx, ty, tw, th = tracker.bbox
            tracker_area = tw * th
            
            # 计算中心点距离
            cx1, cy1 = x + w//2, y + h//2
            cx2, cy2 = tx + tw//2, ty + th//2
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # 检查是否重叠或接近
            if distance < 50 or (new_area > tracker_area * 0.5 and new_area < tracker_area * 1.5):
                return True
        return False
    
    def _cleanup_lost_trackers(self):
        """清理丢失的跟踪器"""
        self.trackers = [t for t in self.trackers if t.lost_count < Config.TRACKER_MAX_LOST]

if __name__ == "__main__":
    input_video = "game_recording.mp4"
    output_video = "tracking_output.mp4"
    
    processor = VideoProcessor(input_video, output_video)
    processor.process_video()
    print(f"处理完成! 输出保存至: {output_video}")