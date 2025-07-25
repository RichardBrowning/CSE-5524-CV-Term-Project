import cv2
from tqdm import tqdm

class VideoProcessorIO:
    def __init__(self, input_path, output_path):
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
    
    def read_frame(self):
        """读取下一帧"""
        ret, frame = self.cap.read()
        return ret, frame
    
    def write_frame(self, frame):
        """写入帧到输出视频"""
        self.out.write(frame)
    
    def release(self):
        """释放资源"""
        self.cap.release()
        self.out.release()
    
    def get_progress(self):
        """获取处理进度"""
        return tqdm(total=self.total_frames, desc="Processing Video")