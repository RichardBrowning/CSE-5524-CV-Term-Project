import cv2
import numpy as np
import argparse
from tqdm import tqdm  # 进度条工具

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=True):
        # 视频输入/输出设置
        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.display_preview = display_preview
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建视频写入器（保持原始分辨率）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # 运动检测相关变量
        self.prev_gray = None
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 15  # MHI衰减时间（秒）
        
    def process_video(self):
        """主处理循环"""
        progress = tqdm(total=self.total_frames, desc="Processing Video")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 运动检测核心处理
            processed_frame = self._detect_motion(frame)
            
            # 写入输出视频
            self.out.write(processed_frame)
            if self.display_preview:
                cv2.imshow("Motion Detection Preview", processed_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # 按ESC或Q退出
                    break
            progress.update(1)
        
        progress.close()
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
    
    def _detect_motion(self, frame):
        """运动目标检测实现"""
        # 1. 转换为灰度图并降噪
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. 初始化前一帧（第一次运行）
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # 3. 计算帧间差分
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
        
        # 4. 更新运动历史图像(MHI)
        self.mhi = np.where(motion_mask == 1, self.tau, np.maximum(self.mhi - 1, 0))
        
        # 5. 从MHI提取运动区域
        mhi_visual = self._visualize_mhi()
        contours = self._extract_motion_regions()
        
        # 6. 可视化结果
        output_frame = frame.copy()
        cv2.addWeighted(output_frame, 0.7, mhi_visual, 0.3, 0, output_frame)
        
        # 绘制检测到的运动区域
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 过滤小噪点
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output_frame, 'Motion', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 更新前一帧
        self.prev_gray = gray
        
        return output_frame
    
    def _visualize_mhi(self):
        """将MHI转换为可视化格式"""
        mhi_norm = cv2.normalize(self.mhi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(mhi_norm, cv2.COLOR_GRAY2BGR)
    
    def _extract_motion_regions(self):
        """从MHI提取运动区域轮廓"""
        mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mhi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

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