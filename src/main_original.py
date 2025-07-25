import cv2
import numpy as np
import argparse
from tqdm import tqdm  # 进度条工具

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=True):
        # Routine: specify the i/o video's properties
        self.cap = cv2.VideoCapture(input_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # was reducing FPS for processing speed and stability, now that does not help
        self.display_preview = display_preview
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Video output setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        # Initialize motion detection variables
        self.prev_gray = None
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 15 # MHI lifetime
        
    def process_video(self):
        progress = tqdm(total=self.total_frames, desc="Processing Video")
        
        """Main loop start"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = self._detect_motion(frame)
            
            # write to output
            self.out.write(processed_frame)
            # preview 
            if self.display_preview:
                cv2.imshow("Motion Detection Preview", processed_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # 按ESC或Q退出
                    break
            progress.update(1)
        """Main loop end"""
        
        progress.close()
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
    
    def _detect_motion(self, frame):
        """cv2: cvtColor (
            frame -> original frame
            cv2.COLOR_BGR2GRAY -> convert to grayscale
            )"""
        """cv2: GaussianBlur (
            gray -> grayscale frame
            (5, 5) -> kernel size for blurring
            0 -> standard deviation in X and Y direction
            )"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # first frame exception
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        # absolute difference between current and previous frame
        frame_diff = cv2.absdiff(gray, self.prev_gray)
        # draw motion mask according to the difference
        _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
        
        # - renew MHI
        self.mhi = np.where(motion_mask == 1, self.tau, np.maximum(self.mhi - 1, 0))
        
        # get area of mothion from MHI
        mhi_visual = self._visualize_mhi()
        contours = self._extract_motion_regions()
        
        # draw MHI on the original frame
        output_frame = frame.copy()
        cv2.addWeighted(output_frame, 0.7, mhi_visual, 0.3, 0, output_frame)

        # draw detected motion regions
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 过滤小噪点
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output_frame, 'Motion', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        self.prev_gray = gray
        return output_frame
    
    def _visualize_mhi(self):
        """Visualize MHI as a color image"""
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

    # Create VideoProcessor instance and process the video
    processor = VideoProcessor(input_video, output_video, display_preview=display_preview)
    processor.process_video()
    print("Processing completed! Output saved to", output_video)