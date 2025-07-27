import cv2
import numpy as np
import argparse
from tqdm import tqdm  # 进度条工具

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=True):
        # Routine: specify the i/o video's properties
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {input_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # was reducing FPS for processing speed and stability, now that does not help
        self.display_preview = display_preview
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Video output setup
        fourcc_writer = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc_writer, self.fps, (self.width, self.height))
        if not self.out.isOpened():
            raise ValueError(f"Error creating the output video: {output_path}")

        # Initialize motion detection variables
        self.prev_gray = None
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 15 # MHI lifetime

        # tracker list next target ID, max lost count not here

        # but introduce bg_subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    def process_video(self):
        progress = tqdm(total=self.total_frames, desc="Processing Video")

        """Main loop start"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # NEW: PREPROCESSING FRAME - convert to greyscale and apply gaussian blur
            # frame_copy = frame.copy()
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

            # MOTION DETECTION
            # get area of motion from MHI
            mhi_visual = self._visualize_mhi()
            processed_frame = frame.copy()
            # 这是MHI（0.3的opacity）叠加到原始帧（0.7的opacity）上，第四个参数0代表没有偏移，processed_frame是输出图像
            cv2.addWeighted(processed_frame, 0.7, mhi_visual, 0.3, 0, processed_frame)
            detected_objects = self._detect_motion(frame, gray)

            # 新增：重叠检测与完全包含处理
            detected_objects = self._remove_contained_rectangles(detected_objects)

            # draw detected objects on the frame
            for (x, y, w, h) in detected_objects:
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(processed_frame, 'Motion', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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

    def _remove_contained_rectangles(self, rects):
        """
        移除完全被其他矩形包含的矩形
        :param rects: 矩形列表，格式为[(x, y, w, h), ...]
        :return: 过滤后的矩形列表
        """
        if len(rects) < 2:
            return rects

        # 按面积从大到小排序
        sorted_rects = sorted(rects, key=lambda r: r[2]*r[3], reverse=True)
        to_remove = set()

        # 检查每个矩形是否被更大的矩形完全包含
        for i in range(len(sorted_rects)):
            for j in range(i+1, len(sorted_rects)):
                if self._is_contained(sorted_rects[j], sorted_rects[i]):
                    to_remove.add(j)

        # 创建新列表，排除需要移除的矩形
        filtered_rects = [sorted_rects[i] for i in range(len(sorted_rects)) if i not in to_remove]
        return filtered_rects

    def _is_contained(self, rectA, rectB):
        """
        检查矩形A是否完全包含在矩形B中
        :param rectA: 矩形A (x, y, w, h)
        :param rectB: 矩形B (x, y, w, h)
        :return: True如果A完全包含在B中，否则False
        """
        # 矩形A的边界
        a_x1, a_y1 = rectA[0], rectA[1]
        a_x2, a_y2 = a_x1 + rectA[2], a_y1 + rectA[3]

        # 矩形B的边界
        b_x1, b_y1 = rectB[0], rectB[1]
        b_x2, b_y2 = b_x1 + rectB[2], b_y1 + rectB[3]

        # 检查A是否完全在B内
        return (a_x1 >= b_x1 and a_y1 >= b_y1 and
                a_x2 <= b_x2 and a_y2 <= b_y2)

    def _detect_motion(self, frame, gray):
        try:
            foreground_mask = self.bg_subtractor.apply(gray)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # apply morphological (形态学) operations to reduce noise: 1. open op to remove small noise, 2. close op to fill small holes
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

            # first frame exception
            if self.prev_gray is None:
                self.prev_gray = gray
                return []

            # absolute difference between current and previous frame
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            # draw motion mask according to the difference
            _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
            # NEW: ADD BACKGROUND SUBTRACTION MASK
            _, bg_mask = cv2.threshold(foreground_mask, 127, 1, cv2.THRESH_BINARY)
            # NEW: combine motion mask and background mask
            combined_mask = cv2.bitwise_and(motion_mask, bg_mask)

            # - renew MHI
            self.mhi = np.where(combined_mask == 1, self.tau, np.maximum(self.mhi - 1, 0))

            # from MHI extract silhouettes of moving objects
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            final_mask = cv2.bitwise_and(mhi_mask, foreground_mask)
            """
            # get area of motion from MHI
            mhi_visual = self._visualize_mhi()
            """
            # NOTE: TODO: try returning this instead of frame with overlays
            # create binary mask from MHI
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            # combine MHI mask with foreground mask
            final_mask = cv2.bitwise_and(mhi_mask, foreground_mask)
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # NOTE: extract_motion_regions() returns contours in Sequence[MatLike], where MatLike is np.ndarray or cv2.Mat

            """
            # draw MHI on the original frame
            output_frame = frame.copy()
            cv2.addWeighted(output_frame, 0.7, mhi_visual, 0.3, 0, output_frame)
            """
            # create detected motion regions/(objects?)
            detected_objects = []
            # draw detected motion regions

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # 过滤小噪点
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_objects.append((x, y, w, h))
                    """
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output_frame, 'Motion', (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.prev_gray = gray
            return output_frame
            """
            return detected_objects
        except Exception as e:
            print(f"Error during motion detection: {e}")
            return frame

    def _visualize_mhi(self):
        """Visualize MHI as a color image"""
        mhi_norm = cv2.normalize(self.mhi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(mhi_norm, cv2.COLOR_GRAY2BGR)


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