import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random

class TargetTracker:
    """Mean-Shift"""
    def __init__(self, target_id, bbox, frame):
        self.id = target_id
        self.bbox = bbox  # [x, y, w, h]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.lost_count = 0 
        self.original_size = (bbox[2], bbox[3]) 
        
        self._init_meanshift_tracker(frame)
    
    def _init_meanshift_tracker(self, frame):
        """Initialize Mean-Shift tracker"""
        x, y, w, h = self.bbox
        roi = frame[y:y+h, x:x+w]

        # Set up ROI histogram (H and S channels)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Save initial position
        self.track_window = (x, y, w, h)

    def update(self, frame):
        """Update target position using Mean-Shift"""
        try:
            x, y, w, h = self.bbox
            track_window = (x, y, w, h)

            # Set termination criteria
            termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 1)

            # Convert to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Compute back projection (using H and S channels)
            dst = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)

            # Apply Mean-Shift to get new position
            _, track_window = cv2.meanShift(dst, track_window, termination_criteria)

            # Update bounding box position but keep original size
            new_x, new_y, new_w, new_h = track_window
            self.bbox = [new_x, new_y, self.original_size[0], self.original_size[1]]

            # Compute back projection response value
            response = np.mean(dst[new_y:new_y+self.original_size[1], new_x:new_x+self.original_size[0]])

            # Check if response is losing
            if response < 5:  # Low response indicates target may be lost
                self.lost_count += 1
                return False
            
            self.lost_count = 0
            return True
            
        except Exception as e:
            print(f"Error updating Mean-Shift: {str(e)}")
            self.lost_count += 1
            return False
    
    def draw(self, frame):
        """Draw frame and ID"""
        x, y, w, h = self.bbox

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)

        # Draw target ID
        cv2.putText(frame, f"ID:{self.id}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        
        return frame

class VideoProcessor:
    def __init__(self, input_path, output_path, display_preview=False):
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not self.out.isOpened():
            raise ValueError(f"无法创建输出视频: {output_path}")
        
        self.display_preview = display_preview
        
        self.prev_gray = None
        self.mhi = np.zeros((self.height, self.width), dtype=np.float32)
        self.tau = 10  # MHI lifetime

        self.trackers = []  # Active tracker list
        self.next_id = 1    # Next target ID
        self.tracker_max_lost = 5  # Target lost threshold

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        
    def process_video(self):
        """Main processing loop"""
        progress = tqdm(total=self.total_frames, desc="Processing Video")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Preprocessing - Denoising
            processed_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Motion detection
            detected_objects = self._detect_motion(frame, gray)

            # Update existing trackers
            self._update_trackers(frame)

            # Create new trackers for untracked objects
            self._create_new_trackers(detected_objects, frame)

            # Cleanup lost trackers
            self._cleanup_lost_trackers()

            # Draw all active trackers
            for tracker in self.trackers:
                processed_frame = tracker.draw(processed_frame)

            # Write output video
            self.out.write(processed_frame)

            # Display preview
            if self.display_preview:
                preview = cv2.resize(processed_frame, (1280, 720))
                cv2.imshow('Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            progress.update(1)

            self.prev_gray = gray
        
        progress.close()
        self.cap.release()
        self.out.release()
        if self.display_preview:
            cv2.destroyAllWindows()
    
    def _detect_motion(self, frame, gray):
        try:
            # Background subtraction
            fg_mask = self.bg_subtractor.apply(frame)

            # Morphological operations to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            # Update motion history image (MHI)
            if self.prev_gray is not None:
                frame_diff = cv2.absdiff(gray, self.prev_gray)
                _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                self.mhi = np.where(motion_mask == 255, self.tau, np.maximum(self.mhi - 1, 0))

            # Combine background subtraction and MHI
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(fg_mask, mhi_mask)

            # Extract contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter small contours and invalid detections
            detected_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Filter out too large or too small areas
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)

                # Ignore small objects at the edges
                if w < 40 or h < 40:
                    continue

                # Ignore top area of the image
                if y < self.height * 0.2:
                    continue

                # Expand bounding box to ensure full object is included
                expansion = 5
                x = max(0, x - expansion)
                y = max(0, y - expansion)
                w = min(self.width - x, w + 2*expansion)
                h = min(self.height - y, h + 2*expansion)
                
                detected_objects.append((x, y, w, h))
            
            return detected_objects
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return []
    
    def _update_trackers(self, frame):
        """Update existing trackers"""
        for tracker in self.trackers:
            try:
                # Update tracker
                success = tracker.update(frame)

                # If tracking fails, increment lost count
                if not success:
                    tracker.lost_count += 1
            except Exception as e:
                print(f"跟踪器更新错误: {str(e)}")
                tracker.lost_count += 1
    
    def _create_new_trackers(self, detected_objects, frame):
        """Create new trackers for untracked objects"""
        for obj in detected_objects:
            x, y, w, h = obj
            
            # How to check if object tracker are overlapping?
            already_tracked = False
            for tracker in self.trackers:
                try:
                    tx, ty, tw, th = tracker.bbox

                    # Calculate overlap area
                    x_overlap = max(0, min(x+w, tx+tw) - max(x, tx))
                    y_overlap = max(0, min(y+h, ty+th) - max(y, ty))
                    overlap_area = x_overlap * y_overlap

                    # Overlap area should not exceeds 12%(40% originally) or smaller -> tracked
                    min_area = min(w*h, tw*th)
                    if min_area > 0 and overlap_area > min_area * 0.12: # 40%
                        already_tracked = True
                        break
                except:
                    continue
            
            if not already_tracked:
                try:
                    # Create new tracker
                    new_tracker = TargetTracker(self.next_id, [x, y, w, h], frame)
                    self.trackers.append(new_tracker)
                    self.next_id += 1
                except Exception as e:
                    print(f"Cannot create new tracker: {str(e)}")

    def _cleanup_lost_trackers(self):
        """Cleanup lost trackers"""
        self.trackers = [t for t in self.trackers if t.lost_count < self.tracker_max_lost]

def parse_args():
    parser = argparse.ArgumentParser(description='Video motion detection and tracking processor')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output video file')
    parser.add_argument('--display_preview', action='store_true', default=True,
                       help='Display preview of the processed video')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    input_video = args.input
    output_video = args.output
    if not input_video or not output_video:
        raise ValueError("Input and output video paths must be specified.")
    display_preview = args.display_preview
    
    processor = VideoProcessor(input_video, output_video, display_preview=display_preview)
    processor.process_video()
    print("Processing completed! Output saved to", output_video)