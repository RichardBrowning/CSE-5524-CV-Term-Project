import cv2
import numpy as np
import argparse
from tqdm import tqdm
import random

class TargetTracker:
    """基于KLT光流的目标跟踪器"""
    def __init__(self, target_id, bbox, frame):
        self.id = target_id
        self.bbox = bbox  # [x, y, w, h]
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.lost_count = 0 
        self.stable_count = 0 
        self.last_position = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
        
        # init KLT features reference
        self._init_klt_features(frame)
    
    def _init_klt_features(self, frame):
        """Initialize KLT features"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.bbox

            # Ensure ROI is valid
            if w <= 10 or h <= 10 or y+h > gray.shape[0] or x+w > gray.shape[1]:
                self.features = None
                return

            # Create center region mask
            mask = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w//2, h//2
            cv2.circle(mask, (center_x, center_y), min(w, h)//3, 255, -1)
            
            roi_gray = gray[y:y+h, x:x+w]
            
            """goodFeaturesToTrack(
                roi_gray -> ROI in grayscale
                maxCorners=50 -> maximum number of corners to return
                qualityLevel=0.05 -> quality level for corner detection
                minDistance=min(w, h)//5 -> minimum distance between corners
                mask=mask -> mask to limit feature detection to the center region
                blockSize=min(w, h)//10 -> size of the neighborhood considered for corner detection
            )"""
            features = cv2.goodFeaturesToTrack(
                roi_gray,
                maxCorners=50,
                qualityLevel=0.05,  # Higher quality threshold
                minDistance=min(w, h)//5,  # Dynamic distance
                mask=mask,
                blockSize=min(w, h)//10
            )
            
            if features is not None and len(features) > 0:
                # Reshape features to 2D array and adjust coordinates
                features = features.reshape(-1, 2)
                features[:, 0] += x
                features[:, 1] += y
                self.features = features
            else:
                self.features = np.empty((0, 2))
        except Exception as e:
            print(f"Error initializing KLT features: {str(e)}")
            self.features = np.empty((0, 2))

    def update(self, prev_gray, current_gray, frame):
        """Update the tracker with the current frame and previous frame in grayscale."""
        # ensure features not empty
        if self.features is None or len(self.features) < 5:
            self._init_klt_features(frame)
            if self.features is None or len(self.features) < 5:
                self.lost_count += 1
                return False

        # ensure previous frame features exist
        if not hasattr(self, 'features_prev') or self.features_prev is None:
            self.features_prev = self.features.copy()
        
        # Convert features to the required format
        prev_pts = self.features_prev.reshape(-1, 1, 2).astype(np.float32)
        
        try:
            # Calculate optical flow using Lucas-Kanade method
            """calcOpticalFlowPyrLK(
                prev_gray -> previous frame in grayscale
                current_gray -> current frame in grayscale
                prev_pts -> previous feature points
                None -> no initial guess for new points
                winSize=(25, 25) -> window size for optical flow calculation
                maxLevel=3 -> number of pyramid levels to use
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03) -> termination criteria
                flags=cv2.OPFLOW_LK_GET_MIN_EIGENVALS -> use minimum eigenvalue for stability
            )"""
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, current_gray, 
                prev_pts, 
                None, 
                winSize=(25, 25),  # increased window size for better stability
                maxLevel=3,  # increased pyramid levels
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS  # use minimum eigenvalue for stability
            )
            
            if status is None:
                self.lost_count += 1
                return False
                
            # Check if enough points are successfully tracked
            success_indices = status.ravel() == 1
            if np.sum(success_indices) < 5:  # need enough points
                self.lost_count += 1
                return False

            # Update current feature points
            self.features = new_features[success_indices].reshape(-1, 2)
            prev_success = prev_pts[success_indices].reshape(-1, 2)

            # Calculate displacement and filter outliers
            displacement = self.features - prev_success
            displacement_magnitude = np.linalg.norm(displacement, axis=1)
            median_mag = np.median(displacement_magnitude)
            std_mag = np.std(displacement_magnitude)

            # Filter out outliers (using a more relaxed range)
            valid_indices = (displacement_magnitude > median_mag - 2.0 * std_mag) & \
                           (displacement_magnitude < median_mag + 2.0 * std_mag)
            
            if np.sum(valid_indices) < 5:
                self.lost_count += 1
                return False
                
            valid_displacement = displacement[valid_indices]
            mean_shift = np.median(valid_displacement, axis=0)

            # Update bounding box position
            self.bbox[0] += int(mean_shift[0])
            self.bbox[1] += int(mean_shift[1])

            # Update center point
            current_center = (self.bbox[0] + self.bbox[2]//2, self.bbox[1] + self.bbox[3]//2)

            # Check if the object is moving stably
            dx = current_center[0] - self.last_position[0]
            dy = current_center[1] - self.last_position[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance > 2:  # increased threshold for stable movement
                self.stable_count = 0
            else:
                self.stable_count += 1
                
            self.last_position = current_center
            
            # Check if the object is lost
            self.features_prev = self.features.copy()
            self.lost_count = 0
            return True
            
        except Exception as e:
            print(f"Optic Flow Error: {str(e)}")
            self.lost_count += 1
            return False
    
    def draw(self, frame):
        """Draw the object on the frame."""
        x, y, w, h = self.bbox
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.color, 2)
        
        cv2.putText(frame, f"ID:{self.id}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)
        
        for point in self.features:
            cv2.circle(frame, tuple(point.astype(int)), 3, (0, 255, 0), -1)
        
        center_x = self.bbox[0] + self.bbox[2]//2
        center_y = self.bbox[1] + self.bbox[3]//2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
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
        self.tau = 8  # MHI lifetime

        self.trackers = []  # Active tracker list
        self.next_id = 1    # Next object ID
        self.tracker_max_lost = 10  # Tracker lost threshold

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        
    def process_video(self):
        """Main loop"""
        progress = tqdm(total=self.total_frames, desc="Processing Video")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processed_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0) 
            
            detected_objects = self._detect_motion(frame, gray)
            
            if self.prev_gray is not None:
                self._update_trackers(self.prev_gray, gray, frame)
            
            self._create_new_trackers(detected_objects, frame)
            
            self._cleanup_lost_trackers()
            
            for tracker in self.trackers:
                processed_frame = tracker.draw(processed_frame)  

            self.out.write(processed_frame)
            
            if self.display_preview:
                preview = cv2.resize(processed_frame, (1280, 720))
                cv2.imshow('KLT Tracking Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            progress.update(1)
            
            # Update previous frame
            self.prev_gray = gray
        
        progress.close()
        self.cap.release()
        self.out.release()
        if self.display_preview:
            cv2.destroyAllWindows()
    
    def _detect_motion(self, frame, gray):
        """运动目标检测"""
        try:
            # apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # renew MHI
            if self.prev_gray is not None:
                frame_diff = cv2.absdiff(gray, self.prev_gray)
                _, motion_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
                self.mhi = np.where(motion_mask == 255, self.tau, np.maximum(self.mhi - 1, 0))
            
            # get motion area from MHI
            mhi_mask = (self.mhi > 0).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(fg_mask, mhi_mask)
            
            # find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # filter contours
            detected_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 800 or area > 50000:  # 过滤过大或过小的区域
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # filter small bounds
                if w < 50 or h < 50:
                    continue

                # filter top bounds
                if y < self.height * 0.2:
                    continue
                
                # Ensure bounding box is within frame limits
                x = max(0, x)
                y = max(0, y)
                w = min(self.width - x, w)
                h = min(self.height - y, h)
                
                detected_objects.append((x, y, w, h))
            
            return detected_objects
            
        except Exception as e:
            print(f"Error detecting motion: {str(e)}")
            return []
    
    def _update_trackers(self, prev_gray, current_gray, frame):
        """update existing trackers"""
        for tracker in self.trackers:
            try:
                success = tracker.update(prev_gray, current_gray, frame)
                
                # if tracker loss update the lost count
                if not success:
                    tracker.lost_count += 1
                else:
                    # reduce lost count if successfully updated
                    tracker.lost_count = max(0, tracker.lost_count - 1)
            except Exception as e:
                print(f"Error updating tracker: {str(e)}")
                tracker.lost_count += 1
    
    def _create_new_trackers(self, detected_objects, frame):
        """create new trackers for detected objects"""
        for obj in detected_objects:
            x, y, w, h = obj
            
            # How to check if object frame is overlapping?
            already_tracked = False
            for tracker in self.trackers:
                try:
                    tx, ty, tw, th = tracker.bbox
                    center_x = tx + tw//2
                    center_y = ty + th//2
                    
                    # Check if the center of the detected object is within the tracker bounding box
                    if (x <= center_x <= x+w) and (y <= center_y <= y+h):
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
    parser = argparse.ArgumentParser(description='Video motion detection and KLT tracking processor')
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