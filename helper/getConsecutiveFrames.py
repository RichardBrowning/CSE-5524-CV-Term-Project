import cv2
import sys
import os
import random

def extract_consecutive_frames(video_path, output_dir, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Read first frame index
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < num_frames:
        print(f"Error: Video has only {frame_count} frames, but {num_frames} are required.")
        cap.release()
        return

    # Select a random starting frame index so that num_frames can be read consecutively
    start_idx = random.randint(0, frame_count - num_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    print(f"Extracting {num_frames} frames starting from frame {start_idx}")
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame {i}")
            break
        output_path = os.path.join(output_dir, f"frame_{i+1}.png")
        cv2.imwrite(output_path, frame)
        print(f"Saved {output_path}")

    cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python getConsecutiveFrames.py <video_path> <output_dir>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_consecutive_frames(video_path, output_dir)