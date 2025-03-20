import cv2
import os
import numpy as np
import torch
import time
from facenet_pytorch import MTCNN

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN with CUDA
mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)

# Define paths
VIDEO_FOLDER = r"C:\Users\KIIT\Documents\Minor\videos"
OUTPUT_FOLDER = os.path.join(r"C:\Users\KIIT\Documents\Minor", "processed_videos")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_video(video_path, output_path, frame_limit=100, frame_skip=2):
    """Extract face from every N-th frame and create a face-only video."""
    cap = cv2.VideoCapture(video_path)
    face_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🔹 Processing {video_path} - Total Frames: {total_frames}")

    frame_count = 0
    processed_count = 0

    start_time = time.time()  # Start timing

    while cap.isOpened() and processed_count < frame_limit:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)  # Skip frames
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster detection (e.g., 50% smaller)
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Convert BGR to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face and get bounding box
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])  # Get first detected face

            # Extract face and ensure valid bounds
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                # Resize for consistency
                face_resized = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LANCZOS4)
                face_frames.append(face_resized)
                processed_count += 1

        frame_count += frame_skip  # Skip frames for efficiency

    cap.release()

    # Convert extracted faces into a new video
    if len(face_frames) > 0:
        height, width, _ = face_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 15  # Lower FPS to reduce file size
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for face in face_frames:
            video_writer.write(face)

        video_writer.release()
        print(f"✅ Face-only video saved: {output_path} | Extracted Frames: {len(face_frames)}")
    else:
        print(f"❌ No faces detected in {video_path}")

    end_time = time.time()  # End timing
    print(f"⏳ Processing Time for {video_path}: {end_time - start_time:.2f} seconds")

def process_all_videos(video_folder, output_folder, frame_limit=100, frame_skip=2):
    """Process all videos in a folder and generate face-only videos."""
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    if not video_files:
        print("❌ No videos found in the folder!")
        return

    start_time_all = time.time()  # Start timing for all videos

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_{video_file}")

        print(f"▶ Processing: {video_file}")
        process_video(video_path, output_path, frame_limit, frame_skip)

    end_time_all = time.time()  # End timing for all videos
    print(f"🚀 Total Processing Time for all videos: {end_time_all - start_time_all:.2f} seconds")

# Process all videos in the folder
process_all_videos(VIDEO_FOLDER, OUTPUT_FOLDER)
