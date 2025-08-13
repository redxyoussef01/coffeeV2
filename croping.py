import cv2
import numpy as np
import os
import csv
import datetime

# --- CONFIG ---
VIDEO_FOLDER = "videos"  # Folder containing input videos
OUTPUT_FOLDER = "motion_clips"  # Folder to store motion videos
CSV_FILE = "video_log.csv"  # Tracks processed videos
MIN_MOVEMENT_FRAMES = 5  # Frames to confirm movement
MOVEMENT_THRESHOLD = 200  # Pixel change threshold in ROI
FPS_SAFETY_MARGIN = 10  # Frames before/after motion to keep
MIN_CLIP_DURATION_SEC = 1  # Minimum clip length in seconds

# Define region of interest (ROI)
ROI_X, ROI_Y, ROI_W, ROI_H = 0, 400, 1900, 300  # (x, y, width, height)

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Load or create CSV log ---
video_status = {}
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_status[row["video_name"]] = row["processed"] == "True"

# Add any new videos to CSV list with processed=False
for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        if file not in video_status:
            video_status[file] = False

# --- Helper: Adjust filename based on motion start time ---
def add_minutes_to_filename(base_name, start_frame, fps):
    parts = base_name.split("_")
    if len(parts) < 4:
        return base_name  # fallback if unexpected format
    
    date_str = parts[-2]
    time_str = parts[-1]
    
    try:
        dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    except ValueError:
        return base_name  # fallback
    
    # Add minutes based on motion start time
    minutes_to_add = int((start_frame / fps) // 60)
    seconds_to_add = int((start_frame / fps) % 60)
    dt_new = dt + datetime.timedelta(minutes=minutes_to_add, seconds=seconds_to_add)
    
    parts[-2] = dt_new.strftime("%Y%m%d")
    parts[-1] = dt_new.strftime("%H%M%S")
    return "_".join(parts)

# --- Process videos ---
for video_name, processed in video_status.items():
    video_base_name = os.path.splitext(video_name)[0]
    if processed:
        print(f"⏩ Skipping already processed: {video_name}")
        continue

    video_path = os.path.join(VIDEO_FOLDER, video_name)
    print(f"▶ Processing: {video_base_name}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    min_frames_required = int(MIN_CLIP_DURATION_SEC * fps)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    motion_detected = False
    motion_frames_counter = 0
    clip_count = 0
    out = None
    frames_after_motion = 0
    clip_frame_counter = 0
    motion_start_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if out is not None:
                out.release()
                if clip_frame_counter < min_frames_required:
                    os.remove(clip_path)
                    print(f"🗑 Deleted short clip: {clip_path}")
            break

        roi_frame = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        fgmask = fgbg.apply(roi_frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        movement = np.sum(fgmask == 255)

        if movement > MOVEMENT_THRESHOLD:
            motion_frames_counter += 1
            frames_after_motion = 0
            if not motion_detected and motion_frames_counter >= MIN_MOVEMENT_FRAMES:
                motion_detected = True
                clip_count += 1
                clip_frame_counter = 0
                motion_start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                adjusted_name = add_minutes_to_filename(video_base_name, motion_start_frame, fps)
                clip_path = f"{OUTPUT_FOLDER}/{adjusted_name}.mp4"
                out = cv2.VideoWriter(
                    clip_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, (width, height)
                )
                print(f"📹 Started recording motion {clip_count} in {video_base_name}")
        else:
            if motion_detected:
                frames_after_motion += 1
                if frames_after_motion >= FPS_SAFETY_MARGIN:
                    motion_detected = False
                    out.release()
                    if clip_frame_counter < min_frames_required:
                        os.remove(clip_path)
                        print(f"🗑 Deleted short clip: {clip_path}")
                    out = None
                    print(f"✅ Finished recording motion {clip_count} in {video_base_name}")

        if motion_detected and out is not None:
            out.write(frame)
            clip_frame_counter += 1

    cap.release()
    print(f"Done with {video_name}, saved {clip_count} clips.")

    video_status[video_name] = True

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "processed"])
        writer.writeheader()
        for name, proc in video_status.items():
            writer.writerow({"video_name": name, "processed": proc})

print("✅ All videos processed.")
