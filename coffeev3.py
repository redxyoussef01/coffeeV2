import cv2
import mediapipe as mp
import numpy as np
import requests
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

# --- CONFIG ---
VIDEO_SOURCE = "input2.mp4"   # or 0 for webcam
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
COUNTING_LINE_Y = 600
SEND_EVERY_N_FRAMES = 10
PROXIMITY_THRESHOLD_PX = 100     # when an object reappears within this pixel distance, reuse old ID
REAPPEAR_TIME_LIMIT_FRAMES = 100 # within this many frames we consider reassigning the old ID
LOST_KEEP_FRAMES = REAPPEAR_TIME_LIMIT_FRAMES

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

def detect_hands(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hands_found = False
    if results.multi_hand_landmarks:
        hands_found = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    return frame, hands_found

# --- Initialize YOLO model ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# Note: ultralytics YOLO object sometimes does not need .to(), but leaving as your setup.
yolo_model = YOLO('best5.pt').to(device)  # change path if needed

# --- Initialize DeepSORT ---
deepsort = DeepSort(max_age=LOST_KEEP_FRAMES, n_init=2, max_cosine_distance=0.4)

# --- Memory for lost objects (for reassigning old IDs) ---
lost_tracks = defaultdict(lambda: deque(maxlen=10))  # keep last few locations
lost_last_seen_frame = {}  # old_id -> last seen frame index

# id_map: maps current deepsort internal id -> stable_id (used for display and counting)
id_map = {}          # current_track_id -> stable_id
stable_id_counter = 0

# reverse map to find which stable_id corresponds to lost tracks
stable_last_positions = {}   # stable_id -> (cx, cy, last_seen_frame)

# --- Enhanced counting class (same as your earlier design) ---
class EnhancedLineZoneCounter:
    def __init__(self, line_y, crossing_tolerance=5):
        self.line_y = line_y
        self.crossing_tolerance = crossing_tolerance
        self.object_trajectories = defaultdict(list)  # stable_id -> list of (x,y)
        self.crossed_objects = set()
        self.crossing_objects = set()
        self.detected_objects = set()
        self.total_count = 0

    def update(self, detections):
        """
        detections: list of dicts with keys: 'id' (stable_id), 'bbox' (cx, cy, w, h)
        """
        current_detected = set()
        for det in detections:
            sid = det['id']
            cx, cy, w, h = det['bbox']
            current_detected.add(sid)
            self.detected_objects.add(sid)
            self.object_trajectories[sid].append((cx, cy))
            if len(self.object_trajectories[sid]) > 10:
                self.object_trajectories[sid].pop(0)

            if len(self.object_trajectories[sid]) >= 2:
                prev_y = self.object_trajectories[sid][-2][1]
                curr_y = cy

                # crossing state
                if abs(curr_y - self.line_y) <= self.crossing_tolerance:
                    self.crossing_objects.add(sid)
                else:
                    self.crossing_objects.discard(sid)

                # crossing from top to bottom
                if prev_y <= self.line_y and curr_y > self.line_y:
                    if sid not in self.crossed_objects:
                        self.crossed_objects.add(sid)
                        self.total_count += 1
                        print(f"Object {sid} crossed the line! Total count: {self.total_count}")

        # We do not purge trajectories here; we keep them until memory constraints handled elsewhere

    def get_object_state(self, stable_id):
        if stable_id in self.crossed_objects:
            return 'crossed'
        elif stable_id in self.crossing_objects:
            return 'crossing'
        else:
            return 'detected'

    def get_count(self):
        return self.total_count

    def get_stats(self):
        return {
            'total_detected': len(self.detected_objects),
            'currently_crossing': len(self.crossing_objects),
            'total_crossed': len(self.crossed_objects),
            'final_count': self.total_count
        }

def display_enhanced_results(frame, smoothed_detections, class_names, counter):
    for det in smoothed_detections:
        sid = det['id']
        class_id = det['class_id']
        # ensure confidence is numeric
        confidence = det.get('confidence', 0.0) or 0.0
        x1, y1, x2, y2 = det['original_bbox']

        state = counter.get_object_state(sid)
        if state == 'crossed':
            color = (0, 255, 0)
            state_text = "CROSSED"
        elif state == 'crossing':
            color = (0, 165, 255)
            state_text = "CROSSING"
        else:
            color = (255, 0, 0)
            state_text = "DETECTED"

        # safer class name access (yolo_model.names is usually a dict)
        class_name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else (class_names[class_id] if class_id < len(class_names) else str(class_id))
        label = f"ID:{sid} {class_name}: {confidence:.2f}"
        state_label = f"State: {state_text}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 30),
                      (x1 + max(label_size[0], 120), y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, state_label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw trajectory
        if sid in counter.object_trajectories:
            traj = counter.object_trajectories[sid]
            if len(traj) > 1:
                for j in range(1, len(traj)):
                    pt1 = (int(traj[j-1][0]), int(traj[j-1][1]))
                    pt2 = (int(traj[j][0]), int(traj[j][1]))
                    cv2.line(frame, pt1, pt2, color, 2)

        # Draw centroid
        cx, cy, _, _ = det['bbox']
        cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

    return frame

def send_frame_to_server(frame, endpoint_url="http://127.0.0.1:4002/video_frame"):
    try:
        _, encoded_frame = cv2.imencode('.jpg', frame)
        response = requests.post(endpoint_url, data=encoded_frame.tobytes(), timeout=5)
        # If server returns JSON:
        try:
            return response.json()
        except Exception:
            return {'status_code': response.status_code}
    except requests.exceptions.RequestException as e:
        print(f"Error sending frame to server: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def send_shutdown_signal(endpoint_url="http://127.0.0.1:4002/shutdown"):
    try:
        response = requests.post(endpoint_url, timeout=5)
        print("Shutdown signal sent successfully")
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error sending shutdown signal: {e}")
        return None

# --- Main detection + tracking function using DeepSORT + reassign logic ---
def detect_and_track_objects_deepsort(frame, frame_index):
    """
    Run YOLO detections and track with DeepSORT. Return list of display-ready detections
    where each detection uses a stable_id (reassigned when reappearing objects match lost positions).
    """
    global stable_id_counter, id_map, lost_tracks, lost_last_seen_frame, stable_last_positions

    # Run YOLO on the frame (model returns a list of Results)
    results = yolo_model(frame, conf=0.5 , iou=0.5, verbose=False)

    ds_detections = []  # list of tuples (tlwh, conf, class)
    raw_boxes_info = [] # store raw info to map later: (tlwh, conf, class, x1,y1,x2,y2)
    for r in results:
        for box in r.boxes:
            # guard & cast values
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
            except Exception:
                continue
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            cls = int(box.cls[0]) if box.cls is not None else 0
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            tlwh = [x1, y1, w, h]
            ds_detections.append((tlwh, conf, cls))
            raw_boxes_info.append((tlwh, conf, cls, x1, y1, x2, y2))

    # Pass detections to DeepSORT
    tracks = deepsort.update_tracks(ds_detections, frame=frame)  # returns list of Track objects

    smoothed = []
    # First build a list of new (internal) track ids and their centers
    current_frame_centers = {}
    for trk in tracks:
        if not trk.is_confirmed():
            continue
        tid_internal = trk.track_id
        # get bounding box from track object:
        try:
            # try common API names safely
            if hasattr(trk, "to_ltrb"):
                ltrb = trk.to_ltrb()
            elif hasattr(trk, "to_tlbr"):
                ltrb = trk.to_tlbr()
            else:
                # fallback: skip if we can't get box
                continue
            x1, y1, x2, y2 = map(int, ltrb)
        except Exception:
            # fallback: skip this track
            continue

        # compute center
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        current_frame_centers[tid_internal] = (cx, cy, x1, y1, x2, y2, trk)

    # Attempt to reassign stable IDs:
    for tid_internal, (cx, cy, x1, y1, x2, y2, trk_obj) in current_frame_centers.items():
        # ensure we always produce numeric confidence and class
        det_conf_raw = getattr(trk_obj, 'det_conf', None)
        det_conf = float(det_conf_raw) if det_conf_raw not in (None, False) else 0.0
        det_cls_raw = getattr(trk_obj, 'det_class', None)
        det_cls = int(det_cls_raw) if det_cls_raw is not None else 0

        if tid_internal in id_map:
            stable_id = id_map[tid_internal]
            # update stable last pos
            stable_last_positions[stable_id] = (cx, cy, frame_index)
            smoothed.append({
                'id': stable_id,
                'class_id': det_cls,
                'confidence': det_conf or 0.0,
                'bbox': (cx, cy, x2 - x1, y2 - y1),
                'original_bbox': (x1, y1, x2, y2)
            })
            continue

        # We're seeing this internal tracker id for the first time
        reassigned = False
        for stable_id, (ox, oy, last_frame) in list(stable_last_positions.items()):
            if frame_index - last_frame <= REAPPEAR_TIME_LIMIT_FRAMES:
                dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist <= PROXIMITY_THRESHOLD_PX:
                    # Reassign: treat this internal id as the old stable id
                    id_map[tid_internal] = stable_id
                    stable_last_positions[stable_id] = (cx, cy, frame_index)
                    reassigned = True
                    # clear lost_tracks for that stable_id since it's back
                    if stable_id in lost_tracks:
                        lost_tracks.pop(stable_id, None)
                        lost_last_seen_frame.pop(stable_id, None)
                    break

        if not reassigned:
            # Create a new stable id
            stable_id_counter += 1
            new_sid = stable_id_counter
            id_map[tid_internal] = new_sid
            stable_last_positions[new_sid] = (cx, cy, frame_index)

        # Build smoothed detection entry using the assigned stable id
        stable_id = id_map[tid_internal]
        smoothed.append({
            'id': stable_id,
            'class_id': det_cls,
            'confidence': det_conf or 0.0,
            'bbox': (cx, cy, x2 - x1, y2 - y1),
            'original_bbox': (x1, y1, x2, y2)
        })

    # Update lost_tracks for internal ids that disappeared: compare current id_map keys with tracks present
    present_internal_ids = set(current_frame_centers.keys())
    mapped_internal_ids = set(id_map.keys())
    disappeared_internal = mapped_internal_ids - present_internal_ids

    # For each disappeared internal id, move trace to lost_tracks under its stable id
    for internal_id in list(disappeared_internal):
        stable_id = id_map.get(internal_id)
        if stable_id is None:
            id_map.pop(internal_id, None)
            continue
        # find last known pos for stable_id
        last_pos = stable_last_positions.get(stable_id)
        if last_pos:
            lx, ly, last_frame_idx = last_pos
            # store in lost_tracks
            lost_tracks[stable_id].append((lx, ly, last_frame_idx))
            lost_last_seen_frame[stable_id] = last_frame_idx
        # remove the internal->stable mapping (internal track died)
        id_map.pop(internal_id, None)

    # Purge old lost_stable entries older than limit
    for stable_id, (_, _, last_frame) in list(stable_last_positions.items()):
        if frame_index - last_frame > REAPPEAR_TIME_LIMIT_FRAMES:
            stable_last_positions.pop(stable_id, None)
            lost_tracks.pop(stable_id, None)
            lost_last_seen_frame.pop(stable_id, None)

    return smoothed

# --- Main video loop ---
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    counter = EnhancedLineZoneCounter(line_y=COUNTING_LINE_Y, crossing_tolerance=5)
    frame_count = 0

    print("Starting video processing and DeepSORT tracking... (press 'q' to quit)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        original_frame = frame.copy()
        frame = cv2.resize(frame, (DESIRED_WIDTH, DESIRED_HEIGHT))
        frame_count += 1

        # draw counting line
        cv2.line(frame, (0, COUNTING_LINE_Y), (frame.shape[1], COUNTING_LINE_Y), (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, COUNTING_LINE_Y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # HUD
        stats = counter.get_stats()
        cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Crossed: {stats['final_count']}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Currently Crossing: {stats['currently_crossing']}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"Total Detected: {stats['total_detected']}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # legend
        legend_x, legend_y = frame.shape[1] - 200, 20
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10), (frame.shape[1] - 10, legend_y + 80), (0, 0, 0), -1)
        cv2.putText(frame, "Legend:", (legend_x, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (legend_x, legend_y + 20), (legend_x + 15, legend_y + 30), (255, 0, 0), -1)
        cv2.putText(frame, "Detected", (legend_x + 20, legend_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (legend_x, legend_y + 35), (legend_x + 15, legend_y + 45), (0, 165, 255), -1)
        cv2.putText(frame, "Crossing", (legend_x + 20, legend_y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(frame, (legend_x, legend_y + 50), (legend_x + 15, legend_y + 60), (0, 255, 0), -1)
        cv2.putText(frame, "Crossed", (legend_x + 20, legend_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Hand detection (only run once we resized)
        frame_for_hands = frame.copy()
        frame_for_hands, hands_found = detect_hands(frame_for_hands)

        objects_detected = False
        smoothed_detections = []

        # Run object detection & tracking only when hands are present (per your original logic)
        if hands_found:
            smoothed_detections = detect_and_track_objects_deepsort(frame, frame_count)
            if smoothed_detections:
                objects_detected = True
                counter.update(smoothed_detections)
                frame = display_enhanced_results(frame, smoothed_detections, yolo_model.names, counter)

        # Send ORIGINAL frame occasionally when objects detected
        if objects_detected and (frame_count % SEND_EVERY_N_FRAMES == 0):
            print(f"Frame {frame_count}: sending original frame with {len(smoothed_detections)} objects")
            resp = send_frame_to_server(original_frame)
            if resp:
                print("Server response:", resp)
            else:
                print("No server response")

        # server status text
        status_y = 550
        if objects_detected:
            cv2.putText(frame, f"SERVER: Sending ORIGINAL frames (every {SEND_EVERY_N_FRAMES}th)",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SERVER: Waiting for detections",
                        (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('DeepSORT Hand+Object Detection', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    print("Sending shutdown signal...")
    send_shutdown_signal()
    final_stats = counter.get_stats()
    print("\n=== Final Statistics ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total objects detected: {final_stats['total_detected']}")
    print(f"Total objects crossed: {final_stats['final_count']}")
    print(f"Objects currently crossing: {final_stats['currently_crossing']}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
