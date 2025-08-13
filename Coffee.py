import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO
import numpy as np
import requests
from collections import defaultdict

# --- Part 1: MediaPipe Hand Detection Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

def detect_hands(frame):
    """
    Detects hands in a given frame and draws the landmarks.
    Args:
        frame (np.ndarray): The input video frame.
    Returns:
        tuple: A tuple containing the processed frame and a boolean indicating if hands were found.
    """
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

# --- Part 2: YOLO Object Detection and Tracking with Kalman Filter ---
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
yolo_model = YOLO('best5.pt').to(device)

kalman_filters = {}

def create_kalman_filter(bbox):
    """Initializes a Kalman filter for a new object."""
    x, y, w, h = bbox
    kf = cv2.KalmanFilter(8, 4)
    
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]
    ], np.float32)

    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0
    kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(8, 1)
    
    return kf

def detect_and_track_objects_yolo(frame, model):
    """
    Detects and tracks objects with YOLO/ByteTrack and applies a Kalman filter.
    Returns:
        tuple: A tuple containing the processed frame and a list of smoothed detections.
    """
    global kalman_filters
    
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack_custom.yaml",
        conf=0.7,
        iou=0.5,
        verbose=False
    )
    
    smoothed_detections = []
    detected_ids = set()

    for r in results:
        boxes = r.boxes
        if boxes.id is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                track_id = int(boxes.id[i])
                
                detected_ids.add(track_id)
                
                bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], np.float32)
                
                if track_id not in kalman_filters:
                    kalman_filters[track_id] = create_kalman_filter(bbox_center)

                kf = kalman_filters[track_id]
                kf.predict()
                kf.correct(bbox_center.reshape(4, 1))
                
                predicted_state = kf.statePost
                smooth_x = predicted_state[0, 0]
                smooth_y = predicted_state[1, 0]
                smooth_w = predicted_state[2, 0]
                smooth_h = predicted_state[3, 0]
                
                smoothed_detections.append({
                    'id': track_id,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': (smooth_x, smooth_y, smooth_w, smooth_h),
                    'original_bbox': (x1, y1, x2, y2)
                })

    lost_ids = list(kalman_filters.keys() - detected_ids)
    for track_id in lost_ids:
        del kalman_filters[track_id]
        
    return frame, smoothed_detections

# --- Part 3: Enhanced Counting Logic with States ---
class EnhancedLineZoneCounter:
    """Enhanced class to handle object counting with multiple states."""
    def __init__(self, line_y, crossing_tolerance=5):
        self.line_y = line_y
        self.crossing_tolerance = crossing_tolerance
        self.object_trajectories = defaultdict(list)
        self.crossed_objects = set()      # Objects that have crossed the line
        self.crossing_objects = set()     # Objects currently crossing the line
        self.detected_objects = set()     # All detected objects
        self.total_count = 0
    
    def update(self, detections):
        """Updates trajectories and counts objects crossing the line."""
        current_detected = set()
        
        for det in detections:
            track_id = det['id']
            smooth_x, smooth_y, smooth_w, smooth_h = det['bbox']
            
            # Use the center of the bounding box
            bbox_center_y = smooth_y
            current_detected.add(track_id)
            self.detected_objects.add(track_id)
            
            self.object_trajectories[track_id].append((smooth_x, bbox_center_y))
            
            # Keep only the last 10 trajectory points to save memory
            if len(self.object_trajectories[track_id]) > 10:
                self.object_trajectories[track_id].pop(0)

            # Check for line crossing
            if len(self.object_trajectories[track_id]) >= 2:
                prev_y = self.object_trajectories[track_id][-2][1]
                curr_y = bbox_center_y
                
                # Check if object is currently crossing the line (within tolerance)
                if abs(curr_y - self.line_y) <= self.crossing_tolerance:
                    self.crossing_objects.add(track_id)
                else:
                    self.crossing_objects.discard(track_id)
                
                # Check for line crossing from top to bottom
                if prev_y <= self.line_y and curr_y > self.line_y:
                    if track_id not in self.crossed_objects:
                        self.crossed_objects.add(track_id)
                        self.total_count += 1
                        print(f"Object {track_id} crossed the line! Total count: {self.total_count}")
        
        # Clean up trajectories for objects that are no longer detected
        # Keep trajectories for recently lost objects for a few frames
        all_ids = set(self.object_trajectories.keys())
        lost_ids = all_ids - current_detected
        for lost_id in lost_ids:
            # Only remove if object has been lost for more than 5 frames
            if len(self.object_trajectories[lost_id]) > 0:
                # Mark as disappeared but keep trajectory for a while
                pass
    
    def get_object_state(self, track_id):
        """Returns the state of an object: 'detected', 'crossing', or 'crossed'"""
        if track_id in self.crossed_objects:
            return 'crossed'
        elif track_id in self.crossing_objects:
            return 'crossing'
        else:
            return 'detected'
    
    def get_count(self):
        return self.total_count
    
    def get_stats(self):
        """Returns detailed statistics"""
        return {
            'total_detected': len(self.detected_objects),
            'currently_crossing': len(self.crossing_objects),
            'total_crossed': len(self.crossed_objects),
            'final_count': self.total_count
        }

def display_enhanced_results(frame, smoothed_detections, class_names, counter):
    """
    Draws bounding boxes and labels with color-coded states for the smoothed detections.
    """
    for det in smoothed_detections:
        track_id = det['id']
        class_id = det['class_id']
        confidence = det['confidence']
        x1, y1, x2, y2 = det['original_bbox']
        
        # Get object state and determine color
        state = counter.get_object_state(track_id)
        if state == 'crossed':
            color = (0, 255, 0)      # Green for crossed
            state_text = "CROSSED"
        elif state == 'crossing':
            color = (0, 165, 255)    # Orange for crossing
            state_text = "CROSSING"
        else:
            color = (255, 0, 0)      # Blue for detected
            state_text = "DETECTED"

        class_name = class_names[class_id]
        label = f"ID:{track_id} {class_name}: {confidence:.2f}"
        state_label = f"State: {state_text}"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw main label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 30), 
                     (x1 + max(label_size[0], 120), y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw state label
        cv2.putText(frame, state_label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw trajectory
        if track_id in counter.object_trajectories:
            trajectory = counter.object_trajectories[track_id]
            if len(trajectory) > 1:
                for j in range(1, len(trajectory)):
                    pt1 = (int(trajectory[j-1][0]), int(trajectory[j-1][1]))
                    pt2 = (int(trajectory[j][0]), int(trajectory[j][1]))
                    cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw centroid
        smooth_x, smooth_y, _, _ = det['bbox']
        cv2.circle(frame, (int(smooth_x), int(smooth_y)), 4, color, -1)

    return frame

def send_frame_to_server(frame, endpoint_url="http://127.0.0.1:4002/video_frame"):
    """
    Send frame to server endpoint
    Args:
        frame: OpenCV frame to send
        endpoint_url: Server endpoint URL
    Returns:
        dict: Server response or None if failed
    """
    try:
        _, encoded_frame = cv2.imencode('.jpg', frame)
        response = requests.post(endpoint_url, data=encoded_frame.tobytes(), timeout=5)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending frame to server: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def send_shutdown_signal(endpoint_url="http://127.0.0.1:4002/shutdown"):
    """Send shutdown signal to server"""
    try:
        response = requests.post(endpoint_url, timeout=5)
        print("Shutdown signal sent successfully")
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error sending shutdown signal: {e}")
        return None

# --- Part 4: Main Function and Video Loop ---
def main():
    cap = cv2.VideoCapture("input1.mp4")  # Change to name of ur video for webcam input

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    desired_width = 1280
    desired_height = 720
    
    # Define the counting line's y-coordinate outside the loop.
    counting_line_y = 600
    counter = EnhancedLineZoneCounter(line_y=counting_line_y, crossing_tolerance=5)
    
    # Frame sending variables
    frame_count = 0
    send_every_n_frames = 5  # Send every 5th frame when objects are detected to avoid overwhelming the server
    last_detection_count = 0

    print("Starting video processing and server communication...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Store original frame BEFORE any processing
        original_frame = frame.copy()
        
        frame = cv2.resize(frame, (desired_width, desired_height))
        frame_count += 1

        # --- Draw persistent elements first ---
        # Draw counting line
        cv2.line(frame, (0, counting_line_y), (frame.shape[1], counting_line_y), (0, 255, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (10, counting_line_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Get statistics
        stats = counter.get_stats()
        
        # Draw enhanced counter display
        cv2.rectangle(frame, (10, 10), (400, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Crossed: {stats['final_count']}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Currently Crossing: {stats['currently_crossing']}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"Total Detected: {stats['total_detected']}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw legend
        legend_x, legend_y = frame.shape[1] - 200, 20
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10), (frame.shape[1] - 10, legend_y + 80), (0, 0, 0), -1)
        cv2.putText(frame, "Legend:", (legend_x, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blue for detected
        cv2.rectangle(frame, (legend_x, legend_y + 20), (legend_x + 15, legend_y + 30), (255, 0, 0), -1)
        cv2.putText(frame, "Detected", (legend_x + 20, legend_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Orange for crossing
        cv2.rectangle(frame, (legend_x, legend_y + 35), (legend_x + 15, legend_y + 45), (0, 165, 255), -1)
        cv2.putText(frame, "Crossing", (legend_x + 20, legend_y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Green for crossed
        cv2.rectangle(frame, (legend_x, legend_y + 50), (legend_x + 15, legend_y + 60), (0, 255, 0), -1)
        cv2.putText(frame, "Crossed", (legend_x + 20, legend_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Perform detection and tracking conditionally ---
        frame, hands_found = detect_hands(frame)
        objects_detected = False
        smoothed_detections = []
        
        if hands_found:
            frame, smoothed_detections = detect_and_track_objects_yolo(frame, yolo_model)
            
            if smoothed_detections:
                objects_detected = True
                counter.update(smoothed_detections)
                frame = display_enhanced_results(frame, smoothed_detections, yolo_model.names, counter)

        # --- Send ORIGINAL frame to server when objects are detected ---
        if objects_detected:
            # Send frame every N frames to avoid overwhelming the server
            if frame_count % send_every_n_frames == 0:
                print(f"Frame {frame_count}: Sending ORIGINAL frame with {len(smoothed_detections)} detected objects...")
                
                # Send the ORIGINAL frame without any processing/annotations
                response = send_frame_to_server(original_frame)
                if response:
                    print(f"Server response: {response}")
                else:
                    print("Failed to get response from server")
        
        # Add server communication status to display
        server_status_y = 550
        if objects_detected:
            cv2.putText(frame, f"SERVER: Sending ORIGINAL frames (every {send_every_n_frames}th)", 
                       (10, server_status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SERVER: Waiting for detections", 
                       (10, server_status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Enhanced Hand and Object Detection with Server Communication', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Send shutdown signal to server
    print("Sending shutdown signal to server...")
    send_shutdown_signal()

    # Print final statistics
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