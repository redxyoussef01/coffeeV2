import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO
import numpy as np
import requests
from collections import defaultdict, deque
import math
import time

# --- Part 1: MediaPipe Hand Detection Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_hand_center(hand_landmarks, frame_width, frame_height):
    """Calculate the center point of a hand from landmarks"""
    x_coords = []
    y_coords = []
    
    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x * frame_width)
        y_coords.append(landmark.y * frame_height)
    
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    return (int(center_x), int(center_y))

def detect_hands_with_tracking(frame):
    """
    Detects hands in a given frame and returns hand information.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    hand_detections = []
    
    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            hand_center = get_hand_center(hand_landmarks, frame.shape[1], frame.shape[0])
            hand_label = handedness.classification[0].label
            
            hand_info = {
                'id': idx,
                'center': hand_center,
                'landmarks': hand_landmarks,
                'label': hand_label,
                'associated_object': None,
                'associated_class': None,
                'last_associated_object': None,
                'last_associated_class': None
            }
            
            hand_detections.append(hand_info)
            
            cv2.circle(frame, hand_center, 8, (255, 255, 0), -1)
            cv2.putText(frame, f"Hand {idx} ({hand_label})", 
                        (hand_center[0] + 10, hand_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return frame, hand_detections

# --- Enhanced Hand Memory System ---
class HandMemoryTracker:
    def __init__(self, memory_duration=30, proximity_threshold=150):
        self.hand_memories = {}  # hand_id -> {'last_object': obj_id, 'last_class': class_id, 'timestamp': time, 'confidence': float}
        self.memory_duration = memory_duration  # frames to remember
        self.proximity_threshold = proximity_threshold
        
    def update_hand_memory(self, hand_id, object_id, class_id, confidence=1.0):
        """Update memory of what object a hand was associated with"""
        self.hand_memories[hand_id] = {
            'last_object': object_id,
            'last_class': class_id,
            'timestamp': time.time(),
            'confidence': confidence,
            'frame_count': 0
        }
    
    def get_hand_memory(self, hand_id):
        """Get the last remembered object for a hand"""
        if hand_id in self.hand_memories:
            memory = self.hand_memories[hand_id]
            # Check if memory is still valid (not too old)
            if memory['frame_count'] < self.memory_duration:
                return memory['last_object'], memory['last_class']
        return None, None
    
    def age_memories(self):
        """Age all memories by one frame"""
        to_remove = []
        for hand_id in self.hand_memories:
            self.hand_memories[hand_id]['frame_count'] += 1
            if self.hand_memories[hand_id]['frame_count'] >= self.memory_duration:
                to_remove.append(hand_id)
        
        for hand_id in to_remove:
            del self.hand_memories[hand_id]
    
    def clear_hand_memory(self, hand_id):
        """Clear memory for a specific hand"""
        if hand_id in self.hand_memories:
            del self.hand_memories[hand_id]
            print(f"Hand memory cleared for hand {hand_id}")
    
    def get_memory_info(self, hand_id):
        """Get detailed memory information for debugging"""
        if hand_id in self.hand_memories:
            memory = self.hand_memories[hand_id]
            frames_left = self.memory_duration - memory['frame_count']
            return {
                'object_id': memory['last_object'],
                'class_id': memory['last_class'],
                'frames_remaining': frames_left,
                'confidence': memory['confidence']
            }
        return None

# --- Enhanced Object Tracker with Hand-Based Recovery ---
class EnhancedObjectTracker:
    def __init__(self, max_disappeared=30, distance_threshold=100, iou_threshold=0.3, confidence_threshold=0.5):
        self.next_object_id = 0
        self.objects = {}  # object_id -> {'center': (x,y), 'bbox': (x1,y1,x2,y2), 'class_id': int, 'confidence': float, 'age': int, 'tracking_mode': 'detection'/'hand'}
        self.disappeared = {}  # object_id -> frames_disappeared
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.object_history = {}  # object_id -> deque of recent positions
        self.max_history = 10
        
        # NEW: Hand-based tracking for lost objects
        self.hand_tracked_objects = {}  # object_id -> {'hand_id': int, 'last_hand_pos': (x,y), 'offset': (x,y), 'confidence_decay': float}
        self.hand_tracking_confidence_decay = 0.95  # Decay confidence each frame when hand-tracking
        self.min_hand_tracking_confidence = 0.1  # Stop hand-tracking below this confidence
        self.hand_tracking_max_frames = 45  # Maximum frames to track using hand before giving up

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def register(self, detection):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            'center': detection['center'],
            'bbox': detection['bbox'],
            'class_id': detection['class_id'],
            'confidence': detection['confidence'],
            'age': 0,
            'tracking_mode': 'detection'
        }
        self.disappeared[self.next_object_id] = 0
        self.object_history[self.next_object_id] = deque(maxlen=self.max_history)
        self.object_history[self.next_object_id].append(detection['center'])
        
        print(f"Registered new object ID {self.next_object_id} of class {detection['class_id']} at {detection['center']}")
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking"""
        if object_id in self.objects:
            print(f"Deregistered object ID {object_id} of class {self.objects[object_id]['class_id']}")
            del self.objects[object_id]
            del self.disappeared[object_id]
            if object_id in self.object_history:
                del self.object_history[object_id]
            if object_id in self.hand_tracked_objects:
                del self.hand_tracked_objects[object_id]

    def predict_next_position(self, object_id):
        """Predict next position based on movement history"""
        if object_id not in self.object_history or len(self.object_history[object_id]) < 2:
            return self.objects[object_id]['center']
        
        history = list(self.object_history[object_id])
        if len(history) >= 3:
            # Use velocity from last 2 positions
            velocity_x = history[-1][0] - history[-2][0]
            velocity_y = history[-1][1] - history[-2][1]
            predicted_x = history[-1][0] + velocity_x
            predicted_y = history[-1][1] + velocity_y
            return (int(predicted_x), int(predicted_y))
        
        return history[-1]

    def start_hand_tracking(self, object_id, hand_id, hand_center):
        """Start tracking a lost object using the associated hand"""
        if object_id in self.objects:
            obj_center = self.objects[object_id]['center']
            # Calculate offset between object and hand
            offset = (obj_center[0] - hand_center[0], obj_center[1] - hand_center[1])
            
            self.hand_tracked_objects[object_id] = {
                'hand_id': hand_id,
                'last_hand_pos': hand_center,
                'offset': offset,
                'confidence_decay': 1.0,
                'frames_tracked': 0
            }
            
            # Update object tracking mode
            self.objects[object_id]['tracking_mode'] = 'hand'
            print(f"Started hand-tracking for object {object_id} using hand {hand_id} (offset: {offset})")

    def update_hand_tracked_objects(self, hand_detections):
        """Update objects that are being tracked by hands"""
        objects_to_remove = []
        
        for object_id, tracking_info in self.hand_tracked_objects.items():
            if object_id not in self.objects:
                objects_to_remove.append(object_id)
                continue
                
            hand_id = tracking_info['hand_id']
            tracking_info['frames_tracked'] += 1
            
            # Find the associated hand
            associated_hand = None
            for hand in hand_detections:
                if hand['id'] == hand_id:
                    associated_hand = hand
                    break
            
            if associated_hand is None:
                # Hand is lost, stop hand-tracking
                print(f"Hand {hand_id} lost, stopping hand-tracking for object {object_id}")
                objects_to_remove.append(object_id)
                continue
            
            # Check if we've been hand-tracking too long
            if tracking_info['frames_tracked'] > self.hand_tracking_max_frames:
                print(f"Hand-tracking timeout for object {object_id} (>{self.hand_tracking_max_frames} frames)")
                objects_to_remove.append(object_id)
                continue
            
            # Update object position based on hand movement
            hand_center = associated_hand['center']
            offset = tracking_info['offset']
            new_object_center = (
                hand_center[0] + offset[0],
                hand_center[1] + offset[1]
            )
            
            # Decay confidence
            tracking_info['confidence_decay'] *= self.hand_tracking_confidence_decay
            if tracking_info['confidence_decay'] < self.min_hand_tracking_confidence:
                print(f"Hand-tracking confidence too low for object {object_id}")
                objects_to_remove.append(object_id)
                continue
            
            # Update object position and properties
            old_bbox = self.objects[object_id]['bbox']
            bbox_width = old_bbox[2] - old_bbox[0]
            bbox_height = old_bbox[3] - old_bbox[1]
            
            # Create new bbox centered on predicted position
            new_bbox = (
                new_object_center[0] - bbox_width // 2,
                new_object_center[1] - bbox_height // 2,
                new_object_center[0] + bbox_width // 2,
                new_object_center[1] + bbox_height // 2
            )
            
            # Update object
            self.objects[object_id].update({
                'center': new_object_center,
                'bbox': new_bbox,
                'confidence': tracking_info['confidence_decay'],  # Use decayed confidence
                'age': self.objects[object_id]['age'] + 1,
                'tracking_mode': 'hand'
            })
            
            # Update history
            self.object_history[object_id].append(new_object_center)
            
            # Update tracking info
            tracking_info['last_hand_pos'] = hand_center
            
            print(f"Hand-tracking: Object {object_id} updated to {new_object_center} (confidence: {tracking_info['confidence_decay']:.3f}, frames: {tracking_info['frames_tracked']})")
        
        # Clean up finished hand-tracking
        for object_id in objects_to_remove:
            if object_id in self.hand_tracked_objects:
                del self.hand_tracked_objects[object_id]
            # Return to normal tracking mode or remove if too old
            if object_id in self.objects:
                self.objects[object_id]['tracking_mode'] = 'detection'
                self.disappeared[object_id] += 1

    def update(self, detections, hand_detections=None):
        """Enhanced tracking with IoU, confidence filtering, and hand-based recovery"""
        # First, update hand-tracked objects if we have hand data
        if hand_detections:
            self.update_hand_tracked_objects(hand_detections)
        
        # Filter detections by confidence
        filtered_detections = [d for d in detections if d['confidence'] >= self.confidence_threshold]
        
        if not filtered_detections:
            # No valid detections, age all existing objects
            objects_to_check_for_hand_tracking = []
            
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Check if we should start hand-tracking for this object
                if (object_id not in self.hand_tracked_objects and 
                    self.disappeared[object_id] == 1 and  # Just disappeared
                    hand_detections and object_id in self.objects):
                    objects_to_check_for_hand_tracking.append(object_id)
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Try to start hand-tracking for newly disappeared objects
            if hand_detections:
                self.try_start_hand_tracking(objects_to_check_for_hand_tracking, hand_detections)
                
            return self.get_tracked_objects()

        if not self.objects:
            # No existing objects, register all detections
            for detection in filtered_detections:
                self.register(detection)
        else:
            # Match detections to existing objects
            object_ids = list(self.objects.keys())
            
            # Separate hand-tracked objects from detection-tracked objects for matching
            detection_tracked_ids = [oid for oid in object_ids if self.objects[oid]['tracking_mode'] == 'detection']
            hand_tracked_ids = [oid for oid in object_ids if self.objects[oid]['tracking_mode'] == 'hand']
            
            # Create cost matrix combining distance, IoU, and class consistency
            cost_matrix = np.full((len(detection_tracked_ids), len(filtered_detections)), np.inf)
            
            for i, object_id in enumerate(detection_tracked_ids):
                predicted_pos = self.predict_next_position(object_id)
                existing_bbox = self.objects[object_id]['bbox']
                existing_class = self.objects[object_id]['class_id']
                
                for j, detection in enumerate(filtered_detections):
                    # Skip if class doesn't match (objects don't change class)
                    if detection['class_id'] != existing_class:
                        continue
                    
                    # Calculate distance cost
                    distance = calculate_distance(predicted_pos, detection['center'])
                    if distance > self.distance_threshold * 2:  # Allow larger distance for recovery
                        continue
                    
                    # Calculate IoU cost
                    iou = self.calculate_iou(existing_bbox, detection['bbox'])
                    if iou < self.iou_threshold * 0.5:  # More lenient for recovery
                        continue
                    
                    # Combined cost (lower is better)
                    distance_cost = distance / (self.distance_threshold * 2)
                    iou_cost = 1.0 - iou
                    confidence_bonus = detection['confidence']
                    
                    # Weighted combination
                    cost_matrix[i, j] = (0.4 * distance_cost + 0.4 * iou_cost - 0.2 * confidence_bonus)
            
            # Check if hand-tracked objects can be recovered with detections
            for object_id in hand_tracked_ids:
                existing_class = self.objects[object_id]['class_id']
                existing_center = self.objects[object_id]['center']
                
                for j, detection in enumerate(filtered_detections):
                    if detection['class_id'] != existing_class:
                        continue
                    
                    distance = calculate_distance(existing_center, detection['center'])
                    if distance < self.distance_threshold:
                        # Recovery! Switch back to detection-based tracking
                        print(f"RECOVERY: Object {object_id} switched from hand-tracking back to detection-based tracking")
                        self.objects[object_id].update({
                            'center': detection['center'],
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'tracking_mode': 'detection'
                        })
                        self.disappeared[object_id] = 0
                        self.object_history[object_id].append(detection['center'])
                        
                        # Remove from hand tracking
                        if object_id in self.hand_tracked_objects:
                            del self.hand_tracked_objects[object_id]
                        
                        # Remove this detection from available detections
                        filtered_detections.remove(detection)
                        break
            
            # Hungarian algorithm alternative: greedy assignment for detection-tracked objects
            used_objects = set()
            used_detections = set()
            
            # Find best matches
            assignments = []
            for _ in range(min(len(detection_tracked_ids), len(filtered_detections))):
                min_cost = np.inf
                best_obj_idx = -1
                best_det_idx = -1
                
                for i in range(len(detection_tracked_ids)):
                    if i in used_objects:
                        continue
                    for j in range(len(filtered_detections)):
                        if j in used_detections:
                            continue
                        if cost_matrix[i, j] < min_cost:
                            min_cost = cost_matrix[i, j]
                            best_obj_idx = i
                            best_det_idx = j
                
                if best_obj_idx >= 0 and best_det_idx >= 0 and min_cost < np.inf:
                    assignments.append((best_obj_idx, best_det_idx, min_cost))
                    used_objects.add(best_obj_idx)
                    used_detections.add(best_det_idx)
                else:
                    break
            
            # Update matched objects
            for obj_idx, det_idx, cost in assignments:
                object_id = detection_tracked_ids[obj_idx]
                detection = filtered_detections[det_idx]
                
                # Update object properties
                self.objects[object_id].update({
                    'center': detection['center'],
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'age': self.objects[object_id]['age'] + 1,
                    'tracking_mode': 'detection'
                })
                self.disappeared[object_id] = 0
                self.object_history[object_id].append(detection['center'])
            
            # Handle unmatched existing objects (only detection-tracked ones)
            unmatched_objects = set(range(len(detection_tracked_ids))) - used_objects
            objects_to_check_for_hand_tracking = []
            
            for obj_idx in unmatched_objects:
                object_id = detection_tracked_ids[obj_idx]
                was_disappeared = self.disappeared[object_id]
                self.disappeared[object_id] += 1
                
                # Check if we should start hand-tracking for this newly disappeared object
                if (was_disappeared == 0 and  # Just disappeared
                    object_id not in self.hand_tracked_objects and
                    hand_detections):
                    objects_to_check_for_hand_tracking.append(object_id)
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                else:
                    self.objects[object_id]['age'] += 1
            
            # Try to start hand-tracking for newly disappeared objects
            if hand_detections and objects_to_check_for_hand_tracking:
                self.try_start_hand_tracking(objects_to_check_for_hand_tracking, hand_detections)
            
            # Register new objects from unmatched detections
            unmatched_detections = set(range(len(filtered_detections))) - used_detections
            for det_idx in unmatched_detections:
                detection = filtered_detections[det_idx]
                self.register(detection)
        
        return self.get_tracked_objects()

    def try_start_hand_tracking(self, object_ids, hand_detections):
        """Try to start hand-tracking for disappeared objects"""
        for object_id in object_ids:
            if object_id not in self.objects:
                continue
                
            obj_center = self.objects[object_id]['center']
            
            # Find the closest hand that might be associated with this object
            closest_hand = None
            min_distance = float('inf')
            
            for hand in hand_detections:
                # Check if hand has current or remembered association with this object
                if (hand.get('associated_object') == object_id or 
                    hand.get('last_associated_object') == object_id):
                    
                    distance = calculate_distance(hand['center'], obj_center)
                    if distance < self.distance_threshold * 1.5 and distance < min_distance:
                        min_distance = distance
                        closest_hand = hand
            
            if closest_hand is not None:
                self.start_hand_tracking(object_id, closest_hand['id'], closest_hand['center'])

    def get_tracked_objects(self):
        """Return current tracked objects in the expected format"""
        tracked_objects = {}
        for object_id, obj_data in self.objects.items():
            tracked_objects[object_id] = (obj_data['center'], obj_data['class_id'])
        return tracked_objects

    def get_object_info(self, object_id):
        """Get detailed information about an object"""
        if object_id in self.objects:
            obj = self.objects[object_id]
            return {
                'id': object_id,
                'center': obj['center'],
                'bbox': obj['bbox'],
                'class_id': obj['class_id'],
                'confidence': obj['confidence'],
                'age': obj['age'],
                'disappeared_frames': self.disappeared[object_id],
                'tracking_mode': obj.get('tracking_mode', 'detection'),
                'hand_tracking_info': self.hand_tracked_objects.get(object_id, None)
            }
        return None

def is_gripping(hand_landmarks, object_bbox, frame_dims):
    """Enhanced gripping detection"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    h, w = frame_dims
    thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
    middle_tip_coords = (int(middle_tip.x * w), int(middle_tip.y * h))
    
    # Check pinching gesture
    pinch_distance = calculate_distance(thumb_tip_coords, index_tip_coords)
    
    obj_x1, obj_y1, obj_x2, obj_y2 = object_bbox
    
    # More flexible gripping detection
    fingers_in_bbox = 0
    if obj_x1 <= thumb_tip_coords[0] <= obj_x2 and obj_y1 <= thumb_tip_coords[1] <= obj_y2:
        fingers_in_bbox += 1
    if obj_x1 <= index_tip_coords[0] <= obj_x2 and obj_y1 <= index_tip_coords[1] <= obj_y2:
        fingers_in_bbox += 1
    if obj_x1 <= middle_tip_coords[0] <= obj_x2 and obj_y1 <= middle_tip_coords[1] <= obj_y2:
        fingers_in_bbox += 1
    
    # Consider gripping if pinching AND fingers are near object OR multiple fingers in bbox
    return (pinch_distance < 60 and fingers_in_bbox >= 1) or fingers_in_bbox >= 2

def enhanced_associate_hands_with_objects(hand_detections, object_detections, memory_tracker, proximity_threshold=120):
    """
    Enhanced hand-object association with memory system
    """
    # Reset associations
    for hand in hand_detections:
        hand['associated_object'] = None
        hand['associated_class'] = None
        hand['last_associated_object'] = None
        hand['last_associated_class'] = None
    
    for obj in object_detections:
        obj['associated_hand'] = None

    # Phase 1: Try to maintain existing associations using memory
    for hand in hand_detections:
        hand_id = hand['id']
        remembered_obj_id, remembered_class = memory_tracker.get_hand_memory(hand_id)
        
        if remembered_obj_id is not None:
            # Find the remembered object
            remembered_obj = next((obj for obj in object_detections if obj['id'] == remembered_obj_id), None)
            
            if remembered_obj is not None:
                distance = calculate_distance(hand['center'], remembered_obj['center'])
                
                # If hand is still close to remembered object, maintain association
                if distance < proximity_threshold * 1.5:  # Slightly larger threshold for memory
                    hand['associated_object'] = remembered_obj_id
                    hand['associated_class'] = remembered_class
                    hand['last_associated_object'] = remembered_obj_id
                    hand['last_associated_class'] = remembered_class
                    remembered_obj['associated_hand'] = hand_id
                    
                    # Update memory with current association
                    memory_tracker.update_hand_memory(hand_id, remembered_obj_id, remembered_class)
                    continue

    # Phase 2: Find new associations for unassociated hands
    unassociated_hands = [h for h in hand_detections if h['associated_object'] is None]
    unassociated_objects = [o for o in object_detections if o['associated_hand'] is None]
    
    for hand in unassociated_hands:
        hand_id = hand['id']
        closest_object = None
        min_distance = float('inf')
        
        for obj in unassociated_objects:
            distance = calculate_distance(hand['center'], obj['center'])
            
            if distance < proximity_threshold and distance < min_distance:
                min_distance = distance
                closest_object = obj
        
        if closest_object is not None:
            hand['associated_object'] = closest_object['id']
            hand['associated_class'] = closest_object['class_id']
            hand['last_associated_object'] = closest_object['id']
            hand['last_associated_class'] = closest_object['class_id']
            closest_object['associated_hand'] = hand_id
            
            # Update memory
            memory_tracker.update_hand_memory(hand_id, closest_object['id'], closest_object['class_id'])
            unassociated_objects.remove(closest_object)
    
    # Phase 3: Set memory for hands that lost association but still have memory
    for hand in hand_detections:
        hand_id = hand['id']
        if hand['associated_object'] is None:
            remembered_obj_id, remembered_class = memory_tracker.get_hand_memory(hand_id)
            if remembered_obj_id is not None:
                hand['last_associated_object'] = remembered_obj_id
                hand['last_associated_class'] = remembered_class

    return hand_detections, object_detections

def detect_and_classify_objects_yolo(frame, model):
    """Detects and classifies objects using YOLO"""
    results = model(frame, verbose=False)
    
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:  # Check if boxes exist
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                })
    return frame, detections

# --- Part 3: Enhanced Counting Logic with Memory-Based Tracking ---
class EnhancedHandObjectLineCounter:
    def __init__(self, lines, crossing_tolerance=15):
        self.lines = [{'start': (line[0], line[1]), 'end': (line[2], line[3]), 'count': 0} for line in lines]
        self.crossing_tolerance = crossing_tolerance
        self.hand_trajectories = defaultdict(lambda: deque(maxlen=15))
        self.crossed_items = set()  # (hand_id, object_id, line_y) tuples
        self.total_count = 0
        self.count_by_class = defaultdict(int)
        self.crossing_hands = set()
        self.hand_line_states = {}  # hand_id -> {line_y: 'above'/'below'/'on'}
        
    def update(self, hand_detections, memory_tracker):
        current_hands = set()
        crossing_hands_in_frame = set()
        
        for hand in hand_detections:
            hand_id = hand['id']
            current_hands.add(hand_id)
            
            # Update trajectory
            self.hand_trajectories[hand_id].append(hand['center'])
            
            # Initialize line states for new hands
            if hand_id not in self.hand_line_states:
                self.hand_line_states[hand_id] = {}
            
            # Check line crossings
            if len(self.hand_trajectories[hand_id]) >= 2:
                prev_point = self.hand_trajectories[hand_id][-2]
                curr_point = self.hand_trajectories[hand_id][-1]
                
                for line_info in self.lines:
                    line_y = line_info['start'][1]
                    
                    # Determine current position relative to line
                    if abs(curr_point[1] - line_y) <= self.crossing_tolerance:
                        current_state = 'on'
                        crossing_hands_in_frame.add(hand_id)
                    elif curr_point[1] < line_y:
                        current_state = 'above'
                    else:
                        current_state = 'below'
                    
                    # Get previous state
                    prev_state = self.hand_line_states[hand_id].get(line_y, current_state)
                    
                    # Detect crossing (transition from above to below or vice versa)
                    if (prev_state == 'above' and current_state == 'below') or \
                       (prev_state == 'below' and current_state == 'above') or \
                       (prev_state in ['above', 'below'] and current_state == 'on'):
                        
                        # Determine which object to count
                        object_to_count = None
                        class_to_count = None
                        
                        # Priority 1: Current association
                        if hand['associated_object'] is not None:
                            object_to_count = hand['associated_object']
                            class_to_count = hand['associated_class']
                        # Priority 2: Last remembered association
                        elif hand['last_associated_object'] is not None:
                            object_to_count = hand['last_associated_object']
                            class_to_count = hand['last_associated_class']
                        
                        if object_to_count is not None:
                            crossing_key = (hand_id, object_to_count, line_y)
                            
                            if crossing_key not in self.crossed_items:
                                self.crossed_items.add(crossing_key)
                                self.total_count += 1
                                self.count_by_class[class_to_count] += 1
                                line_info['count'] += 1
                                
                                # Get class name (assuming yolo_model is available)
                                try:
                                    class_name = yolo_model.names[class_to_count]
                                except:
                                    class_name = f"Class_{class_to_count}"
                                
                                association_type = "current" if hand['associated_object'] is not None else "remembered"
                                print(f"CROSSING: Hand {hand_id} with {class_name} (Object ID: {object_to_count}) crossed line at y={line_y}! [Type: {association_type}]")
                                
                                # Clear the hand's memory after successful crossing to prevent duplicate counts
                                memory_tracker.clear_hand_memory(hand_id)
                                print(f"Memory cleared for Hand {hand_id} after crossing line")
                    
                    # Update state
                    self.hand_line_states[hand_id][line_y] = current_state
        
        self.crossing_hands = crossing_hands_in_frame
        
        # Clean up data for lost hands
        lost_hands = set(self.hand_line_states.keys()) - current_hands
        for lost_hand in lost_hands:
            if lost_hand in self.hand_trajectories:
                del self.hand_trajectories[lost_hand]
            if lost_hand in self.hand_line_states:
                del self.hand_line_states[lost_hand]
    
    def get_hand_state(self, hand_id):
        if hand_id in self.crossing_hands:
            return 'crossing'
        else:
            return 'detected'
    
    def get_count(self):
        return self.total_count
    
    def get_stats(self):
        return {
            'total_count': self.total_count,
            'count_by_class': dict(self.count_by_class),
            'currently_crossing': len(self.crossing_hands),
            'total_crossed_items': len(self.crossed_items),
            'line_counts': {f"line_y_{line['start'][1]}": line['count'] for line in self.lines}
        }

def display_enhanced_results(frame, hand_detections, object_detections, class_names, counter, memory_tracker, proximity_threshold, tracker):
    """Enhanced display with memory information and tracking details"""
    
    # Display objects with tracking information
    for obj in object_detections:
        track_id = obj['id']
        class_id = obj['class_id']
        center = obj['center']
        
        # Get detailed tracking info
        tracking_info = tracker.get_object_info(track_id)
        
        if obj['associated_hand'] is not None:
            color = (0, 255, 0)  # Green for associated
            association_text = f"Hand {obj['associated_hand']}"
        else:
            color = (255, 0, 0)  # Red for free
            association_text = "No hand"
        
        class_name = class_names[class_id]
        
        # Enhanced label with tracking mode and age
        if tracking_info:
            tracking_mode = tracking_info.get('tracking_mode', 'detection')
            if tracking_mode == 'hand':
                color = (255, 165, 0)  # Orange for hand-tracked objects
                mode_indicator = "HAND-TRACKED"
                hand_info = tracking_info.get('hand_tracking_info', {})
                if hand_info:
                    confidence = hand_info.get('confidence_decay', 0)
                    frames_tracked = hand_info.get('frames_tracked', 0)
                    association_text = f"Hand {hand_info.get('hand_id', '?')} (Conf:{confidence:.2f}, F:{frames_tracked})"
            else:
                mode_indicator = "DETECTION"
            
            label = f"ID:{track_id} {class_name} ({mode_indicator}, Age:{tracking_info['age']})"
            confidence_text = f"Conf:{tracking_info['confidence']:.2f}"
        else:
            label = f"ID:{track_id} {class_name}"
            confidence_text = ""
        
        cv2.putText(frame, label, (center[0] - 30, center[1] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, association_text, (center[0] - 30, center[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if confidence_text:
            cv2.putText(frame, confidence_text, (center[0] - 30, center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Special indicator for hand-tracked objects
        if tracking_info and tracking_info.get('tracking_mode') == 'hand':
            cv2.circle(frame, center, 8, (255, 165, 0), 3)  # Orange circle
            cv2.putText(frame, "H-TRACK", (center[0] - 25, center[1] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
        else:
            cv2.circle(frame, center, 4, color, -1)
        
        # Draw bounding box for tracked objects
        if tracking_info and 'bbox' in tracking_info:
            x1, y1, x2, y2 = tracking_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Display hands with enhanced information
    for hand in hand_detections:
        hand_id = hand['id']
        center = hand['center']
        state = counter.get_hand_state(hand_id)
        
        # Determine display based on association status
        if hand['associated_object'] is not None:
            if state == 'crossing':
                color = (0, 165, 255)  # Orange for crossing with object
                state_text = "CROSSING WITH OBJ"
            else:
                color = (0, 255, 0)  # Green for with object
                state_text = "WITH OBJECT"
            
            class_name = class_names[hand['associated_class']]
            association_text = f"Obj: {class_name}"
            memory_text = ""
            
        elif hand['last_associated_object'] is not None:
            color = (0, 255, 255)  # Yellow for remembered association
            state_text = "REMEMBERS OBJECT"
            class_name = class_names[hand['last_associated_class']]
            association_text = f"Last: {class_name}"
            
            # Show memory countdown
            memory_info = memory_tracker.hand_memories.get(hand_id, {})
            memory_frames_left = memory_tracker.memory_duration - memory_info.get('frame_count', 0)
            memory_text = f"({memory_frames_left}f)"
            
        else:
            color = (128, 128, 128)  # Gray for free hand
            state_text = "FREE HAND"
            association_text = "No object"
            memory_text = ""
        
        # Draw hand
        cv2.circle(frame, center, 12, color, -1)
        cv2.circle(frame, center, 12, (255, 255, 255), 2)
        
        # Draw text information
        cv2.putText(frame, f"Hand {hand_id} ({hand['label']})", 
                    (center[0] + 15, center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, state_text, 
                    (center[0] + 15, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        cv2.putText(frame, f"{association_text} {memory_text}", 
                    (center[0] + 15, center[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Display additional debugging info for hands with memory
        memory_info = memory_tracker.get_memory_info(hand_id)
        if memory_info and hand['associated_object'] is None and hand['last_associated_object'] is not None:
            debug_y = center[1] + 35
            cv2.putText(frame, f"Confidence: {memory_info['confidence']:.1f}", 
                       (center[0] + 15, debug_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw trajectory for hands with associations (current or remembered)
        if (hand['associated_object'] is not None or hand['last_associated_object'] is not None) and \
           hand_id in counter.hand_trajectories:
            trajectory = list(counter.hand_trajectories[hand_id])
            if len(trajectory) > 1:
                for j in range(1, len(trajectory)):
                    pt1 = trajectory[j-1]
                    pt2 = trajectory[j]
                    cv2.line(frame, pt1, pt2, color, 3)
        
        # Draw proximity circle for hands with associations
        if hand['associated_object'] is not None or hand['last_associated_object'] is not None:
            cv2.circle(frame, center, proximity_threshold, color, 1)
    
    # Draw connection lines between hands and objects
    for hand in hand_detections:
        if hand['associated_object'] is not None:
            for obj in object_detections:
                if obj['id'] == hand['associated_object']:
                    cv2.line(frame, hand['center'], obj['center'], (255, 255, 0), 2)
                    distance = calculate_distance(hand['center'], obj['center'])
                    mid_point = ((hand['center'][0] + obj['center'][0]) // 2,
                                 (hand['center'][1] + obj['center'][1]) // 2)
                    cv2.putText(frame, f"{distance:.0f}px", mid_point,
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    break

    # Display tracking statistics with hand-tracking info
    tracking_stats_y = 250
    cv2.rectangle(frame, (10, tracking_stats_y), (450, tracking_stats_y + 160), (0, 0, 0), -1)
    cv2.putText(frame, "Tracking Statistics:", (20, tracking_stats_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    active_objects = len([obj for obj in object_detections if obj.get('id') is not None])
    cv2.putText(frame, f"Active Objects: {active_objects}", (20, tracking_stats_y + 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    total_tracked = len(tracker.objects) if hasattr(tracker, 'objects') else 0
    cv2.putText(frame, f"Total Tracked: {total_tracked}", (20, tracking_stats_y + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    disappeared_objects = len([obj_id for obj_id, frames in tracker.disappeared.items() if frames > 0]) if hasattr(tracker, 'disappeared') else 0
    cv2.putText(frame, f"Temporarily Lost: {disappeared_objects}", (20, tracking_stats_y + 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # NEW: Hand-tracking statistics
    hand_tracked_count = len(tracker.hand_tracked_objects) if hasattr(tracker, 'hand_tracked_objects') else 0
    cv2.putText(frame, f"Hand-Tracked Objects: {hand_tracked_count}", (20, tracking_stats_y + 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    cv2.putText(frame, f"Next Object ID: {tracker.next_object_id}", (20, tracking_stats_y + 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show hand-tracking details
    if hand_tracked_count > 0:
        cv2.putText(frame, f"Hand-Track Max Frames: {tracker.hand_tracking_max_frames}", (20, tracking_stats_y + 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

    return frame

def send_frame_to_server(frame, endpoint_url="http://127.0.0.1:4002/video_frame"):
    """Send frame to server endpoint"""
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

# --- Part 4: Main Function ---
def main():
    # Initialize YOLO model
    global yolo_model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    yolo_model = YOLO('best5.pt').to(device)
    
    cap = cv2.VideoCapture("input1.mp4")

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    desired_width = 1280
    desired_height = 720
    frame_dims = (desired_height, desired_width)
    
    counting_lines = [
        (0, 300, desired_width, 300),
        (0, 600, desired_width, 600)
    ]
    
    proximity_threshold = 120
    counter = EnhancedHandObjectLineCounter(lines=counting_lines, crossing_tolerance=15)
    
    # Use the enhanced tracker with hand-tracking capabilities
    tracker = EnhancedObjectTracker(
        max_disappeared=30,
        distance_threshold=100,
        iou_threshold=0.3,
        confidence_threshold=0.3
    )
    
    memory_tracker = HandMemoryTracker(memory_duration=30, proximity_threshold=proximity_threshold)
    
    frame_count = 0
    send_every_n_frames = 5
    
    print(f"Starting enhanced hand-object detection and counting with hand-based tracking recovery...")
    print(f"Proximity threshold: {proximity_threshold} pixels")
    print(f"Memory duration: {memory_tracker.memory_duration} frames")
    print(f"Object tracking - Distance threshold: {tracker.distance_threshold}px, IoU threshold: {tracker.iou_threshold}")
    print(f"Confidence threshold: {tracker.confidence_threshold}")
    print(f"Hand-tracking max frames: {tracker.hand_tracking_max_frames}")
    print(f"Hand-tracking confidence decay: {tracker.hand_tracking_confidence_decay}")
    for line in counting_lines:
        print(f"Counting line at y={line[1]}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        original_frame = frame.copy()
        frame = cv2.resize(frame, (desired_width, desired_height))
        frame_count += 1

        # Draw counting lines
        for line in counting_lines:
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 3)
            cv2.putText(frame, f"Line at y={line[1]}", (10, line[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detect hands
        frame, hand_detections = detect_hands_with_tracking(frame)
        
        # Age memories
        memory_tracker.age_memories()
        
        # Detect objects if hands are present
        object_detections = []
        if hand_detections:
            frame, yolo_detections = detect_and_classify_objects_yolo(frame, yolo_model)
            
            # Use enhanced tracker with hand detections for recovery
            tracked_objects = tracker.update(yolo_detections, hand_detections)
            
            if tracked_objects:
                for obj_id, (center, class_id) in tracked_objects.items():
                    # Get additional object info from tracker
                    obj_info = tracker.get_object_info(obj_id)
                    if obj_info:
                        object_detections.append({
                            'id': obj_id,
                            'center': center,
                            'class_id': class_id,
                            'associated_hand': None,
                            'bbox': obj_info['bbox'],
                            'confidence': obj_info['confidence'],
                            'age': obj_info['age']
                        })

            # Enhanced association with memory
            hand_detections, object_detections = enhanced_associate_hands_with_objects(
                hand_detections, object_detections, memory_tracker, proximity_threshold)
            
            # Update counter
            counter.update(hand_detections, memory_tracker)
            
            # Display results with enhanced tracking info including hand-tracking
            frame = display_enhanced_results(
                frame, hand_detections, object_detections, 
                yolo_model.names, counter, memory_tracker, proximity_threshold, tracker)
        
        # Display statistics
        stats = counter.get_stats()
        
        # Main stats panel
        cv2.rectangle(frame, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Crossings: {stats['total_count']}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Line counts
        stats_y = 60
        for line_y, count in stats['line_counts'].items():
            cv2.putText(frame, f"  {line_y}: {count}", (20, stats_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            stats_y += 25
        
        # Additional stats
        cv2.putText(frame, f"Active Memories: {len(memory_tracker.hand_memories)}", (20, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        stats_y += 20
        cv2.putText(frame, f"Currently Crossing: {stats['currently_crossing']}", (20, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        stats_y += 20
        cv2.putText(frame, f"Proximity Threshold: {proximity_threshold}px", (20, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        stats_y += 20
        # NEW: Hand-tracking status
        hand_tracked_count = len(tracker.hand_tracked_objects) if hasattr(tracker, 'hand_tracked_objects') else 0
        cv2.putText(frame, f"Hand-Tracked Objects: {hand_tracked_count}", (20, stats_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Class-specific counts
        if stats['count_by_class']:
            class_display_y = stats_y + 30
            cv2.putText(frame, "Count by Class:", (20, class_display_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for class_id, count in stats['count_by_class'].items():
                class_display_y += 25
                class_name = yolo_model.names[class_id]
                cv2.putText(frame, f"  {class_name}: {count}", (30, class_display_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Enhanced Legend with hand-tracking info
        legend_x, legend_y = frame.shape[1] - 350, 20
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10), (frame.shape[1] - 10, legend_y + 240), (0, 0, 0), -1)
        cv2.putText(frame, "System Legend:", (legend_x, legend_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Hand state legend
        cv2.circle(frame, (legend_x + 10, legend_y + 30), 8, (128, 128, 128), -1)
        cv2.putText(frame, "Free Hand", (legend_x + 25, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(frame, (legend_x + 10, legend_y + 50), 8, (0, 255, 0), -1)
        cv2.putText(frame, "With Object", (legend_x + 25, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(frame, (legend_x + 10, legend_y + 70), 8, (0, 255, 255), -1)
        cv2.putText(frame, "Remembers Object", (legend_x + 25, legend_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(frame, (legend_x + 10, legend_y + 90), 8, (0, 165, 255), -1)
        cv2.putText(frame, "Crossing Line", (legend_x + 25, legend_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Object state legend
        cv2.putText(frame, "Objects:", (legend_x, legend_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (legend_x + 10, legend_y + 125), (legend_x + 25, legend_y + 135), (0, 255, 0), -1)
        cv2.putText(frame, "Detection", (legend_x + 30, legend_y + 133), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (legend_x + 10, legend_y + 145), (legend_x + 25, legend_y + 155), (255, 165, 0), -1)
        cv2.putText(frame, "Hand-Tracked", (legend_x + 30, legend_y + 153), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(frame, (legend_x + 150, legend_y + 125), (legend_x + 165, legend_y + 135), (255, 0, 0), -1)
        cv2.putText(frame, "Free", (legend_x + 170, legend_y + 133), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Tracking info
        cv2.putText(frame, f"Tracking Info:", (legend_x, legend_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"IoU Thresh: {tracker.iou_threshold}", (legend_x, legend_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Conf Thresh: {tracker.confidence_threshold}", (legend_x, legend_y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Hand-Track Max: {tracker.hand_tracking_max_frames}f", (legend_x, legend_y + 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        # Server communication
        hands_with_associations = [h for h in hand_detections if h['associated_object'] is not None or h['last_associated_object'] is not None]
        if hands_with_associations and frame_count % send_every_n_frames == 0:
            print(f"Frame {frame_count}: Sending frame with {len(hands_with_associations)} hand-object associations...")
            response = send_frame_to_server(original_frame)
            if response:
                print(f"Server response: {response}")

        # Server status display
        server_status_y = 700
        if hands_with_associations:
            cv2.putText(frame, f"SERVER: Active - {len(hands_with_associations)} associations", 
                        (10, server_status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "SERVER: Waiting for associations", 
                        (10, server_status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Enhanced Hand-Object Association and Counting System with Hand-Based Recovery', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    print("Sending shutdown signal to server...")
    send_shutdown_signal()

    # Final statistics
    final_stats = counter.get_stats()
    print("\n=== Final Enhanced Hand-Object Crossing Statistics with Hand-Tracking ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total hand-object crossings: {final_stats['total_count']}")
    print(f"Total crossed items: {final_stats['total_crossed_items']}")
    print(f"Final active memories: {len(memory_tracker.hand_memories)}")
    print(f"Final tracked objects: {len(tracker.objects)}")
    print(f"Final hand-tracked objects: {len(tracker.hand_tracked_objects)}")
    print(f"Count by object class:")
    for class_id, count in final_stats['count_by_class'].items():
        print(f"  {yolo_model.names[class_id]}: {count}")
    print(f"Line counts: {final_stats['line_counts']}")
    
    # Tracking analysis
    print(f"\nObject Tracking Analysis:")
    print(f"Total objects ever tracked: {tracker.next_object_id}")
    print(f"Maximum disappearance tolerance: {tracker.max_disappeared} frames")
    print(f"Distance matching threshold: {tracker.distance_threshold}px")
    print(f"IoU matching threshold: {tracker.iou_threshold}")
    
    # Hand-tracking analysis
    print(f"\nHand-Tracking Analysis:")
    print(f"Hand-tracking max frames: {tracker.hand_tracking_max_frames}")
    print(f"Hand-tracking confidence decay rate: {tracker.hand_tracking_confidence_decay}")
    print(f"Minimum hand-tracking confidence: {tracker.min_hand_tracking_confidence}")
    
    if tracker.hand_tracked_objects:
        print("Active hand-tracked objects at end:")
        for obj_id, hand_info in tracker.hand_tracked_objects.items():
            obj_info = tracker.get_object_info(obj_id)
            if obj_info:
                class_name = yolo_model.names[obj_info['class_id']]
                print(f"  Object {obj_id} ({class_name}): Hand {hand_info['hand_id']}, "
                      f"Confidence: {hand_info['confidence_decay']:.3f}, "
                      f"Frames: {hand_info['frames_tracked']}")
    
    # Memory analysis
    if memory_tracker.hand_memories:
        print("\nFinal hand memories:")
        for hand_id, memory in memory_tracker.hand_memories.items():
            class_name = yolo_model.names[memory['last_class']]
            frames_left = memory_tracker.memory_duration - memory['frame_count']
            print(f"  Hand {hand_id}: {class_name} (Object {memory['last_object']}) - {frames_left} frames remaining")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()