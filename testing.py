import cv2
import torch
import numpy as np
import time
import threading
import winsound  # For alert sound on Windows
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace

# Load YOLOv8 Model (Person Detection)
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for a lightweight model

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50)  # Tracks person movement with unique IDs

# Define camera source (0 for webcam)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to 30

# Define an invisible restricted area (bounding box coordinates)
box_x1, box_y1, box_x2, box_y2 = 200, 150, 400, 350  # Define the top-left and bottom-right corners

# Store detected persons who entered the box
entered_ids = {}  # {person_id: timestamp}
cooldown_time = 10  # Cooldown period in seconds

# Global variables for async DeepFace processing
gender_labels = {}  # {person_id: "Gender"}
processing = {}  # Track processing status for each person

def play_alert():
    """Plays an alert sound when a person enters the restricted box."""
    winsound.Beep(1000, 500)  # 1000 Hz frequency for 500ms

def save_screenshot(frame, track_id):
    """Save a screenshot when an alert is triggered."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"alert_{track_id}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Screenshot saved: {filename}")

def analyze_frame(person_id, face_crop):
    """Run DeepFace asynchronously for gender detection."""
    global gender_labels, processing
    processing[person_id] = True  # Prevent multiple calls for the same ID
    
    # Run gender detection
    result = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False)
    
    # Extract gender prediction
    gender_dict = result[0]['gender']
    gender = max(gender_dict, key=gender_dict.get)  # Get highest probability
    
    # Update gender label
    gender_labels[person_id] = gender
    processing[person_id] = False  # Allow next detection

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons with YOLO
    results = model(frame)
    detections = []
    
    # Extract bounding boxes and confidence scores
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) == 0:  # Class 0 = Person
                detections.append([[x1, y1, x2, y2], conf, "person"])
    
    # Track persons using DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Uncomment to visualize the restricted box
    # cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

    # Process tracked persons
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        bbox = track.to_ltwh()  # Get bounding box (left, top, width, height)
        x, y, w, h = map(int, bbox)
        center_x, center_y = x + w // 2, y + h // 2  # Person's center

        # Extract face for gender classification
        face_crop = frame[y:y+h, x:x+w]
        
        # Run DeepFace in a separate thread if not already processing
        if track_id not in processing or not processing[track_id]:
            threading.Thread(target=analyze_frame, args=(track_id, face_crop), daemon=True).start()

        # Check if person is inside the box
        if box_x1 <= center_x <= box_x2 and box_y1 <= center_y <= box_y2:
            current_time = time.time()
            
            # Check cooldown to prevent duplicate alerts
            if track_id not in entered_ids or (current_time - entered_ids[track_id]) > cooldown_time:
                print(f"ALERT 🚨: Person {track_id} entered the restricted area!")
                threading.Thread(target=play_alert, daemon=True).start()  # Play alert sound
                save_screenshot(frame, track_id)  # Save screenshot
                entered_ids[track_id] = current_time  # Update last entry time

        # Draw bounding box and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display gender label
        gender_text = gender_labels.get(track_id, "Detecting...")
        cv2.putText(frame, f"{gender_text}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the video stream
    cv2.imshow("Restricted Area & Gender Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
