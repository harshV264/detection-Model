import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 Model (Person Detection)
model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt' for a lightweight model

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50)  # Tracks person movement with unique IDs

# Define camera source (0 for webcam)
cap = cv2.VideoCapture(0)

# Define a vertical crossing line (adjust as needed)
line_x = 300  # Line's X-coordinate
line_color = (0, 0, 255)  # Red color for boundary line
line_thickness = 2

# Store detected persons who crossed the line
crossed_ids = {}  # {person_id: timestamp}
cooldown_time = 10  # Cooldown period in seconds

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

    # Draw vertical crossing line
    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), line_color, line_thickness)

    # Process tracked persons
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        bbox = track.to_ltwh()  # Get bounding box (left, top, width, height)
        x, y, w, h = map(int, bbox)
        center_x = x + w // 2  # Find the horizontal center of the person

        # Check if person crosses the vertical line
        if center_x >= line_x:
            current_time = time.time()

            # Check cooldown to prevent duplicate alerts
            if track_id not in crossed_ids or (current_time - crossed_ids[track_id]) > cooldown_time:
                print(f"ALERT 🚨: Person {track_id} crossed the boundary!")
                crossed_ids[track_id] = current_time  # Update last crossing time

        # Draw bounding box and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the video stream
    cv2.imshow("Restricted Area Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
