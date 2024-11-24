# # Install required libraries
# !pip install mediapipe opencv-python ultralytics torch
# !pip install -r requirements.txt

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model_path = r'C:\Users\swaro\.qaihm\models\yolov8_det\v1\ultralytics_ultralytics_git\runs\detect\train4\weights\best.pt'  # Replace with your model path
model = YOLO(model_path)

# Move the model to GPU
model.to(device)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Attempt to enable GPU acceleration for MediaPipe Hands
# Note: This may require building MediaPipe from source with GPU support
# For demonstration, we'll proceed with default settings
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera, or use a file path for a video file

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Optionally resize frame for faster processing
    # frame = cv2.resize(frame, (640, 480))

    # Perform inference with YOLOv8
    results = model(frame, device=device)

    # Process the results
    for result in results:
        boxes = result.boxes  # Bounding boxes
        names = result.names  # Class names

        if boxes is not None and boxes.xyxy is not None:
            # No need to move tensors to CPU since we're using GPU
            pred = boxes.xyxy  # Tensor on GPU
            confs = boxes.conf
            cls_ids = boxes.cls.int()  # Class IDs as integers

            # Convert tensors to CPU NumPy arrays for OpenCV functions
            pred_np = pred.detach().cpu().numpy()
            confs_np = confs.detach().cpu().numpy()
            cls_ids_np = cls_ids.detach().cpu().numpy()

            # Draw bounding boxes and labels
            for i in range(len(pred_np)):
                x1, y1, x2, y2 = map(int, pred_np[i])
                conf = confs_np[i]
                cls = cls_ids_np[i]
                label = names[cls] if cls in names else 'Unknown'
                color = (0, 255, 0)  # Green color for the bounding box

                # Ensure coordinates are within frame dimensions
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label_text = f'{label} {conf:.2f}'
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                output_file = "transcription_output.txt"  # File to save all transcriptions
                transcriptions = []  # Store all transcription results
                with open(output_file, "w") as f:
                    f.write("\n".join(transcriptions))

    # Process the frame with MediaPipe Hands
    # Note: MediaPipe Hands may still run on CPU
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(frame_rgb)

    # Draw hand landmarks and check fingertip position
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, c = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw a circle at the index fingertip
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

            # Check if the fingertip is within any bounding box
            if 'pred_np' in locals():
                for i in range(len(pred_np)):
                    x1, y1, x2, y2 = map(int, pred_np[i])
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        label = names[cls_ids_np[i]] if cls_ids_np[i] in names else 'Unknown'
                        cv2.putText(frame, label, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('YOLOv8 and MediaPipe Hands Integration', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
