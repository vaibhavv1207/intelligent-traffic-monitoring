import cv2
import sys
import torch
import numpy as np
from collections import deque
from torchvision import transforms
sys.path.append("../detection")

from detector import VehicleDetector
from accident_detector import AccidentCNNLSTM

# Paths
VIDEO_PATH  = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\samples\traffic.mp4"
MODEL_PATH  = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\modules\accident\accident_model.pth"

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH  = 10
THRESHOLD   = 0.75  # confidence threshold to trigger alert

# Load model
model = AccidentCNNLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)
print("Accident model loaded!")

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

detector    = VehicleDetector()
frame_buffer = deque(maxlen=SEQ_LENGTH)
cap          = cv2.VideoCapture(VIDEO_PATH)
accident_active = False
alert_frames    = 0

print("Running accident detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add frame to buffer
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb)
    frame_buffer.append(tensor)

    # Only predict when buffer is full
    if len(frame_buffer) == SEQ_LENGTH:
        sequence = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(sequence)
            probs  = torch.softmax(output, dim=1)
            accident_prob = probs[0][0].item()  # class 0 = Accident

        accident_active = accident_prob > THRESHOLD

        if accident_active:
            alert_frames += 1

    # Draw detections
    detections = detector.detect(frame)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show accident alert banner
    if accident_active:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 200), -1)
        cv2.putText(frame, "ACCIDENT DETECTED — ALERTING AUTHORITIES!",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)

    # Show probability
    if len(frame_buffer) == SEQ_LENGTH:
        cv2.putText(frame, f"Accident probability: {accident_prob:.2f}",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Phase 6 - Accident Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal accident alerts triggered: {alert_frames} frames")