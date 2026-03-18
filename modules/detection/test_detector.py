import cv2
import sys
from detector import VehicleDetector

detector = VehicleDetector()

video_path = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\samples\traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
    sys.exit()

print("Running detection... press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    detections = detector.detect(frame)
    frame = detector.draw(frame, detections)
    frame = cv2.resize(frame, (960, 540))
    # Show count on screen
    cv2.putText(frame, f"Vehicles detected: {len(detections)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()