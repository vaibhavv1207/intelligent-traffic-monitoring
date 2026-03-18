import cv2
from detector import VehicleDetector

detector = VehicleDetector()

# Test on webcam (press Q to quit)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    frame = detector.draw(frame, detections)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()