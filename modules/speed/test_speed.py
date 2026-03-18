import cv2
import sys
sys.path.append("../detection")

from detector import VehicleDetector
from speed_estimator import SpeedEstimator
from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\samples\traffic.mp4"

detector = VehicleDetector()
estimator = SpeedEstimator(fps=30, speed_limit=50)
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    # Format detections for DeepSort: ([x1,y1,w,h], confidence, class)
    ds_input = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        ds_input.append(([x1, y1, w, h], d["confidence"], d["class"]))

    tracks = tracker.update_tracks(ds_input, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()  # (x1, y1, x2, y2)
        bbox = tuple(map(int, bbox))

        speed = estimator.estimate_speed(track_id, bbox)
        if speed is not None:
            frame = estimator.draw_speed(frame, track_id, bbox, speed)

    frame = cv2.resize(frame, (960, 540))
    cv2.putText(frame, f"Speed limit: {estimator.speed_limit} km/h", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Speed Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()