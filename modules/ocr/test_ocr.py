import cv2
import sys
sys.path.append("../detection")
sys.path.append("../speed")

from detector import VehicleDetector
from speed_estimator import SpeedEstimator
from plate_reader import PlateReader
from deep_sort_realtime.deepsort_tracker import DeepSort

video_path = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\samples\traffic.mp4"

detector = VehicleDetector()
estimator = SpeedEstimator(fps=30, speed_limit=50)
plate_reader = PlateReader()
tracker = DeepSort(max_age=30)

print("Starting... EasyOCR loading (first time is slow)")
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    ds_input = []
    detections = detector.detect(frame)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        ds_input.append(([x1, y1, w, h], d["confidence"], d["class"]))

    tracks = tracker.update_tracks(ds_input, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = tuple(map(int, track.to_ltrb()))
        speed = estimator.estimate_speed(track_id, bbox)

        if speed is None:
            continue

        frame = estimator.draw_speed(frame, track_id, bbox, speed)

        if estimator.is_overspeeding(track_id) and frame_count % 15 == 0:
            plate = plate_reader.read_plate(frame, bbox)
            if plate:
                plate_reader.log_violation(track_id, plate, speed)

        if track_id in plate_reader.violations:
            x1, y1, x2, y2 = bbox
            plate_text = plate_reader.violations[track_id]["plate"]
            cv2.putText(frame, f"PLATE: {plate_text}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    frame = cv2.resize(frame, (960, 540))
    cv2.putText(frame, f"Violations: {len(plate_reader.violations)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Traffic Monitor - OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- Violation Report ---")
for tid, info in plate_reader.violations.items():
    print(f"ID:{tid} | Plate:{info['plate']} | Speed:{info['speed']} km/h")