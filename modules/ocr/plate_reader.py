import cv2
import easyocr
import numpy as np


class PlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.violations = {}

    def preprocess(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return thresh

    def read_plate(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        pad = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        processed = self.preprocess(crop)
        results = self.reader.readtext(processed)

        if not results:
            return None

        best = max(results, key=lambda x: x[2])
        text = best[1].strip().upper()
        confidence = best[2]

        if confidence > 0.2 and len(text) > 2:
            return text
        return None

    def log_violation(self, track_id, plate, speed):
        if track_id not in self.violations:
            self.violations[track_id] = {
                "plate": plate,
                "speed": speed
            }
            print(f"VIOLATION — ID:{track_id} | Plate:{plate} | Speed:{speed} km/h")