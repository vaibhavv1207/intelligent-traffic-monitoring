import cv2
import numpy as np
from collections import defaultdict


class SpeedEstimator:
    def __init__(self, fps=30, speed_limit=50):
        self.fps = fps
        self.speed_limit = speed_limit  # km/h
        self.track_history = defaultdict(list)  # track_id -> list of centers
        self.speeds = {}  # track_id -> speed in km/h

        # Pixels per meter (calibrate this for your video)
        # For now using an estimate — we'll refine with homography later
        self.pixels_per_meter = 8.0

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def estimate_speed(self, track_id, bbox):
        center = self.get_center(bbox)
        self.track_history[track_id].append(center)

        # Need at least 2 points to calculate speed
        if len(self.track_history[track_id]) < 2:
            return None

        # Use last 5 frames for smoother speed
        history = self.track_history[track_id][-5:]
        if len(history) < 2:
            return None

        # Pixel distance between first and last point in history
        x1, y1 = history[0]
        x2, y2 = history[-1]
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Convert to real-world distance (meters)
        real_distance = pixel_distance / self.pixels_per_meter

        # Time elapsed (seconds)
        time_elapsed = len(history) / self.fps

        # Speed in m/s then convert to km/h
        speed_ms = real_distance / time_elapsed
        speed_kmh = speed_ms * 3.6

        self.speeds[track_id] = round(speed_kmh, 1)
        return self.speeds[track_id]

    def is_overspeeding(self, track_id):
        speed = self.speeds.get(track_id, 0)
        return speed > self.speed_limit

    def draw_speed(self, frame, track_id, bbox, speed):
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255) if self.is_overspeeding(track_id) else (0, 255, 0)
        label = f"ID:{track_id} {speed} km/h"
        if self.is_overspeeding(track_id):
            label += " OVERSPEED!"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame