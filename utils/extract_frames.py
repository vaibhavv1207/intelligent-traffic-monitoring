import cv2
import os

video_paths = [
    r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\raw\ambulance1.mp4",
    r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\raw\ambulance2.mp4",
    r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\raw\ambulance3.mp4",
]

output_dir = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\ambulance_dataset\images\train"
os.makedirs(output_dir, exist_ok=True)

for video_path in video_paths:
    if not os.path.exists(video_path):
        print(f"Skipping {video_path} — file not found")
        continue

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    video_name = os.path.basename(video_path).replace(".mp4", "")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 10 == 0:
            filename = f"{video_name}_frame{saved:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved += 1

    cap.release()
    print(f"{video_name}: extracted {saved} frames")

print(f"\nDone! Total images saved in: {output_dir}")