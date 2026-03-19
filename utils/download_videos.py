import yt_dlp
import os

# Add YouTube URLs here — search these on YouTube and copy the URLs:
# "ambulance flashing lights india traffic"
# "ambulance emergency india road"
# "indian ambulance siren road dashcam"

VIDEOS = [
    {
        "url": "https://youtu.be/g8bdycR6YnI?si=MGthQebsm8wi9Ks-",
        "name": "ambulance1"
    },
    {
        "url": "https://youtube.com/shorts/qQdZV3Y90JI?si=Y-NpqE5v2oPBrUrx",
        "name": "ambulance2"
    },
    {
        "url": "https://youtube.com/shorts/H2kpIWMEnEM?si=qvkFn7s62jNSCaPf",
        "name": "ambulance3"
    },
    # {
    #     "url": "https://www.youtube.com/watch?v=PASTE_URL_4_HERE",
    #     "name": "ambulance4"
    # },
]

output_dir = r"C:\Users\vaibh\Desktop\intelligent-traffic-monitoring\data\raw"
os.makedirs(output_dir, exist_ok=True)

def download_video(url, name):
    output_path = os.path.join(output_dir, f"{name}.mp4")

    if os.path.exists(output_path):
        print(f"{name}.mp4 already exists, skipping...")
        return

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    print(f"\nDownloading {name}...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"{name}.mp4 downloaded successfully!")
    except Exception as e:
        print(f"Failed to download {name}: {e}")

for video in VIDEOS:
    if "PASTE_URL" in video["url"]:
        print(f"Skipping {video['name']} — URL not set yet")
        continue
    download_video(video["url"], video["name"])

print("\nAll downloads complete!")
print(f"Videos saved in: {output_dir}")
# ```

# Now here's how to get the YouTube URLs:

# **Step 1** — Go to YouTube and search:
# ```
# ambulance flashing lights india traffic
# ```

# **Step 2** — Open a video → copy the URL from address bar. It looks like:
# ```
# https://www.youtube.com/watch?v=xK2uElmcgqQ