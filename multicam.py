import cv2
from ultralytics import YOLO
import datetime
import numpy as np

# ─── SETTINGS ──────────────────────────────────────────
NVR_IP   = "192.168.1.100"   # ← change this tonight
PASSWORD = "admin"            # ← change this if different

CAMERAS = {
    "Front Door":  f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/101",
    "Backyard":    f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/201",
    "Driveway":    f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/301",
    "Side Gate":   f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/401",
    "Garage":      f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/501",
    "Street":      f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/601",
    "Backgate":    f"rtsp://admin:{PASSWORD}@{NVR_IP}:554/Streaming/Channels/701",
}

TRACK    = ["person", "car", "truck", "bicycle", "motorcycle"]
LOG_FILE = "detections.log"
# ───────────────────────────────────────────────────────

model = YOLO("yolov8n.pt")

# Open all camera streams
caps = {}
for name, url in CAMERAS.items():
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print(f"✅ Connected: {name}")
    else:
        print(f"❌ Failed:    {name}")
    caps[name] = cap

def log_detection(camera, label, count):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{now}] {camera} → {label}: {count}\n")

def process_frame(frame, cam_name):
    results = model(frame, verbose=False)
    counts = {}

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            if label not in TRACK:
                continue

            counts[label] = counts.get(label, 0) + 1

            # Draw detection box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw camera name
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(frame, cam_name, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw counts
    y = 55
    for label, count in counts.items():
        cv2.putText(frame, f"{label}: {count}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 25
        log_detection(cam_name, label, count)

    return frame

def build_grid(frames, grid_cols=3):
    """Arrange frames into a grid"""
    if not frames:
        return None

    # Resize all frames to same size
    cell_w, cell_h = 640, 360
    resized = []
    for f in frames:
        resized.append(cv2.resize(f, (cell_w, cell_h)))

    # Pad to fill grid
    grid_cols = min(grid_cols, len(resized))
    grid_rows = (len(resized) + grid_cols - 1) // grid_cols
    while len(resized) < grid_rows * grid_cols:
        resized.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    rows = []
    for r in range(grid_rows):
        row = resized[r * grid_cols:(r + 1) * grid_cols]
        rows.append(np.hstack(row))
    return np.vstack(rows)

print("\n🎥 Security Dashboard Running — Press Q to quit\n")

while True:
    frames = []

    for name, cap in caps.items():
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = process_frame(frame, name)
        else:
            # Show offline placeholder
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"{name} - OFFLINE", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        frames.append(frame)

    grid = build_grid(frames, grid_cols=3)
    if grid is not None:
        cv2.imshow("🔒 Security Dashboard", grid)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()
print("\n✅ Dashboard closed. Check detections.log for history.")
