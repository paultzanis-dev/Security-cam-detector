#!/usr/bin/env python3
"""
Security Camera Web Dashboard
Run this alongside or instead of multicam.py to get a browser-based dashboard.
Visit http://localhost:5000 in your browser.
"""

from flask import Flask, Response, render_template_string, jsonify
import cv2
from ultralytics import YOLO
import datetime
import numpy as np
import threading
import os
import base64
import json

app = Flask(__name__)

# ─── SETTINGS ──────────────────────────────────────────
NVR_IP   = "192.168.0.224"
from urllib.parse import quote
PASSWORD = quote("pao1908pao1908?", safe="")

CAMERAS = {
    "Front Door":  f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/101",
    "Backyard":    f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/201",
    "Driveway":    f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/301",
    "Side Gate":   f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/401",
    "Garage":      f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/501",
    "Street":      f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/601",
    "Backgate":    f"rtsp://PaulTz:{PASSWORD}@{NVR_IP}:8554/Streaming/Channels/701",
}

TRACK = ["person", "car", "truck", "bicycle", "motorcycle"]
SNAPSHOT_DIR = os.path.expanduser("~/projects/security-cam/snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ─── SHARED STATE ──────────────────────────────────────
model = YOLO("yolov8n.pt")
latest_frames = {}      # cam_name -> jpeg bytes
detection_counts = {}   # cam_name -> {label: count}
recent_detections = []  # list of {time, camera, label, snapshot}
state_lock = threading.Lock()

# ─── DETECTION LOGIC ───────────────────────────────────
def process_frame(frame, cam_name):
    results = model(frame, verbose=False)
    counts = {}
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls)]
            if label not in TRACK:
                continue
            conf = float(box.conf[0])
            if conf < 0.4:
                continue
            counts[label] = counts.get(label, 0) + 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if label == "person" else (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Save snapshot if person detected
            if label == "person":
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"{cam_name.replace(' ', '_')}_{ts}.jpg"
                snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
                cv2.imwrite(snap_path, frame)
                with state_lock:
                    recent_detections.insert(0, {
                        "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        "date": datetime.datetime.now().strftime("%b %d"),
                        "camera": cam_name,
                        "label": label,
                        "snapshot": snap_name
                    })
                    if len(recent_detections) > 50:
                        recent_detections.pop()

    with state_lock:
        detection_counts[cam_name] = counts
    return frame

def camera_thread(cam_name, url):
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = process_frame(frame, cam_name)
        else:
            # Offline placeholder
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"{cam_name} - OFFLINE", (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60), 2)
            cap.release()
            cap = cv2.VideoCapture(url)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with state_lock:
            latest_frames[cam_name] = jpeg.tobytes()

# ─── ROUTES ────────────────────────────────────────────
def generate_stream(cam_name):
    while True:
        with state_lock:
            frame = latest_frames.get(cam_name)
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video/<cam_name>")
def video_feed(cam_name):
    return Response(generate_stream(cam_name),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/counts")
def api_counts():
    with state_lock:
        return jsonify(detection_counts)

@app.route("/api/detections")
def api_detections():
    with state_lock:
        return jsonify(recent_detections[:20])

@app.route("/snapshot/<filename>")
def snapshot(filename):
    path = os.path.join(SNAPSHOT_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    return "", 404

@app.route("/")
def index():
    cam_names = list(CAMERAS.keys())
    return render_template_string(HTML_TEMPLATE, cam_names=cam_names)

# ─── HTML TEMPLATE ─────────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Security Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #1e1e2e;
    --accent: #00ff88;
    --accent2: #ff6b35;
    --text: #e2e2e8;
    --muted: #6b6b80;
    --person: #00ff88;
    --vehicle: #ff6b35;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--accent);
  }

  .logo-dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }

  .header-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--muted);
  }

  .main {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 0;
    height: calc(100vh - 57px);
  }

  .cameras-panel {
    padding: 20px;
    overflow-y: auto;
  }

  .cameras-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }

  .cam-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    position: relative;
  }

  .cam-card:hover {
    border-color: var(--accent);
    transition: border-color 0.2s;
  }

  .cam-feed {
    width: 100%;
    aspect-ratio: 16/9;
    object-fit: cover;
    display: block;
    background: #000;
  }

  .cam-footer {
    padding: 8px 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .cam-name {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .cam-counts {
    display: flex;
    gap: 6px;
  }

  .count-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(0,255,136,0.1);
    color: var(--accent);
    border: 1px solid rgba(0,255,136,0.2);
  }

  .count-badge.vehicle {
    background: rgba(255,107,53,0.1);
    color: var(--vehicle);
    border-color: rgba(255,107,53,0.2);
  }

  /* Right sidebar */
  .sidebar {
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .sidebar-section {
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }

  .section-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  .detections-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
  }

  .detection-item {
    display: flex;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    align-items: flex-start;
  }

  .detection-item:last-child { border-bottom: none; }

  .detection-snap {
    width: 60px;
    height: 45px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid var(--border);
    flex-shrink: 0;
    background: var(--border);
  }

  .detection-info { flex: 1; min-width: 0; }

  .detection-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    text-transform: capitalize;
  }

  .detection-cam {
    font-size: 11px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
  }

  .detection-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-top: 3px;
  }

  .empty-state {
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    padding: 24px 0;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .stat-label { font-size: 12px; color: var(--muted); }
  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    SECURITY · LIVE
  </div>
  <div class="header-time" id="clock">--:--:--</div>
</header>

<div class="main">
  <div class="cameras-panel">
    <div class="cameras-grid">
      {% for name in cam_names %}
      <div class="cam-card">
        <img class="cam-feed"
             src="/video/{{ name }}"
             alt="{{ name }}">
        <div class="cam-footer">
          <span class="cam-name">{{ name }}</span>
          <div class="cam-counts" id="counts-{{ loop.index0 }}" data-cam="{{ name }}"></div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="sidebar">
    <div class="sidebar-section">
      <div class="section-title">Summary</div>
      <div class="stat-row">
        <span class="stat-label">Cameras online</span>
        <span class="stat-value" id="stat-online">{{ cam_names|length }}</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">People detected</span>
        <span class="stat-value" id="stat-people">0</span>
      </div>
      <div class="stat-row">
        <span class="stat-label">Vehicles detected</span>
        <span class="stat-value" id="stat-vehicles">0</span>
      </div>
    </div>

    <div class="section-title" style="padding: 16px 16px 0">Recent Detections</div>
    <div class="detections-list" id="detections-list">
      <div class="empty-state">Watching for activity…</div>
    </div>
  </div>
</div>

<script>
// Clock
function updateClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}
setInterval(updateClock, 1000);
updateClock();

// Poll detection counts
async function updateCounts() {
  try {
    const res = await fetch('/api/counts');
    const data = await res.json();
    let totalPeople = 0, totalVehicles = 0;

    document.querySelectorAll('[data-cam]').forEach(el => {
      const cam = el.dataset.cam;
      const counts = data[cam] || {};
      el.innerHTML = '';
      let people = counts['person'] || 0;
      let vehicles = (counts['car']||0) + (counts['truck']||0) +
                     (counts['bicycle']||0) + (counts['motorcycle']||0);
      totalPeople += people;
      totalVehicles += vehicles;
      if (people > 0) el.innerHTML += `<span class="count-badge">👤 ${people}</span>`;
      if (vehicles > 0) el.innerHTML += `<span class="count-badge vehicle">🚗 ${vehicles}</span>`;
    });

    document.getElementById('stat-people').textContent = totalPeople;
    document.getElementById('stat-vehicles').textContent = totalVehicles;
  } catch(e) {}
}

// Poll recent detections
async function updateDetections() {
  try {
    const res = await fetch('/api/detections');
    const items = await res.json();
    const list = document.getElementById('detections-list');
    if (items.length === 0) {
      list.innerHTML = '<div class="empty-state">Watching for activity…</div>';
      return;
    }
    list.innerHTML = items.map(d => `
      <div class="detection-item">
        <img class="detection-snap" src="/snapshot/${d.snapshot}" alt="snap"
             onerror="this.style.background='#1e1e2e'">
        <div class="detection-info">
          <div class="detection-label">${d.label}</div>
          <div class="detection-cam">${d.camera}</div>
          <div class="detection-time">${d.date} · ${d.time}</div>
        </div>
      </div>`).join('');
  } catch(e) {}
}

setInterval(updateCounts, 1000);
setInterval(updateDetections, 3000);
updateCounts();
updateDetections();
</script>
</body>
</html>
"""

# ─── MAIN ──────────────────────────────────────────────
if __name__ == "__main__":
    print("🎥 Starting camera threads...")
    for name, url in CAMERAS.items():
        t = threading.Thread(target=camera_thread, args=(name, url), daemon=True)
        t.start()
        print(f"   ✓ {name}")

    print("\n✅ Dashboard ready!")
    print("👉 Open your browser and go to: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
