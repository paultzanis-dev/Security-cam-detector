# 🔒 Security Camera Detector

Live security camera dashboard with AI object detection using YOLOv8.

## What it does
- Connects to 7 LaView PoE security cameras via RTSP
- Detects people, cars, trucks, bicycles in real time
- Shows all cameras in a grid dashboard
- Logs all detections with timestamps to detections.log

## Setup
1. Install dependencies: `pip install ultralytics opencv-python`
2. Set your NVR IP and password in multicam.py
3. Run: `python3 multicam.py`

## Built with
- Python 3.13
- YOLOv8 (Ultralytics)
- OpenCV
