# backend/app.py
import cv2 as cv
import numpy as np
import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import queue
import pickle
import cv2 # Ensure cv2 is imported for VideoWriter
import shutil

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_videos")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRIGGER_FILE = os.path.join(BASE_DIR, "trigger_training.txt")
MODEL_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

def process_video_to_dataset(video_path, username):
    print(f"[DEBUG] Processing video for {username}")
    print(f"[DEBUG] Video Path: {video_path}")

    # 1. Check Model
    if not os.path.exists(YUNET_PATH):
        print(f"[ERROR] Model missing at: {YUNET_PATH}")
        return 0
    
    try:
        # Lower threshold to 0.6
        detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)
    except Exception as e:
        print(f"[ERROR] Failed to load AI Model: {e}")
        return 0

    # 2. Check Video File
    if not os.path.exists(video_path):
        print("[ERROR] Video file does not exist on disk!")
        return 0
    
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        print("[ERROR] Uploaded video is 0 bytes (Empty)!")
        return 0

    # 3. Open Video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] OpenCV cannot open the video file. Missing FFmpeg?")
        return 0

    user_path = os.path.join(DATASET_DIR, username)
    os.makedirs(user_path, exist_ok=True)
    
    count = 0
    frames_read = 0
    margin = 0.25 
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frames_read += 1
        
        h_img, w_img = frame.shape[:2]
        detector.setInputSize((w_img, h_img))
        _, faces = detector.detect(frame)
        
        if faces is not None:
             for face in faces:
                confidence = face[-1]
                if confidence < 0.6: continue 

                box = face[:4].astype(int)
                x, y, w, h = box
                
                # Padding
                x_new = max(0, int(x - w * margin))
                y_new = max(0, int(y - h * margin))
                w_new = min(w_img - x_new, int(w * (1 + 2 * margin)))
                h_new = min(h_img - y_new, int(h * (1 + 2 * margin)))
                
                crop = frame[y_new:y_new+h_new, x_new:x_new+w_new]
                
                if crop.size > 0:
                    img_name = f"{count}.jpg"
                    cv.imwrite(os.path.join(user_path, img_name), crop)
                    count += 1
                    if count >= 100: break 
        if count >= 100: break
        
    cap.release()
    
    if frames_read == 0:
        print("[ERROR] Video was opened but contained 0 frames.")
    elif count == 0:
        print("[WARNING] Frames were read but NO faces were detected.")

    return count

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if 'video' not in request.files: return jsonify({"status": "error"}), 400
    file = request.files['video']
    username = request.form['user_id']
    
    # 1. Save Raw Temp File (For AI Processing)
    temp_path = os.path.join(TEMP_DIR, f"{username}.webm")
    file.save(temp_path)
    
    # 2. Save Copy for Admin Dashboard (Foolproof Method)
    # We simply copy the raw webm file to the dataset folder.
    # We name it .mp4 so the frontend accepts it (Browsers will play it regardless of extension)
    dataset_path = os.path.join(DATASET_DIR, f"{username}.mp4")
    try:
        shutil.copy(temp_path, dataset_path)
        print(f"[UPLOAD] Saved Admin Video: {dataset_path}")
    except Exception as e:
        print(f"[ERROR] Copy failed: {e}")

    return jsonify({"status": "success"})

@app.route("/process_pending_video", methods=["POST"])
def process_pending_video():
    username = request.json.get("username")
    video_path = os.path.join(TEMP_DIR, f"{username}.webm")
    
    print(f"\n[START] Processing Request for: {username}")
    
    if not os.path.exists(video_path):
        print("[ERROR] Video path not found.")
        return jsonify({"status": "error", "message": "Video missing"}), 404
        
    count = process_video_to_dataset(video_path, username)
    
    with open(TRIGGER_FILE, "w") as f: f.write("start")
    
    # Clean up temp file (Raw WebM), but keep the MP4 in dataset for Admin logs?
    # Usually we delete temp, but if you want logs to work forever, keep the MP4.
    if os.path.exists(video_path): os.remove(video_path)
    
    print(f"[DONE] Finished processing. Count: {count}\n")
    return jsonify({"status": "success", "count": count})

@app.route("/delete_temp_video", methods=["POST"])
def delete_temp_video():
    username = request.json.get("username")
    # Delete the temp file
    video_path = os.path.join(TEMP_DIR, f"{username}.webm")
    if os.path.exists(video_path): os.remove(video_path)
    
    # Optional: Delete the Admin video too if denied?
    # admin_video = os.path.join(DATASET_DIR, f"{username}.mp4")
    # if os.path.exists(admin_video): os.remove(admin_video)
    
    return jsonify({"status": "success"})

if __name__ == "__main__":
    print("[INFO] Enrollment App Running on Port 5002 (DEBUG MODE)")
    app.run(host="0.0.0.0", port=5002, threaded=True)