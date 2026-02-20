# backend/app.py
import cv2 as cv
import numpy as np
import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- ABSOLUTE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_videos")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRIGGER_FILE = os.path.join(BASE_DIR, "trigger_training.txt")
MODEL_DIR = os.path.join(BASE_DIR, "models") # Looks for models folder here

# Path to ONNX model
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def process_video_to_dataset(video_path, username):
    # Use absolute path for model
    if not os.path.exists(YUNET_PATH):
        print(f"[ERROR] Model not found at: {YUNET_PATH}")
        return 0
        
    detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.9, 0.3, 5000)
    cap = cv.VideoCapture(video_path)
    user_path = os.path.join(DATASET_DIR, username)
    os.makedirs(user_path, exist_ok=True)
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is not None:
             for face in faces:
                box = face[:4].astype(int)
                x, y, w_box, h_box = box
                if w_box > 0 and h_box > 0:
                    crop = frame[y:y+h_box, x:x+w_box]
                    if crop.size > 0:
                        cv.imwrite(os.path.join(user_path, f"{count}.jpg"), crop)
                        count += 1
                        if count >= 200: break
        if count >= 200: break
    cap.release()
    return count

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if 'video' not in request.files: return jsonify({"status": "error"}), 400
    file = request.files['video']
    username = request.form['user_id']
    save_path = os.path.join(TEMP_DIR, f"{username}.webm")
    file.save(save_path)
    return jsonify({"status": "success"})

@app.route("/process_pending_video", methods=["POST"])
def process_pending_video():
    username = request.json.get("username")
    video_path = os.path.join(TEMP_DIR, f"{username}.webm")
    
    if not os.path.exists(video_path):
        return jsonify({"status": "error", "message": "Video missing"}), 404
        
    count = process_video_to_dataset(video_path, username)
    
    # Trigger Training
    with open(TRIGGER_FILE, "w") as f:
        f.write("start")
        
    # Delete Video
    if os.path.exists(video_path):
        os.remove(video_path) # This should work now!
    
    return jsonify({"status": "success", "count": count})

@app.route("/delete_temp_video", methods=["POST"])
def delete_temp_video():
    username = request.json.get("username")
    video_path = os.path.join(TEMP_DIR, f"{username}.webm")
    if os.path.exists(video_path):
        os.remove(video_path)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, threaded=True)