#app.py -> Camera App with Face Capture and Training Trigger

###################################################################################################################################



import cv2 as cv
import numpy as np
import time
import os
import threading
import queue
import pickle
from flask import Flask, render_template, Response, jsonify, request

# --- CONFIGURATION ---
ESP32_URL = "http://192.168.0.196:81/stream"
SAVE_DIR = "dataset"
DB_FILE = "face_encodings2.pickle"
MAX_IMAGES = 200
TIME_LIMIT_SEC = 60
CONFIDENCE_THRESHOLD = 0.9
BLUR_THRESHOLD = 35 
FRAME_SKIP_RATE = 3 
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

app = Flask(__name__)

outputFrame = None
lock = threading.Lock()

is_capturing = False
capture_user_id = "unknown"
capture_start_time = 0
images_captured = 0
capture_message = "Ready"

# Queue for saving images
save_queue = queue.Queue()

def image_saver_worker():
    while True:
        file_path, image_data = save_queue.get()
        if file_path is None: break
        try:
            cv.imwrite(file_path, image_data)
        except: pass
        save_queue.task_done()

t_saver = threading.Thread(target=image_saver_worker, daemon=True)
t_saver.start()

def initialize_models(width=320, height=320):
    detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (width, height), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")
    return detector, recognizer

def camera_processing_thread():
    global outputFrame, is_capturing, images_captured, capture_message, capture_start_time

    cap = None
    detector = None
    recognizer = None
    frame_count = 0

    print(f"[APP] Connecting to {ESP32_URL}")
    cap = cv.VideoCapture(ESP32_URL, cv.CAP_FFMPEG)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        detector, recognizer = initialize_models(320, 240)
    else:
        print("[APP] Camera connection failed (Will keep trying)")

    while True:
        try:
            if not cap.isOpened():
                time.sleep(1.0)
                cap = cv.VideoCapture(ESP32_URL, cv.CAP_FFMPEG)
                if cap.isOpened():
                    detector, recognizer = initialize_models(320, 240)
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue
            
            frame_count += 1
            if frame_count % FRAME_SKIP_RATE != 0:
                with lock: outputFrame = frame.copy()
                continue

            h, w = frame.shape[:2]
            
            # AI Processing
            img_small = cv.resize(frame, (320, 240))
            detector.setInputSize((320, 240))
            faces = detector.detect(img_small)
            
            scale_x = w / 320
            scale_y = h / 240
            
            if faces[1] is not None:
                sorted_faces = sorted(faces[1], key=lambda x: x[2]*x[3], reverse=True)
                face = sorted_faces[0]
                confidence = face[-1]
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    box = list(map(int, face[:4]))
                    box[0] = int(box[0] * scale_x)
                    box[1] = int(box[1] * scale_y)
                    box[2] = int(box[2] * scale_x)
                    box[3] = int(box[3] * scale_y)
                    
                    cv.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

                    if is_capturing:
                        face_scaled = face.copy()
                        face_scaled[:4] = box
                        for k in range(4, 14):
                            face_scaled[k] *= (scale_x if k % 2 == 0 else scale_y)
                            
                        face_aligned = recognizer.alignCrop(frame, face_scaled)
                        gray = cv.cvtColor(face_aligned, cv.COLOR_BGR2GRAY)
                        score = cv.Laplacian(gray, cv.CV_64F).var()
                        
                        elapsed = time.time() - capture_start_time
                        if elapsed > TIME_LIMIT_SEC or images_captured >= MAX_IMAGES:
                            is_capturing = False
                            capture_message = "Done. Click Train."
                        elif score > BLUR_THRESHOLD:
                            user_path = os.path.join(SAVE_DIR, capture_user_id)
                            os.makedirs(user_path, exist_ok=True)
                            fname = os.path.join(user_path, f"{capture_user_id}_{images_captured:04d}.jpg")
                            save_queue.put((fname, face_aligned))
                            images_captured += 1
                            capture_message = f"Capturing... {int(TIME_LIMIT_SEC - elapsed)}s"
                        else:
                            cv.putText(frame, "Blurry", (box[0], box[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            with lock: outputFrame = frame.copy()

        except Exception:
            pass

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv.imencode(".jpg", outputFrame, [int(cv.IMWRITE_JPEG_QUALITY), 70])
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_capture", methods=["POST"])
def start_capture():
    global is_capturing, capture_user_id, images_captured, capture_start_time, capture_message
    data = request.json
    user_id = data.get("user_id", "unknown").strip().replace(" ", "_")
    if not user_id: return jsonify({"status": "error"})
    
    capture_user_id = user_id
    images_captured = 0
    capture_start_time = time.time()
    is_capturing = True
    capture_message = "Starting..."
    return jsonify({"status": "success"})

@app.route("/train_model", methods=["POST"])
def train_model():
    # --- HERE IS THE MAGIC FIX ---
    # Instead of training, we write a trigger file and exit.
    # The Supervisor script will see this file.
    with open("trigger_training.txt", "w") as f:
        f.write("start")
    
    return jsonify({
        "status": "success", 
        "message": "System restarting to Train... Please wait 15 seconds."
    })

@app.route("/get_status")
def get_status():
    global images_captured, is_capturing, capture_message
    user_count = 0
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                user_count = len(pickle.load(f))
        except: pass

    return jsonify({
        "count": images_captured,
        "max": MAX_IMAGES,
        "capturing": is_capturing,
        "message": capture_message,
        "total_users": user_count
    })

if __name__ == "__main__":
    t = threading.Thread(target=camera_processing_thread)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)