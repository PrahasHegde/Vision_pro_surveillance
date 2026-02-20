# main1.py -> Main Application: Real-Time Face Recognition from ESP32-CAM Stream
#-------------------------------------------------------------------------------


# IMPORTS
import cv2 as cv
import numpy as np
import os
import pickle
import time
import threading
import gc
from flask import Flask, Response, render_template_string


#CONFIGURATION
ESP32_URL = "http://192.168.0.196:81/stream"
DB_FILE = "face_encodings2.pickle"
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"
MATCH_THRESHOLD = 0.4  
CONFIDENCE_THRESHOLD = 0.85
FRAME_SKIP_RATE = 5 


app = Flask(__name__)

# Global State
outputFrame = None
lock = threading.Lock()
face_db = {}



def load_database():
    global face_db
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                face_db = pickle.load(f)
            print(f"[INFO] Database loaded. Tracking {len(face_db)} users.")
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
            face_db = {}
    else:
        print("[WARNING] No database found.")
        face_db = {}

def match_face(feature_vector):
    max_score = 0.0
    best_match = "Unknown"

    for name, db_vector in face_db.items():
        score = np.dot(feature_vector, db_vector)
        if score > max_score:
            max_score = score
            best_match = name

    if max_score >= MATCH_THRESHOLD:
        return best_match, max_score
    else:
        return "Unknown", max_score

def recognition_thread():
    global outputFrame, lock

    detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 240), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

    print(f"[INFO] Connecting to stream: {ESP32_URL}")
    cap = cv.VideoCapture(ESP32_URL, cv.CAP_FFMPEG)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    
    frame_count = 0
    last_results = [] 

    while True:
        try:
            ret, frame = cap.read()
            
            if not ret:
                cap.release()
                time.sleep(2.0)
                cap = cv.VideoCapture(ESP32_URL, cv.CAP_FFMPEG)
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                continue

            h, w = frame.shape[:2]
            time.sleep(0.01) 
            
            if frame_count % 100 == 0:
                gc.collect()

            if frame_count % FRAME_SKIP_RATE == 0:
                last_results = [] 
                
                img_small = cv.resize(frame, (320, 240))
                detector.setInputSize((320, 240))
                faces = detector.detect(img_small)

                scale_x = w / 320
                scale_y = h / 240

                if faces[1] is not None:
                    for face in faces[1]:
                        confidence = face[-1]
                        if confidence >= CONFIDENCE_THRESHOLD:
                            box_small = face[:4]
                            box = list(map(int, [
                                box_small[0] * scale_x,
                                box_small[1] * scale_y,
                                box_small[2] * scale_x,
                                box_small[3] * scale_y
                            ]))
                            
                            face_scaled = face.copy()
                            face_scaled[:4] = box
                            for k in range(4, 14):
                                face_scaled[k] *= (scale_x if k % 2 == 0 else scale_y)

                            # --- RECOGNITION BLOCK ---
                            face_aligned = recognizer.alignCrop(frame, face_scaled)
                            face_feature = recognizer.feature(face_aligned)
                            
                            # >>> THE FIX: NORMALIZE VECTOR <<<
                            query_feat = face_feature[0]
                            norm = np.linalg.norm(query_feat)
                            if norm > 0:
                                query_feat = query_feat / norm
                            
                            name, score = match_face(query_feat)
                            # -------------------------------

                            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
                            last_results.append((box, name, score, color))
                            
                            del face_aligned, face_feature

            frame_count += 1

            for result in last_results:
                (box, name, score, color) = result
                cv.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                
                # Create Label
                label = f"{name} ({int(score*100)}%)"
                
                cv.rectangle(frame, (box[0], box[1]-25), (box[0]+box[2], box[1]), color, -1)
                cv.putText(frame, label, (box[0]+5, box[1]-5), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            with lock:
                outputFrame = frame.copy()

        except Exception as e:
            print(f"[CRITICAL ERROR] Loop crashed: {e}")
            time.sleep(1)

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv.imencode(".jpg", outputFrame, [int(cv.IMWRITE_JPEG_QUALITY), 60])
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)


# SIMPLE UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Live Recognition</title>
    <style>
        body { background: #1a1a1a; color: white; font-family: sans-serif; text-align: center; }
        img { border: 4px solid #333; border-radius: 8px; max-width: 100%; margin-top: 20px; }
        .info { color: #888; font-size: 0.9em; margin-bottom: 10px;}
    </style>
</head>
<body>
    <h1>Security Feed: Live Recognition</h1>
    <div class="info">Stability Mode: Active</div>
    <img src="{{ url_for('video_feed') }}">
</body>
</html>
"""


# ROUTES
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    load_database()
    t = threading.Thread(target=recognition_thread)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
