<<<<<<< HEAD
#main_main -> code for running entire system- liveness detection, face recognition, and lock control
=======
#complete code for liveness detection and face recognition access control system with arduino integration and UI
>>>>>>> 40cb91ecf27a26c5de84c76a085e457390578696

import cv2 as cv
import numpy as np
import os
import pickle
import time
import threading
import sys
import gc
import serial
from flask import Flask, Response, render_template_string, jsonify
<<<<<<< HEAD
import serial
from flask import Flask, Response, render_template_string, jsonify
# from flask_cors import CORS 
=======
>>>>>>> 40cb91ecf27a26c5de84c76a085e457390578696

# ==========================================================
# CONFIGURATION
# ==========================================================
<<<<<<< HEAD
URL_LEFT  = "http://192.168.0.6:81/stream"
URL_RIGHT = "http://192.168.0.5:81/stream"

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response
# --------------------------------------

class AppState:
    STATUS = "IDLE"
=======
URL_LEFT  = "http://192.168.0.196:81/stream"
URL_RIGHT = "http://192.168.0.197:81/stream"
>>>>>>> 40cb91ecf27a26c5de84c76a085e457390578696

NPZ_PATH = 'stereo_calibration.npz'
DB_FILE = "face_encodings2.pickle"
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

# --- ARDUINO CONFIGURATION (NEW) ---
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
# -----------------------------------

# --- OPTIMIZATION SETTINGS ---
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
SCALE_FACTOR = 0.5 
PROC_W = int(DISPLAY_WIDTH * SCALE_FACTOR)
PROC_H = int(DISPLAY_HEIGHT * SCALE_FACTOR)

SKIP_FRAMES = 3 
# -----------------------------

MATCH_THRESHOLD = 0.4
LIVENESS_FRAMES_REQUIRED = 4
DEPTH_THRESHOLD_METERS = 0.025
SWAP_CAMERAS = True

# ==========================================================
# GLOBAL STATE
# ==========================================================
app = Flask(__name__)

class AppState:
    STATUS = "IDLE"
    MESSAGE = "SYSTEM INITIALIZING"
    USER_NAME = ""
    LIVENESS_PROGRESS = 0
    LAST_UNLOCK_TIME = 0  # To prevent spamming the serial port
    
state = AppState()
outputFrame = None
lock = threading.Lock()
face_db = {}
arduino = None

# ==========================================================
# ARDUINO SETUP
# ==========================================================
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for Arduino restart
    print(f"[SUCCESS] Connected to Lock Hardware on {SERIAL_PORT}")
except Exception as e:
    print(f"[WARNING] Arduino not found: {e}. Running in Simulation Mode.")

def trigger_solenoid():
    """Sends signal to Arduino to open the lock"""
    if arduino and arduino.is_open:
        try:
            arduino.write(b'O') # Send 'O' byte
            print("[HARDWARE] Solenoid Triggered")
        except Exception as e:
            print(f"[ERROR] Serial write failed: {e}")

# ==========================================================
# INITIALIZATION (UNCHANGED)
# ==========================================================
if not os.path.exists(NPZ_PATH):
    sys.exit(f"CRITICAL: {NPZ_PATH} not found.")

data = np.load(NPZ_PATH)
K_L, D_L = data['mtx_l'], data['dist_l']
K_R, D_R = data['mtx_r'], data['dist_r']
R_cal, T_cal = data['R'], data['T']

f_pixel = (K_L[0,0] + K_L[1,1]) / 2.0
B_meter = abs(T_cal[0][0])

R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K_L, D_L, K_R, D_R, (DISPLAY_WIDTH, DISPLAY_HEIGHT), R_cal, T_cal, alpha=0)
map1_L, map2_L = cv.initUndistortRectifyMap(K_L, D_L, R1, P1, (DISPLAY_WIDTH, DISPLAY_HEIGHT), cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(K_R, D_R, R2, P2, (DISPLAY_WIDTH, DISPLAY_HEIGHT), cv.CV_16SC2)

stereo = cv.StereoSGBM_create(
    minDisparity=-8, numDisparities=16*5, blockSize=5,
    P1=8 * 3 * 5**2, P2=32 * 3 * 5**2,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
)

face_detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (PROC_W, PROC_H), 0.7, 0.3, 1)
face_recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)

# ==========================================================
# HELPER FUNCTIONS (UNCHANGED)
# ==========================================================
def get_depth_at_point(disp_map, x, y):
    if x < 0 or x >= PROC_W or y < 0 or y >= PROC_H: return None
    roi = disp_map[max(0, y-2):min(PROC_H, y+3), max(0, x-2):min(PROC_W, x+3)]
    valid = roi[roi > 1.0]
    if len(valid) == 0: return None
    d_small = np.median(valid)
    d_full_equiv = d_small * (1 / SCALE_FACTOR) 
    return (f_pixel * B_meter) / d_full_equiv

def match_face_embedding(feature_vector):
    max_score = 0.0
    best_match = "Unknown"
    norm = np.linalg.norm(feature_vector)
    if norm > 0: feature_vector /= norm
    for name, db_vector in face_db.items():
        score = np.dot(feature_vector, db_vector)
        if score > max_score:
            max_score = score
            best_match = name
    if max_score >= MATCH_THRESHOLD:
        return best_match, max_score
    return "Unknown", max_score

# ==========================================================
# PROCESSING THREAD
# ==========================================================
def processing_thread():
    global outputFrame, state
    
    capL = cv.VideoCapture(URL_LEFT)
    capR = cv.VideoCapture(URL_RIGHT)
    capL.set(cv.CAP_PROP_BUFFERSIZE, 1)
    capR.set(cv.CAP_PROP_BUFFERSIZE, 1)

    liveness_counter = 0
    frame_idx = 0
    state.STATUS = "CHECKING_LIVENESS"
    
    cached_faces = None
    cached_name = "Scanning..."
    cached_color = (100, 100, 100)

    print("[INFO] System Started.")

    while True:
        if not capL.grab() or not capR.grab():
            time.sleep(0.005)
            continue
            
        _, frameL = capL.retrieve()
        _, frameR = capR.retrieve()

        if frameL is None or frameR is None: continue

        frameL = cv.resize(frameL, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        frameR = cv.resize(frameR, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if SWAP_CAMERAS:
            frameL, frameR = frameR, frameL

        rect_L = cv.remap(frameL, map1_L, map2_L, cv.INTER_LINEAR)
        
        frame_idx += 1
        run_heavy_logic = (frame_idx % SKIP_FRAMES == 0)
        vis_frame = rect_L.copy()

        if run_heavy_logic:
            rect_R = cv.remap(frameR, map1_R, map2_R, cv.INTER_LINEAR)
            small_L = cv.resize(rect_L, (PROC_W, PROC_H))
            small_R = cv.resize(rect_R, (PROC_W, PROC_H))
            
            face_detector.setInputSize((PROC_W, PROC_H))
            _, faces = face_detector.detect(small_L)
            
            cached_faces = faces

            if state.STATUS == "CHECKING_LIVENESS":
                state.MESSAGE = "ALIGN FACE"
                
                grayL = cv.cvtColor(small_L, cv.COLOR_BGR2GRAY)
                grayR = cv.cvtColor(small_R, cv.COLOR_BGR2GRAY)
                disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
                
                is_real = False
                if faces is not None:
                    for face in faces:
                        re_x, re_y = int(face[4]), int(face[5])
                        le_x, le_y = int(face[6]), int(face[7])
                        n_x, n_y   = int(face[8]), int(face[9])
                        
                        z_re = get_depth_at_point(disparity, re_x, re_y)
                        z_le = get_depth_at_point(disparity, le_x, le_y)
                        z_nose = get_depth_at_point(disparity, n_x, n_y)
                        
                        if z_re and z_le and z_nose:
                            avg_eye_z = (z_re + z_le) / 2.0
                            protrusion = avg_eye_z - z_nose
                            
                            if protrusion > DEPTH_THRESHOLD_METERS:
                                is_real = True
                                
                if is_real:
                    liveness_counter += 1
                else:
                    liveness_counter = max(0, liveness_counter - 1)
                
                state.LIVENESS_PROGRESS = int((liveness_counter / LIVENESS_FRAMES_REQUIRED) * 100)
                if liveness_counter >= LIVENESS_FRAMES_REQUIRED:
                    state.STATUS = "RECOGNIZING"
                    state.MESSAGE = "LIVENESS CONFIRMED"
                    liveness_counter = 0

            elif state.STATUS == "RECOGNIZING":
                if faces is not None:
                    face_small = faces[0]
                    face_full = face_small.copy()
                    face_full[:14] = face_full[:14] * (1 / SCALE_FACTOR)
                    
                    face_aligned = face_recognizer.alignCrop(rect_L, face_full)
                    face_feat = face_recognizer.feature(face_aligned)
                    
                    name, score = match_face_embedding(face_feat[0])
                    
                    if name != "Unknown":
                        state.STATUS = "GRANTED"
                        state.USER_NAME = name
                        state.MESSAGE = f"WELCOME, {name.upper()}"
                        cached_color = (0, 255, 0)
                        
                        # --- TRIGGER LOCK (NEW LOGIC) ---
                        current_time = time.time()
                        # Only trigger if we haven't triggered in the last 5 seconds
                        if current_time - state.LAST_UNLOCK_TIME > 5:
                            trigger_solenoid()
                            state.LAST_UNLOCK_TIME = current_time
                        # --------------------------------
                    else:
                        state.STATUS = "DENIED"
                        state.MESSAGE = "UNAUTHORIZED ACCESS"
                        cached_color = (0, 0, 255)
                    
                    cached_name = name

            if frame_idx % 100 == 0:
                gc.collect()

        # --- DRAWING OVERLAY ---
        if cached_faces is not None:
            face = cached_faces[0]
            box = (face[:4] / SCALE_FACTOR).astype(int)
            
            # Draw professional corners instead of full box
            l = 30
            x, y, w, h = box[0], box[1], box[2], box[3]
            
            if state.STATUS == "CHECKING_LIVENESS":
                color = (255, 255, 0) 
            else:
                color = cached_color

            # Top Left
            cv.line(vis_frame, (x, y), (x + l, y), color, 2)
            cv.line(vis_frame, (x, y), (x, y + l), color, 2)
            # Top Right
            cv.line(vis_frame, (x + w, y), (x + w - l, y), color, 2)
            cv.line(vis_frame, (x + w, y), (x + w, y + l), color, 2)
            # Bottom Left
            cv.line(vis_frame, (x, y + h), (x + l, y + h), color, 2)
            cv.line(vis_frame, (x, y + h), (x, y + h - l), color, 2)
            # Bottom Right
            cv.line(vis_frame, (x + w, y + h), (x + w - l, y + h), color, 2)
            cv.line(vis_frame, (x + w, y + h), (x + w, y + h - l), color, 2)

            if state.STATUS != "CHECKING_LIVENESS":
                cv.putText(vis_frame, cached_name, (x, y - 10), 
                           cv.FONT_HERSHEY_DUPLEX, 0.7, color, 1)

        if state.STATUS in ["GRANTED", "DENIED"]:
             if frame_idx % 60 == 0: 
                state.STATUS = "CHECKING_LIVENESS"
                state.LIVENESS_PROGRESS = 0
                cached_faces = None
                cached_name = ""

        with lock:
            outputFrame = vis_frame.copy()

# ==========================================================
# FLASK & NEW PROFESSIONAL UI
# ==========================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ACCESS CONTROL</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #09090b;
            --panel: rgba(24, 24, 27, 0.7);
            --primary: #06b6d4;
            --success: #10b981;
            --danger: #ef4444;
            --text-main: #f4f4f5;
            --text-mute: #a1a1aa;
        }
        
        body {
            background-color: var(--bg-dark);
            background-image: radial-gradient(circle at 50% 0%, #1e1e24 0%, var(--bg-dark) 70%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
            margin: 0;
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        .dashboard {
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 20px;
            width: 90%;
            max-width: 1200px;
            height: 85vh;
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        /* Video Section */
        .video-container {
            position: relative;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.9;
        }

        .overlay-text {
            position: absolute;
            top: 20px;
            left: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--primary);
            background: rgba(0,0,0,0.6);
            padding: 5px 10px;
            border-radius: 4px;
            border-left: 2px solid var(--primary);
        }

        /* Sidebar */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .card {
            background: var(--panel);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .header {
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 15px;
            margin-bottom: 15px;
        }
        
        h1 { margin: 0; font-size: 1.1rem; letter-spacing: 1px; color: var(--text-main); }
        h2 { margin: 0; font-size: 0.75rem; color: var(--text-mute); font-family: 'JetBrains Mono'; text-transform: uppercase; }

        .status-display {
            text-align: center;
            padding: 30px 0;
        }

        #status-icon {
            font-size: 3rem;
            margin-bottom: 10px;
            display: block;
        }

        #message {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }

        .status-granted { color: var(--success); text-shadow: 0 0 20px rgba(16, 185, 129, 0.4); }
        .status-denied { color: var(--danger); text-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
        .status-idle { color: var(--text-mute); }
        .status-scan { color: var(--primary); animation: pulse 1.5s infinite; }

        /* Progress Bar */
        .progress-wrapper {
            margin-top: auto;
        }
        
        .progress-label {
            display: flex;
            justify-content: space-between;
            font-family: 'JetBrains Mono';
            font-size: 0.7rem;
            margin-bottom: 8px;
            color: var(--text-mute);
        }

        .progress-track {
            height: 6px;
            background: #27272a;
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: var(--primary);
            width: 0%;
            transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Responsive */
        @media (max-width: 800px) {
            .dashboard { grid-template-columns: 1fr; height: auto; }
            .video-container { height: 300px; }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="video-container">
            <div class="overlay-text">LIVE FEED // STEREO_CAM_01</div>
            <img src="{{ url_for('video_feed') }}">
        </div>

        <div class="sidebar">
            <div class="card header">
                <h2>System Status</h2>
<<<<<<< HEAD
                <h1>VISION PRO</h1>
=======
                <h1>SENTRY V2.0</h1>
>>>>>>> 40cb91ecf27a26c5de84c76a085e457390578696
            </div>

            <div class="card status-display">
                <span id="status-icon">🔒</span>
                <div id="message">INITIALIZING...</div>
            </div>

            <div class="card" style="flex-grow: 1; display: flex; flex-direction: column; justify-content: flex-end;">
                <div class="progress-wrapper">
                    <div class="progress-label">
                        <span>LIVENESS CHECK</span>
                        <span id="pct">0%</span>
                    </div>
                    <div class="progress-track">
                        <div id="progress" class="progress-fill"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const msgEl = document.getElementById('message');
        const iconEl = document.getElementById('status-icon');
        const progEl = document.getElementById('progress');
        const pctEl = document.getElementById('pct');

        setInterval(() => {
            fetch('/status').then(r => r.json()).then(data => {
                msgEl.innerText = data.message;
                progEl.style.width = data.progress + "%";
                pctEl.innerText = data.progress + "%";

                msgEl.className = ""; // Reset class
                
                if(data.status === "GRANTED") {
                    msgEl.classList.add("status-granted");
                    iconEl.innerText = "🔓";
                } else if(data.status === "DENIED") {
                    msgEl.classList.add("status-denied");
                    iconEl.innerText = "🚫";
                } else if(data.status === "CHECKING_LIVENESS") {
                    msgEl.classList.add("status-scan");
                    iconEl.innerText = "👁️";
                } else {
                    msgEl.classList.add("status-idle");
                    iconEl.innerText = "🔒";
                }
            });
        }, 500);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                if outputFrame is None: 
                    time.sleep(0.01)
                    continue
                (flag, encodedImage) = cv.imencode(".jpg", outputFrame, [int(cv.IMWRITE_JPEG_QUALITY), 60])
                if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

<<<<<<< HEAD


@app.route("/status")
def status_api():
    # 1. Create the data
    data = {
        "status": state.STATUS, 
        "message": state.MESSAGE, 
        "progress": state.LIVENESS_PROGRESS
    }
    
    # 2. Turn it into a Flask Response object
    response = jsonify(data)
    
    # 3. Manually STAMP the permission header onto this specific response
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response

@app.route("/reload_db")
def reload_db_api():
    global face_db
    print("[SYSTEM] Reloading Face Database...")
    try:
        with open(DB_FILE, "rb") as f:
            with lock: # Thread safe update
                face_db = pickle.load(f)
        print(f"[SYSTEM] Reload Complete. Users: {list(face_db.keys())}")
        return jsonify({"status": "success", "message": "Database reloaded"})
    except Exception as e:
        print(f"[ERROR] Reload failed: {e}")
        return jsonify({"status": "error"}), 500
    
# Add this endpoint to handle the Manual Unlock button
@app.route('/unlock', methods=['POST'])
def manual_unlock():
    """Handles manual unlock requests from Admin Dashboard"""
    global arduino
    try:
        # Check if arduino exists and is open
        if arduino and arduino.is_open:
            arduino.write(b'O')  # Send 'O' to unlock
            print("[HARDWARE] Manual Unlock Triggered via Dashboard")
            
            # Send success response with CORS header
            response = jsonify({"status": "success", "message": "Door Unlocked"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            print("[ERROR] Arduino not connected")
            response = jsonify({"status": "error", "message": "Hardware not connected"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500
            
    except Exception as e:
        print(f"[ERROR] Manual unlock failed: {e}")
        response = jsonify({"status": "error", "message": str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
=======
@app.route("/status")
def status_api():
    return jsonify({"status": state.STATUS, "message": state.MESSAGE, "progress": state.LIVENESS_PROGRESS})
>>>>>>> 40cb91ecf27a26c5de84c76a085e457390578696

if __name__ == "__main__":
    t = threading.Thread(target=processing_thread)
    t.daemon = True
    t.start()
<<<<<<< HEAD

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
=======
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
>>>>>>> 40cb91ecf27a26c5de84c76a085e457390578696
