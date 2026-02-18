# IT'S WORKING - Optimized Stereo Vision Face Recognition System
#liveness + face recognition integration without the arduino relay and lock control

import cv2 as cv
import numpy as np
import os
import pickle
import time
import threading
import sys
import gc
from flask import Flask, Response, render_template_string, jsonify

# ==========================================================
# CONFIGURATION
# ==========================================================
URL_LEFT  = "http://192.168.0.196:81/stream"
URL_RIGHT = "http://192.168.0.197:81/stream"

NPZ_PATH = 'stereo_calibration.npz'
DB_FILE = "face_encodings2.pickle"
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

# --- OPTIMIZATION SETTINGS ---
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
SCALE_FACTOR = 0.5  # Process at 50% scale (320x240) for speed
PROC_W = int(DISPLAY_WIDTH * SCALE_FACTOR)
PROC_H = int(DISPLAY_HEIGHT * SCALE_FACTOR)

SKIP_FRAMES = 3     # Run AI only every 3 frames
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
    MESSAGE = "Initializing..."
    USER_NAME = ""
    LIVENESS_PROGRESS = 0
    
state = AppState()
outputFrame = None
lock = threading.Lock()
face_db = {}

# ==========================================================
# INITIALIZATION
# ==========================================================
if not os.path.exists(NPZ_PATH):
    sys.exit(f"CRITICAL: {NPZ_PATH} not found.")

# Load Calibration
data = np.load(NPZ_PATH)
K_L, D_L = data['mtx_l'], data['dist_l']
K_R, D_R = data['mtx_r'], data['dist_r']
R_cal, T_cal = data['R'], data['T']

f_pixel = (K_L[0,0] + K_L[1,1]) / 2.0
B_meter = abs(T_cal[0][0])

# Rectification Maps (Computed at Full Resolution)
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K_L, D_L, K_R, D_R, (DISPLAY_WIDTH, DISPLAY_HEIGHT), R_cal, T_cal, alpha=0)
map1_L, map2_L = cv.initUndistortRectifyMap(K_L, D_L, R1, P1, (DISPLAY_WIDTH, DISPLAY_HEIGHT), cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(K_R, D_R, R2, P2, (DISPLAY_WIDTH, DISPLAY_HEIGHT), cv.CV_16SC2)

# Stereo Matcher (Tuned for Low Res)
stereo = cv.StereoSGBM_create(
    minDisparity=-8,            # Scaled down from -16
    numDisparities=16*5,        # Reduced disparities for speed
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Load AI Models (Input size set to small resolution)
face_detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (PROC_W, PROC_H), 0.7, 0.3, 1)
face_recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

# Load Database
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def get_depth_at_point(disp_map, x, y):
    """ Reads disparity from small map and calculates Depth """
    if x < 0 or x >= PROC_W or y < 0 or y >= PROC_H: return None
    
    # Grab small window
    roi = disp_map[max(0, y-2):min(PROC_H, y+3), max(0, x-2):min(PROC_W, x+3)]
    valid = roi[roi > 1.0]
    
    if len(valid) == 0: return None
    d_small = np.median(valid)
    
    # MATH FIX: Disparity at half-res is half of full-res disparity
    # Z = (f * B) / d_full  =>  Z = (f * B) / (d_small * 2)
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
# PROCESSING THREAD (OPTIMIZED)
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
    
    # Variables to hold previous results during skipped frames
    cached_faces = None
    cached_name = "Scanning..."
    cached_box = None
    cached_color = (100, 100, 100)

    print("[INFO] System Started. Processing at low-res, Displaying high-res.")

    while True:
        if not capL.grab() or not capR.grab():
            time.sleep(0.005) # Prevent CPU burn loop
            continue
            
        _, frameL = capL.retrieve()
        _, frameR = capR.retrieve()

        if frameL is None or frameR is None: continue

        # Resize Inputs
        frameL = cv.resize(frameL, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        frameR = cv.resize(frameR, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if SWAP_CAMERAS:
            frameL, frameR = frameR, frameL

        # 1. Rectification (Fast Remap)
        rect_L = cv.remap(frameL, map1_L, map2_L, cv.INTER_LINEAR)
        
        # --- FRAME SKIPPING LOGIC ---
        frame_idx += 1
        run_heavy_logic = (frame_idx % SKIP_FRAMES == 0)

        # Visual Frame (What the user sees)
        vis_frame = rect_L.copy()

        if run_heavy_logic:
            # OPTIMIZATION: Create Small Images for AI/Stereo
            rect_R = cv.remap(frameR, map1_R, map2_R, cv.INTER_LINEAR)
            
            small_L = cv.resize(rect_L, (PROC_W, PROC_H))
            small_R = cv.resize(rect_R, (PROC_W, PROC_H))
            
            # Detect Faces on SMALL image
            face_detector.setInputSize((PROC_W, PROC_H))
            _, faces = face_detector.detect(small_L)
            
            cached_faces = faces # Store for visualization

            if state.STATUS == "CHECKING_LIVENESS":
                state.MESSAGE = "ALIGN FACE FOR SCAN"
                
                # Compute Stereo on SMALL images (Fast!)
                grayL = cv.cvtColor(small_L, cv.COLOR_BGR2GRAY)
                grayR = cv.cvtColor(small_R, cv.COLOR_BGR2GRAY)
                disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
                
                is_real = False
                if faces is not None:
                    for face in faces:
                        # Landmarks on small image
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
                    state.MESSAGE = "LIVENESS CONFIRMED..."
                    liveness_counter = 0

            elif state.STATUS == "RECOGNIZING":
                if faces is not None:
                    # Scale face box back to FULL size for recognition crop
                    face_small = faces[0]
                    face_full = face_small.copy()
                    face_full[:14] = face_full[:14] * (1 / SCALE_FACTOR) # Scale coordinates up
                    
                    # Align & Crop from FULL resolution image
                    face_aligned = face_recognizer.alignCrop(rect_L, face_full)
                    face_feat = face_recognizer.feature(face_aligned)
                    
                    name, score = match_face_embedding(face_feat[0])
                    
                    if name != "Unknown":
                        state.STATUS = "GRANTED"
                        state.USER_NAME = name
                        state.MESSAGE = f"ACCESS GRANTED: {name}"
                        cached_color = (0, 255, 0)
                    else:
                        state.STATUS = "DENIED"
                        state.MESSAGE = "UNKNOWN USER"
                        cached_color = (0, 0, 255)
                    
                    cached_name = name

            # Garbage Collect occasionally to prevent RPi crash
            if frame_idx % 100 == 0:
                gc.collect()

        # --- DRAWING OVERLAY (Using Cached Data) ---
        if cached_faces is not None:
            face = cached_faces[0]
            # Convert small coords to display coords
            box = (face[:4] / SCALE_FACTOR).astype(int)
            
            if state.STATUS == "CHECKING_LIVENESS":
                color = (255, 255, 0) # Cyan
                cv.rectangle(vis_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            else:
                cv.rectangle(vis_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), cached_color, 2)
                cv.putText(vis_frame, cached_name, (box[0], box[1]-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, cached_color, 2)

        # Reset logic
        if state.STATUS in ["GRANTED", "DENIED"]:
             if frame_idx % 60 == 0: # Wait ~2 seconds (assuming 30fps)
                state.STATUS = "CHECKING_LIVENESS"
                state.LIVENESS_PROGRESS = 0
                cached_faces = None
                cached_name = ""

        with lock:
            outputFrame = vis_frame.copy()

# ==========================================================
# FLASK & UI (Same as before)
# ==========================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SecureEntry 2.0</title>
    <style>
        :root { --primary: #00f3ff; --danger: #ff0055; --bg: #0b0c15; --panel: #151725; }
        body { background: var(--bg); color: #fff; font-family: 'Segoe UI', monospace; margin: 0; display: flex; flex-direction: column; align-items: center; height: 100vh; overflow: hidden; }
        header { width: 100%; padding: 20px; background: var(--panel); border-bottom: 2px solid #333; display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; font-size: 1.2rem; letter-spacing: 2px; text-transform: uppercase; color: var(--primary); }
        .main-container { display: flex; flex-direction: column; align-items: center; margin-top: 40px; position: relative; }
        .video-wrapper { position: relative; border: 4px solid #333; padding: 5px; background: var(--panel); border-radius: 4px; box-shadow: 0 0 50px rgba(0,0,0,0.5); }
        img { display: block; max-width: 800px; width: 100%; height: auto; }
        .hud-overlay { position: absolute; top: 20px; left: 20px; right: 20px; display: flex; justify-content: space-between; pointer-events: none; }
        .status-badge { background: rgba(0,0,0,0.7); padding: 8px 16px; border-left: 3px solid var(--primary); font-weight: bold; }
        .progress-container { width: 100%; max-width: 800px; height: 6px; background: #333; margin-top: 0; position: relative; }
        .progress-bar { height: 100%; background: var(--primary); width: 0%; transition: width 0.3s ease; box-shadow: 0 0 10px var(--primary); }
        .info-panel { margin-top: 20px; text-align: center; }
        #message { font-size: 1.5rem; font-weight: bold; margin-bottom: 10px; }
        .granted { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
        .denied { color: var(--danger); text-shadow: 0 0 10px var(--danger); }
    </style>
</head>
<body>
    <header><h1>Stereo Vision Gate</h1><div>SYS.OPTIMIZED</div></header>
    <div class="main-container">
        <div class="video-wrapper">
            <img src="{{ url_for('video_feed') }}">
            <div class="hud-overlay"><div class="status-badge">CAM_01: ACTIVE</div><div class="status-badge">LOW_LATENCY: ON</div></div>
        </div>
        <div class="progress-container"><div id="progress" class="progress-bar"></div></div>
        <div class="info-panel"><div id="message">INITIALIZING...</div></div>
    </div>
    <script>
        setInterval(() => {
            fetch('/status').then(r => r.json()).then(data => {
                const msgEl = document.getElementById('message');
                const progEl = document.getElementById('progress');
                msgEl.innerText = data.message;
                progEl.style.width = data.progress + "%";
                msgEl.className = "";
                if(data.status === "GRANTED") msgEl.classList.add("granted");
                if(data.status === "DENIED") msgEl.classList.add("denied");
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
                # Lower quality slightly to save bandwidth/CPU
                (flag, encodedImage) = cv.imencode(".jpg", outputFrame, [int(cv.IMWRITE_JPEG_QUALITY), 60])
                if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.03) # Cap stream at ~30fps
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status_api():
    return jsonify({"status": state.STATUS, "message": state.MESSAGE, "progress": state.LIVENESS_PROGRESS})

if __name__ == "__main__":
    t = threading.Thread(target=processing_thread)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)