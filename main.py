import cv2 as cv
import numpy as np
import os
import pickle
import time
import threading
import sys
import serial
import json
import logging
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from collections import deque

# ==========================================================
# CONFIGURATION & PATHS
# ==========================================================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 

CALIB_DIR = os.path.join(BASE_DIR, 'stereo5_maps.npz')
DB_FILE   = os.path.join(BASE_DIR, "face_encodings.pickle")
LOG_FILE  = os.path.join(DATA_DIR, "user_logs.json") 
USER_DETAILS_FILE = os.path.join(DATA_DIR, "user_details.json") # <--- NEW: Sync Source

URL_LEFT  = "http://192.168.0.6:81/stream"
URL_RIGHT = "http://192.168.0.5:81/stream"

YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

LIVENESS_MIN, LIVENESS_MAX = 0.012, 0.080 
CONSENSUS_FRAMES, MATCH_THRESHOLD, SKIP_FRAMES = 3, 0.40, 2 

# ==========================================================
# LOGGING & SYNC UTILS
# ==========================================================
def log_event(name, action):
    """Saves event to data/user_logs.json"""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": name, "action": action
    }
    try:
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        data = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                try: data = json.load(f)
                except: data = []
        data.insert(0, entry)
        with open(LOG_FILE, "w") as f: json.dump(data[:50], f, indent=4)
        # print(f"[LOG] {action}: {name}")
    except Exception as e: print(f"[LOG ERROR] {e}")

# ==========================================================
# INITIALIZATION
# ==========================================================
try:
    mapLx = np.load(os.path.join(CALIB_DIR, "stereoMapL_x.npy"))
    mapLy = np.load(os.path.join(CALIB_DIR, "stereoMapL_y.npy"))
    mapRx = np.load(os.path.join(CALIB_DIR, "stereoMapR_x.npy"))
    mapRy = np.load(os.path.join(CALIB_DIR, "stereoMapR_y.npy"))
    Q = np.load(os.path.join(CALIB_DIR, "Q.npy"))
    f_pixel, baseline = Q[2, 3], abs(1.0 / Q[3, 2])
    # print("[SUCCESS] Stereo Maps loaded.")
except Exception as e:
    sys.exit(f"CRITICAL: Initialization failed: {e}")

arduino = None
for port in ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1']:
    try:
        arduino = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        print(f"[SUCCESS] Lock Connected on {port}")
        break 
    except: continue

# ==========================================================
# STATE & DATABASE
# ==========================================================
class AppState:
    STATUS, MESSAGE, USER_NAME, PROGRESS, LAST_EVENT = "IDLE", "WAITING", "", 0, 0

state = AppState()
outputFrame, lock = None, threading.Lock()
face_db, last_db_check = {}, 0
liveness_history = deque(maxlen=CONSENSUS_FRAMES)

def load_database():
    global face_db, last_db_check
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f: face_db = pickle.load(f)
        last_db_check = os.path.getmtime(DB_FILE)
        # print(f"♻️ Database Reloaded: {len(face_db)} users.")

load_database()

face_detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 240), 0.7, 0.3, 1)
face_recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")
stereo = cv.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

# ==========================================================
# CORE UTILS
# ==========================================================
def trigger_lock():
    if arduino:
        try:
            arduino.write(b'O')
            return True
        except: return False
    return False

def get_depth(disp, x, y):
    h, w = disp.shape
    if x < 5 or x >= w-5 or y < 5 or y >= h-5: return None
    roi = disp[y-2:y+3, x-2:x+3]
    valid = roi[roi > 16.0]
    return (f_pixel * baseline) / (np.median(valid) / 16.0) if len(valid) > 0 else None

# --- NEW SYNC FUNCTION ---
def sync_user_database():
    """Checks user_details.json and removes deleted users from Pickle file"""
    global face_db
    try:
        if not os.path.exists(USER_DETAILS_FILE) or not os.path.exists(DB_FILE): return

        # 1. Get valid users from JSON (Source of Truth)
        with open(USER_DETAILS_FILE, "r") as f:
            user_list = json.load(f)
            # Assuming 'name' is the key. Adjust if you use 'username' or 'id'.
            valid_names = [u.get('name') for u in user_list if u.get('name')]

        # 2. Check for ghosts in Pickle
        current_db_names = list(face_db.keys())
        deleted_count = 0
        
        for db_name in current_db_names:
            if db_name not in valid_names:
                del face_db[db_name] # Remove from memory
                deleted_count += 1
        
        # 3. If we deleted anyone, save the Pickle file back to disk
        if deleted_count > 0:
            print(f"[SYNC] Removing {deleted_count} deleted users from Database...")
            with open(DB_FILE, "wb") as f:
                pickle.dump(face_db, f)
            # print("[SYNC] Database Updated Successfully.")
            
    except Exception as e:
        print(f"[SYNC ERROR] {e}")
# -------------------------

# ==========================================================
# MAIN PROCESSING
# ==========================================================
def processing_thread():
    global outputFrame, state, face_db, last_db_check
    capL, capR = cv.VideoCapture(URL_LEFT), cv.VideoCapture(URL_RIGHT)
    frame_count = 0
    
    while True:
        # 1. Hot Reload & SYNC (Every 100 frames)
        if frame_count % 100 == 0:
            sync_user_database() # <--- SYNC CHECK
            if os.path.exists(DB_FILE):
                if os.path.getmtime(DB_FILE) > last_db_check: load_database()

        if not capL.grab() or not capR.grab(): 
            time.sleep(0.01); continue
        _, frameL = capL.retrieve(); _, frameR = capR.retrieve()
        if frameL is None or frameR is None: continue

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0: continue

        # Resize to standard calibration size (640x480)
        frameL = cv.resize(frameL, (640, 480))
        frameR = cv.resize(frameR, (640, 480))

        # Create Rectified frames for System Logic
        rectL = cv.remap(frameL, mapLx, mapLy, cv.INTER_LINEAR)
        rectR = cv.remap(frameR, mapRx, mapRy, cv.INTER_LINEAR)
        
        # Use Raw frame for Visuals (No Black Borders)
        vis_frame = frameL.copy() 

        # --- DOUBLE DETECTION LOGIC ---
        # A. Detect on VISUAL Frame (Raw) for Box Drawing
        face_detector.setInputSize((vis_frame.shape[1], vis_frame.shape[0]))
        _, faces_vis = face_detector.detect(vis_frame)

        # B. Detect on SYSTEM Frame (Rectified) for Liveness/Recog
        face_detector.setInputSize((rectL.shape[1], rectL.shape[0]))
        _, faces_sys = face_detector.detect(rectL)

        # Logic Processing (Uses System/Rectified Data)
        if faces_sys is not None:
            face_sys = faces_sys[0]
            
            if state.STATUS in ["IDLE", "CHECKING_LIVENESS"]:
                state.STATUS, state.MESSAGE = "CHECKING_LIVENESS", "VERIFYING LIVENESS..."
                disp = cv.medianBlur(stereo.compute(cv.cvtColor(rectL, cv.COLOR_BGR2GRAY), cv.cvtColor(rectR, cv.COLOR_BGR2GRAY)).astype(np.float32), 3)
                z_re, z_le, z_no = get_depth(disp, int(face_sys[4]), int(face_sys[5])), get_depth(disp, int(face_sys[6]), int(face_sys[7])), get_depth(disp, int(face_sys[8]), int(face_sys[9]))
                is_real = LIVENESS_MIN < (((z_re + z_le)/2.0) - z_no) < LIVENESS_MAX if (z_re and z_le and z_no) else False
                
                liveness_history.append(is_real)
                state.PROGRESS = int((sum(liveness_history)/CONSENSUS_FRAMES)*100)
                if len(liveness_history) == CONSENSUS_FRAMES and all(liveness_history): state.STATUS = "RECOGNIZING"
            
            elif state.STATUS == "RECOGNIZING":
                state.MESSAGE = "SCANNING IDENTITY..."
                feat = face_recognizer.feature(face_recognizer.alignCrop(rectL, face_sys))[0]
                feat /= np.linalg.norm(feat)
                best_match, best_score = "Unknown", 0
                for name, db_feat in face_db.items():
                    score = np.dot(feat, db_feat)
                    if score > best_score: best_score, best_match = score, name
                
                if best_score >= MATCH_THRESHOLD:
                    state.STATUS, state.MESSAGE, state.USER_NAME = "GRANTED", f"WELCOME {best_match.upper()}", best_match
                    log_event(best_match, "ACCESS GRANTED")
                    trigger_lock()
                else:
                    state.STATUS, state.MESSAGE, state.USER_NAME = "DENIED", "ACCESS DENIED", "Unauthorized"
                    if time.time() - state.LAST_EVENT > 4: log_event("Unknown", "DENIED")
                state.LAST_EVENT = time.time()

        # Visuals Drawing (Uses Visual/Raw Data - Perfect Box Alignment)
        if faces_vis is not None:
            face_vis = faces_vis[0]
            box, color = face_vis[:4].astype(int), (255, 165, 0)
            if state.STATUS == "GRANTED": color = (0, 255, 0)
            elif state.STATUS == "DENIED": color = (0, 0, 255)
            cv.rectangle(vis_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            if state.USER_NAME: cv.putText(vis_frame, state.USER_NAME.upper(), (box[0], box[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Reset if no faces found in either
        if faces_vis is None and faces_sys is None:
            if state.STATUS not in ["GRANTED", "DENIED"]:
                state.STATUS, state.MESSAGE, state.USER_NAME, state.PROGRESS = "IDLE", "WAITING FOR FACE", "", 0
                liveness_history.clear()

        if state.STATUS in ["GRANTED", "DENIED"] and (time.time() - state.LAST_EVENT > 4):
            state.STATUS, state.USER_NAME = "IDLE", ""

        with lock: outputFrame = vis_frame.copy()
        time.sleep(0.01)

# ==========================================================
# FLASK SERVER
# ==========================================================
app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/")
def index():
    return jsonify({"status": "running", "message": "Backend is active. Use the React Dashboard."})

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with lock:
                if outputFrame is None: 
                    time.sleep(0.05); continue
                _, enc = cv.imencode(".jpg", outputFrame, [int(cv.IMWRITE_JPEG_QUALITY), 75])
            yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status_api(): return jsonify({"status": state.STATUS, "message": state.MESSAGE, "progress": state.PROGRESS})

@app.route("/logs")
def get_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f: return jsonify(json.load(f))
    return jsonify([])

@app.route('/unlock', methods=['POST'])
def manual_unlock():
    log_event("Admin", "MANUAL UNLOCK")
    if trigger_lock(): return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500

if __name__ == "__main__":
    threading.Thread(target=processing_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)