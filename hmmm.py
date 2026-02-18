import cv2 as cv
import numpy as np
import os
import pickle
import time
import threading
import sys
import gc
import serial
import json
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request, redirect, url_for, session
from functools import wraps

# ==========================================================
# CONFIGURATION
# ==========================================================
URL_LEFT  = "http://192.168.0.196:81/stream"
URL_RIGHT = "http://192.168.0.197:81/stream"

NPZ_PATH = 'stereo_calibration.npz'
DB_FILE = "face_encodings2.pickle"
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

# Optimization
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
SCALE_FACTOR = 0.5 
PROC_W = int(DISPLAY_WIDTH * SCALE_FACTOR)
PROC_H = int(DISPLAY_HEIGHT * SCALE_FACTOR)
SKIP_FRAMES = 3 

# Security Logic
MATCH_THRESHOLD = 0.4
LIVENESS_FRAMES_REQUIRED = 4
DEPTH_THRESHOLD_METERS = 0.025
SWAP_CAMERAS = True

# Admin Credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123" # CHANGE THIS

# ==========================================================
# GLOBAL STATE & STORAGE
# ==========================================================
app = Flask(__name__)
app.secret_key = "super_secret_sentry_key" # Required for sessions

class AppState:
    STATUS = "IDLE"
    MESSAGE = "SYSTEM INITIALIZING"
    USER_NAME = ""
    LIVENESS_PROGRESS = 0
    LAST_UNLOCK_TIME = 0
    CURRENT_FRAME_EMBEDDING = None # Used for enrollment capture

state = AppState()
outputFrame = None
lock = threading.Lock()
arduino = None

# Data Stores
face_db = {}         # Active Users: {name: embedding}
pending_users = []   # Requests: [{'id': uuid, 'name': name, 'embedding': vector, 'time': str}]
access_logs = []     # Logs: [{'time': str, 'event': str, 'status': str}]

# ==========================================================
# HARDWARE & CV SETUP
# ==========================================================
# (Arduino and CV Initialization kept mostly identical to your original code)
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[SUCCESS] Connected to Lock Hardware on {SERIAL_PORT}")
except Exception:
    print(f"[WARNING] Arduino not found. Running in Simulation Mode.")

def trigger_solenoid():
    if arduino and arduino.is_open:
        try:
            arduino.write(b'O')
        except Exception as e:
            print(f"[ERROR] Serial write failed: {e}")

if not os.path.exists(NPZ_PATH): sys.exit(f"CRITICAL: {NPZ_PATH} not found.")

# Load Calibration
data = np.load(NPZ_PATH)
K_L, D_L = data['mtx_l'], data['dist_l']
K_R, D_R = data['mtx_r'], data['dist_r']
R_cal, T_cal = data['R'], data['T']
f_pixel = (K_L[0,0] + K_L[1,1]) / 2.0
B_meter = abs(T_cal[0][0])

R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K_L, D_L, K_R, D_R, (DISPLAY_WIDTH, DISPLAY_HEIGHT), R_cal, T_cal, alpha=0)
map1_L, map2_L = cv.initUndistortRectifyMap(K_L, D_L, R1, P1, (DISPLAY_WIDTH, DISPLAY_HEIGHT), cv.CV_16SC2)
map1_R, map2_R = cv.initUndistortRectifyMap(K_R, D_R, R2, P2, (DISPLAY_WIDTH, DISPLAY_HEIGHT), cv.CV_16SC2)

stereo = cv.StereoSGBM_create(minDisparity=-8, numDisparities=16*5, blockSize=5, P1=8*3*5**2, P2=32*3*5**2, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
face_detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (PROC_W, PROC_H), 0.7, 0.3, 1)
face_recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)

# ==========================================================
# CORE PROCESSING LOOP
# ==========================================================
def get_depth_at_point(disp_map, x, y):
    if x < 0 or x >= PROC_W or y < 0 or y >= PROC_H: return None
    roi = disp_map[max(0, y-2):min(PROC_H, y+3), max(0, x-2):min(PROC_W, x+3)]
    valid = roi[roi > 1.0]
    if len(valid) == 0: return None
    d_small = np.median(valid)
    return (f_pixel * B_meter) / (d_small * (1 / SCALE_FACTOR))

def match_face_embedding(feature_vector):
    max_score = 0.0
    best_match = "Unknown"
    norm = np.linalg.norm(feature_vector)
    if norm > 0: feature_vector /= norm
    
    # Store current embedding for potential enrollment
    state.CURRENT_FRAME_EMBEDDING = feature_vector.copy()

    for name, db_vector in face_db.items():
        score = np.dot(feature_vector, db_vector)
        if score > max_score:
            max_score = score
            best_match = name
    
    if max_score >= MATCH_THRESHOLD:
        return best_match, max_score
    return "Unknown", max_score

def log_event(event, status):
    """Adds an entry to the access log."""
    # Prevent spamming logs (simple debounce)
    if access_logs and access_logs[0]['event'] == event and \
       (datetime.now() - datetime.strptime(access_logs[0]['time'], "%Y-%m-%d %H:%M:%S")).seconds < 5:
        return

    entry = {
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'event': event,
        'status': status
    }
    access_logs.insert(0, entry) # Add to top
    if len(access_logs) > 50: access_logs.pop() # Keep last 50

def processing_thread():
    global outputFrame, state
    capL = cv.VideoCapture(URL_LEFT)
    capR = cv.VideoCapture(URL_RIGHT)
    liveness_counter = 0
    frame_idx = 0
    state.STATUS = "CHECKING_LIVENESS"
    cached_faces, cached_name, cached_color = None, "Scanning...", (100, 100, 100)

    while True:
        if not capL.grab() or not capR.grab():
            time.sleep(0.005)
            continue
        _, frameL = capL.retrieve()
        _, frameR = capR.retrieve()
        if frameL is None: continue

        frameL = cv.resize(frameL, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        frameR = cv.resize(frameR, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        if SWAP_CAMERAS: frameL, frameR = frameR, frameL
        
        rect_L = cv.remap(frameL, map1_L, map2_L, cv.INTER_LINEAR)
        frame_idx += 1
        vis_frame = rect_L.copy()

        if frame_idx % SKIP_FRAMES == 0:
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
                    face = faces[0]
                    # Logic to detect protrusion (simplified from your code)
                    z_re = get_depth_at_point(disparity, int(face[4]), int(face[5]))
                    z_le = get_depth_at_point(disparity, int(face[6]), int(face[7]))
                    z_nose = get_depth_at_point(disparity, int(face[8]), int(face[9]))
                    
                    if z_re and z_le and z_nose:
                        protrusion = ((z_re + z_le) / 2.0) - z_nose
                        if protrusion > DEPTH_THRESHOLD_METERS: is_real = True

                if is_real: liveness_counter += 1
                else: liveness_counter = max(0, liveness_counter - 1)
                
                state.LIVENESS_PROGRESS = int((liveness_counter / LIVENESS_FRAMES_REQUIRED) * 100)
                if liveness_counter >= LIVENESS_FRAMES_REQUIRED:
                    state.STATUS = "RECOGNIZING"
                    liveness_counter = 0

            elif state.STATUS == "RECOGNIZING":
                if faces is not None:
                    face_full = faces[0].copy()
                    face_full[:14] *= (1 / SCALE_FACTOR)
                    face_aligned = face_recognizer.alignCrop(rect_L, face_full)
                    face_feat = face_recognizer.feature(face_aligned)
                    
                    name, score = match_face_embedding(face_feat[0])
                    
                    if name != "Unknown":
                        state.STATUS = "GRANTED"
                        state.MESSAGE = f"WELCOME {name}"
                        cached_color = (0, 255, 0)
                        log_event(f"User {name}", "GRANTED")
                        
                        curr = time.time()
                        if curr - state.LAST_UNLOCK_TIME > 5:
                            trigger_solenoid()
                            state.LAST_UNLOCK_TIME = curr
                    else:
                        state.STATUS = "DENIED"
                        state.MESSAGE = "UNAUTHORIZED"
                        cached_color = (0, 0, 255)
                        log_event("Unknown User", "DENIED")
                    cached_name = name

        # Reset status periodically
        if state.STATUS in ["GRANTED", "DENIED"] and frame_idx % 60 == 0:
            state.STATUS = "CHECKING_LIVENESS"
            state.LIVENESS_PROGRESS = 0

        # Draw Overlay
        if cached_faces is not None:
            face = cached_faces[0]
            box = (face[:4] / SCALE_FACTOR).astype(int)
            color = (255, 255, 0) if state.STATUS == "CHECKING_LIVENESS" else cached_color
            cv.rectangle(vis_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            if state.STATUS != "CHECKING_LIVENESS":
                cv.putText(vis_frame, cached_name, (box[0], box[1]-10), cv.FONT_HERSHEY_DUPLEX, 0.7, color, 1)

        with lock: outputFrame = vis_frame.copy()

# ==========================================================
# FLASK WEB INTERFACE
# ==========================================================

# --- CSS STYLES (Shared across templates) ---
COMMON_CSS = """
<style>
    :root { --bg: #09090b; --panel: #18181b; --primary: #06b6d4; --text: #f4f4f5; }
    body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
    .container { text-align: center; width: 100%; max-width: 400px; padding: 20px; }
    .btn { display: block; width: 100%; padding: 15px; margin: 10px 0; background: var(--panel); border: 1px solid #333; color: var(--text); font-weight: bold; cursor: pointer; border-radius: 8px; transition: 0.2s; text-decoration: none; }
    .btn:hover { background: var(--primary); color: #000; }
    .btn-danger { border-color: #ef4444; color: #ef4444; }
    .btn-danger:hover { background: #ef4444; color: white; }
    .btn-success { border-color: #10b981; color: #10b981; }
    .btn-success:hover { background: #10b981; color: white; }
    input { width: 90%; padding: 15px; margin: 10px 0; background: #000; border: 1px solid #333; color: white; border-radius: 8px; }
    h1 { letter-spacing: 2px; margin-bottom: 30px; font-family: 'JetBrains Mono', monospace; }
    .flash { color: #ef4444; margin-bottom: 10px; }
    
    /* Dashboard Specific */
    .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; max-width: 1200px; width: 95%; height: 90vh; text-align: left; }
    .panel { background: var(--panel); border: 1px solid #333; border-radius: 12px; padding: 20px; overflow-y: auto; }
    .video-box img { width: 100%; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { padding: 10px; border-bottom: 1px solid #333; font-size: 0.9rem; }
    th { color: #a1a1aa; text-align: left; }
    .tick { color: #10b981; cursor: pointer; font-size: 1.2rem; margin-right: 10px; }
    .cross { color: #ef4444; cursor: pointer; font-size: 1.2rem; }
</style>
"""

# --- TEMPLATES ---
TEMPLATES = {
    "HOME": """
    <!DOCTYPE html><html><head><title>Sentry Home</title>""" + COMMON_CSS + """</head><body>
        <div class="container">
            <h1>SENTRY V2.0</h1>
            <a href="/user_view" class="btn">User View</a>
            <a href="/enroll" class="btn">New User</a>
            <a href="/login" class="btn" style="border-color: var(--primary)">Admin</a>
        </div>
    </body></html>
    """,
    
    "LOGIN": """
    <!DOCTYPE html><html><head><title>Admin Login</title>""" + COMMON_CSS + """</head><body>
        <div class="container">
            <h1>ADMIN ACCESS</h1>
            {% if error %}<div class="flash">{{ error }}</div>{% endif %}
            <form method="POST">
                <input type="text" name="username" placeholder="Username" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit" class="btn">Sign In</button>
            </form>
            <a href="/" class="btn btn-danger">Back</a>
        </div>
    </body></html>
    """,

    "ENROLL": """
    <!DOCTYPE html><html><head><title>New User</title>""" + COMMON_CSS + """</head><body>
        <div class="container">
            <h1>NEW USER REQUEST</h1>
            <p style="color: #a1a1aa; margin-bottom: 20px;">Look at the camera. Ensure your face is detected.</p>
            <form method="POST">
                <input type="text" name="fullname" placeholder="Enter Full Name" required>
                <button type="submit" class="btn btn-success">Submit Request</button>
            </form>
            <a href="/" class="btn btn-danger">Cancel</a>
        </div>
    </body></html>
    """,

    "ADMIN": """
    <!DOCTYPE html><html><head><title>Admin Dashboard</title>""" + COMMON_CSS + """</head><body>
        <div class="dashboard-grid">
            <div style="display: flex; flex-direction: column; gap: 20px;">
                <div class="panel video-box">
                    <h3 style="margin-top:0">LIVE FEED</h3>
                    <img src="{{ url_for('video_feed') }}">
                </div>
                <div class="panel" style="flex-grow: 1;">
                    <h3 style="margin-top:0">ACCESS LOGS</h3>
                    <table>
                        <thead><tr><th>Time</th><th>Event</th><th>Status</th></tr></thead>
                        <tbody>
                            {% for log in logs %}
                            <tr>
                                <td style="color: #a1a1aa">{{ log.time }}</td>
                                <td>{{ log.event }}</td>
                                <td style="color: {{ '#10b981' if log.status == 'GRANTED' else '#ef4444' }}">{{ log.status }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="panel">
                <h3 style="margin-top:0">ACCESS REQUESTS</h3>
                {% if not requests %}
                    <p style="color: #a1a1aa; font-style: italic;">No pending requests.</p>
                {% else %}
                    <table>
                        <thead><tr><th>Name</th><th>Action</th></tr></thead>
                        <tbody>
                            {% for req in requests %}
                            <tr>
                                <td>{{ req.name }}</td>
                                <td>
                                    <a href="/approve/{{ loop.index0 }}" class="tick">✔</a>
                                    <a href="/deny/{{ loop.index0 }}" class="cross">✘</a>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                {% endif %}
                
                <div style="margin-top: 50px;">
                    <a href="/" class="btn">Home Page</a>
                    <a href="/logout" class="btn btn-danger">Sign Out</a>
                </div>
            </div>
        </div>
    </body></html>
    """
}

# --- ROUTES ---

@app.route("/")
def index():
    return render_template_string(TEMPLATES["HOME"])

@app.route("/user_view")
def user_view():
    # Reuses the professional UI you already had, kept simple here
    return render_template_string(HTML_TEMPLATE_USER) # Defined at bottom

@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        name = request.form.get("fullname")
        if state.CURRENT_FRAME_EMBEDDING is not None:
            # Save request to pending list
            req = {
                'name': name,
                'embedding': state.CURRENT_FRAME_EMBEDDING,
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            pending_users.append(req)
            return f"""<body style='background:#09090b;color:white;text-align:center;font-family:sans-serif;padding-top:100px;'>
                       <h1>Request Sent</h1><p>Admin must approve '{name}'</p>
                       <a href='/' style='color:#06b6d4'>Return Home</a></body>"""
        else:
            return f"""<body style='background:#09090b;color:white;text-align:center;font-family:sans-serif;padding-top:100px;'>
                       <h1 style='color:red'>Face Not Detected</h1><p>Please look at the camera and try again.</p>
                       <a href='/enroll' style='color:#06b6d4'>Try Again</a></body>"""
    return render_template_string(TEMPLATES["ENROLL"])

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid Credentials"
    return render_template_string(TEMPLATES["LOGIN"], error=error)

@app.route("/admin")
def admin_dashboard():
    if not session.get('admin'):
        return redirect(url_for('login'))
    return render_template_string(TEMPLATES["ADMIN"], logs=access_logs, requests=pending_users)

@app.route("/approve/<int:req_id>")
def approve_user(req_id):
    if not session.get('admin'): return redirect(url_for('login'))
    if 0 <= req_id < len(pending_users):
        user = pending_users.pop(req_id)
        face_db[user['name']] = user['embedding']
        # Save to disk
        with open(DB_FILE, "wb") as f:
            pickle.dump(face_db, f)
        log_event(f"Admin approved {user['name']}", "SYSTEM")
    return redirect(url_for('admin_dashboard'))

@app.route("/deny/<int:req_id>")
def deny_user(req_id):
    if not session.get('admin'): return redirect(url_for('login'))
    if 0 <= req_id < len(pending_users):
        pending_users.pop(req_id)
    return redirect(url_for('admin_dashboard'))

@app.route("/logout")
def logout():
    session.pop('admin', None)
    return redirect(url_for('index'))

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

# User View Template (Your original nice UI)
HTML_TEMPLATE_USER = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTRY USER VIEW</title>
    <style>
        body { background: #09090b; color: #f4f4f5; font-family: 'Inter', sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .video-container { border: 2px solid #333; border-radius: 12px; overflow: hidden; margin-bottom: 20px; }
        img { display: block; max-width: 100%; }
        .btn { padding: 10px 20px; background: #333; color: white; text-decoration: none; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="video-container">
        <img src="/video_feed">
    </div>
    <h1 style="font-family: monospace;">SECURE GATEWAY</h1>
    <a href="/" class="btn">Return Home</a>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=processing_thread)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)