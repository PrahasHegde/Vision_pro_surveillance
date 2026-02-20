# z.py -> Final code for Vision Pro Access Control System (version 1) == not part of main code
#--------------------------------------------------------------------------------------------



# IMPORTS
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
from collections import deque



# CONFIGURATION
URL_LEFT  = "http://192.168.0.6:81/stream"
URL_RIGHT = "http://192.168.0.5:81/stream"
CALIB_DIR = 'stereo2_maps.npz' 
DB_FILE = "face_encodings2.pickle"
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"


# TUNING PARAMETERS
LIVENESS_MIN = 0.012  # 1.2cm minimum protrusion
LIVENESS_MAX = 0.050  # 6cm maximum protrusion
CONSENSUS_FRAMES = 3  # Successive frames required

# RECOGNITION TUNING
MATCH_THRESHOLD = 0.40 # Adjust based on your environment
SCALE_FACTOR = 0.5     # Processing scale for speed



# INITIALIZATION
print("[SYSTEM] Booting up Vision Pro...")

if not os.path.exists(CALIB_DIR):
    sys.exit(f"CRITICAL: Calibration folder {CALIB_DIR} not found.")

try:
    mapLx = np.load(os.path.join(CALIB_DIR, "stereoMapL_x.npy"))
    mapLy = np.load(os.path.join(CALIB_DIR, "stereoMapL_y.npy"))
    mapRx = np.load(os.path.join(CALIB_DIR, "stereoMapR_x.npy"))
    mapRy = np.load(os.path.join(CALIB_DIR, "stereoMapR_y.npy"))
    Q = np.load(os.path.join(CALIB_DIR, "Q.npy"))
    print("[SUCCESS] Stereo Maps and Q-Matrix loaded.")
except Exception as e:
    sys.exit(f"CRITICAL: Data corruption in .npy files: {e}")

f_pixel = Q[2, 3]
baseline = abs(1.0 / Q[3, 2])

# Global State Management
app = Flask(__name__)
class AppState:
    STATUS = "IDLE"
    MESSAGE = "WAITING FOR USER"
    USER_NAME = ""
    PROGRESS = 0
    LAST_EVENT = 0

state = AppState()
outputFrame = None
lock = threading.Lock()
face_db = {}
liveness_history = deque(maxlen=CONSENSUS_FRAMES)

# Load Models
face_detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 240), 0.7, 0.3, 1)
face_recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

# Optimize Stereo Matcher
stereo = cv.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=7,
    P1=8*3*7**2, P2=32*3*7**2, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

# Load Face DB
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        face_db = pickle.load(f)
    print(f"[INFO] Face database loaded: {list(face_db.keys())}")

# Hardware
arduino = None
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(1)
except: print("[WARN] Running without Arduino.")



# CORE LOGIC
def get_depth(disp, x, y):
    h, w = disp.shape
    if x < 5 or x >= w-5 or y < 5 or y >= h-5: return None
    roi = disp[y-2:y+3, x-2:x+3]
    valid = roi[roi > 16.0]
    if len(valid) == 0: return None
    return (f_pixel * baseline) / (np.median(valid) / 16.0)

def processing_thread():
    global outputFrame, state
    capL = cv.VideoCapture(URL_LEFT)
    capR = cv.VideoCapture(URL_RIGHT)
    
    while True:
        if not capL.grab() or not capR.grab(): continue
        _, frameL = capL.retrieve(); _, frameR = capR.retrieve()
        if frameL is None or frameR is None: continue

        # Rectify for depth accuracy
        rectL = cv.remap(frameL, mapLx, mapLy, cv.INTER_LINEAR)
        rectR = cv.remap(frameR, mapRx, mapRy, cv.INTER_LINEAR)
        
        vis_frame = rectL.copy()
        h, w = rectL.shape[:2]
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(rectL)

        if faces is not None:
            face = faces[0]
            
            # LIVENESS CHECK
            if state.STATUS in ["IDLE", "CHECKING_LIVENESS"]:
                state.STATUS = "CHECKING_LIVENESS"
                state.MESSAGE = "VERIFYING LIVENESS..."
                
                grayL = cv.cvtColor(rectL, cv.COLOR_BGR2GRAY)
                grayR = cv.cvtColor(rectR, cv.COLOR_BGR2GRAY)
                disp = stereo.compute(grayL, grayR).astype(np.float32)
                
                z_re = get_depth(disp, int(face[4]), int(face[5]))
                z_le = get_depth(disp, int(face[6]), int(face[7]))
                z_no = get_depth(disp, int(face[8]), int(face[9]))

                is_real = False
                if z_re and z_le and z_no:
                    diff = ((z_re + z_le)/2.0) - z_no
                    if LIVENESS_MIN < diff < LIVENESS_MAX:
                        is_real = True
                
                liveness_history.append(is_real)
                state.PROGRESS = int((sum(liveness_history)/CONSENSUS_FRAMES)*100)
                
                if len(liveness_history) == CONSENSUS_FRAMES and all(liveness_history):
                    state.STATUS = "RECOGNIZING" # Move to next phase
                    state.MESSAGE = "LIVENESS OK - SCANNING IDENTITY"
            
            # FACE RECOGNITION
            elif state.STATUS == "RECOGNIZING":
                # Align and crop the confirmed 3D face
                aligned = face_recognizer.alignCrop(rectL, face)
                feat = face_recognizer.feature(aligned)[0]
                feat /= np.linalg.norm(feat)

                best_match = "Unknown"
                best_score = 0
                for name, db_feat in face_db.items():
                    score = np.dot(feat, db_feat)
                    if score > best_score:
                        best_score = score
                        best_match = name
                
                if best_score >= MATCH_THRESHOLD:
                    state.STATUS = "GRANTED"
                    state.MESSAGE = f"ACCESS GRANTED: {best_match.upper()}"
                    state.USER_NAME = best_match
                    if arduino: arduino.write(b'O')
                else:
                    state.STATUS = "DENIED"
                    state.MESSAGE = "UNAUTHORIZED USER"
                
                state.LAST_EVENT = time.time()

            # Drawing face box
            box = face[:4].astype(int)
            color = (0, 255, 0) if state.STATUS == "GRANTED" else (0, 255, 255)
            cv.rectangle(vis_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            
        else:
            # Reset if face leaves frame
            if state.STATUS not in ["GRANTED", "DENIED"]:
                state.STATUS = "IDLE"
                state.MESSAGE = "WAITING FOR FACE"
                liveness_history.clear()
                state.PROGRESS = 0

        # Auto-reset after GRANTED/DENIED event
        if state.STATUS in ["GRANTED", "DENIED"] and (time.time() - state.LAST_EVENT > 4):
            state.STATUS = "IDLE"
            liveness_history.clear()

        with lock:
            outputFrame = vis_frame.copy()



# STANDARD UI
HTML_TEMPLATE = """
<!DOCTYPE html><html><head><title>Vision Pro Access</title>
<style>
    body { background: #000; color: #0cf; font-family: 'Courier New', monospace; text-align: center; }
    .status-panel { margin-top: 20px; border: 1px solid #0cf; padding: 15px; display: inline-block; min-width: 400px; }
    .progress-bar { width: 100%; height: 10px; background: #222; border-radius: 5px; margin-top: 10px; }
    .progress-fill { height: 100%; background: #0f0; width: 0%; transition: 0.3s; }
    .GRANTED { color: #0f0; border-color: #0f0; }
    .DENIED { color: #f00; border-color: #f00; }
</style></head><body>
    <h1>STEREO VISION ACCESS CONTROL</h1>
    <div class="status-panel" id="panel">
        <div id="stat" style="font-size: 24px;">IDLE</div>
        <div id="msg">Awaiting Input...</div>
        <div class="progress-bar"><div id="bar" class="progress-fill"></div></div>
    </div>
    <br><br><img src="/video_feed" style="width: 60%; border: 1px solid #333;">
    <script>
        setInterval(() => {
            fetch('/status').then(r => r.json()).then(d => {
                document.getElementById('stat').innerText = d.status;
                document.getElementById('panel').className = 'status-panel ' + d.status;
                document.getElementById('msg').innerText = d.message;
                document.getElementById('bar').style.width = d.progress + '%';
            });
        }, 300);
    </script>
</body></html>
"""



# ROUTES
@app.route("/")
def index(): return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with lock:
                if outputFrame is None: continue
                _, enc = cv.imencode(".jpg", outputFrame, [int(cv.IMWRITE_JPEG_QUALITY), 70])
            yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status_api():
    return jsonify({"status": state.STATUS, "message": state.MESSAGE, "progress": state.PROGRESS})

if __name__ == "__main__":
    threading.Thread(target=processing_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
