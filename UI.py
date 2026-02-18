#combination.py + code to retract the lock via Arduino


import cv2
import numpy as np
import os
import sys
import pickle
import threading
import time
import gc
import serial
from flask import Flask, Response, render_template, jsonify

# ==========================================================
# 1. TUNING CONFIGURATION
# ==========================================================
REC_THRESHOLD = 0.55  
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 9600

# ==========================================================
# 2. SYSTEM SETUP
# ==========================================================
URL_LEFT  = "http://192.168.0.196:81/stream" 
URL_RIGHT = "http://192.168.0.197:81/stream"

# Resolution must balance Speed vs Calibration Accuracy
PROCESS_W, PROCESS_H = 320, 240 # Increased slightly for better stereo match
DISPLAY_W, DISPLAY_H = 640, 480
SCALE_X = DISPLAY_W / PROCESS_W
SCALE_Y = DISPLAY_H / PROCESS_H
SWAP_CAMERAS = True 

NPZ_PATH = 'stereo_calibration.npz'
YUNET_MODEL = 'models/face_detection_yunet_2023mar.onnx'
SFACE_MODEL = 'models/face_recognition_sface_2021dec.onnx'
DB_FILE = "face_encodings2.pickle"

app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()
arduino = None 

system_state = {
    "status": "System Loading...", 
    "mode": "SCANNING",
    "name": "---"
}

# ==========================================================
# 3. HARDWARE & UTILS
# ==========================================================
def init_arduino():
    global arduino
    if arduino is not None and arduino.is_open: return
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"[HARDWARE] Connected to Lock on {SERIAL_PORT}")
    except: pass

def send_unlock_signal():
    global arduino
    try:
        if arduino and arduino.is_open:
            arduino.write(b'O')
            return True
    except:
        if arduino: arduino.close()
        arduino = None
    return False

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
    def start(self):
        t = threading.Thread(target=self.update); t.daemon = True; t.start()
        return self
    def update(self):
        while not self.stopped:
            if not self.stream.isOpened(): time.sleep(1); continue
            ret, frame = self.stream.read()
            with self.lock:
                if ret: self.ret, self.frame = ret, frame
            time.sleep(0.01)
    def read(self):
        with self.lock: return self.ret, self.frame.copy() if self.frame is not None else None

# ==========================================================
# 4. MAIN LOGIC
# ==========================================================
def processing_thread():
    global output_frame, system_state
    init_arduino()
    
    # LOAD RESOURCES
    try:
        data = np.load(NPZ_PATH)
        K_L, D_L = data['mtx_l'], data['dist_l']
        K_R, D_R = data['mtx_r'], data['dist_r']
        R, T = data['R'], data['T']
        
        # Scale Calibration Matrix for Processing Resolution
        Sx, Sy = PROCESS_W / 640.0, PROCESS_H / 480.0
        K_L[0,0]*=Sx; K_L[1,1]*=Sy; K_L[0,2]*=Sx; K_L[1,2]*=Sy
        K_R[0,0]*=Sx; K_R[1,1]*=Sy; K_R[0,2]*=Sx; K_R[1,2]*=Sy

        # EPIPOLAR RECTIFICATION MAPS
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_L, D_L, K_R, D_R, (PROCESS_W, PROCESS_H), R, T, alpha=0)
        map1_L, map2_L = cv2.initUndistortRectifyMap(K_L, D_L, R1, P1, (PROCESS_W, PROCESS_H), cv2.CV_16SC2)
        map1_R, map2_R = cv2.initUndistortRectifyMap(K_R, D_R, R2, P2, (PROCESS_W, PROCESS_H), cv2.CV_16SC2)
        
        # STEREO MATCHER (Updated settings for better results)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,          # Changed from -16 to 0 (Safer)
            numDisparities=32,       # Must be divisible by 16
            blockSize=5,
            P1=8*3*5**2, P2=32*3*5**2,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
        )

        detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (PROCESS_W, PROCESS_H), 0.6, 0.3, 1)
        recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        
        db = {}
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "rb") as f: db = pickle.load(f)

        f_pixel = (K_L[0,0] + K_L[1,1]) / 2.0
        B_meter = abs(T[0][0])
        print(f"[DEBUG] Baseline: {B_meter}, Focal: {f_pixel}") # Check console for this!
        
    except Exception as e:
        print(f"[ERROR] Loading Failed: {e}")
        return

    camL = VideoStream(URL_LEFT).start()
    camR = VideoStream(URL_RIGHT).start()
    time.sleep(2.0)

    frame_count = 0
    access_until = 0
    
    while True:
        if frame_count % 30 == 0: gc.collect()
        frame_count += 1
        
        retL, rawL = camL.read()
        retR, rawR = camR.read()
        if not retL or not retR: time.sleep(0.05); continue
        
        if SWAP_CAMERAS: rawL, rawR = rawR, rawL
        try: vis_frame = cv2.resize(rawL, (DISPLAY_W, DISPLAY_H))
        except: continue

        # --- UPDATE UI STATE ---
        if time.time() < access_until:
             system_state["mode"] = "ACCESS GRANTED"
        else:
             system_state["mode"] = "SCANNING"

        # --- ACCESS GRANTED OVERLAY ---
        if time.time() < access_until:
            cv2.rectangle(vis_frame, (0, 0), (DISPLAY_W, 60), (0, 255, 0), -1)
            cv2.putText(vis_frame, f"GRANTED: {system_state['name']}", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
            with frame_lock: output_frame = vis_frame
            time.sleep(0.03)
            continue

        # --- SCANNING ---
        if frame_count % 3 == 0:
            try:
                smallL = cv2.resize(rawL, (PROCESS_W, PROCESS_H))
                smallR = cv2.resize(rawR, (PROCESS_W, PROCESS_H))
                
                # RECTIFY (Fixes the Epipolar Lines)
                rect_L = cv2.remap(smallL, map1_L, map2_L, cv2.INTER_LINEAR)
                rect_R = cv2.remap(smallR, map1_R, map2_R, cv2.INTER_LINEAR)
                
                detector.setInputSize((PROCESS_W, PROCESS_H))
                _, faces = detector.detect(rect_L)

                if faces is not None:
                    system_state["status"] = "Face Detected"
                    
                    # DEPTH CALCULATION
                    grayL = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
                    grayR = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)
                    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
                    
                    for face in faces:
                        # Extract Nose Depth vs Cheek Depth
                        # Points: 4=Right Eye, 8=Nose Tip
                        pts = {"RE": (int(face[4]), int(face[5])), "N": (int(face[8]), int(face[9]))}
                        z = {}
                        
                        for k, (x, y) in pts.items():
                            if 0 <= x < PROCESS_W and 0 <= y < PROCESS_H:
                                d = disp[y, x]
                                if d > 0.1: # Avoid division by zero
                                    z_val = (f_pixel * B_meter) / d
                                    z[k] = z_val
                        
                        is_real = False
                        # LIVENESS LOGIC:
                        # Real faces have a nose closer than the eyes.
                        # Flat screens have equal depth.
                        if "N" in z and "RE" in z:
                            diff = abs(z["RE"] - z["N"])
                            # DEBUG PRINT (Check your console!)
                            # print(f"[DEPTH] Nose: {z['N']:.2f}, Diff: {diff:.4f}") 
                            
                            # Threshold: 0.005 works for meters. If units are mm, use 5.0
                            if diff > 0.005: 
                                is_real = True

                        b = list(map(int, face[:4]))
                        sb = [int(b[0]*SCALE_X), int(b[1]*SCALE_Y), int(b[2]*SCALE_X), int(b[3]*SCALE_Y)]

                        if is_real:
                            # RECOGNIZE
                            align = recognizer.alignCrop(rect_L, face) # Use rect_L for better alignment
                            feat = recognizer.feature(align)[0]
                            feat /= np.linalg.norm(feat)
                            
                            max_score = 0
                            final_name = "Unknown"
                            
                            for name, db_feat in db.items():
                                score = np.dot(feat, db_feat)
                                if score > max_score:
                                    max_score = score
                                    final_name = name
                            
                            system_state["name"] = final_name
                            
                            if max_score > REC_THRESHOLD:
                                send_unlock_signal()
                                access_until = time.time() + 4.0
                                break
                            else:
                                color = (0, 255, 255) # Yellow for Unknown
                                cv2.rectangle(vis_frame, (sb[0], sb[1]), (sb[0]+sb[2], sb[1]+sb[3]), color, 2)
                        else:
                            # SPOOF DETECTED
                            system_state["status"] = "Spoof Detected"
                            cv2.rectangle(vis_frame, (sb[0], sb[1]), (sb[0]+sb[2], sb[1]+sb[3]), (0, 0, 255), 2)
                            cv2.putText(vis_frame, "FAKE", (sb[0], sb[1]-10), 0, 0.7, (0, 0, 255), 2)
                else:
                    system_state["status"] = "Searching..."
                    system_state["name"] = "---"

            except Exception as e:
                print(f"Error: {e}")
                init_arduino()

        with frame_lock: output_frame = vis_frame

@app.route("/")
def index(): return render_template('index.html')

@app.route("/status")
def status(): return jsonify(system_state)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                if output_frame is None: time.sleep(0.1); continue
                _, enc = cv2.imencode(".jpg", output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=processing_thread); t.daemon = True; t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)