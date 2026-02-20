# x.py -> copy of the liveness detection by epipolar geometry code == part of main project code 
#----------------------------------------------------------------------------------------------


# IMPORTS
import cv2
import numpy as np
import os
import sys
import time
from collections import deque



# CONFIGURATION
URL_LEFT  = "http://192.168.0.6:81/stream"
URL_RIGHT = "http://192.168.0.5:81/stream"

CALIB_FILE  = "stereo2_maps.npz"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"

# TUNING PARAMETERS
# Real face if nose is between 1.5cm and 8.0cm ahead of eyes
LIVENESS_MIN = 0.015 
LIVENESS_MAX = 0.050 
CONSENSUS_FRAMES = 3  # Number of frames to check before deciding



# SETUP
if not os.path.exists(YUNET_MODEL):
    sys.exit(f"Error: {YUNET_MODEL} not found.")

try:
    cv_file = np.load(CALIB_FILE)
    mapLx, mapLy = cv_file["stereoMapL_x"], cv_file["stereoMapL_y"]
    mapRx, mapRy = cv_file["stereoMapR_x"], cv_file["stereoMapR_y"]
    Q = cv_file["Q"]
except IOError:
    sys.exit("Error: stereo1_maps.npz not found.")

f_pixel = Q[2, 3]
baseline = abs(1.0 / Q[3, 2])

face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (0, 0), 0.7, 0.3, 1)

# Fast Stereo Matcher
matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=16*6, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, uniquenessRatio=10, 
    speckleWindowSize=100, speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

def get_depth(disparity_map, x, y):
    h, w = disparity_map.shape
    if x < 2 or x >= w-2 or y < 2 or y >= h-2: return None
    roi = disparity_map[y-2:y+3, x-2:x+3]
    valid = roi[roi > 16.0]
    if len(valid) == 0: return None
    d_val = np.median(valid) / 16.0
    return (f_pixel * baseline) / d_val


# MAIN LOOP
print(f"Liveness Consensus Mode. Baseline: {baseline:.4f}m")
print(f"Consensus Required: {CONSENSUS_FRAMES} frames")

capL = cv2.VideoCapture(URL_LEFT)
capR = cv2.VideoCapture(URL_RIGHT)
capL.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capR.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# History of "True/False" judgments
history = deque(maxlen=CONSENSUS_FRAMES)

while True:
    if not capL.grab() or not capR.grab(): continue
    _, imgL_raw = capL.retrieve()
    _, imgR_raw = capR.retrieve()
    if imgL_raw is None or imgR_raw is None: continue

    # Rectify & Compute
    rectL = cv2.remap(imgL_raw, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR_raw, mapRx, mapRy, cv2.INTER_LINEAR)
    
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    
    disp = matcher.compute(grayL, grayR).astype(np.float32)
    disp = cv2.medianBlur(disp, 5)

    # Detect Face
    h, w = rectL.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(rectL)

    if faces is not None:
        for face in faces:
            # Landmarks
            re = (int(face[4]), int(face[5]))
            le = (int(face[6]), int(face[7]))
            no = (int(face[8]), int(face[9]))

            z_re = get_depth(disp, re[0], re[1])
            z_le = get_depth(disp, le[0], le[1])
            z_no = get_depth(disp, no[0], no[1])

            if z_re and z_le and z_no:
                avg_eyes = (z_re + z_le) / 2.0
                diff = avg_eyes - z_no

                # Determine Frame Status
                is_real_frame = False
                if diff > LIVENESS_MIN and diff < LIVENESS_MAX:
                    is_real_frame = True
                
                # Update History
                history.append(is_real_frame)

                # Calculate Consensus (Only status changes if ALL last 3 frames agree)
                status_text = "Analyzing..."
                color = (255, 0, 0) # Blue (Uncertain)

                if len(history) == CONSENSUS_FRAMES:
                    if all(history): # All True
                        status_text = "REAL FACE ✅"
                        color = (0, 255, 0) # Green
                    elif not any(history): # All False
                        status_text = "FAKE FACE ❌"
                        color = (0, 0, 255) # Red
                    else:
                        # Mixed results (e.g., True, False, True) - maintain caution
                        status_text = "Scanning..." 

                # Print & Draw
                print(f"Status: {status_text} | Diff: {diff:.3f}m | Eyes: {avg_eyes:.3f}m | Nose: {z_no:.3f}m")

                # Visuals
                box = face[0:4].astype(int)
                cv2.rectangle(rectL, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
                cv2.putText(rectL, status_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.circle(rectL, re, 5, (255, 0, 0), -1)
                cv2.circle(rectL, le, 5, (255, 0, 0), -1)
                cv2.circle(rectL, no, 5, (0, 255, 0), -1)

    cv2.imshow("RPi Liveness Check", rectL)
    if cv2.waitKey(1) == ord('q'): break

capL.release()

capR.release()
cv2.destroyAllWindows()
