# epi.py -> main code for liveness detection using epipolar geometry == part of the main code for the project
#------------------------------------------------------------------------------------------------------------


# IMPORTS
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import sys
from collections import deque


# CONFIGURATION
NPZ_PATH = 'stereo_calibration.npz'
YUNET_MODEL = 'face_detection_yunet_2023mar.onnx'

#CAMERA URL's
URL_LEFT  = "http://192.168.0.196:81/stream"
URL_RIGHT = "http://192.168.0.197:81/stream"

#RESOLUTION
WIDTH, HEIGHT = 640, 480
SWAP_CAMERAS = True
NUM_DISPARITIES = 16 * 10
BLOCK_SIZE = 5
REQUIRED_REAL_FRAMES = 3



# LOAD MODELS & CALIBRATION
if not os.path.exists(YUNET_MODEL):
    sys.exit("YuNet model missing")

face_detector = cv2.FaceDetectorYN.create(
    YUNET_MODEL, "", (WIDTH, HEIGHT),
    score_threshold=0.7,
    nms_threshold=0.3,
    top_k=1
)

data = np.load(NPZ_PATH)
K_L, D_L = data['mtx_l'], data['dist_l']
K_R, D_R = data['mtx_r'], data['dist_r']
R, T     = data['R'], data['T']

f_pixel = (K_L[0,0] + K_L[1,1]) / 2.0
B_meter = abs(T[0][0])

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K_L, D_L, K_R, D_R, (WIDTH, HEIGHT), R, T, alpha=0
)

map1_L, map2_L = cv2.initUndistortRectifyMap(
    K_L, D_L, R1, P1, (WIDTH, HEIGHT), cv2.CV_16SC2
)
map1_R, map2_R = cv2.initUndistortRectifyMap(
    K_R, D_R, R2, P2, (WIDTH, HEIGHT), cv2.CV_16SC2
)

#SGBM PIPELINE
stereo = cv2.StereoSGBM_create(
    minDisparity=-16,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=8 * 3 * BLOCK_SIZE**2,
    P2=32 * 3 * BLOCK_SIZE**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)



# HELPERS
def get_disp(disp, x, y, w=7):
    h, W = disp.shape
    x0, x1 = max(0, x-w), min(W, x+w+1)
    y0, y1 = max(0, y-w), min(h, y+w+1)
    roi = disp[y0:y1, x0:x1]
    roi = roi[roi > 1.0]
    return np.median(roi) if len(roi) else 0.0

def disp_to_depth(d):
    return (f_pixel * B_meter) / d if d > 0 else None


# MAIN LOOP
capL = cv2.VideoCapture(URL_LEFT)
capR = cv2.VideoCapture(URL_RIGHT)

real_history = deque(maxlen=REQUIRED_REAL_FRAMES)

while True:
    if not capL.grab() or not capR.grab():
        continue

    _, frameL = capL.retrieve()
    _, frameR = capR.retrieve()
    if frameL is None or frameR is None:
        continue

    frameL = cv2.resize(frameL, (WIDTH, HEIGHT))
    frameR = cv2.resize(frameR, (WIDTH, HEIGHT))
    if SWAP_CAMERAS:
        frameL, frameR = frameR, frameL

    rect_L = cv2.remap(frameL, map1_L, map2_L, cv2.INTER_LINEAR)
    rect_R = cv2.remap(frameR, map1_R, map2_R, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rect_L, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rect_R, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disparity = cv2.medianBlur(disparity, 5)

    face_detector.setInputSize((WIDTH, HEIGHT))
    _, faces = face_detector.detect(rect_L)

    is_real_frame = False

    if faces is not None:
        for face in faces:
            depths = {}

            landmarks = {
                "RE": (int(face[4]), int(face[5])),
                "LE": (int(face[6]), int(face[7])),
                "N" : (int(face[8]), int(face[9]))
            }

            for name, (x, y) in landmarks.items():
                d = get_disp(disparity, x, y)
                z = disp_to_depth(d)
                if z is not None:
                    depths[name] = z

            eye_depths = [depths[k] for k in ("RE", "LE") if k in depths]
            nose_z = depths.get("N", None)

            if nose_z is not None and len(eye_depths) >= 1:
                eye_avg_z = sum(eye_depths) / len(eye_depths)
                diff = abs(eye_avg_z - nose_z)

                if diff > nose_z:
                    is_real_frame = True

    real_history.append(is_real_frame)

    if sum(real_history) == REQUIRED_REAL_FRAMES:
        cv2.putText(rect_L, "Real Face", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3)

    disp_vis = cv2.normalize(disparity, None, 0, 255,
                             cv2.NORM_MINMAX, cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    cv2.imshow("Stereo Liveness Detection",
               np.hstack((rect_L, disp_color)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
