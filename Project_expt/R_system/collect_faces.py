import cv2 as cv
import numpy as np
import time
import os
import sys

# --- CONFIGURATION ---
USER_ID = "User_01"  # Change this for each person (e.g., "John_Doe")
SAVE_DIR = f"dataset/{USER_ID}"
MAX_IMAGES = 200
TIME_LIMIT_SEC = 30
CONFIDENCE_THRESHOLD = 0.9  # High threshold to ensure only clear faces are saved

# Model Paths (Ensure these files are in the same directory)
YUNET_PATH = "face_detection_yunet_2023mar.onnx"
SFACE_PATH = "face_recognition_sface_2021dec.onnx"

def initialize_models():
    # Load YuNet (Face Detection)
    detector = cv.FaceDetectorYN.create(
        YUNET_PATH,
        "",
        (320, 320),  # Input size will be updated dynamically
        0.9, 0.3, 5000
    )
    
    # Load SFace (Face Recognition) - Used here specifically for Alignment
    recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")
    
    return detector, recognizer

def main():
    # 1. Setup Directories
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"[INFO] Created directory: {SAVE_DIR}")
    else:
        print(f"[INFO] Saving to existing directory: {SAVE_DIR}")

    # 2. Initialize Camera
    # On RPi 5, index 0 usually maps to the USB cam or CSI cam via libcamera-hello
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit()

    # Get actual frame geometry
    frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 3. Initialize AI Models
    detector, recognizer = initialize_models()
    detector.setInputSize((frame_w, frame_h))

    print(f"\n[STARTING] Collecting {MAX_IMAGES} images for {TIME_LIMIT_SEC} seconds...")
    print("Please follow the movement instructions displayed on screen.")
    time.sleep(2) # Give user a moment to prepare

    start_time = time.time()
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check timers and limits
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT_SEC or count >= MAX_IMAGES:
            break

        # Inference
        # YuNet expects distinct detection calls
        faces = detector.detect(frame)
        
        # Draw GUI info
        cv.putText(frame, f"Count: {count}/{MAX_IMAGES}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Time: {int(TIME_LIMIT_SEC - elapsed)}s", (10, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Process detections
        if faces[1] is not None:
            # Get the face with the highest score (usually the first one)
            face = faces[1][0]
            confidence = face[-1]

            # Only process high-quality detections
            if confidence >= CONFIDENCE_THRESHOLD:
                # Bounding box for visualization
                box = list(map(int, face[:4]))
                cv.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)

                # ALIGN AND CROP (Crucial for SFace)
                # SFace 'alignCrop' uses landmarks to straighten the face to 112x112
                face_aligned = recognizer.alignCrop(frame, face)

                # Save the aligned image
                img_name = f"{SAVE_DIR}/{USER_ID}_{count:04d}.jpg"
                cv.imwrite(img_name, face_aligned)
                count += 1
            else:
                cv.putText(frame, "Low Confidence - Adjust Light", (10, 90), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow("Dataset Collection", frame)

        # Exit on 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    print(f"\n[DONE] Collected {count} images in {elapsed:.2f} seconds.")
    print(f"Images saved to: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()