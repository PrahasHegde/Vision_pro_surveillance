# backend/train_worker.py
import os
import time
import pickle
import cv2 as cv
import numpy as np
import requests # Need this to call main_main

# --- ABSOLUTE PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_FILE = os.path.join(BASE_DIR, "face_encodings2.pickle")
TRIGGER_FILE = os.path.join(BASE_DIR, "trigger_training.txt")
MODEL_DIR = os.path.join(BASE_DIR, "models")

YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.9, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

def train():
    print("[TRAINING] Starting...")
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            try: db = pickle.load(f)
            except: db = {}
    else: db = {}

    users = os.listdir(DATASET_DIR)
    for user_id in users:
        user_path = os.path.join(DATASET_DIR, user_id)
        if not os.path.isdir(user_path): continue
        if user_id in db: continue # Skip existing

        print(f"[PROCESSING] {user_id}")
        embeddings = []
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            img = cv.imread(img_path)
            if img is None: continue
            
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is not None:
                face_align = recognizer.alignCrop(img, faces[0])
                feat = recognizer.feature(face_align)
                embeddings.append(feat[0])

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)
            db[user_id] = avg
            print(f"[ADDED] {user_id}")

    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)
    print("[SUCCESS] Database saved.")

    # --- CRITICAL: TELL MAIN SYSTEM TO RELOAD ---
    try:
        requests.get("http://localhost:5000/reload_db")
        print("[INFO] Main System Reloaded Successfully")
    except:
        print("[WARNING] Could not reload Main System (Is it running?)")

def main():
    print("[WORKER] Watching for trigger...")
    while True:
        if os.path.exists(TRIGGER_FILE):
            try:
                with open(TRIGGER_FILE, "r") as f: content = f.read().strip()
                if content == "start":
                    time.sleep(1)
                    train()
                    os.remove(TRIGGER_FILE)
            except Exception as e:
                print(f"[ERROR] {e}")
        time.sleep(2)

if __name__ == "__main__":
    main()