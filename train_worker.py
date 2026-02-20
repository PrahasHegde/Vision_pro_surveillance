# backend/train_worker.py -> backend code to convert images to an embedding vector == part of main code
#-----------------------------------------------------------------------------------------------------

# IMPORTS
import os
import time
import pickle
import cv2 as cv
import numpy as np
import requests
import gc
import random


# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DB_FILE = os.path.join(BASE_DIR, "face_encodings.pickle") 
TRIGGER_FILE = os.path.join(BASE_DIR, "trigger_training.txt")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# MODEL PATHS
YUNET_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")

# Load models
detector = cv.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.9, 0.3, 5000)
recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")

def train():
    print("[WORKER] Training started...")
    
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f: db = pickle.load(f)
        except: db = {}
    else: db = {}

    users = os.listdir(DATASET_DIR)
    updated = False

    for user_id in users:
        if user_id in db: continue 
        
        user_path = os.path.join(DATASET_DIR, user_id)
        if not os.path.isdir(user_path): continue

        print(f"[WORKER] Processing new user: {user_id}")
        embeddings = []
        
        # Get all images
        all_images = os.listdir(user_path)
        
        # CRASH PREVENTER 1: LIMIT TO 20 IMAGES
        # We only take 20 random images. This is enough for accuracy
        if len(all_images) > 20:
            images_to_process = random.sample(all_images, 20)
        else:
            images_to_process = all_images
            
        # print(f"[WORKER] Selected {len(images_to_process)} images for training.")

        for i, img_name in enumerate(images_to_process):
            try:
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
                
                # CRASH PREVENTER 2: SLEEP BETWEEN IMAGES
                # Let CPU cool down after every single image
                time.sleep(0.05) 
                
            except Exception as e:
                print(f"[WARNING] Skipped image {img_name}: {e}")

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)
            db[user_id] = avg
            updated = True
            # print(f"[WORKER] Added {user_id}")
            
            # Save and Clean RAM immediately
            with open(DB_FILE, "wb") as f: pickle.dump(db, f)
            del embeddings
            gc.collect()

    if updated:
        try:
            requests.get("http://localhost:5000/reload_db", timeout=2)
            print("[WORKER] Main System Reloaded.")
        except:
            pass

def main():
    print("[WORKER] Ready. Waiting for trigger...")
    while True:
        if os.path.exists(TRIGGER_FILE):
            try:
                with open(TRIGGER_FILE, "r") as f: content = f.read().strip()
                if content == "start":
                    time.sleep(2) # Wait for file writes to finish
                    train()
                    if os.path.exists(TRIGGER_FILE):
                        os.remove(TRIGGER_FILE)
            except Exception as e:
                print(f"[ERROR] Worker error: {e}")
        time.sleep(3)

if __name__ == "__main__":
    main()
