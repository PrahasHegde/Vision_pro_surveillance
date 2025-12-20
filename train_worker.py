# train_worker.py -> to Process Captured Images and Update Face Embedding Database


#####################################################################################################################################

import cv2 as cv
import numpy as np
import os
import pickle
import sys

# --- CONFIGURATION ---
SAVE_DIR = "dataset"
DB_FILE = "face_encodings2.pickle"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

def main():
    print("-" * 50)
    print("[TRAINING WORKER] Starting Process...")
    print("-" * 50)

    if not os.path.exists(SFACE_PATH):
        print("[ERROR] Model file missing.")
        return

    try:
        recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")
    except Exception as e:
        print(f"[ERROR] Model Load Failed: {e}")
        return

    database = {}
    
    if not os.path.exists(SAVE_DIR):
        print("[ERROR] Dataset folder missing.")
        return

    users = [d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d))]
    count = 0
    
    for user_name in users:
        print(f" -> Processing user: {user_name}")
        user_path = os.path.join(SAVE_DIR, user_name)
        image_files = os.listdir(user_path)
        embeddings = []

        if not image_files: continue
        
        for img_file in image_files:
            try:
                img_path = os.path.join(user_path, img_file)
                img = cv.imread(img_path)
                if img is None: continue
                
                feat = recognizer.feature(img)
                embeddings.append(feat.flatten())
            except: pass

        if embeddings:
            embeddings_np = np.array(embeddings)
            avg_embedding = np.mean(embeddings_np, axis=0)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            database[user_name] = avg_embedding
            count += 1

    try:
        with open(DB_FILE, "wb") as f:
            pickle.dump(database, f)
        print("-" * 50)
        print(f"[SUCCESS] Database Updated. Total Users: {count}")
        print("-" * 50)
    except Exception as e:
        print(f"[ERROR] Saving Pickle: {e}")

if __name__ == "__main__":
    main()