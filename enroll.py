# enroll.py -> one-time code to get face embeddings from captured images and save to database (not part of main code)
#-------------------------------------------------------------------------------------------------------------------


# IMPORTS
import cv2 as cv
import numpy as np
import os
import pickle
import sys


# CONFIGURATION
SAVE_DIR = "dataset"
DB_FILE = "face_encodings.pickle"
SFACE_PATH = "models/face_recognition_sface_2021dec.onnx"

def main():
    print("--- FACE RECOGNITION TRAINING TOOL ---")
    
    # Check if Model Exists
    if not os.path.exists(SFACE_PATH):
        print(f"[ERROR] Model file not found: {SFACE_PATH}")
        print("Please download it using: wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
        sys.exit()

    # Check if Dataset Exists
    if not os.path.exists(SAVE_DIR):
        print(f"[ERROR] Dataset directory '{SAVE_DIR}' not found.")
        print("Run the capture app first to collect images.")
        sys.exit()

    # Initialize SFace Model
    try:
        recognizer = cv.FaceRecognizerSF.create(SFACE_PATH, "")
    except Exception as e:
        print(f"[ERROR] Failed to load SFace model: {e}")
        sys.exit()

    database = {}
    
    # Get List of Users
    users = [d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d))]
    
    if not users:
        print("[WARNING] No user folders found in dataset!")
        sys.exit()

    print(f"[INFO] Found {len(users)} users: {users}")
    print("-" * 40)

    # Process Each User
    for user_name in users:
        print(f"Processing '{user_name}'...", end=" ", flush=True)
        
        user_path = os.path.join(SAVE_DIR, user_name)
        image_files = os.listdir(user_path)
        
        embeddings = []
        
        if not image_files:
            print("[SKIPPED] (Empty folder)")
            continue

        # Process every image in the user's folder
        for img_file in image_files:
            # Filter for valid image extensions
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(user_path, img_file)
            img = cv.imread(img_path)
            
            if img is None:
                continue
            
            # EXTRACT FEATURES
            # The capture app already saved these as aligned 112x112 crops
            # so we just feed them straight into the recognizer.
            feat = recognizer.feature(img)
            
            if feat is not None:
                embeddings.append(feat.flatten())

        # Average and Normalize
        if embeddings:
            embeddings_np = np.array(embeddings)
            
            # Calculate Mean Vector 
            avg_embedding = np.mean(embeddings_np, axis=0)
            
            # Normalize
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            # Add to Database
            database[user_name] = avg_embedding
            print(f"[DONE] ({len(embeddings)} images merged)")
        else:
            print("[FAILED] No valid features found.")

    # Save to Disk
    print("-" * 40)
    if database:
        try:
            with open(DB_FILE, "wb") as f:
                pickle.dump(database, f)
            print(f"[SUCCESS] Database saved to '{DB_FILE}'")
            print(f"          Total Enrolled Users: {len(database)}")
        except Exception as e:
            print(f"[ERROR] Could not write to file: {e}")
    else:
        print("[ERROR] Database is empty. Nothing saved.")

if __name__ == "__main__":
    main()
