import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# ================= CONFIGURATION =================
# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'dataset_neo')
PKL_PATH = os.path.join(SCRIPT_DIR, 'embeddings_insight.pkl')
# =================================================

def train():
    # Initialize InsightFace
    # 'buffalo_l' is the SOTA model pack (Detection + Recognition)
    # providers=['CPUExecutionProvider'] forces it to use CPU. Use 'CUDAExecutionProvider' if you have an NVIDIA GPU.
    print("Initializing InsightFace (Buffalo_L)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    known_encodings = {} # Format: {'Name': [emb1, emb2...]}
    
    # Check if train dir exists
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Training directory not found at {TRAIN_DIR}")
        return

    print(f"Starting training on: {TRAIN_DIR}")
    
    for person_name in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        known_encodings[person_name] = []
        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # InsightFace reads images via OpenCV (BGR format)
            img = cv2.imread(img_path)
            if img is None: continue

            # Get faces
            faces = app.get(img)

            if len(faces) > 0:
                # InsightFace returns a 512D embedding
                # We take the first face detected (faces[0])
                embedding = faces[0].embedding
                known_encodings[person_name].append(embedding)

    print(f"Saving {len(known_encodings)} people to {PKL_PATH}...")
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(known_encodings, f)
    print("Done.")

if __name__ == "__main__":
    train()