import face_recognition
import cv2
import os
import pickle

from pathlib import Path

# CONFIG
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = str((BASE_DIR.parent / 'face_recognition' / 'dataset_neo'))
PKL_PATH = str((BASE_DIR / 'embeddings_dlib.pkl'))

# Validate training directory
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}.\nPlease ensure the folder exists or update TRAIN_DIR in the script.")

def train():
    known_encodings = {}  # Format: {'Name': [emb1, emb2...]}

    print("Starting Dlib training...")
    
    for person_name in os.listdir(TRAIN_DIR):
        person_dir = os.path.join(TRAIN_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        known_encodings[person_name] = []
        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Load image using face_recognition library
            image = face_recognition.load_image_file(img_path)
            
            # Detect faces and get encodings
            # model='hog' is faster, 'cnn' is more accurate but requires GPU
            encodings = face_recognition.face_encodings(image, model='hog')

            if len(encodings) > 0:
                known_encodings[person_name].append(encodings[0])

    print(f"Saving embeddings to {PKL_PATH}...")
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(known_encodings, f)
    print("Done.")

if __name__ == "__main__":
    train()