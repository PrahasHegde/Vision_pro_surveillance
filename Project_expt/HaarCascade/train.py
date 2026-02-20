import cv2
import os
import pickle
import numpy as np

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'dataset_neo')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_haar_lbph.yml')
LABEL_PATH = os.path.join(SCRIPT_DIR, 'labels_haar.pkl')

def train():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # --- TUNING FOR 40% ACCURACY ---
    # radius=2: Looks at pixels 2 steps away (catches bigger features like nose shape)
    # neighbors=8: Standard detail level
    # grid_x=8, grid_y=8: High grid resolution (Strict matching)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)

    faces = []
    ids = []
    label_map = {} 
    current_id = 0

    print(f"Training Tuned Haar+LBPH on {TRAIN_DIR}...")
    
    for person_name in sorted(os.listdir(TRAIN_DIR)):
        person_path = os.path.join(TRAIN_DIR, person_name)
        if not os.path.isdir(person_path): continue

        # Fix names (Prahas vs prahas)
        clean_name = person_name.strip().title()
        
        label_map[current_id] = clean_name
        print(f"Processing {clean_name} (ID: {current_id})...")

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # Detect Face
            faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rects:
                roi = img[y:y+h, x:x+w]
                # Light Correction
                roi = cv2.equalizeHist(roi)
                faces.append(roi)
                ids.append(current_id)

        current_id += 1

    print(f"Training model on {len(faces)} samples...")
    recognizer.train(faces, np.array(ids))

    recognizer.write(MODEL_PATH)
    with open(LABEL_PATH, 'wb') as f:
        pickle.dump(label_map, f)
    
    print("Training Complete. Now run the optimizer script.")

if __name__ == "__main__":
    train()