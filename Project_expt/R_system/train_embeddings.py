# ===========================================================
# train_embeddings.py
# ===========================================================

import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# ============================
# CONFIGURATION
# ============================
DATASET_PATH = "dataset_neo"
EMBEDDINGS_FILE = "embeddings2.pkl"
MODEL_NAME = "buffalo_l"  # works well on CPU
DETECTION_SIZE = (640, 640)

# ============================
# INITIALIZE MODEL
# ============================
print("[INFO] Loading InsightFace model...")
app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

# ============================
# GENERATE EMBEDDINGS
# ============================
embeddings = []
labels = []

print("[INFO] Starting embedding generation...")

for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_path):
        continue

    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        embedding = faces[0].embedding
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        embeddings.append(embedding)
        labels.append(person_name)

print(f"[INFO] Generated {len(embeddings)} embeddings.")

# ============================
# SAVE EMBEDDINGS
# ============================
with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump((embeddings, labels), f)

print(f"[INFO] Embeddings saved to '{EMBEDDINGS_FILE}' successfully.")
