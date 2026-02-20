# ===========================================================
# recognize_face.py
# ===========================================================

import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# ============================
# CONFIGURATION
# ============================
EMBEDDINGS_FILE = "embeddings2.pkl"
MODEL_NAME = "buffalo_l"
THRESHOLD = 0.5  # adjust for strictness (0.55–0.7 good range)
DETECTION_SIZE = (640, 640)

# ============================
# LOAD MODEL & EMBEDDINGS
# ============================
print("[INFO] Loading model and embeddings...")
app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings, labels = pickle.load(f)

embeddings = np.array(embeddings)
# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# ============================
# RECOGNITION LOOP
# ============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not found.")
    exit()

print("[INFO] Starting real-time face recognition...")
print("[INFO] Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Compute cosine similarity
        sims = np.dot(embeddings, embedding)
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        name = labels[best_idx] if best_score > THRESHOLD else "Unknown"

        # Draw detection box and name
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{name} ({best_score:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        print(f"[DEBUG] Detected: {name}, Score: {best_score:.3f}")

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Recognition stopped.")
