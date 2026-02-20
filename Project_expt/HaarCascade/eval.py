import cv2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ================= CONFIGURATION =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'model_haar_lbph.yml')
LABEL_PATH = os.path.join(SCRIPT_DIR, 'labels_haar.pkl')

if os.path.exists(os.path.join(SCRIPT_DIR, 'test_dataset')):
    TEST_DIR = os.path.join(SCRIPT_DIR, 'test_dataset')
elif os.path.exists(os.path.join(os.path.dirname(SCRIPT_DIR), 'test_dataset')):
    TEST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'test_dataset')
else:
    print("CRITICAL ERROR: 'test_dataset' folder not found.")
    exit()

# 500 forces a match. We WANT matches, even if weak.
CONFIDENCE_THRESHOLD = 500
# =================================================

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_haar.py first.")
        return

    # Use default Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    
    with open(LABEL_PATH, 'rb') as f:
        label_map = pickle.load(f)

    # Normalize map for comparison
    id_to_name = {k: v.strip().lower() for k, v in label_map.items()}
    known_names = list(id_to_name.values())

    print(f"Evaluating with LOOSE DETECTOR parameters...")

    y_true = []
    y_pred = []
    
    for person_name in sorted(os.listdir(TEST_DIR)):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        clean_true_name = person_name.strip().lower()

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            # --- KEY CHANGE: RELAXED DETECTION ---
            # scaleFactor 1.05 = Scans image more thoroughly (slower but finds more)
            # minNeighbors 3 = Accepts "weaker" face candidates (Standard is 5)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)

            if len(faces) == 0:
                y_true.append(clean_true_name)
                y_pred.append("no_face")
                continue

            # Take largest face
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            (x, y, w, h) = faces[0]
            roi = img[y:y+h, x:x+w]
            roi = cv2.equalizeHist(roi)

            label_id, distance = recognizer.predict(roi)

            # FORCE GUESS
            if distance < CONFIDENCE_THRESHOLD:
                if label_id in label_map:
                    pred_name = label_map[label_id].strip().lower()
                    y_pred.append(pred_name)
                else:
                    y_pred.append("unknown")
            else:
                y_pred.append("unknown")

            y_true.append(clean_true_name)

    # Calculate Accuracy
    if not y_true: return
    
    # Filter out "no_face" for accuracy calculation to see 'Raw Model Performance'
    # valid_preds = [(t, p) for t, p in zip(y_true, y_pred) if p != 'no_face']
    # if valid_preds:
    #     vt, vp = zip(*valid_preds)
    #     raw_acc = accuracy_score(vt, vp)
    # else: 
    #     raw_acc = 0.0

    acc = accuracy_score(y_true, y_pred)
    labels = sorted(list(set(y_true + y_pred)))

    print("\n" + "="*50)
    print("FINAL HAAR RESULTS")
    print("="*50)
    print(f"Accuracy: {acc*100:.2f}%")
    print("="*50)
    
    # Calculate FAR/FRR
    tp = 0; tn = 0; fp = 0; fn = 0
    # Impostors are those NOT in the training map
    impostors = ["messi", "ronaldo"] 
    
    for t, p in zip(y_true, y_pred):
        if t in impostors:
            if p in known_names: fp += 1 # Security Risk
            else: tn += 1 # Correct Reject
        else:
            if p == t: tp += 1
            else: fn += 1 # Miss/Reject

    total_imp = tn + fp
    total_gen = tp + fn
    far = (fp / total_imp * 100) if total_imp > 0 else 0
    frr = (fn / total_gen * 100) if total_gen > 0 else 0

    print(f"FAR: {far:.2f}% (Risk)")
    print(f"FRR: {frr:.2f}% (Rejection)")
    print("-" * 50)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # Plot
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Oranges, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig('haar_confusion_matrix_loose.png')
    plt.show()

if __name__ == "__main__":
    evaluate()