import cv2
import os
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(SCRIPT_DIR, 'embeddings_insight.pkl')

if os.path.exists(os.path.join(SCRIPT_DIR, 'test_dataset')):
    TEST_DIR = os.path.join(SCRIPT_DIR, 'test_dataset')
elif os.path.exists(os.path.join(os.path.dirname(SCRIPT_DIR), 'test_dataset')):
    TEST_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'test_dataset')
else:
    print("CRITICAL ERROR: 'test_dataset' folder not found.")
    exit()

# Threshold
COSINE_THRESHOLD = 0.40 
# =================================================

def compute_cosine_similarity(embed1, embed2):
    embed1 = embed1.flatten()
    embed2 = embed2.flatten()
    norm1 = np.linalg.norm(embed1)
    norm2 = np.linalg.norm(embed2)
    return np.dot(embed1, embed2) / (norm1 * norm2)

def evaluate():
    if not os.path.exists(PKL_PATH):
        print("Pickle file missing.")
        return

    print("Loading Embeddings...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    known_names = []
    known_embeddings = []
    
    # Load data
    if isinstance(data, dict):
        for name, embs in data.items():
            if not isinstance(embs, list): embs = [embs]
            for emb in embs:
                known_names.append(name)
                known_embeddings.append(emb)
    elif isinstance(data, list):
         for item in data:
            if len(item) == 2:
                known_names.append(item[0])
                known_embeddings.append(item[1])

    print(f"Loaded {len(known_embeddings)} known faces.")

    # 1. INITIALIZE APP TO GET MODELS
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 2. EXTRACT JUST THE RECOGNITION MODEL (ArcFace)
    # We loop through models to find the one that handles recognition
    rec_model = None
    for model in app.models.values():
        # Look for the model that outputs 512-dim embeddings
        if hasattr(model, 'input_shape') and model.input_shape[1] == 112:
            rec_model = model
            break
            
    if rec_model is None:
        print("Error: Could not find recognition model inside InsightFace.")
        # Fallback: usually the last model or specifically named
        rec_model = app.models['recognition']

    print("Recognition Model Found. Bypassing Detector...")

    y_true = []
    y_pred = []
    
    print(f"\n{'TRUE LABEL':<15} | {'PREDICTED':<15} | {'SCORE':<6} | {'RESULT'}")
    print("-" * 65)

    for person_name in sorted(os.listdir(TEST_DIR)):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        clean_true_name = person_name.strip().lower()
        count = 0
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            # Load Image (BGR)
            img = cv2.imread(img_path) 
            if img is None: continue

            # --- CRITICAL FIX: DIRECT INFERENCE ---
            # We skip app.get(img) and go straight to embedding
            # Note: rec_model.get_feat(img) handles the 112x112 input
            try:
                test_emb = rec_model.get_feat(img).flatten()
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

            # Match Embedding
            max_score = -1
            best_match = "unknown"

            for i, known_emb in enumerate(known_embeddings):
                score = compute_cosine_similarity(test_emb, known_emb)
                if score > max_score:
                    max_score = score
                    best_match = known_names[i].strip().lower()

            final_pred = "unknown"
            if max_score >= COSINE_THRESHOLD:
                final_pred = best_match

            y_true.append(clean_true_name)
            y_pred.append(final_pred)

            if count < 3:
                symbol = "✅" if clean_true_name == final_pred else "❌"
                if clean_true_name in ["messi", "ronaldo"] and final_pred == "unknown":
                    symbol = "✅ (Reject)"
                print(f"{clean_true_name:<15} | {final_pred:<15} | {max_score:.2f}   | {symbol}")
                count += 1

    # Accuracy
    if not y_true: return
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate Metrics
    tp = 0; tn = 0; fp = 0; fn = 0
    known_classes = set([n.lower() for n in known_names])
    
    for t, p in zip(y_true, y_pred):
        if t in known_classes:
            if p == t: tp += 1
            else: fn += 1
        else:
            if p == "unknown": tn += 1
            else: fp += 1
            
    far = (fp / (tn + fp) * 100) if (tn + fp) > 0 else 0
    frr = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0

    print("\n" + "="*50)
    print(f"FINAL DIRECT EVALUATION")
    print("="*50)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"FAR:      {far:.2f}%")
    print(f"FRR:      {frr:.2f}%")
    print("="*50)
    
    # Detailed Report
    labels = sorted(list(set(y_true + y_pred)))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    # Plot
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Greens, xticks_rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()