import face_recognition
import cv2
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(SCRIPT_DIR, 'embeddings_dlib.pkl')

# ROBUST PATH FINDING
path_a = os.path.join(SCRIPT_DIR, 'test_dataset')
path_b = os.path.join(os.path.dirname(SCRIPT_DIR), 'test_dataset')

if os.path.exists(path_a):
    TEST_DIR = path_a
elif os.path.exists(path_b):
    TEST_DIR = path_b
else:
    print("CRITICAL ERROR: 'test_dataset' not found.")
    exit()

# Threshold (Dlib Default: 0.6. Lower is stricter.)
TOLERANCE = 0.45 
# =================================================

def calculate_biometric_metrics(y_true, y_pred, known_classes):
    """
    Calculates FAR, FRR, and Accuracy based on Genuine vs Impostor attempts.
    """
    tp = 0  # True Positive: Known User -> Correctly Identified
    tn = 0  # True Negative: Impostor -> Correctly Rejected (Unknown)
    fp = 0  # False Positive: Impostor -> Incorrectly Identified as User
    fn = 0  # False Negative: Known User -> Incorrectly Rejected (Unknown)
    wa = 0  # Wrong Accept: Known User A -> Incorrectly Identified as User B

    # Convert known_classes to a set for fast lookup
    known_set = set(known_classes)

    for true_label, pred_label in zip(y_true, y_pred):
        # 1. ANALYZE GENUINE ATTEMPTS (User is in the database)
        if true_label in known_set:
            if pred_label == true_label:
                tp += 1
            elif pred_label == "Unknown":
                fn += 1  # False Rejection
            else:
                wa += 1  # Wrong Acceptance (Misidentification)

        # 2. ANALYZE IMPOSTOR ATTEMPTS (User is NOT in database, e.g., Messi)
        else:
            if pred_label == "Unknown":
                tn += 1  # True Rejection
            else:
                fp += 1  # False Acceptance (Security Breach!)

    # Calculate Rates
    total_genuine = tp + fn + wa
    total_impostor = tn + fp

    # Avoid division by zero
    frr = (fn / total_genuine) * 100 if total_genuine > 0 else 0
    far = (fp / total_impostor) * 100 if total_impostor > 0 else 0
    accuracy = ((tp + tn) / (total_genuine + total_impostor)) * 100 if (total_genuine + total_impostor) > 0 else 0
    
    return {
        "FAR": far,
        "FRR": frr,
        "Accuracy": accuracy,
        "Counts": {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "WA": wa}
    }

def plot_biometric_chart(metrics):
    """Plots FAR vs FRR vs Accuracy"""
    labels = ['FAR (Security Risk)', 'FRR (Inconvenience)', 'Overall Accuracy']
    values = [metrics['FAR'], metrics['FRR'], metrics['Accuracy']]
    colors = ['#d62728', '#ff7f0e', '#2ca02c'] # Red, Orange, Green

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors)

    # Add text on top
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'Biometric Performance Metrics\n(Threshold: {TOLERANCE})')
    
    print("Saving biometric chart to 'biometric_metrics.png'...")
    plt.tight_layout()
    plt.savefig('biometric_metrics.png')
    plt.show()

def evaluate():
    # --- LOAD EMBEDDINGS ---
    if not os.path.exists(PKL_PATH):
        print("Error: Pickle file not found.")
        return

    print(f"Loading Dlib embeddings from: {PKL_PATH}")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    # Extract Known Names
    known_names = []
    known_encodings = []
    
    # Support both Dict and List formats
    if isinstance(data, dict):
        for name, embs in data.items():
            if not isinstance(embs, list): embs = [embs]
            for emb in embs:
                known_names.append(name)
                known_encodings.append(emb)
    elif isinstance(data, list):
        for item in data:
            if len(item) == 2:
                known_names.append(item[0])
                known_encodings.append(item[1])

    # Unique list of people the model actually knows
    known_classes = list(set(known_names))
    print(f"Model knows these people: {known_classes}")

    # --- EVALUATION LOOP ---
    y_true = []
    y_pred = []

    print(f"Starting evaluation on: {TEST_DIR}")
    for person_name in sorted(os.listdir(TEST_DIR)):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            try:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
            except:
                continue

            if len(encodings) == 0:
                y_true.append(person_name)
                y_pred.append("No_Face_Detected")
                continue
            
            test_emb = encodings[0]
            distances = face_recognition.face_distance(known_encodings, test_emb)
            
            if len(distances) == 0:
                y_pred.append("Unknown")
                y_true.append(person_name)
                continue

            best_match_index = np.argmin(distances)
            min_distance = distances[best_match_index]

            if min_distance <= TOLERANCE:
                y_pred.append(known_names[best_match_index])
            else:
                y_pred.append("Unknown")
            
            y_true.append(person_name)

    # --- CALCULATE & DISPLAY METRICS ---
    if not y_true:
        print("No data to evaluate.")
        return

    bio_metrics = calculate_biometric_metrics(y_true, y_pred, known_classes)
    
    print("\n" + "="*50)
    print("BIOMETRIC PERFORMANCE REPORT")
    print("="*50)
    print(f"False Acceptance Rate (FAR): {bio_metrics['FAR']:.2f}%  <-- Security Risk (Impostors accepted)")
    print(f"False Rejection Rate  (FRR): {bio_metrics['FRR']:.2f}%  <-- Inconvenience (Users rejected)")
    print(f"Overall Accuracy:            {bio_metrics['Accuracy']:.2f}%")
    print("-" * 50)
    print(f"True Positives (Correct):    {bio_metrics['Counts']['TP']}")
    print(f"True Negatives (Rejections): {bio_metrics['Counts']['TN']}")
    print(f"False Positives (Breach):    {bio_metrics['Counts']['FP']}")
    print(f"False Negatives (Missed):    {bio_metrics['Counts']['FN']}")
    print(f"Misidentifications:          {bio_metrics['Counts']['WA']}")
    print("="*50)

    # Plot
    plot_biometric_chart(bio_metrics)

if __name__ == "__main__":
    evaluate()