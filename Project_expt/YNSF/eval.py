import cv2
import numpy as np
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
# FILE PATHS
PKL_PATH = os.path.join('models', 'face_encodings.pickle')
TEST_DIR = 'test_dataset'
YUNET_MODEL = os.path.join('models', 'face_detection_yunet_2023mar.onnx')
SFACE_MODEL = os.path.join('models', 'face_recognition_sface_2021dec.onnx')

# Threshold
COSINE_THRESHOLD = 0.40
# =================================================

def load_models():
    try:
        detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL, "", (320, 320), 0.9, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
        return detector, recognizer
    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}")
        exit()

def get_embedding(img, detector, recognizer):
    img_h, img_w, _ = img.shape
    detector.setInputSize((img_w, img_h))
    faces = detector.detect(img)
    
    if faces[1] is None:
        return None
    
    face_coords = faces[1][0]
    aligned_face = recognizer.alignCrop(img, face_coords)
    embedding = recognizer.feature(aligned_face)
    return embedding

def plot_metrics_chart(report_dict, labels):
    """
    Generates a grouped bar chart for Precision, Recall, and F1-Score.
    """
    # Extract data for plotting
    classes = [label for label in labels if label in report_dict]
    precision = [report_dict[c]['precision'] for c in classes]
    recall = [report_dict[c]['recall'] for c in classes]
    f1 = [report_dict[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))  # label locations
    width = 0.25  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#4c72b0')
    rects2 = ax.bar(x, recall, width, label='Recall', color='#55a868')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score', color='#c44e52')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_title('Model Performance Metrics by Person')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)  # Set y-axis limit slightly above 1
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    print("Saving metrics chart to 'metrics_chart.png'...")
    plt.savefig('metrics_chart.png')
    plt.show()

def evaluate():
    # --- LOAD EMBEDDINGS ---
    print(f"Loading embeddings from {PKL_PATH}...")
    if not os.path.exists(PKL_PATH):
        print(f"Error: {PKL_PATH} not found.")
        return

    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    known_names = []
    known_embeddings = []

    if isinstance(data, dict):
        for name, embs in data.items():
            if not isinstance(embs, list): embs = [embs]
            for emb in embs:
                emb_np = np.array(emb, dtype=np.float32)
                if len(emb_np.shape) == 1: emb_np = emb_np.reshape(1, -1)
                known_names.append(name)
                known_embeddings.append(emb_np)
    else:
        print("Error: PKL file structure is not a dictionary.")
        return

    print(f"Loaded {len(known_embeddings)} embeddings.")

    # --- EVALUATION LOOP ---
    detector, recognizer = load_models()
    y_true = []
    y_pred = []

    if not os.path.exists(TEST_DIR):
        print(f"Error: {TEST_DIR} not found.")
        return

    print("Starting evaluation...")
    for person_name in sorted(os.listdir(TEST_DIR)):
        person_dir = os.path.join(TEST_DIR, person_name)
        if not os.path.isdir(person_dir): continue

        for img_name in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, img_name))
            if img is None: continue

            test_emb = get_embedding(img, detector, recognizer)
            
            if test_emb is None:
                y_true.append(person_name)
                y_pred.append("No_Face_Detected")
                continue
            
            if len(test_emb.shape) == 1: test_emb = test_emb.reshape(1, -1)

            max_score = 0
            best_match_name = "Unknown"

            for i, known_emb in enumerate(known_embeddings):
                score = recognizer.match(test_emb, known_emb, cv2.FaceRecognizerSF_FR_COSINE)
                if score > max_score:
                    max_score = score
                    best_match_name = known_names[i]

            final_prediction = "Unknown" if max_score < COSINE_THRESHOLD else best_match_name
            y_true.append(person_name)
            y_pred.append(final_prediction)

    # --- METRICS & PLOTTING ---
    if not y_true:
        print("No images evaluated.")
        return

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    labels = sorted(list(set(y_true + y_pred)))
    
    # Get report as a dictionary so we can access the numbers for plotting
    report_dict = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # 1. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax1)
    ax1.set_title(f"Confusion Matrix (Threshold: {COSINE_THRESHOLD})")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show() # Show first plot

    # 2. Plot Metrics Bar Chart
    print("Generating Metrics Chart...")
    # Filter out 'accuracy', 'macro avg', 'weighted avg' to plot only classes
    class_labels = [l for l in labels if l in report_dict] 
    plot_metrics_chart(report_dict, class_labels)

if __name__ == "__main__":
    evaluate()