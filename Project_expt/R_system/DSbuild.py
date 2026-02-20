import cv2
import os
import time
import numpy as np
from insightface.app import FaceAnalysis

# ============================
# CONFIGURATION
# ============================
DATASET_DIR = "dataset_neo"
PERSON_NAME = input("Enter the person's name: ").strip()
NUM_IMAGES = 200
DELAY_BETWEEN = 0.25  # seconds between captures
MODEL_NAME = "buffalo_l"


# Pose & Expression instructions
instructions = [
    "Look straight",
    "Look left",
    "Look right",
    "Look up",
    "Look down",
    "Smile 😊",
    "Angry 😠",
    "Neutral 😐",
    "Tilt head left ↖️",
    "Tilt head right ↗️"
]

# ============================
# SETUP
# ============================
person_path = os.path.join(DATASET_DIR, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)

# Initialize InsightFace detector (SCRFD + ArcFace)
app = FaceAnalysis(name=MODEL_NAME, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))  # Reduced detection size for better performance

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not found.")
    exit()

print(f"[INFO] Starting capture for {PERSON_NAME}...")

# ============================
# HELPER FUNCTIONS
# ============================
def is_blurry(image, threshold=50.0):  # Reduced threshold
    """Return True if image is blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold

def get_padded_face(frame, face, padding_percent=40):
    """Get face region with padding around it."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    
    # Calculate padding
    width = x2 - x1
    height = y2 - y1
    padding_x = int(width * (padding_percent/100))
    padding_y = int(height * (padding_percent/100))
    
    # Apply padding with bounds checking
    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(w, x2 + padding_x)
    y2 = min(h, y2 + padding_y)
    
    return frame[y1:y2, x1:x2]

# ============================
# CAPTURE LOOP
# ============================
count = 0
inst_index = 0
total_instructions = len(instructions)
last_prompt_time = time.time()

while count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    original_frame = frame.copy()
    faces = app.get(frame, max_num=1)

    # Update instructions
    if time.time() - last_prompt_time > 6 and inst_index < total_instructions - 1:
        inst_index += 1
        last_prompt_time = time.time()

    instruction_text = instructions[inst_index]
    
    # Draw guide rectangle
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    rect_size = min(w, h) // 1.5
    guide_x1 = int(center_x - rect_size//2)
    guide_y1 = int(center_y - rect_size//2)
    guide_x2 = int(center_x + rect_size//2)
    guide_y2 = int(center_y + rect_size//2)
    cv2.rectangle(frame, (guide_x1, guide_y1), (guide_x2, guide_y2), (255, 255, 255), 2)

    # Draw instructions
    cv2.putText(frame, f"Instruction: {instruction_text}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Images Captured: {count}/{NUM_IMAGES}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    if faces:
        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Simplified capture logic - just check if face is detected
        face_crop = get_padded_face(original_frame, face, padding_percent=50)
        
        if face_crop.size > 0:  # If face crop is valid
            filename = os.path.join(person_path, f"{PERSON_NAME}_{count+1:03d}.jpg")
            cv2.imwrite(filename, face_crop)
            count += 1
            print(f"[INFO] Saved image {count}/{NUM_IMAGES}")
            
            # Add debug information
            print(f"Face detected at: ({x1},{y1}) to ({x2},{y2})")
            
            # Reduced delay between captures
            time.sleep(0.1)
    else:
        cv2.putText(frame, "No face detected!", (30, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Dataset Capture", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Done! {count} high-quality images saved for {PERSON_NAME} at '{person_path}'")
