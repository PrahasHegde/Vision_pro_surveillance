import cv2
import os
import time

# -------- CONFIG --------
PERSON_NAME = "Ronaldo"     # change for each person
NUM_IMAGES = 30
SAVE_DIR = "test_dataset"
CAMERA_ID = 0
FACE_SIZE = (112, 112)
# ------------------------

# Create directory
person_dir = os.path.join(SAVE_DIR, PERSON_NAME)
os.makedirs(person_dir, exist_ok=True)

# Load YuNet face detector
detector = cv2.FaceDetectorYN.create(
    "models\\face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

cap = cv2.VideoCapture(CAMERA_ID)
count = 0

print(f"[INFO] Capturing {NUM_IMAGES} images for {PERSON_NAME}")
print("[INFO] Press 'q' to quit early")

while cap.isOpened() and count < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        x, y, bw, bh = faces[0][:4].astype(int)

        # safety clipping
        x, y = max(0, x), max(0, y)
        face = frame[y:y+bh, x:x+bw]

        if face.size > 0:
            face = cv2.resize(face, FACE_SIZE)

            img_path = os.path.join(person_dir, f"{count:03d}.jpg")
            cv2.imwrite(img_path, face)
            count += 1

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {count}/{NUM_IMAGES}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            time.sleep(0.3)  # delay for variation

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("[INFO] Capture complete")
