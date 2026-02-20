from pathlib import Path
import face_recognition
import pickle

# Paths for dataset and output
DATASET_DIR = Path('dataset')
ENCODINGS_PATH = Path('encodings.pkl')

known_encodings = []
known_names = []

for person_dir in DATASET_DIR.iterdir():
    if not person_dir.is_dir():
        continue
    name = person_dir.name
    for img_path in person_dir.glob('*'):
        # Load image and detect face
        image = face_recognition.load_image_file(img_path)
        boxes = face_recognition.face_locations(image, model='hog')
        encodings = face_recognition.face_encodings(image, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

# Save encodings to disk
with open(ENCODINGS_PATH, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
