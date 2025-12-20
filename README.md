# Face_Recognition_YuNet_SFace

Simple real-time face detection + recognition pipeline using YuNet (detector) and SFace (embedding) in ONNX format.

## Requirements
- Python 3.8+
- pip packages: opencv-python, onnxruntime, numpy, pillow, scipy, scikit-learn, tqdm

## Setup
1. Clone:
   git clone https://github.com/PrahasHegde/Face_Recognition_YuNet_SFace.git
2. Create venv (optional) and install:
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt

## Models
Place ONNX models in the models/ folder:
- models/yunet.onnx  # YuNet face detector
- models/sface.onnx  # SFace (or other face embedding model)

## Project layout (recommended)
- models/ — ONNX model files
- data/known/<person_name>/ — images for enrollment
- data/unknown/ — test images
- embeddings/ — saved embeddings / centroids
- scripts/ — detect.py, extract_embeddings.py, enroll.py, recognize.py

## Quick usage
- Detect faces:
  python scripts/detect.py --input path/to/image.jpg --model models/yunet.onnx --out results/
- Extract embeddings:
  python scripts/extract_embeddings.py --input data/known --embedder models/sface.onnx --out embeddings/enrolled_embeddings.npy
- Create centroids (enroll):
  python scripts/enroll.py --embeddings embeddings/enrolled_embeddings.npy --out embeddings/centroids.npy
- Recognize (image or webcam):
  python scripts/recognize.py --detector models/yunet.onnx --embedder models/sface.onnx --db embeddings/centroids.npy --threshold 0.4

## Notes
- Match preprocessing (input size, normalization, channel order) to each model.
- Normalize embeddings (L2) and use cosine similarity for matching.
- Thresholds are model-dependent — validate on a holdout set.

## License
MIT — add full LICENSE file if desired.

If you want, I can generate simple starter scripts (detect.py, extract_embeddings.py, enroll.py, recognize.py) to go with this README.
