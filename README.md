# YOLOv11 Concrete-Defect Segmentation API

FastAPI service that serves a YOLOv11 model for 8 concrete defect classes:
- Crack, ACrack, Efflorescence, WConccor, Spalling, Wetspot, Rust, ExposedRebars.

## Features
- **/predict/image** → upload image, get annotated image + counts
- **/predict/video** → upload video, get annotated MP4 + *_counts.txt
- Polygons/contours only (no shaded masks, no boxes, no confidences)
- Dockerfile included for deployment

## Quick Start (local)
```bash
# clone
git clone https://github.com/natnaeltaye/yolo11-fastapi-app.git
cd yolo11-fastapi-app

# create venv & install
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# set model URL
export MODEL_URL="https://<your-hosted-model>/best.onnx"

# run API
uvicorn app.main:app --host 0.0.0.0 --port 10000 --reload
