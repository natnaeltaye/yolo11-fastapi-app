import requests
import os
import cv2
import numpy as np
import onnxruntime
from pathlib import Path
from collections import defaultdict
from datetime import datetime

IM_SIZE = 640
CLASSES = {
    0: "Crack", 1: "ACrack", 2: "Efflorescence", 3: "WConccor",
    4: "Spalling", 5: "Wetspot", 6: "Rust", 7: "ExposedRebars"
}

COLORS = {
    0: (255, 0, 0),       # Red
    1: (0, 255, 0),       # Green
    2: (0, 0, 255),       # Blue
    3: (0, 255, 255),     # Yellow
    4: (255, 0, 255),     # Magenta
    5: (255, 255, 0),     # Cyan
    6: (128, 0, 255),     # Purple
    7: (0, 128, 255),     # Orange
}

def load_model(model_path):
    url = "https://drive.google.com/uc?export=download&id=13xdczmFe4pnw8r8IKm2EITjA1oyZKdXK"
    if not model_path.exists():
        print("Downloading model from", url)
        response = requests.get(url)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return onnxruntime.InferenceSession(str(model_path))


def draw_polygon_and_labels(image, masks, classes):
    defect_counts = defaultdict(int)
    for mask, cls_id in zip(masks, classes):
        name = CLASSES.get(cls_id, f"class_{cls_id}")
        color = COLORS.get(cls_id, (0, 255, 0))
        defect_counts[name] += 1

        # Get contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) > 2:
                cv2.drawContours(image, [cnt], -1, color, 2)
                x, y = cnt[0][0]
                cv2.putText(
                    image, name,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2, cv2.LINE_AA
                )

    y0 = 25
    for i, (k, v) in enumerate(defect_counts.items()):
        text = f"{k}: {v}"
        cv2.putText(image, text, (10, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return image, defect_counts

def predict_image(model: YOLO, file_bytes: bytes, filename: str) -> tuple[bytes, Path]:
    img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    orig_h, orig_w = img.shape[:2]

    results = model.predict(img, imgsz=IM_SIZE, conf=0.25, verbose=False)
    res = results[0]
    masks = res.masks.data.cpu().numpy() if res.masks is not None else []
    classes = res.boxes.cls.int().tolist() if res.boxes is not None else []

    masks_resized = [cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) for m in masks]

    img, defect_counts = draw_polygon_and_labels(img, masks_resized, classes)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{Path(filename).stem}_{now}.jpg"
    cv2.imwrite(str(out_path), img)

    count_path = out_dir / f"{Path(filename).stem}_{now}_counts.txt"
    with open(count_path, "w") as f:
        for k, v in defect_counts.items():
            f.write(f"{k}: {v}\n")

    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes(), out_path

# ------------------------------------------------------------------
# helper: IoU between two binary masks
def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union else 0.0


def predict_video(model: YOLO, file_path: str | Path) -> Path:
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
    video_out = out_dir / f"{file_path.stem}_{now}.mp4"
    txt_out   = out_dir / f"{file_path.stem}_{now}_counts.txt"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (w, h))

    # IoU‑based deduplication
    stored_masks = defaultdict(list)   # {class_name: [binary_mask, …]}
    final_counts = defaultdict(int)
    IOU_THRESH   = 0.2               # change if needed

    POLY_THICK = 3
    FONT_SCALE = 0.9
    FONT_THICK = 2

    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        print(f"Processing frame {frame_idx}")

        res = model.predict(frame, imgsz=IM_SIZE, conf=0.25, verbose=False)[0]
        masks   = res.masks.data.cpu().numpy() if res.masks is not None else []
        classes = res.boxes.cls.int().tolist()    if res.boxes is not None else []

        masks = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks]

        for mask, cls_id in zip(masks, classes):
            label = CLASSES.get(cls_id, f"class_{cls_id}")
            color = COLORS.get(cls_id, (0, 255, 0))

            # --- IoU deduplication ---
            duplicate = False
            for stored in stored_masks[label]:
                if mask_iou(mask, stored) > IOU_THRESH:
                    duplicate = True
                    break
            if not duplicate:
                stored_masks[label].append(mask.astype(np.uint8))
                final_counts[label] += 1

            # draw mask polygon for visualization
            cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if len(cnt) < 3:
                    continue
                cv2.polylines(frame, [cnt], True, color, POLY_THICK)
                x, y, _, _ = cv2.boundingRect(cnt)
                cv2.putText(frame, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                            color, FONT_THICK, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()

    with open(txt_out, "w") as f:
        for cls, cnt in sorted(final_counts.items()):
            f.write(f"{cls}: {cnt}\n")

    print(f"[VIDEO] saved to {video_out}")
    print(f"[COUNTS] saved to {txt_out}")
    return video_out
