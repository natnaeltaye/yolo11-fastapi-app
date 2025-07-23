from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pathlib import Path
from app.utils import load_model, predict_image, predict_video
import tempfile, shutil
import io

app = FastAPI(
    title="YOLOv11 Concrete‑Defect Segmentation API",
    summary="Predicts masks on images & videos.",
)

MODEL = load_model(Path(__file__).parent / "models" / "best.onnx")

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>Concrete Defect API is healthy ✅ – go to /docs to try it</h2>"

@app.post("/predict/image")
async def segment_image(file: UploadFile = File(...)):
    raw = await file.read()
    result_bytes, saved_path = predict_image(MODEL, raw, file.filename)
    print(f"[IMAGE] Saved to {saved_path}")
    return StreamingResponse(io.BytesIO(result_bytes), media_type="image/jpeg")

@app.post("/predict/video")
async def segment_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    video_path = predict_video(MODEL, tmp_path)
    tmp_path.unlink(missing_ok=True)

    print(f"[VIDEO] Saved to {video_path}")
    print(f"[COUNTS] saved to outputs\\{Path(video_path).stem}_counts.txt")
    
    return JSONResponse({
        "video": str(video_path),
        "counts": f"outputs\\{Path(video_path).stem}_counts.txt"
    })
