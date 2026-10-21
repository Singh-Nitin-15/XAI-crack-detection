"""FastAPI application with prediction endpoints."""
import os
import sys
import uuid
import socket
import tempfile
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import Prediction, init_db, get_db
from backend.storage import get_storage, LocalStorage, MinIOStorage
from ai.explain import predict_with_heatmap

app = FastAPI(title="XAI Crack Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = get_storage()

TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())
os.makedirs(TEMP_DIR, exist_ok=True)

if isinstance(storage, LocalStorage):
    UPLOAD_DIR = os.path.join(str(storage.base_dir), "uploads")
    HEATMAP_DIR = os.path.join(str(storage.base_dir), "heatmaps")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)

init_db()


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/whoami")
def whoami():
    return {"hostname": socket.gethostname()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = Path(file.filename or "image.jpg").suffix or ".jpg"
    uid = str(uuid.uuid4())
    upload_name = f"{uid}{ext}"
    heatmap_name = f"{uid}_heatmap.jpg"
    image_key = f"uploads/{upload_name}"
    heatmap_key = f"heatmaps/{heatmap_name}"

    tmp_upload = os.path.join(TEMP_DIR, upload_name)
    tmp_heatmap = os.path.join(TEMP_DIR, heatmap_name)

    content = await file.read()
    with open(tmp_upload, "wb") as f:
        f.write(content)

    try:
        label, confidence, _ = predict_with_heatmap(tmp_upload, tmp_heatmap)
    except Exception as e:
        for p in [tmp_upload, tmp_heatmap]:
            if os.path.exists(p):
                os.unlink(p)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    storage.save(tmp_upload, image_key)
    storage.save(tmp_heatmap, heatmap_key)

    for p in [tmp_upload, tmp_heatmap]:
        if os.path.exists(p):
            os.unlink(p)

    record = Prediction(
        filename=file.filename or "unknown",
        label=label,
        confidence=round(confidence, 4),
        image_path=image_key,
        heatmap_path=heatmap_key,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": record.filename,
        "label": record.label,
        "confidence": record.confidence,
        "image_url": storage.get_url(image_key),
        "heatmap_url": storage.get_url(heatmap_key),
        "created_at": record.created_at.isoformat(),
    }


@app.get("/history")
def history(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(Prediction).order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "label": r.label,
            "confidence": r.confidence,
            "image_url": storage.get_url(r.image_path),
            "heatmap_url": storage.get_url(r.heatmap_path),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in records
    ]


@app.get("/files/{file_path:path}")
def serve_file(file_path: str):
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    if isinstance(storage, LocalStorage):
        full_path = storage.get_full_path(file_path)
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(str(full_path), media_type=content_type)

    elif isinstance(storage, MinIOStorage):
        try:
            response = storage.get_object(file_path)
            data = BytesIO(response.read())
            response.close()
            response.release_conn()
            return StreamingResponse(data, media_type=content_type)
        except Exception:
            raise HTTPException(status_code=404, detail="File not found in storage")

    raise HTTPException(status_code=500, detail="Unknown storage backend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
