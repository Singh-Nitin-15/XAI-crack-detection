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
from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError
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

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())
os.makedirs(TEMP_DIR, exist_ok=True)

if isinstance(storage, LocalStorage):
    UPLOAD_DIR = os.path.join(str(storage.base_dir), "uploads")
    HEATMAP_DIR = os.path.join(str(storage.base_dir), "heatmaps")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)

init_db()


def _iso_utc(dt):
    """Timestamps are stored as UTC, but the DB column is timezone-naive, so
    values come back without an offset and browsers would read them as local
    time. Attach UTC explicitly so the ISO string ends in +00:00."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


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

    # Only keep known image extensions: the stored file is later served with a
    # content type guessed from its extension (e.g. ".html" would be served as
    # a web page).
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    uid = str(uuid.uuid4())
    upload_name = f"{uid}{ext}"
    heatmap_name = f"{uid}_heatmap.jpg"
    image_key = f"uploads/{upload_name}"
    heatmap_key = f"heatmaps/{heatmap_name}"

    tmp_upload = os.path.join(TEMP_DIR, upload_name)
    tmp_heatmap = os.path.join(TEMP_DIR, heatmap_name)

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB} MB)")
    with open(tmp_upload, "wb") as f:
        f.write(content)

    try:
        # Model inference is CPU-heavy; run it in a worker thread so the event
        # loop keeps serving /health and /history meanwhile.
        label, confidence, _ = await run_in_threadpool(predict_with_heatmap, tmp_upload, tmp_heatmap)
    except UnidentifiedImageError:
        for p in [tmp_upload, tmp_heatmap]:
            if os.path.exists(p):
                os.unlink(p)
        raise HTTPException(status_code=400, detail="File is not a valid image")
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
        "created_at": _iso_utc(record.created_at),
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
            "created_at": _iso_utc(r.created_at),
        }
        for r in records
    ]


@app.get("/files/{file_path:path}")
def serve_file(file_path: str):
    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

    if isinstance(storage, LocalStorage):
        try:
            full_path = storage.get_full_path(file_path)
        except ValueError:
            raise HTTPException(status_code=404, detail="File not found")
        if not full_path.is_file():
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
