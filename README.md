# Explainable AI Crack Detection

Structural crack detection with explainable AI. Upload a photo of a concrete surface and get a crack / no-crack prediction, a confidence score, and a Grad-CAM heatmap showing which regions of the image drove the decision.

## Architecture

```
Browser → React frontend (nginx, :3000) → FastAPI backend (:8000) → ResNet50 + Grad-CAM
                                                  ↓
                                   PostgreSQL (history) + local file storage (images, heatmaps)
```

| Component  | Technology                                          |
|------------|-----------------------------------------------------|
| AI model   | PyTorch ResNet50, Grad-CAM                          |
| Backend    | FastAPI, SQLAlchemy, Uvicorn                        |
| Frontend   | React 19, Vite, served by nginx                     |
| Database   | PostgreSQL (Docker) / SQLite (local dev)            |
| Storage    | Local filesystem (Docker volume). MinIO is supported but optional. |
| Containers | Docker, Docker Compose                              |

## Project Structure

```
xai-crack-detection/
├── ai/
│   ├── explain.py            # Inference + Grad-CAM heatmap
│   ├── train.py              # ResNet50 transfer-learning script
│   ├── evaluate.py           # Test-set metrics (accuracy, precision, recall, F1)
│   ├── download_dataset.py   # Optional: fetch a 1,000-image sample from Kaggle
│   └── model.pth             # Trained weights (not in repo, ~90 MB)
├── backend/
│   ├── main.py               # FastAPI endpoints
│   ├── models.py             # SQLAlchemy models
│   ├── storage.py            # Storage abstraction (local / MinIO)
│   └── Dockerfile
├── frontend/
│   ├── src/App.jsx           # UI
│   ├── src/App.css, index.css
│   ├── nginx.conf
│   └── Dockerfile
├── tests/
│   └── smoke_test.py         # API smoke test
├── docker-compose.yml
├── requirements.txt          # Pinned Python dependencies
├── run.bat / run.sh          # Start the backend without Docker
├── .env.example
└── README.md
```

## Setup

### 1. Clone and add the model

```bash
git clone <repo-url>
cd xai-crack-detection
```

The trained weights are not stored in git. Place `model.pth` in `ai/`:

```
ai/model.pth    # ~90 MB ResNet50 weights
```

### 2. Run with Docker Compose

```bash
docker compose up --build -d
```

The first build takes 5–10 minutes (PyTorch is large). Then open:

| Service  | URL                         |
|----------|-----------------------------|
| Frontend | http://localhost:3000       |
| Backend  | http://localhost:8000       |
| API docs | http://localhost:8000/docs  |

Check that it is working:

```bash
curl http://localhost:8000/health
python tests/smoke_test.py
```

Prediction history and stored images live in Docker volumes and survive restarts. `docker compose down -v` deletes them.

### 3. Local development (without Docker)

```bash
pip install -r requirements.txt
# Backend (uses SQLite and ./storage by default)
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Frontend, in a separate terminal
cd frontend && npm install && npm run dev
```

## Using the app

1. Drop or choose an image: JPG, PNG, WebP or BMP, up to 20 MB.
2. Click **Run analysis**.
3. The result panel shows the classification, the confidence, and the input next to its Grad-CAM heatmap. Switch to **Overlay** to fade the heatmap over the photo with a slider.
4. Every prediction is saved in the **Inspection log**. Click a row to reopen it.

Phone photos are rotated using their EXIF orientation, so the heatmap lines up with the image as you see it.

## API Endpoints

| Method | Endpoint        | Description                                                        |
|--------|-----------------|--------------------------------------------------------------------|
| POST   | `/predict`      | Upload an image (`file` form field) → label, confidence, image and heatmap URLs |
| GET    | `/history`      | Recent predictions, newest first (`skip`, `limit`; default 20)     |
| GET    | `/health`       | Health check                                                       |
| GET    | `/whoami`       | Hostname (container ID)                                            |
| GET    | `/files/{path}` | Serve stored uploads and heatmaps                                  |

`/predict` returns `400` for an unsupported file type or a file that is not a valid image, and `413` for files over `MAX_UPLOAD_MB`. Timestamps are ISO 8601 in UTC (`+00:00`).

Example:

```bash
curl -F "file=@crack.jpg" http://localhost:8000/predict
```

```json
{
  "id": 2,
  "filename": "crack.jpg",
  "label": "positive",
  "confidence": 0.9962,
  "image_url": "/files/uploads/<uuid>.jpg",
  "heatmap_url": "/files/heatmaps/<uuid>_heatmap.jpg",
  "created_at": "2026-11-03T17:07:38.348544+00:00"
}
```

## Environment Variables

| Variable           | Default                      | Description                                    |
|--------------------|------------------------------|------------------------------------------------|
| `DATABASE_URL`     | `sqlite:///./predictions.db` | Database URL. For PostgreSQL use `postgresql+psycopg2://…` |
| `STORAGE_TYPE`     | `local`                      | `local` or `minio`                             |
| `STORAGE_DIR`      | `./storage`                  | Folder for uploads and heatmaps (local storage) |
| `TEMP_DIR`         | system temp folder           | Scratch folder used while processing uploads   |
| `MAX_UPLOAD_MB`    | `20`                         | Maximum upload size                            |
| `CRACK_THRESHOLD`  | `0.5`                        | P(crack) at or above which an image is labelled positive |
| `PORT`             | `8000`                       | Backend port (used by `python -m backend.main`) |
| `VITE_API_URL`     | `http://localhost:8000`      | Backend URL baked into the frontend build      |
| `MINIO_ENDPOINT`   | `localhost:9000`             | MinIO only: server address                     |
| `MINIO_ACCESS_KEY` | `minioadmin`                 | MinIO only: access key                         |
| `MINIO_SECRET_KEY` | `minioadmin`                 | MinIO only: secret key                         |
| `MINIO_BUCKET`     | `crack-detection`            | MinIO only: bucket name                        |
| `MINIO_SECURE`     | `false`                      | MinIO only: use HTTPS                          |

Docker Compose sets `DATABASE_URL`, `STORAGE_TYPE=local`, `STORAGE_DIR`, `TEMP_DIR` and `CRACK_THRESHOLD` for you.

## Training and Evaluation

```bash
# Optional: download a 1,000-image sample (needs the Kaggle CLI and login).
# Refuses to overwrite an existing ai/dataset/ unless you pass --force.
python ai/download_dataset.py

# Train. Expects ai/dataset/{train,val}/{negative,positive}/.
# The current ai/model.pth is backed up to ai/model.pth.bak first.
python -m ai.train --epochs 10

# Evaluate on a folder with negative/ and positive/ subfolders
python -m ai.evaluate path/to/test
```

## Model Details

- **Architecture**: ResNet50 (torchvision) with `fc = Linear(2048, 2)`
- **Input**: 224×224 RGB, ImageNet normalization `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- **Classes**: index 0 = negative (no crack), index 1 = positive (crack)
- **Explainability**: Grad-CAM on `model.layer4[-1]`
- **Training**: Transfer learning from ImageNet-pretrained weights, fine-tuned in Google Colab with a GPU
- **Dataset**: 40,000 concrete crack images (20k positive, 20k negative)

## Results and Limitations

**Training dataset**: 40,000 images (20k crack / 20k no-crack concrete surfaces).

**External dataset test** (different source, unseen domain):

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 66.67% |
| Precision | 96.24% |
| Recall    | 64.18% |
| F1 Score  | 77.00% |

**Confusion matrix** (rows = actual, cols = predicted):

```
           Predicted
            Neg   Pos
Actual Neg [ 184,  37]
Actual Pos [ 528, 946]
```

The model has high precision (few false positives) but lower recall: it misses some cracks. The accuracy drop from training to the external dataset shows **domain shift**. The model is sensitive to differences in surface texture, lighting and crack appearance across datasets. This is expected for a model trained on a single dataset.

`evaluate.py` labels by the higher-scoring class, which matches the API only at the default `CRACK_THRESHOLD=0.5`.

## Future Work

> **Not implemented**: planned for future iterations.

- **K3s cluster**: Multi-node deployment with K3s for horizontal scaling
- **Load balancing**: Traefik ingress with sticky sessions
- **Monitoring**: Prometheus + Grafana dashboards for model and infra metrics
- **YOLO localization**: Object detection to localize crack regions with bounding boxes instead of classification only
