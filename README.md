# Explainable AI Crack Detection

Structural crack detection with explainable AI. Upload a concrete surface image → get a crack/no-crack prediction with a Grad-CAM heatmap highlighting the regions the model focused on.

## Architecture

```
User → React Frontend (Vite) → FastAPI Backend → AI (ResNet50 + Grad-CAM)
                                      ↓
                              PostgreSQL + MinIO
```

| Component  | Technology                              |
|------------|-----------------------------------------|
| AI Model   | PyTorch ResNet50, Grad-CAM              |
| Backend    | FastAPI, SQLAlchemy, Uvicorn             |
| Frontend   | React 19, Vite                          |
| Database   | SQLite (dev) / PostgreSQL (prod)        |
| Storage    | Local filesystem (dev) / MinIO (prod)   |
| Containers | Docker, Docker Compose                  |

## Project Structure

```
xai-crack-detection/
├── ai/
│   ├── explain.py          # Grad-CAM inference + heatmap
│   ├── train.py            # ResNet50 training script (Colab)
│   ├── evaluate.py         # Test set evaluation with metrics
│   └── model.pth           # Trained weights (not in repo)
├── backend/
│   ├── main.py             # FastAPI endpoints
│   ├── models.py           # SQLAlchemy models
│   ├── storage.py          # Storage abstraction (local/MinIO)
│   └── Dockerfile
├── frontend/
│   ├── src/App.jsx         # Main React component
│   ├── src/App.css
│   ├── nginx.conf          # Production nginx config
│   └── Dockerfile
├── tests/
│   └── smoke_test.py       # API smoke test
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Clone and add model

```bash
git clone <repo-url>
cd xai-crack-detection
```

Place the trained `model.pth` file in `ai/`:

```
ai/model.pth    # ~90 MB ResNet50 weights
```

### 2. Run with Docker Compose

```bash
docker compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| Frontend  | http://localhost:3000       |
| Backend   | http://localhost:8000       |
| API Docs  | http://localhost:8000/docs  |
| MinIO UI  | http://localhost:9001       |

### 3. Local development (without Docker)

```bash
pip install -r requirements.txt
# Start backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
# Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

## API Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| POST   | `/predict`       | Upload image → prediction + heatmap      |
| GET    | `/history`       | Prediction history                       |
| GET    | `/health`        | Health check                             |
| GET    | `/whoami`        | Hostname (container ID)                  |
| GET    | `/files/{path}`  | Serve uploaded images and heatmaps       |

## Environment Variables

| Variable         | Default                      | Description                    |
|------------------|------------------------------|--------------------------------|
| `DATABASE_URL`   | `sqlite:///./predictions.db` | Database connection string     |
| `STORAGE_TYPE`   | `local`                      | `local` or `minio`            |
| `MINIO_ENDPOINT` | `localhost:9000`             | MinIO server address           |
| `MINIO_ACCESS_KEY`| `minioadmin`                | MinIO access key               |
| `MINIO_SECRET_KEY`| `minioadmin123`             | MinIO secret key               |
| `MINIO_BUCKET`   | `crack-detection`            | MinIO bucket name              |
| `CRACK_THRESHOLD`| `0.5`                        | P(crack) threshold for positive|
| `VITE_API_URL`   | `http://localhost:8000`      | Backend URL for frontend       |
| `PORT`           | `8000`                       | Backend port                   |

## Model Details

- **Architecture**: ResNet50 (torchvision) with `fc = Linear(2048, 2)`
- **Input**: 224×224 RGB, ImageNet normalization `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`
- **Classes**: index 0 = negative (no crack), index 1 = positive (crack)
- **Explainability**: Grad-CAM on `model.layer4[-1]`
- **Training**: Transfer learning from ImageNet pretrained weights, fine-tuned in Google Colab with GPU
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

The model has high precision (few false positives) but lower recall — it misses some cracks. The accuracy drop from training to the external dataset shows **domain shift**: the model is sensitive to differences in surface texture, lighting, and crack appearance across datasets. This is expected for a model trained on a single dataset.

## Future Work

> **Not implemented** — planned for future iterations.

- **K3s cluster**: Multi-node deployment with K3s for horizontal scaling
- **Load balancing**: Traefik ingress with sticky sessions
- **Monitoring**: Prometheus + Grafana dashboards for model and infra metrics
- **YOLO localization**: Object detection to localize crack regions with bounding boxes instead of classification-only
