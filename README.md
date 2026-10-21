# Explainable AI Service for Structural Crack Detection

## Architecture
User → React Frontend → FastAPI Backend → AI Service (ResNet18 + Grad-CAM) → PostgreSQL + MinIO

## Project Structure
```
xai-crack-detection/
├── ai/                     # AI model training and inference
│   ├── train.py            # ResNet18 transfer learning trainer
│   ├── explain.py          # Grad-CAM heatmap generation
│   └── model.pth           # Trained model weights (generated)
├── backend/                # FastAPI REST API
│   ├── main.py             # API endpoints
│   ├── models.py           # SQLAlchemy models
│   ├── storage.py          # Storage abstraction (local/MinIO)
│   └── Dockerfile          # Backend container
├── frontend/               # React + Vite
│   ├── src/App.jsx         # Main app component
│   ├── src/App.css         # Styling
│   ├── nginx.conf          # Production nginx config
│   └── Dockerfile          # Frontend container (build + nginx)
├── k8s/                    # Kubernetes manifests (future)
├── scripts/                # Deployment scripts (future)
├── docker-compose.yml      # Full stack composition
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .dockerignore           # Docker build exclusions
├── run.bat / run.sh        # Quick start scripts
└── README.md
```

## Quick Start — Local Development

### 1. Train the model
```bash
pip install -r requirements.txt
python -m ai.train
```

### 2. Start backend
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Start frontend
```bash
cd frontend && npm install && npm run dev
```

## Quick Start — Docker Compose

```bash
docker compose up --build
```

| Service   | URL                        |
|-----------|----------------------------|
| Frontend  | http://localhost:3000       |
| Backend   | http://localhost:8000       |
| API Docs  | http://localhost:8000/docs  |
| MinIO UI  | http://localhost:9001       |

## API Endpoints
| Method | Endpoint    | Description                           |
|--------|-------------|---------------------------------------|
| POST   | /predict    | Upload image → crack prediction + heatmap |
| GET    | /history    | Retrieve prediction history           |
| GET    | /whoami     | Returns hostname (for load balancing) |
| GET    | /health     | Health check                          |

## Environment Variables
| Variable        | Default                    | Description                |
|-----------------|----------------------------|----------------------------|
| DATABASE_URL    | sqlite:///./predictions.db | Database connection string |
| STORAGE_TYPE    | local                      | "local" or "minio"         |
| MINIO_ENDPOINT  | localhost:9000             | MinIO server address       |
| MINIO_ACCESS_KEY| minioadmin                 | MinIO access key           |
| MINIO_SECRET_KEY| minioadmin123              | MinIO secret key           |
| VITE_API_URL    | http://localhost:8000      | Backend URL for frontend   |

## Tech Stack
- **AI**: PyTorch, ResNet18, Grad-CAM
- **Backend**: FastAPI, SQLAlchemy, Uvicorn
- **Frontend**: React, Vite
- **Database**: SQLite (dev) → PostgreSQL (prod)
- **Storage**: Local filesystem → MinIO (prod)
- **Containers**: Docker, Docker Compose
- **Infra**: K3s, Traefik, Prometheus + Grafana (future weeks)
