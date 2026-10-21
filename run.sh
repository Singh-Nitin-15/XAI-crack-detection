#!/bin/bash
# Run script for XAI Crack Detection Service (Week 1)
set -e
echo "=== XAI Crack Detection Service ==="

if [ ! -f "ai/model.pth" ]; then
    echo "[1/2] Training model..."
    python -m ai.train
    echo ""
fi

echo "[2/2] Starting FastAPI server on port 8000..."
echo "API docs: http://localhost:8000/docs"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
