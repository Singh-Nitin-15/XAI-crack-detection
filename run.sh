#!/bin/bash
set -e
echo "=== XAI Crack Detection Service ==="

if [ ! -f "ai/model.pth" ]; then
    echo "ERROR: ai/model.pth not found. Place the trained model file there first."
    exit 1
fi

echo "Starting FastAPI server on port 8000..."
echo "API docs: http://localhost:8000/docs"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
