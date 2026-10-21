@echo off
REM Run script for XAI Crack Detection Service (Week 1)
echo === XAI Crack Detection Service ===

REM Check if model exists, train if not
if not exist "ai\model.pth" (
    echo [1/2] Training model...
    python -m ai.train
    echo.
)

echo [2/2] Starting FastAPI server on port 8000...
echo API docs: http://localhost:8000/docs
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
