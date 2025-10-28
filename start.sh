#!/bin/bash
set -e

# Activate virtual environment if needed
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Run the FastAPI app
echo "Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port=${PORT:-10000}
