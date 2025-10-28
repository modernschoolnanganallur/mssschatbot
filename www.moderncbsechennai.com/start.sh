#!/bin/bash
set -e

echo "Starting Modern School FastAPI server (with Ollama llama3.2)..."

# Optional: Verify Ollama is reachable
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "⚠️ Warning: Ollama service not reachable on port 11434."
  echo "Make sure Ollama is running and llama3.2 model is pulled."
fi

# Run FastAPI app
exec uvicorn api:app --host=0.0.0.0 --port=${PORT:-8080}
