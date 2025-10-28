#!/bin/bash
set -e

echo "Starting Modern School Chatbot (ChatGPT-4-mini)"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY not set"
  exit 1
fi

# Use PORT provided by Cloud Run or default 8080
exec uvicorn api:app --host=0.0.0.0 --port="${PORT:-8080}" --proxy-headers
