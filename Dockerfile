# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# System deps (add if you need e.g., libpq-dev, build-essential, etc.)
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# System deps (add libgomp1)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*


# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# If your repo uses a start script, keep this:
# CMD ["bash", "start.sh"]

# Otherwise, if it's a Flask/Django/Gunicorn app, use (edit module:app):
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]

# replace the last line with this:
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
# In Dockerfile (replace last line)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
