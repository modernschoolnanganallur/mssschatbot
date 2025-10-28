# === Base Image ===
FROM python:3.11-slim

# === Set working directory ===
WORKDIR /app

# === Environment variables ===
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# === Install system dependencies ===
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# === Copy requirements if any ===
COPY requirements.txt .

# === Install Python dependencies ===
RUN pip install --no-cache-dir -r requirements.txt

# === Copy project files ===
COPY . .

# === Start app ===
CMD ["bash", "start.sh"]
