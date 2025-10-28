# Use official Python image
FROM python:3.11-slim

# Avoid running as root
ARG USER=app
ARG UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    APP_HOME=/app

# Create non-root user and workdir
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid ${UID} ${USER}

WORKDIR ${APP_HOME}

# Copy only dependency files first for caching
COPY requirements.txt ./requirements.txt

# Install deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Ensure start script is executable
RUN chown -R ${USER}:${USER} ${APP_HOME} \
    && chmod +x ./start.sh

USER ${USER}

# Port used by Cloud Run; can be overridden by env var PORT
ENV PORT=8080
EXPOSE 8080

# Default APP_MODULE if not provided (module:app)
ENV APP_MODULE=api:app
ENV WORKERS=2

# Use start.sh so behaviour is configurable
CMD ["./start.sh"]
