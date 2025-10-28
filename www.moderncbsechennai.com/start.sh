#!/usr/bin/env bash
set -euo pipefail

# Defaults
: "${PORT:=8080}"
: "${APP_MODULE:=api:app}"
: "${WORKERS:=2}"
: "${TIMEOUT:=120}"

# Detect whether app is ASGI (FastAPI) or WSGI (Flask)
# Prefer Gunicorn with Uvicorn worker for ASGI apps
# If uvicorn is present and APP_MODULE looks like module:app, run gunicorn -k uvicorn.workers.UvicornWorker
if python - <<PY >/dev/null 2>&1
try:
    import importlib
    mod, attr = "${APP_MODULE}".split(":",1)
    m = importlib.import_module(mod)
    getattr(m, attr)
    print("OK")
except Exception as e:
    raise SystemExit(1)
PY
then
    # If uvicorn workers available, use them
    if python -c "import importlib,sys
try:
    import uvicorn, gunicorn
    print('HAS')
except Exception:
    sys.exit(1)" >/dev/null 2>&1; then
        exec gunicorn "${APP_MODULE}" -k uvicorn.workers.UvicornWorker \
            --bind "0.0.0.0:${PORT}" \
            --workers "${WORKERS}" \
            --threads 4 \
            --timeout "${TIMEOUT}" \
            --chdir /app \
            --access-logfile "-" \
            --error-logfile "-"
    else
        # fallback to uvicorn directly if gunicorn/uvicorn not installed
        exec uvicorn "${APP_MODULE}" --host 0.0.0.0 --port "${PORT}" --workers "${WORKERS}"
    fi
else
    # If the module cannot be imported, still attempt to run gunicorn with given APP_MODULE
    exec gunicorn "${APP_MODULE}" --bind "0.0.0.0:${PORT}" --workers "${WORKERS}" --timeout "${TIMEOUT}"
fi
