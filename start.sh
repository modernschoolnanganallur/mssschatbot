#!/usr/bin/env bash
set -e

# ensure folders exist
mkdir -p vectorstore sessions
#
# # run uvicorn on Render's $PORT
exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
#
