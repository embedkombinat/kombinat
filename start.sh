#!/bin/sh
set -e
dbmate up
exec uvicorn kombinat.main:app --host 0.0.0.0 --port ${PORT:-8000}
