FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev libxml2-dev libxslt1-dev curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
COPY . /app
EXPOSE 8000
ENV APP_MODULE=backend.app:app HOST=0.0.0.0 PORT=8000 WORKERS=2
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:${PORT}/health || exit 1
CMD exec gunicorn "$APP_MODULE" -k uvicorn.workers.UvicornWorker -w "${WORKERS}" -b "${HOST}:${PORT}" --timeout 120 --access-logfile '-' --error-logfile '-'
