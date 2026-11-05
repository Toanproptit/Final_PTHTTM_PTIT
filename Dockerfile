FROM node:22-alpine AS frontend-build
WORKDIR /build
COPY package.json package-lock.json vite.config.js ./
COPY frontend ./frontend
RUN npm ci && npm run build

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend-build /build/web ./web
RUN chmod +x /app/scripts/docker-entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=5 \
  CMD curl --fail http://127.0.0.1:8000/api/health || exit 1
ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
