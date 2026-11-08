# Pattern Lens, the live pattern visualizer. Build from the repo root:
#   docker build -f service/viz/Dockerfile -t pattern-lens .
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 WORK_DIR=/tmp/osu-viz MODELS_DIR=/app/models
WORKDIR /app
COPY service/viz/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && python -c "import numpy, scipy, sklearn, pandas; print('import ok', numpy.__version__)"
COPY osu_patterns osu_patterns
COPY models/slots.pkl models/live_heads.pkl models/note_model*.npz models/note_model*.json models/
COPY models/similar models/similar
COPY service/__init__.py service/__init__.py
COPY service/viz service/viz
RUN useradd -r -u 1000 lens && mkdir -p /tmp/osu-viz && chown lens /tmp/osu-viz
USER lens
EXPOSE 8770
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8770/health',timeout=4)"
CMD ["uvicorn", "service.viz.app:app", "--host", "0.0.0.0", "--port", "8770", "--proxy-headers", "--forwarded-allow-ips", "*"]
