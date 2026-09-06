FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends clang libgomp1 libsdl2-2.0-0 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY doom/requirements.txt /app/doom/requirements.txt
RUN pip install --no-cache-dir -r doom/requirements.txt
COPY doom /app/doom
COPY doom_learning /app/doom_learning
COPY doom_learning_v6 /app/doom_learning_v6
RUN python -m doom.build_kernel && useradd --uid 10001 --create-home runner && mkdir -p /state && chown -R runner:runner /app /state
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
USER runner
EXPOSE 8766
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request,json,time; d=json.load(urllib.request.urlopen('http://127.0.0.1:8766/health',timeout=4)); assert d['status']=='running' and time.time()*1000-d['generated_at_ms']<15000"
CMD ["python","-m","doom.server","--model","experimental-v6","--learning","--bind","0.0.0.0","--port","8766","--scenario","combat_survival","--reward","off","--audit-dir","/state/training-v1/audit","--checkpoint-dir","/state/training-v1/checkpoints","--checkpoint-seconds","300","--resume"]
