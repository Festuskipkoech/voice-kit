FROM python:3.11-slim

WORKDIR /app

# gateway is lightweight — no ML models loaded here
# whisper runs in stt service, chatterbox runs in tts service
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    numpy \
    pyyaml \
    anthropic \
    redis \
    setuptools \
    openai

COPY voicekit /app/voicekit
COPY runtime/gateway/main.py /app/main.py
COPY runtime/gateway/config.py /app/config.py
COPY runtime/gateway/dependencies.py /app/dependencies.py
COPY runtime/gateway/remote_pipeline.py /app/remote_pipeline.py
COPY runtime/gateway/routes /app/routes
COPY runtime/gateway/services /app/services

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "main.py"]
