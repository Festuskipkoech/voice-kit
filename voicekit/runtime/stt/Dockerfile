FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    numpy \
    pyyaml

RUN pip install --no-cache-dir faster-whisper

COPY voicekit /app/voicekit
COPY runtime/stt/server.py /app/server.py

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["python", "server.py"]
