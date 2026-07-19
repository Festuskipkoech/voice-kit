FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    numpy \
    pyyaml \
    setuptools

RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir chatterbox-tts

COPY voicekit /app/voicekit
COPY runtime/tts/server.py /app/server.py

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

CMD ["python", "server.py"]
