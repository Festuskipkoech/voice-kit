FROM python:3.11-slim

WORKDIR /app

# espeak-ng is a required system dependency for Kokoro phonemization
# without it Kokoro raises: RuntimeError: espeak-ng not found
RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# install base dependencies first
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    numpy \
    pyyaml \
    soundfile \
    setuptools

# install torch CPU — separate step to use whl index
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# install TTS models — each in separate step for layer caching
# kokoro is fast on CPU — the recommended default for streaming
RUN pip install --no-cache-dir kokoro>=0.9.4

# chatterbox — voice cloning support, slower on CPU
RUN pip install --no-cache-dir chatterbox-tts

COPY voicekit /app/voicekit
COPY runtime/tts/server.py /app/server.py

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

CMD ["python", "server.py"]
