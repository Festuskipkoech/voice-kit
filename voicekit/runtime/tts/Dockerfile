FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    espeak-ng \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    numpy \
    pyyaml \
    soundfile \
    setuptools \
    onnxruntime \
    kokoro-onnx>=0.5.0

# download model files once at build time
# they live at /app/models/ and are reused across container restarts
RUN mkdir -p /app/models && \
    wget -q -O /app/models/kokoro-v1.0.onnx \
        https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx && \
    wget -q -O /app/models/voices-v1.0.bin \
        https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY voicekit /app/voicekit
COPY runtime/tts/server.py /app/server.py

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

CMD ["python", "server.py"]
