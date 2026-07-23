"""
TTS Service

Accepts a single phrase over WebSocket, streams back WAV audio chunks.

Protocol:
    client -> server  text: {"type": "phrase", "text": "complete phrase"}
    client -> server  text: {"type": "end"}
    server -> client  binary: WAV audio chunk (first chunk has WAV header)
    server -> client  binary: raw float32 PCM chunks (subsequent chunks)
    server -> client  text:   {"type": "done"}
    server -> client  text:   {"type": "error", "message": "..."}

One phrase per WebSocket connection. The gateway opens a new connection
for each phrase — this keeps the server stateless and naturally concurrent.
"""
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TTS] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

TTS_MODEL = os.environ.get("VOICEKIT_TTS_MODEL")
TTS_VOICE = os.environ.get("VOICEKIT_TTS_VOICE")

provider = None

def _load_provider(model: str, voice: str):
    if model == "chatterbox-turbo":
        from voicekit.providers.tts.chatterbox import ChatterboxTTS
        return ChatterboxTTS(voice=voice)

    elif model == "kokoro":
        from voicekit.providers.tts.kokoro import KokoroTTS
        return KokoroTTS(voice=voice)

    else:
        raise ValueError(
            f"Unknown TTS model: '{model}'. "
            f"Available: chatterbox-turbo, kokoro"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    log.info(f"Loading TTS model: {TTS_MODEL}")

    try:
        provider = _load_provider(TTS_MODEL, TTS_VOICE)
        await provider.load()
        log.info(f"TTS model ready: {TTS_MODEL} voice={TTS_VOICE}")
    except Exception as e:
        log.error(f"Failed to load TTS model: {e}")
        sys.exit(1)

    yield

    log.info("TTS service stopped")

app = FastAPI(title="voicekit-tts", lifespan=lifespan)

@app.get("/health")
async def health():
    if provider is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    ready = await provider.health()
    if not ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {
        "status": "ok",
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
    }

@app.websocket("/tts")
async def tts_endpoint(ws: WebSocket):
    await ws.accept()

    audio_chunks_sent = 0
    phrase = None

    try:
        async for raw in ws.iter_text():
            data = json.loads(raw)

            if data.get("type") == "phrase":
                phrase = data.get("text", "").strip()

            elif data.get("type") == "end":
                break

        if not phrase:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "no phrase received before end signal",
            }))
            return

        log.info(f"Synthesising: '{phrase[:60]}'")

        async for audio_chunk in provider.synthesize(phrase):
            await ws.send_bytes(audio_chunk)
            audio_chunks_sent += 1

        await ws.send_text(json.dumps({"type": "done"}))
        log.info(f"TTS done: {audio_chunks_sent} chunks for '{phrase[:40]}'")

    except WebSocketDisconnect:
        log.info("TTS client disconnected")
    except Exception as e:
        log.exception(f"TTS error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")