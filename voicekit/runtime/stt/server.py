"""
STT Service

Accepts raw audio over WebSocket, returns transcript tokens.

Protocol:
  client - server  binary frames: raw float32 PCM audio chunks
  client - server  text frame: {"type": "end"} signals end of audio
  server - client  text frames: {"type": "token", "text": "..."}
  server - client  text frame: {"type": "done", "transcript": "..."}
  server - client  text frame: {"type": "error", "message": "..."}
"""
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, "/app")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STT] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

STT_MODEL = os.environ.get("VOICEKIT_STT_MODEL", "whisper")
STT_VARIANT = os.environ.get("VOICEKIT_STT_VARIANT", "small")

provider = None

def _load_provider(model: str, variant: str):
    """
    Load the correct STT provider based on model name.
    Uses the provider registry pattern — adding a new STT model
    means adding it to the registry, not modifying this server.
    """
    if model == "whisper":
        from voicekit.providers.stt.whisper import WhisperSTT
        return WhisperSTT(variant=variant)

    else:
        raise ValueError(
            f"Unknown STT model: '{model}'. "
            f"Available: whisper, simulated"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    log.info(f"Loading STT model: {STT_MODEL} ({STT_VARIANT})")

    try:
        provider = _load_provider(STT_MODEL, STT_VARIANT)
        await provider.load()
        log.info(f"STT model ready: {STT_MODEL}/{STT_VARIANT}")
    except Exception as e:
        log.error(f"Failed to load STT model: {e}")
        sys.exit(1)

    yield

    log.info("STT service stopped")

app = FastAPI(title="voicekit-stt", lifespan=lifespan)

@app.get("/health")
async def health():
    if provider is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    ready = await provider.health()
    if not ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {
        "status": "ok",
        "model": STT_MODEL,
        "variant": STT_VARIANT,
    }

@app.websocket("/stt")
async def stt_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("STT session started")

    audio_queue: asyncio.Queue = asyncio.Queue()
    transcript_parts = []

    async def receive_audio():
        while True:
            try:
                message = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                log.warning("STT session timed out waiting for audio")
                await audio_queue.put(None)
                return

            if message["type"] == "websocket.disconnect":
                await audio_queue.put(None)
                return

            if "bytes" in message and message["bytes"]:
                chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                await audio_queue.put(chunk)

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                if data.get("type") == "end":
                    await audio_queue.put(None)
                    return

    async def audio_stream():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            yield chunk

    try:
        receive_task = asyncio.create_task(receive_audio())

        async for token in provider.transcribe(audio_stream()):
            token_text = token.strip()
            if token_text:
                transcript_parts.append(token_text)
                await ws.send_text(json.dumps({
                    "type": "token",
                    "text": token_text,
                }))

        await receive_task

        full_transcript = " ".join(transcript_parts)
        await ws.send_text(json.dumps({
            "type": "done",
            "transcript": full_transcript,
        }))
        log.info(f"STT done: '{full_transcript}'")

    except WebSocketDisconnect:
        log.info("STT client disconnected")
    except Exception as e:
        log.exception(f"STT error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")