"""
TTS Service
 
Accepts text tokens over WebSocket, streams back WAV audio chunks.
 
Protocol:
  client server  text frame: {"type": "token", "text": "word "}
  client server  text frame: {"type": "end"} signals end of text
  server client  binary frames: WAV audio chunks
  server client  text frame: {"type": "done"}
  server client  text frame: {"type": "error", "message": "..."}
"""
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TTS] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

TTS_MODEL = os.environ.get("VOICEKIT_TTS_MODEL", "simulated")
TTS_VOICE = os.environ.get("VOICEKIT_TTS_VOICE", "default")

provider = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider
    log.info(f"Loading TTS model: {TTS_MODEL}")

    if TTS_MODEL == "simulated":
        sys.path.insert(0, "/app")
        from voicekit.providers.tts.simulated import SimulatedTTS
        provider = SimulatedTTS(voice=TTS_VOICE)
    else:
        log.error(f"Unknown TTS model: {TTS_MODEL}")
        sys.exit(1)

    await provider.load()
    log.info("TTS model ready")
    yield

app = FastAPI(title="voicekit-tts", lifespan=lifespan)

@app.get("/health")
async def health():
    if provider is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    ready = await provider.health()
    if not ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {"status": "ok", "model": TTS_MODEL, "voice": TTS_VOICE}

@app.websocket("/tts")
async def tts_websocket(ws: WebSocket):
    await ws.accept()
    log.info("TTS session started")

    token_queue: asyncio.Queue = asyncio.Queue()
    audio_chunks_sent = 0

    async def receive_tokens():
        while True:

            try:
                message = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except  asyncio.TimeoutError:
                log.warning("TTS session timed out waiting for tokens")
                await token_queue.put(None)
                return

            if message["type"] == "websocket.disconnect":
                await token_queue(None)
                return
            
            if "text" in message and message["text"]:
                data = json.loads(message["text"])

                if data.get("type") == "token":
                    await token_queue.put(data.get("text", ""))
                
                elif data.get("type") == "end":
                    await token_queue.put(None)
                    return

    async def token_stream():
        while True:
            token = await token_queue.get()
            if token is None:
                break
            yield token

    try: 
        receive_task = asyncio.create_task(receive_tokens())

        async for audio_chunk in provider.synthesize(token_stream()):
            await ws.send_bytes(audio_chunk)
            audio_chunks_sent += 1

        await receive_task
        
        await ws.send_text(json.dumps({"type": "done"}))

        log.info(f"TTS done: {audio_chunks_sent} audio chunks sent")
    
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

