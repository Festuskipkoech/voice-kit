"""
Gateway Service
 
The gateway is what a voice agent project connects to.
It accepts audio from the caller, runs the full pipeline,
and streams audio back. It holds the session state — conversation
history and pipeline instance — for the duration of a call.
 
Protocol:
  client server  binary frames: raw float32 PCM audio (16kHz mono)
  client server  text frame: {"type": "end_of_speech"}
  server client  binary frames: WAV audio chunks (agent response)
  server client  text frame: {"type": "transcript", "text": "..."}
  server client  text frame: {"type": "response", "text": "..."}
  server client  text frame: {"type": "metrics", ...}
  server client  text frame: {"type": "ready"} (sent when pipeline loaded)
  server client  text frame: {"type": "error", "message": "..."}
"""
import asyncio
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

sys.path.insert(0, "/app")

from voicekit.config import VoiceConfig, STTConfig, TTSConfig, VADConfig, LLMConfig
from voicekit.pipeline import VoicePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GW] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

_sessions: dict[str, VoicePipeline] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="voicekit-gateway", lifespan=lifespan)

def _config_from_env() -> VoiceConfig:
    return VoiceConfig(
        project=os.environ.get("VOICEKIT_PROJECT", "voicekit"),
        stt=STTConfig(
            model=os.environ.get("VOICEKIT_STT_MODEL", "simulated"),
            variant=os.environ.get("VOICEKIT_STT_VARIANT", "small"),
        ),
        tts=TTSConfig(
            model=os.environ.get("VOICEKIT_TTS_MODEL", "simulated"),
            voice=os.environ.get("VOICEKIT_TTS_VOICE", "default"),
        ),
        vad=VADConfig(
            enabled=os.environ.get("VOICEKIT_VAD_ENABLED", "true").lower() == "true",
            sensitivity=float(os.environ.get("VOICEKIT_VAD_SENSITIVITY", "0.5")),
        ),
        llm=LLMConfig(
            provider=os.environ.get("VOICEKIT_LLM_PROVIDER", "simulated"),
            model=os.environ.get("VOICEKIT_LLM_MODEL", "simulated"),
            api_key=os.environ.get("VOICEKIT_LLM_API_KEY", ""),
        ),
        system_prompt=os.environ.get(
            "VOICEKIT_SYSTEM_PROMPT",
            "You are a helpful voice assistant. Keep responses concise and natural.",
        ),
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
    } 
 
@app.get("/sessions")
async def sessions():
    return {
        "active": len(_sessions),
        "ids": list(_sessions.keys()),
    }

@app.websocket("/session")
async def session_endpoint(ws: WebSocket):
    await ws.accept()

    session_id = str(uuid.uuid4())[:8]
    log.info(f"[{session_id}] Session opened")

    config =_config_from_env()
    pipeline = VoicePipeline(config)
    _sessions[session_id] = pipeline

    try:
        # load models — this blocks until ready
        await pipeline.load()
        await ws.send_text(json.dumps({"type": "ready", "session_id": session_id}))
        log.info(f"[{session_id}] Pipeline loaded, session ready")

        # main session loop — each iteration is one voice turn
        while True:
            audio_in: asyncio.Queue = asyncio.Queue()
            audio_out: asyncio.Queue = asyncio.Queue()

            # receive audio from client until end_of_speech signal
            end_of_speech = asyncio.Event()

            async def receive_loop():
                while not end_of_speech.is_set():
                    try:
                        message =await asyncio.wait_for(
                            ws.receive(), timeout=60.0
                        )
                    except asyncio.TimeoutError:
                         log.warning(f"[{session_id}] Receive timeout")
                         end_of_speech.set()
                         return                    

                    if message["type"] == "websocket.disconnect":
                        end_of_speech.set()
                        return
 
                    if "bytes" in message and message["bytes"]:
                        chunk = np.frombuffer(
                            message["bytes"], dtype=np.float32
                        )
                        await audio_in.put(chunk)
 
                    elif "text" in message and message["text"]:
                        data = json.loads(message["text"])
                        if data.get("type") == "end_of_speech":
                            end_of_speech.set()
            
            async def send_audio_loop():
                while not end_of_speech.is_set() or not audio_out.empty():
                    try:
                        chunk = await asyncio.wait_for(
                            audio_out.get(), timeout=0.1
                        )
                        await ws.send_bytes(chunk)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

            receive_task = asyncio.create_task(receive_loop())
            send_task = asyncio.create_task(send_audio_loop())

            await receive_task

            # run the pipeline turn
            metrics = await pipeline.run_turn(audio_in, audio_out)

            # drain remaining audio output
            while not audio_out.empty():
                chunk = audio_out.get_nowait()
                await ws.send_bytes9(chunk)

            send_task.cancel()

            # send transcript and metrics back to client
            await ws.send_text(json.dumps({
                "type": "transcript",
                "text": metrics.transcript,
            }))
            await ws.send_text(json.dumps({
                "type": "response",
                "text": metrics.response,
            }))
            await ws.send_text(json.dumps({
                "type": "metrics",
                "stt_ms": round(metrics.stt_ms),
                "llm_first_token_ms": round(metrics.llm_first_token_ms),
                "llm_total_ms": round(metrics.llm_total_ms),
                "tts_first_chunk_ms": round(metrics.tts_first_chunk_ms),
                "total_ms": round(metrics.total_ms),
            }))
 
            log.info(f"[{session_id}] Turn complete — {metrics}")

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Client disconnected")
    except Exception as e:
        log.exception(f"[{session_id}] Session error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass

        finally:
            _sessions.pop(session_id, None)
            log.info(f"[{session_id}] Session closed")

if _-name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
