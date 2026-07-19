"""
Session route.

WebSocket /session endpoint.
One pipeline turn per loop iteration.

True end-to-end streaming:
    pipeline.run_turn() places audio chunks into audio_out as TTS
    generates them. We drain audio_out concurrently using
    asyncio.gather so chunks are sent to the client immediately
    as they arrive — not after the turn completes.

    Order of messages to client:
        binary: audio chunk 1      ← arrives as TTS generates
        binary: audio chunk 2
        ...
        binary: audio chunk N      ← last chunk
        text:   transcript         ← what the user said
        text:   response           ← what the agent said
        text:   metrics            ← timing breakdown

    The client can start playing audio on chunk 1 while TTS
    is still generating chunks 2..N.
"""
import sys
sys.path.insert(0, "/app")

import asyncio
import json
import logging
import time
import uuid

import numpy as np
import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from dependencies import get_redis_ws, get_pipeline_ws
from remote_pipeline import AUDIO_DONE
from services.ping import ping_loop, PING_INTERVAL, PING_TIMEOUT
from services.session import (
    register_session,
    update_session_state,
    complete_turn,
    cleanup_session,
)

log = logging.getLogger(__name__)

router = APIRouter()

IDLE_TIMEOUT = 1800    # 30 minutes
TURN_TIMEOUT = 300     # 5 minutes for CPU inference

@router.websocket("/session")
async def session_endpoint(
    ws: WebSocket,
    redis: aioredis.Redis = Depends(get_redis_ws),
    pipeline=Depends(get_pipeline_ws),
):
    await ws.accept()

    session_id = str(uuid.uuid4())[:8]
    session_start = time.time()
    close_reason = "disconnect"

    log.info(json.dumps({
        "event": "session_open",
        "session_id": session_id,
        "ts": session_start,
    }))

    await register_session(redis, session_id)

    last_turn_time = time.time()
    last_pong_time = [time.time()]

    ping_task = asyncio.create_task(
        ping_loop(
            ws,
            session_id,
            last_pong_time,
            pipeline.turn_in_progress,
        )
    )

    try:
        await ws.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
        }))

        while True:

            if time.time() - last_turn_time > IDLE_TIMEOUT:
                close_reason = "idle_timeout"
                log.info(f"[{session_id}] Idle timeout")
                try:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "Session idle timeout.",
                    }))
                except RuntimeError:
                    pass
                break

            audio_in: asyncio.Queue = asyncio.Queue()
            audio_out: asyncio.Queue = asyncio.Queue()
            end_of_speech = asyncio.Event()

            async def receive_turn():
                while not end_of_speech.is_set():
                    try:
                        message = await asyncio.wait_for(
                            ws.receive(),
                            timeout=float(PING_INTERVAL + PING_TIMEOUT),
                        )
                    except asyncio.TimeoutError:
                        continue
                    except RuntimeError:
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
                        elif data.get("type") == "pong":
                            last_pong_time[0] = time.time()

            await asyncio.create_task(receive_turn())

            if audio_in.empty():
                continue

            await update_session_state(redis, session_id, "processing")

            # concurrent audio sender
            # drains audio_out and sends to client as chunks arrive
            # runs simultaneously with pipeline.run_turn()
            # stops when it receives the AUDIO_DONE sentinel
            async def send_audio_stream():
                chunks_sent = 0
                while True:
                    chunk = await audio_out.get()
                    if chunk is AUDIO_DONE:
                        log.info(
                            f"[{session_id}] Audio stream complete: "
                            f"{chunks_sent} chunks sent"
                        )
                        break
                    try:
                        await ws.send_bytes(chunk)
                        chunks_sent += 1
                    except RuntimeError:
                        log.warning(
                            f"[{session_id}] Client disconnected "
                            f"during audio stream at chunk {chunks_sent}"
                        )
                        break

            try:
                # run pipeline and audio sender concurrently
                # pipeline puts chunks into audio_out as TTS generates them
                # send_audio_stream() sends each chunk to client immediately
                # pipeline puts AUDIO_DONE sentinel when finished
                # send_audio_stream() stops on sentinel
                results = await asyncio.wait_for(
                    asyncio.gather(
                        pipeline.run_turn(audio_in, audio_out),
                        send_audio_stream(),
                        return_exceptions=True,
                    ),
                    timeout=float(TURN_TIMEOUT),
                )
            except asyncio.TimeoutError:
                log.warning(
                    f"[{session_id}] Turn timeout after {TURN_TIMEOUT}s"
                )
                try:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": "Response timed out. Please try again.",
                    }))
                except RuntimeError:
                    pass
                await update_session_state(redis, session_id, "waiting")
                continue

            # extract metrics from gather results
            # results[0] = PipelineMetrics, results[1] = None (send_audio return)
            metrics = results[0]

            if isinstance(metrics, Exception):
                log.error(f"[{session_id}] Pipeline error: {metrics}")
                try:
                    await ws.send_text(json.dumps({
                        "type": "error",
                        "message": str(metrics),
                    }))
                except RuntimeError:
                    pass
                continue

            # send text messages after all audio has been streamed
            try:
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
                    "tts_first_chunk_ms": round(metrics.tts_first_chunk_ms),
                    "total_ms": round(metrics.total_ms),
                }))
            except RuntimeError:
                log.warning(
                    f"[{session_id}] Client disconnected before "
                    f"text messages were sent"
                )
                break

            await complete_turn(redis, session_id, pipeline.history)
            last_turn_time = time.time()

            log.info(json.dumps({
                "event": "turn_complete",
                "session_id": session_id,
                "stt_ms": round(metrics.stt_ms),
                "llm_first_token_ms": round(metrics.llm_first_token_ms),
                "tts_first_chunk_ms": round(metrics.tts_first_chunk_ms),
                "total_ms": round(metrics.total_ms),
                "transcript_length": len(metrics.transcript),
                "turn_count": len(pipeline.history) // 2,
                "ts": time.time(),
            }))

    except WebSocketDisconnect:
        close_reason = "disconnect"
        log.info(f"[{session_id}] Client disconnected")

    except Exception as e:
        close_reason = "error"
        log.exception(f"[{session_id}] Unexpected error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass

    finally:
        ping_task.cancel()
        await cleanup_session(redis, session_id)

        log.info(json.dumps({
            "event": "session_close",
            "session_id": session_id,
            "reason": close_reason,
            "turn_count": len(pipeline.history) // 2,
            "duration_s": round(time.time() - session_start),
            "ts": time.time(),
        }))