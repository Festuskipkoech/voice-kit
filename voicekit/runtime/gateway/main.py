"""
Gateway entry point.

The gateway is lightweight — it does NOT load any ML models.
Whisper runs in the STT service. Chatterbox runs in the TTS service.
The gateway handles session management, LLM calls, and routing only.

This is the correct production architecture:
    Gateway   - fast startup, no models, handles many sessions
    STT :8001 - Whisper loaded once, serves all transcription
    TTS :8002 - Chatterbox loaded once, serves all synthesis
    Redis     - session state and conversation history
"""

import sys
sys.path.insert(0, "/app")

import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI

from config import config_from_env, configure_llm_key
from remote_pipeline import RemotePipeline
from routes.health import router as health_router
from routes.session import router as session_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GW] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
        1. Build config from environment
        2. Set provider API key
        3. Connect to Redis
        4. Initialise RemotePipeline — verifies STT and TTS are reachable
           No model loading. Startup completes in seconds.

    Shutdown:
        Close Redis connection cleanly.
    """
    config = config_from_env()
    configure_llm_key(config)

    redis_client = aioredis.Redis(
        host=os.environ.get("REDIS_HOST", "redis"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        decode_responses=True,
        max_connections=20,
    )
    app.state.redis = redis_client
    log.info("Redis connected")

    # RemotePipeline — lightweight, verifies service connectivity only
    pipeline = RemotePipeline(config)
    await pipeline.load()
    app.state.pipeline = pipeline
    app.state.config = config

    log.info(
        f"Gateway ready — "
        f"STT: {os.environ.get('STT_WS_URL', 'ws://stt:8001/stt')} "
        f"TTS: {os.environ.get('TTS_WS_URL', 'ws://tts:8002/tts')} "
        f"LLM: {config.llm.provider}/{config.llm.model}"
    )

    yield

    await redis_client.aclose()
    log.info("Gateway stopped")

app = FastAPI(title="voicekit-gateway", lifespan=lifespan)

app.include_router(health_router)
app.include_router(session_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )