"""
Gateway dependencies.

FastAPI dependency functions that inject shared resources into routes.

Resources live on app.state — set once during lifespan, never mutated
after startup. This is the FastAPI production standard for shared
stateful resources such as database connections and loaded ML models.

Why app.state over module-level globals:
    - Explicitly scoped to the app instance, not the Python process
    - Testable: set app.state.redis = mock in tests, no module patching
    - Inspectable: app.state is a plain object, visible in debuggers
    - Idiomatic: matches FastAPI documentation and community standards

For HTTP routes, resources are accessed via Request.app.state.
For WebSocket routes, resources are accessed via WebSocket.app.state.
Both are injected automatically by FastAPI through Depends().
"""
from __future__ import annotations

import redis.asyncio as aioredis
from fastapi import Request, WebSocket

from voicekit.pipeline import VoicePipeline

async def get_redis(request: Request) -> aioredis.Redis:
    """
    Inject the shared Redis connection pool into HTTP route handlers.

    The pool is created once in lifespan and stored on app.state.
    FastAPI passes the current request automatically when this
    function is used as a dependency.
    """
    return request.app.state.redis


async def get_pipeline(request: Request) -> VoicePipeline:
    """
    Inject the shared VoicePipeline into HTTP route handlers.

    The pipeline — with Whisper and Chatterbox loaded in memory —
    is created once in lifespan. Every request gets the same instance.
    Models are never reloaded between requests.
    """
    return request.app.state.pipeline


async def get_redis_ws(websocket: WebSocket) -> aioredis.Redis:
    """
    Inject the shared Redis connection pool into WebSocket route handlers.

    WebSocket handlers receive a WebSocket object, not a Request.
    WebSocket.app gives access to the FastAPI app and its state.
    """
    return websocket.app.state.redis

async def get_pipeline_ws(websocket: WebSocket) -> VoicePipeline:
    """
    Inject the shared VoicePipeline into WebSocket route handlers.
    """
    return websocket.app.state.pipeline