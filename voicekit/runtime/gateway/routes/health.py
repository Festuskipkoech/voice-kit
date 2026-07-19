"""
Health route.

GET /health — returns gateway status and active session count.
Used by Docker healthcheck and external monitoring.
"""
import sys
sys.path.insert(0, "/app")

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends

from dependencies import get_redis
from services.session import active_session_count

router = APIRouter()

@router.get("/health")
async def health(redis: aioredis.Redis = Depends(get_redis)):
    count = await active_session_count(redis)
    return {
        "status": "ok",
        "active_sessions": count,
    }