"""
Session service.

All Redis operations related to session lifecycle in one place.
No FastAPI imports — pure async functions, easily unit tested.
"""
import time

import redis.asyncio as aioredis

async def register_session(
    redis: aioredis.Redis,
    session_id: str,
) -> None:
    """Register a new session in Redis on WebSocket connect."""
    await redis.hset(f"session:{session_id}", mapping={
        "state": "waiting",
        "created_at": str(time.time()),
        "last_active": str(time.time()),
    })
    await redis.sadd("active_sessions", session_id)

async def update_session_state(
    redis: aioredis.Redis,
    session_id: str,
    state: str,
) -> None:
    """Update the session state field."""
    await redis.hset(f"session:{session_id}", "state", state)

async def complete_turn(
    redis: aioredis.Redis,
    session_id: str,
    history: list[dict],
) -> None:
    """
    Persist conversation history and update session metadata
    after a successful pipeline turn.
    History is stored with a 1-hour TTL — sessions that go idle
    for over an hour lose their history, which is acceptable.
    """
    import json

    turn_count = len(history) // 2

    await redis.set(
        f"history:{session_id}",
        json.dumps(history),
        ex=3600,
    )
    await redis.hset(f"session:{session_id}", mapping={
        "state": "waiting",
        "last_active": str(time.time()),
        "turn_count": str(turn_count),
    })

async def cleanup_session(
    redis: aioredis.Redis,
    session_id: str,
) -> None:
    """
    Remove all session data from Redis.

    Called from the finally block in the session route —
    guaranteed to run regardless of how the session exits.
    No session can leak Redis state.
    """
    await redis.delete(f"session:{session_id}")
    await redis.delete(f"history:{session_id}")
    await redis.srem("active_sessions", session_id)


async def active_session_count(redis: aioredis.Redis) -> int:
    """Return the number of currently active sessions."""
    return await redis.scard("active_sessions")