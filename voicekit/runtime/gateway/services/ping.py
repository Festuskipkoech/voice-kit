"""
Ping service.

WebSocket heartbeat — runs as a background task per session.
Distinguishes idle-but-alive sessions from dead connections.

Critical behaviour:
    The ping loop NEVER closes the connection while a pipeline turn
    is in progress. Closing mid-synthesis would cut off audio that
    is still being generated — especially important on CPU where
    Chatterbox synthesis takes 90-120 seconds.

    A user thinking between turns still pongs — connection is alive.
    A dead connection does not pong — close it.
    An active turn — do not ping at all, wait for turn to finish.
"""
import asyncio
import json
import logging
import time

from fastapi import WebSocket

log = logging.getLogger(__name__)

PING_INTERVAL = 30    # seconds between pings when idle
PING_TIMEOUT = 10     # seconds to wait for pong


async def ping_loop(
    ws: WebSocket,
    session_id: str,
    last_pong_time: list[float],
    turn_in_progress: asyncio.Event,
) -> None:
    """
    Send ping every PING_INTERVAL seconds when session is idle.
    Skip ping entirely if a turn is in progress.
    Close connection only if pong not received AND no turn in progress.
    """
    while True:
        await asyncio.sleep(PING_INTERVAL)

        # if a turn is in progress, wait for it to finish
        # do not ping or close during active synthesis
        if turn_in_progress.is_set():
            log.debug(
                f"[{session_id}] Turn in progress — skipping ping"
            )
            continue

        try:
            await ws.send_text(json.dumps({"type": "ping"}))
            log.debug(f"[{session_id}] Ping sent")
        except Exception:
            return

        deadline = time.time() + PING_TIMEOUT
        while time.time() < deadline:
            # if a turn started while waiting for pong, skip close
            if turn_in_progress.is_set():
                break
            if last_pong_time[0] > time.time() - PING_INTERVAL:
                break
            await asyncio.sleep(1.0)
        else:
            # pong never arrived and no turn started
            log.warning(
                f"[{session_id}] Ping timeout — closing dead connection"
            )
            try:
                await ws.close(code=1001, reason="Ping timeout")
            except Exception:
                pass
            return