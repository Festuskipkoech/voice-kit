"""
Integration-style tests for VoicekitClient using a local echo server.

The fake server:
  - accepts one WebSocket connection
  - expects session.config -> replies session.ready
  - echoes binary audio frames back unchanged
  - closes cleanly on disconnect
"""

import json

import numpy as np
import pytest
import pytest_asyncio
import websockets
import websockets.asyncio.server

from voicekit import VoicekitClient, SessionConfig
from voicekit.exceptions import HandshakeError, PolicyError
from voicekit.protocol import FRAME_SAMPLES

# Fake server helpers
async def _server_session_ready(ws):
    """Accept any session.config and reply session.ready."""
    raw = await ws.recv()
    obj = json.loads(raw)
    assert obj["type"] == "session.config"
    await ws.send(json.dumps({"type": "session.ready"}))


async def _echo_server(ws):
    await _server_session_ready(ws)
    async for msg in ws:
        if isinstance(msg, bytes):
            await ws.send(msg)

# Fixtures
@pytest_asyncio.fixture
async def echo_server():
    async with websockets.asyncio.server.serve(_echo_server, "127.0.0.1", 0) as server:
        host, port = server.sockets[0].getsockname()
        yield f"ws://{host}:{port}"


@pytest_asyncio.fixture
async def error_server_1002():
    async def handler(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "session.error", "message": "bad frame"}))
        await ws.close(code=1002)

    async with websockets.asyncio.server.serve(handler, "127.0.0.1", 0) as server:
        host, port = server.sockets[0].getsockname()
        yield f"ws://{host}:{port}"

@pytest_asyncio.fixture
async def policy_server():
    async def handler(ws):
        await ws.recv()
        await ws.send(
            json.dumps({"type": "session.error", "message": "voice override denied"})
        )
        await ws.close(code=1008)

    async with websockets.asyncio.server.serve(handler, "127.0.0.1", 0) as server:
        host, port = server.sockets[0].getsockname()
        yield f"ws://{host}:{port}"

# Tests
@pytest.mark.asyncio
async def test_start_session_success(echo_server):
    async with VoicekitClient(echo_server) as client:
        await client.start_session(SessionConfig(voice="af_heart"))
        assert client.connected


@pytest.mark.asyncio
async def test_start_session_defaults(echo_server):
    async with VoicekitClient(echo_server) as client:
        await client.start_session()  # no config - server uses defaults
        assert client.connected

@pytest.mark.asyncio
async def test_audio_roundtrip(echo_server):
    samples = np.random.uniform(-1.0, 1.0, FRAME_SAMPLES).astype(np.float32)

    async with VoicekitClient(echo_server) as client:
        await client.start_session()
        await client.send_audio(samples)

        received = []
        async for frame in client.receive_audio():
            received.append(frame)
            break  # one echo is enough

    assert len(received) == 1
    np.testing.assert_array_almost_equal(samples, received[0])

@pytest.mark.asyncio
async def test_handshake_error_raises(error_server_1002):
    async with VoicekitClient(error_server_1002) as client:
        with pytest.raises(HandshakeError, match="bad frame"):
            await client.start_session()

@pytest.mark.asyncio
async def test_policy_error_raises(policy_server):
    async with VoicekitClient(policy_server) as client:
        with pytest.raises(PolicyError, match="voice override denied"):
            await client.start_session(SessionConfig(voice="bf_isabella"))

@pytest.mark.asyncio
async def test_not_connected_raises():
    client = VoicekitClient("ws://127.0.0.1:1")  # not connected
    from voicekit.exceptions import NotConnectedError

    with pytest.raises(NotConnectedError):
        await client.send_audio(np.zeros(FRAME_SAMPLES, dtype=np.float32))
