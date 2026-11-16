"""
VoicekitClient - async WebSocket client for the Voicekit server.

Typical usage::

    from voicekit import VoicekitClient, SessionConfig
    from voicekit.audio import AudioCapture, AudioPlayer

    async def main():
        async with VoicekitClient("ws://localhost:8080") as client:
            await client.start_session(SessionConfig(voice="af_heart"))

            async with AudioCapture() as cap, AudioPlayer() as player:
                async def send_loop():
                    async for frame in cap:
                        await client.send_audio(frame)

                async def recv_loop():
                    async for frame in client.receive_audio():
                        await player.play(frame)

                await asyncio.gather(send_loop(), recv_loop())
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import AudioCapture

import numpy as np
import websockets
import websockets.exceptions

from .exceptions import HandshakeError, NotConnectedError, PolicyError, ProtocolError
from .protocol import (
    SessionConfigFrame,
    SessionErrorFrame,
    SessionReadyFrame,
    decode_audio,
    decode_server_text,
    encode_audio,
)

# WebSocket close codes per PROTOCOL.md
_CLOSE_POLICY_VIOLATION = 1008


class VoicekitClient:
    """
    Async WebSocket client for a Voicekit server.

    Use as an async context manager to manage the connection lifetime::

        async with VoicekitClient("ws://localhost:8080") as client:
            await client.start_session(...)
            ...

    The connection is closed cleanly when the block exits, even on error.
    """

    def __init__(self, url: str, *, open_timeout: float = 10.0) -> None:
        """
        Parameters
        ----------
        url:
            WebSocket URL of the server, e.g. ``ws://localhost:8080`` or
            ``wss://host/``.
        open_timeout:
            Seconds to wait for the initial TCP + WebSocket handshake.
        """
        self._url = url
        self._open_timeout = open_timeout
        self._ws: websockets.ClientConnection | None = None

    # Context manager
    async def __aenter__(self) -> "VoicekitClient":
        self._ws = await websockets.connect(
            self._url,
            open_timeout=self._open_timeout,
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # Session handshake
    async def start_session(
        self,
        config: SessionConfigFrame | None = None,
    ) -> None:
        """
        Send ``session.config`` and wait for ``session.ready``.

        Must be called before sending any audio.  Raises HandshakeError (or
        the more specific PolicyError) if the server rejects the config.

        Parameters
        ----------
        config:
            Session overrides.  Pass ``None`` to use all server defaults.
        """
        ws = self._require_ws()
        frame = config if config is not None else SessionConfigFrame()
        await ws.send(frame.encode())

        raw = await ws.recv()
        if not isinstance(raw, str):
            raise ProtocolError(
                "expected a JSON text frame for session.ready, got binary"
            )

        server_frame = decode_server_text(raw)

        if isinstance(server_frame, SessionReadyFrame):
            return

        if isinstance(server_frame, SessionErrorFrame):
            # Inspect the close code on the wire to distinguish policy errors
            # from generic ones.  The server closes immediately after the error
            # frame, so we attempt one more recv to drain the close handshake.
            try:
                await ws.recv()
            except websockets.exceptions.ConnectionClosedError as exc:
                if exc.rcvd is not None and exc.rcvd.code == _CLOSE_POLICY_VIOLATION:
                    raise PolicyError(server_frame.message) from exc
            raise HandshakeError(server_frame.message)

        raise ProtocolError(f"unexpected frame during handshake: {server_frame!r}")

    # Audio I/O
    async def send_audio(self, samples: np.ndarray) -> None:
        """
        Send exactly 512 float32 samples to the server.

        Parameters
        ----------
        samples:
            1-D numpy array of 512 float32 values captured from the microphone.
        """
        ws = self._require_ws()
        await ws.send(encode_audio(samples))

    async def receive_audio(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator that yields audio frames as they arrive from the server.

        Each yielded value is a 1-D float32 numpy array.  The generator
        stops when the connection is closed normally.

        Raises ProtocolError if the server sends an unexpected text frame
        while audio is expected.
        """
        ws = self._require_ws()
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    yield decode_audio(message)
                elif isinstance(message, str):
                    # Could be a late session.error - surface it properly
                    frame = decode_server_text(message)
                    if isinstance(frame, SessionErrorFrame):
                        raise HandshakeError(frame.message)
                    raise ProtocolError(
                        f"unexpected text frame during audio stream: {message!r}"
                    )
        except websockets.exceptions.ConnectionClosedOK:
            return

    # Convenience runner
    async def run(
        self,
        on_audio: Callable[[np.ndarray], Awaitable[None] | None],
        capture: AudioCapture | None = None,
    ) -> None:
        """
        Run send and receive loops concurrently until the connection closes.

        This is the simplest way to use the client for a full duplex session.
        ``start_session`` must be called before ``run``.

        Parameters
        ----------
        on_audio:
            Called with each TTS audio frame received from the server.
            May be a plain function or a coroutine function.
        capture:
            An already-opened ``AudioCapture`` instance.  When provided,
            microphone frames are sent to the server automatically.
            When omitted, only the receive loop runs (useful for testing
            or when the caller manages sending manually).

        Example::

            async with VoicekitClient("ws://localhost:8080") as client:
                await client.start_session(SessionConfig(voice="af_heart"))
                async with AudioCapture() as cap, AudioPlayer() as player:
                    await client.run(on_audio=player.play, capture=cap)
        """

        async def _send_loop() -> None:
            assert capture is not None
            async for frame in capture:
                await self.send_audio(frame)

        async def _recv_loop() -> None:
            async for frame in self.receive_audio():
                result = on_audio(frame)
                if asyncio.iscoroutine(result):
                    await result

        loops = [_recv_loop()]
        if capture is not None:
            loops.append(_send_loop())

        try:
            await asyncio.gather(*loops)
        except asyncio.CancelledError:
            pass

    # Low-level helpers
    def _require_ws(self) -> websockets.ClientConnection:
        if self._ws is None:
            raise NotConnectedError(
                "VoicekitClient is not connected - use 'async with VoicekitClient(...)'"
            )
        return self._ws

    @property
    def connected(self) -> bool:
        """True if the WebSocket connection is currently open."""
        return self._ws is not None and self._ws.state is websockets.State.OPEN
