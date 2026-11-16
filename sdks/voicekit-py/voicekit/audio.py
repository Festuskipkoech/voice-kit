"""
Audio I/O helpers built on sounddevice 0.5.6.

AudioCapture  - reads from the microphone, yields 512-sample f32 frames
                via an asyncio.Queue, decoupling the PortAudio callback
                thread from the event loop.

AudioPlayer   - receives f32 frames from an asyncio.Queue and plays them
                through the default output device.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
import sounddevice as sd

from .protocol import MIC_SAMPLE_RATE, TTS_SAMPLE_RATE, FRAME_SAMPLES


class AudioCapture:
    """
    Streams microphone audio as 512-sample float32 frames.

    Usage::

        async with AudioCapture() as cap:
            async for frame in cap:
                await client.send_audio(frame)
    """

    def __init__(
        self,
        device: Optional[int | str] = None,
        queue_maxsize: int = 64,
    ) -> None:
        self._device = device
        self._queue_maxsize = queue_maxsize
        self._queue: asyncio.Queue[Optional[np.ndarray]] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "AudioCapture":
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=MIC_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        # signal the iterator to stop
        await self._queue.put(None)

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        # indata shape: (FRAME_SAMPLES, 1) - squeeze to 1-D
        frame = indata[:, 0].copy()
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)

    def __aiter__(self) -> "AudioCapture":
        return self

    async def __anext__(self) -> np.ndarray:
        frame = await self._queue.get()
        if frame is None:
            raise StopAsyncIteration
        return frame


class AudioPlayer:
    """
    Plays audio frames received from the server.

    Usage::

        async with AudioPlayer() as player:
            async for audio_frame in received_frames:
                await player.play(audio_frame)
    """

    def __init__(self, device: Optional[int | str] = None) -> None:
        self._device = device
        self._stream: Optional[sd.OutputStream] = None

    async def __aenter__(self) -> "AudioPlayer":
        self._stream = sd.OutputStream(
            samplerate=TTS_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self._device,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def play(self, samples: np.ndarray) -> None:
        """
        Write samples to the output stream.  Runs the blocking write in a
        thread executor so the event loop is not stalled.
        """
        if self._stream is None:
            raise RuntimeError("AudioPlayer is not open")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._stream.write, samples.astype(np.float32))
