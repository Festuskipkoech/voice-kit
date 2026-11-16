"""
Wire-protocol frame definitions and codec.

Text frames are JSON.  Binary frames are raw f32-LE audio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


MIC_SAMPLE_RATE = 16_000  # client -> server: microphone audio
TTS_SAMPLE_RATE = 24_000  # server -> client: TTS output
FRAME_SAMPLES = 512  # mic frames are always exactly 512 samples
FRAME_BYTES = FRAME_SAMPLES * 4  # 4 bytes per f32


# Client -> Server frames
@dataclass
class SessionConfigFrame:
    """
    First frame the client must send.  All override fields are optional;
    omitting them causes the server to use its configured defaults.
    """

    system_prompt: Optional[str] = None
    voice: Optional[str] = None
    vad_threshold: Optional[float] = None
    silence_ms: Optional[int] = None

    def encode(self) -> str:
        payload: dict = {"type": "session.config"}
        if self.system_prompt is not None:
            payload["system_prompt"] = self.system_prompt
        if self.voice is not None:
            payload["voice"] = self.voice
        if self.vad_threshold is not None:
            payload["vad_threshold"] = self.vad_threshold
        if self.silence_ms is not None:
            payload["silence_ms"] = self.silence_ms
        return json.dumps(payload)


# Server -> Client frames
@dataclass
class SessionReadyFrame:
    pass


@dataclass
class SessionErrorFrame:
    message: str

# Codec
def decode_server_text(raw: str) -> SessionReadyFrame | SessionErrorFrame:
    """Parse a JSON text frame from the server."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"server sent malformed JSON: {exc}") from exc

    frame_type = obj.get("type")
    if frame_type == "session.ready":
        return SessionReadyFrame()
    if frame_type == "session.error":
        return SessionErrorFrame(message=obj.get("message", "(no message)"))
    raise ValueError(f"unknown server frame type: {frame_type!r}")

def encode_audio(samples: np.ndarray) -> bytes:
    """
    Pack a numpy float32 array of exactly FRAME_SAMPLES values into wire bytes.
    Raises ValueError if the shape is wrong.
    """
    arr = np.asarray(samples, dtype=np.float32).ravel()
    if arr.size != FRAME_SAMPLES:
        raise ValueError(
            f"audio frame must be exactly {FRAME_SAMPLES} samples, got {arr.size}"
        )
    return arr.astype("<f4").tobytes()

def decode_audio(data: bytes) -> np.ndarray:
    """
    Unpack a binary wire frame into a float32 numpy array.
    The length does not have to be exactly FRAME_BYTES; the server may
    send multi-frame chunks at the end of a TTS utterance.
    """
    n = len(data) // 4
    return np.frombuffer(data, dtype="<f4")[:n].copy()
