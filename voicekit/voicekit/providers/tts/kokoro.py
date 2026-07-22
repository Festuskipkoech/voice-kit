import asyncio
import logging
import io
import re
from typing import AsyncIterator

import numpy as np
import soundfile as sf

from voicekit.providers.base import TTSProvider

FLUSH_CHARS = re.compile(r"[.!?,;:\n]")
MAX_BUFFER_TOKENS = 3
MIN_BUFFER_CHARS = 1
SAMPLE_RATE = 24000
MODEL_PATH = "/app/models/kokoro-v1.0.onnx"
VOICES_PATH = "/app/models/voices-v1.0.bin"

class KokoroTTS(TTSProvider):
    """
    Kokoro TTS via kokoro-onnx — true streaming on CPU.

    Uses create_stream() which is a native async generator that yields
    audio chunks during inference, not after full synthesis completes.
    Each chunk arrives in ~200-500ms on CPU, enabling real-time playback.

    Model files are downloaded once into /app/models/ during Docker build.
    """
    def __init__(self, voice: str = "af_bella", lang: str = "en-us"):
        self.voice = voice
        self.lang = lang
        self._kokoro = None

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from kokoro_onnx import Kokoro
        self._kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

    async def synthesize(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """
        Stream LLM tokens, yield audio bytes as kokoro-onnx generates them.

        Buffers tokens until flush condition then calls create_stream()
        which yields audio chunks progressively during ONNX inference.
        First audio arrives within 200-500ms of first flush on CPU.
        """
        buffer = []
        first_chunk = True

        async def synthesise_phrase(text: str) -> AsyncIterator[bytes]:
            nonlocal first_chunk
            text = text.strip()
            if not text:
                return

            stream = self._kokoro.create_stream(
                text,
                voice=self.voice,
                speed=1.0,
                lang=self.lang,
            )

            async for samples, sample_rate in stream:
                if samples is None or len(samples) == 0:
                    continue

                audio = samples.astype(np.float32)

                if first_chunk:
                    buf = io.BytesIO()
                    sf.write(
                        buf, audio, sample_rate,
                        format="WAV", subtype="FLOAT"
                    )
                    yield buf.getvalue()
                    first_chunk = False
                else:
                    yield audio.tobytes()

        async for token in text_stream:
            buffer.append(token)
            joined = "".join(buffer)

            on_punctuation = bool(FLUSH_CHARS.search(token))   # no minimum — flush immediately
            on_max_tokens = len(buffer) >= MAX_BUFFER_TOKENS

            if on_punctuation or on_max_tokens:
                logging.getLogger(__name__).info(f"TTS flush: '{joined.strip()[:50]}'")
                async for chunk in synthesise_phrase(joined):
                    yield chunk
                buffer = []

        if buffer:
            async for chunk in synthesise_phrase("".join(buffer)):
                yield chunk

    async def health(self) -> bool:
        return self._kokoro is not None