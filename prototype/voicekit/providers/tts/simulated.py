import asyncio
import math
import struct
from typing import AsyncIterator

from voicekit.providers.base import TTSProvider

SAMPLE_RATE = 24000
TONE_HZ = 440


class SimulatedTTS(TTSProvider):
    """
    Simulates a real TTS model without loading anything.

    Realistic behaviour:
    - Buffers incoming text tokens into sentences before synthesising
      so the sentence-boundary logic is exercised under real conditions
    - Generates actual valid WAV audio (a sine wave) so downstream
      audio handling code can be fully tested — not silence, real bytes
    - Streams audio in chunks to simulate progressive generation
    - Applies configurable latency for time-to-first-audio testing
    """

    def __init__(self, voice: str = "default", latency_ms: int = 80):
        self.voice = voice
        self.latency_ms = latency_ms
        self._loaded = False

    async def load(self) -> None:
        await asyncio.sleep(0.3)
        self._loaded = True

    async def synthesize(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        buffer = ""

        async for token in text_stream:
            buffer += token

            # synthesise on sentence boundaries — same logic real TTS uses
            # sending one token at a time produces poor prosody
            # waiting for the full response kills latency
            # sentence boundaries are the right compromise
            if buffer.rstrip().endswith((".", "!", "?", ",")):
                async for chunk in self._synthesize_sentence(buffer.strip()):
                    yield chunk
                buffer = ""

        # flush anything remaining after the stream ends
        if buffer.strip():
            async for chunk in self._synthesize_sentence(buffer.strip()):
                yield chunk

    async def _synthesize_sentence(self, text: str) -> AsyncIterator[bytes]:
        # simulate time to first audio chunk
        await asyncio.sleep(self.latency_ms / 1000)

        # generate a real sine wave proportional to word count
        # this is actual playable audio, not silence
        words = len(text.split())
        duration_seconds = max(words * 0.25, 0.3)
        num_samples = int(SAMPLE_RATE * duration_seconds)

        samples = [
            int(32767 * math.sin(2 * math.pi * TONE_HZ * i / SAMPLE_RATE))
            for i in range(num_samples)
        ]
        raw_pcm = struct.pack(f"{num_samples}h", *samples)
        wav_bytes = self._build_wav(raw_pcm)

        # stream in 100ms chunks to simulate progressive generation
        chunk_size = SAMPLE_RATE * 2 // 10  # 100ms of 16-bit audio
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i:i + chunk_size]
            await asyncio.sleep(0.04)

    def _build_wav(self, pcm_data: bytes) -> bytes:
        num_channels = 1
        bits_per_sample = 16
        byte_rate = SAMPLE_RATE * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(pcm_data)

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            SAMPLE_RATE,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + pcm_data

    async def health(self) -> bool:
        return self._loaded