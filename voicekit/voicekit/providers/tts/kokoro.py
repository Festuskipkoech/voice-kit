"""
KokoroTTS — Kokoro ONNX text-to-speech provider.

Two-stage synthesis:
    1. misaki G2P converts the phrase to phonemes (synchronous, milliseconds)
    2. kokoro-onnx create_stream() synthesises audio from phonemes

Why misaki g2p:
    Kokoro is trained on phoneme sequences, not raw text. Without explicit
    phonemisation, Kokoro does its own internal text-to-phoneme guess which
    stumbles on abbreviations, numbers, proper nouns, and anything unusual.
    misaki handles all standard English correctly. The espeak fallback handles
    out-of-vocabulary words that misaki does not recognise.

    espeak-ng is already a system dependency in the TTS Dockerfile — no new
    Docker dependency is required.

Why no executor:
    create_stream() is a native async generator. It manages its own internal
    threading for ONNX inference and yields chunks as they complete. Wrapping
    it in run_in_executor + a nested event loop would fight the language.
    We simply async for over it directly — clean, idiomatic, and safe.
    Exceptions propagate naturally to the caller (synthesise_phrases in
    remote_pipeline) which is already inside asyncio.gather with
    return_exceptions=True.

Audio output format:
    Chunk 1: complete WAV file (44-byte header + float32 PCM, 24000 Hz mono)
    Chunk 2+: raw float32 PCM bytes, no header, 24000 Hz mono

Model files are downloaded once into /app/models/ during Docker build.
"""
import asyncio
import io
from typing import AsyncIterator

import os
import numpy as np
import soundfile as sf

from voicekit.providers.base import TTSProvider

SAMPLE_RATE = 24000
MODEL_PATH  = os.environ.get("KOKORO_MODEL_PATH",  "models/kokoro-v1.0.onnx")
VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", "models/voices-v1.0.bin")

class KokoroTTS(TTSProvider):
    """
    Kokoro TTS via kokoro-onnx with misaki G2P phonemisation.

    load() initialises both the Kokoro ONNX model and the misaki G2P engine
    once at service startup. Both are reused across all synthesis requests.

    synthesize() accepts a single clean phrase string (already stripped of
    markers and markdown by PhraseStream + clean_for_tts). It converts to
    phonemes via misaki, then streams audio via create_stream().
    """
    def __init__(self, voice: str = "af_bella", lang: str = "en-us") -> None:
        self.voice   = voice
        self.lang    = lang
        self._kokoro = None
        self._g2p    = None

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        from kokoro_onnx import Kokoro
        from misaki import en, espeak

        self._kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

        fallback  = espeak.EspeakFallback(british=False)
        self._g2p = en.G2P(trf=False, british=False, fallback=fallback)

    async def synthesize(self, phrase: str) -> AsyncIterator[bytes]:
        """
        Synthesise a single clean phrase to audio chunks.

        Phrase arrives already cleaned by PhraseStream — no markers,
        no markdown, plain spoken English only.

        Steps:
            1. Strip and validate — skip empty phrases silently
            2. G2P — convert text to phonemes via misaki (sync, fast)
            3. create_stream() — async generator yields audio chunks
            4. First chunk: write WAV header via soundfile, yield complete WAV
               Subsequent chunks: yield raw float32 PCM bytes
        """
        phrase = phrase.strip()
        if not phrase:
            return

        phonemes, _ = self._g2p(phrase)

        first_chunk = True

        async for samples, sample_rate in self._kokoro.create_stream(
            phonemes,
            voice=self.voice,
            speed=1.0,
            lang=self.lang,
            is_phonemes=True,
        ):
            if samples is None or len(samples) == 0:
                continue

            audio = samples.astype(np.float32)

            if first_chunk:
                buf = io.BytesIO()
                sf.write(buf, audio, sample_rate, format="WAV", subtype="FLOAT")
                yield buf.getvalue()
                first_chunk = False
            else:
                yield audio.tobytes()

    async def health(self) -> bool:
        return self._kokoro is not None and self._g2p is not None