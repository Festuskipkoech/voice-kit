
"""
Kokoro TTS provider.
 
82M parameter model, Apache 2.0, runs on CPU at 2x real-time.
Native streaming — yields audio chunks as they are generated.
First audio arrives within 500-800ms of the first sentence boundary.
 
Why Kokoro over Chatterbox for CPU:
    Chatterbox on CPU: 90-150 seconds to first audio chunk
    Kokoro on CPU:500-800ms to first audio chunk
    Kokoro achieves true real-time streaming on CPU hardware.
 
Tradeoff:
    Kokoro does not support voice cloning — 54 fixed voices.
    Chatterbox supports voice cloning from a reference audio clip.
    For streaming latency on CPU, Kokoro is the correct choice.
  
Voices (selected):
    af_bella    — American female, warm
    af_sarah    — American female, clear
    am_adam     — American male, deep
    am_michael  — American male, natural
    bf_emma     — British female, formal
    bm_george   — British male, authoritative
 
Streaming pattern:
    Tokens arrive from LLM one by one.
    Buffer until sentence boundary (. ! ? ,) or buffer > 64 tokens.
    Phonemize buffered text.
    Run Kokoro inference — fast on CPU, RTF ~0.47.
    Yield audio chunks immediately as generator produces them.
    User hears first audio within 500-800ms of first sentence.
"""
import asyncio
import io
import re
from typing import AsyncIterator
 
import numpy as np
import soundfile as sf
 
from voicekit.providers.base import TTSProvider
 
SENTENCE_END = re.compile(r"[.!?,;:]")
MAX_BUFFER_TOKENS = 64
SAMPLE_RATE = 24000 
 
class KokoroTTS(TTSProvider):
    """
    Kokoro TTS — true streaming on CPU.
 
    Uses KPipeline from the kokoro package which returns a generator
    that yields (graphemes, phonemes, audio) tuples. We take the audio
    and yield it directly — no buffering of the full response.
 
    Each sentence is synthesised independently as it arrives from the
    LLM token stream. First audio arrives after the first sentence
    boundary — typically within 500-800ms on CPU.
    """
 
    def __init__(self, voice: str = "af_bella", lang: str = "a"):
        """
        voice — Kokoro voice ID (default: af_bella)
        lang  — language code: 'a' = American English, 'b' = British English
                'j' = Japanese, 'z' = Mandarin Chinese
        """
        self.voice = voice
        self.lang = lang
        self._pipeline = None
 
    async def load(self) -> None:
        """
        Load Kokoro pipeline into memory.
        Runs in executor — first load downloads model weights (~327MB).
        Subsequent loads use the local cache and complete in ~2 seconds.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)
 
    def _load_sync(self) -> None:
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code=self.lang)
 
    async def synthesize(
        self,
        text_stream: AsyncIterator[str],
    ) -> AsyncIterator[bytes]:
        """
        Stream text tokens from LLM, yield WAV audio chunks as generated.
 
        Buffers tokens until a sentence boundary then synthesises
        immediately. Each sentence is synthesised and streamed back
        before the next sentence is buffered — true end-to-end streaming.
 
        Yields:
            chunk 1   — complete WAV bytes (header + float32 PCM)
            chunk 2..N — raw float32 PCM bytes (no header)
 
        This matches the Chatterbox output format exactly so clients
        do not need to change how they handle audio chunks.
        """
        buffer = []
        first_chunk = True
 
        async def synthesise_buffer(text: str) -> AsyncIterator[bytes]:
            nonlocal first_chunk
            text = text.strip()
            if not text:
                return
 
            loop = asyncio.get_event_loop()
 
            # run Kokoro inference in executor — blocks event loop otherwise
            chunks = await loop.run_in_executor(
                None,
                lambda: list(self._synthesise_sync(text))
            )
 
            for audio in chunks:
                if audio is None or len(audio) == 0:
                    continue
 
                if first_chunk:
                    # first chunk — encode as complete WAV with header
                    buf = io.BytesIO()
                    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="FLOAT")
                    yield buf.getvalue()
                    first_chunk = False
                else:
                    # subsequent chunks — raw float32 PCM, no header
                    yield audio.astype(np.float32).tobytes()
 
        async for token in text_stream:
            buffer.append(token)
            joined = "".join(buffer)
 
            # flush on sentence boundary or large buffer
            if SENTENCE_END.search(token) or len(buffer) >= MAX_BUFFER_TOKENS:
                async for chunk in synthesise_buffer(joined):
                    yield chunk
                buffer = []
 
        # flush remaining buffer
        if buffer:
            async for chunk in synthesise_buffer("".join(buffer)):
                yield chunk
 
    def _synthesise_sync(self, text: str):
        """
        Run Kokoro inference synchronously.
        Called inside run_in_executor — must not use async.
        Returns list of numpy arrays (one per internal segment).
        """
        results = []
        for _, _, audio in self._pipeline(text, voice=self.voice):
            if audio is not None and len(audio) > 0:
                results.append(audio)
        return results
 
    async def health(self) -> bool:
        return self._pipeline is not None
