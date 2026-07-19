import asyncio
from typing import AsyncIterator

import numpy as np

from voicekit.providers.base import STTProvider

class WhisperSTT(STTProvider):
    """
    Whisper STT via faster-whisper (CTranslate2 reimplementation).

    Runs on CPU with int8 quantization — no GPU required for small/tiny.
    Model loads once at startup and serves all requests from memory.

    Hallucination suppression:
        Whisper is known to hallucinate on silence and noise —
        producing phrases like 'Thank you.', 'you', 'Bye.' etc.
        This is suppressed via:
            no_speech_threshold=0.6   rejects low-confidence segments
            log_prob_threshold=-1.0   filters low probability outputs
            vad_filter=True           faster-whisper's internal VAD

    Lazy generator fix:
        faster-whisper.transcribe() returns a lazy generator.
        Iterating it outside run_in_executor runs inference on the
        event loop — blocking all other coroutines.
        We force list() inside the executor to evaluate eagerly.

    Variant guide:
        tiny      390MB  ~20ms   development, testing
        base      740MB  ~30ms   low-resource servers
        small     1.5GB  ~150ms  recommended production default
        medium    3GB    ~300ms  higher accuracy
        large-v3  6GB    ~600ms  best accuracy, GPU recommended
    """

    def __init__(self, variant: str = "small"):
        self.variant = variant
        self.model = None

    async def load(self) -> None:
        """
        Load Whisper model into memory.
        Runs in executor — model loading blocks for 3-10 seconds.
        Called once at service startup, never per request.
        """
        from faster_whisper import WhisperModel

        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.variant,
                device="cpu",
                compute_type="int8",
            )
        )

    async def transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        """
        Transcribe audio stream to text tokens.

        Collects all audio chunks first — Whisper works on complete
        utterances not streaming chunks. Silero VAD upstream ensures
        only speech arrives here in production. In tests, VAD may be
        disabled so no_speech_threshold suppresses hallucinations.

        Returns immediately with no output if audio stream is empty
        or if Whisper determines the audio contains no speech.
        """
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        if not chunks:
            return

        audio = np.concatenate(chunks)

        loop = asyncio.get_event_loop()

        # force list() inside executor — faster-whisper returns a lazy
        # generator. iterating outside executor runs inference on the
        # event loop which blocks all other coroutines.
        segments = await loop.run_in_executor(
            None,
            lambda: list(self.model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                no_speech_threshold=0.6,   # suppress hallucinations on silence
                log_prob_threshold=-1.0,   # filter low probability outputs
            )[0])   # [0] = segments generator, [1] = TranscriptionInfo
        )

        for segment in segments:
            text = segment.text.strip()
            if text:
                yield text

    async def health(self) -> bool:
        return self.model is not None