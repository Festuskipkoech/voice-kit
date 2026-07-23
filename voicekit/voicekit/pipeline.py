import asyncio
import time
from typing import AsyncIterator

import numpy as np

from voicekit.config import VoiceConfig
from voicekit.providers.base import STTProvider, TTSProvider, LLMProvider
from voicekit.providers.registry import get_stt, get_tts, get_llm
from voicekit.utils.phrase_splitter import PhraseStream
from voicekit.vad import VADProcessor

class PipelineMetrics:
    """Timing data captured on every turn for observability."""

    def __init__(self):
        self.stt_ms: float = 0
        self.llm_first_token_ms: float = 0
        self.llm_total_ms: float = 0
        self.tts_first_chunk_ms: float = 0
        self.tts_total_ms: float = 0
        self.total_ms: float = 0
        self.transcript: str   = ""
        self.response: str   = ""

    def __str__(self) -> str:
        return (
            f"STT={self.stt_ms:.0f}ms "
            f"LLM_first={self.llm_first_token_ms:.0f}ms "
            f"LLM_total={self.llm_total_ms:.0f}ms "
            f"TTS_first={self.tts_first_chunk_ms:.0f}ms "
            f"total={self.total_ms:.0f}ms"
        )


class VoicePipeline:
    """
    Orchestrates the full voice turn:
        audio in → VAD → STT → LLM → PhraseStream → TTS → audio out

    The pipeline only knows about provider interfaces — never concrete
    implementations. Swapping a model changes the registry and config.
    Nothing here changes.

    Streaming cascade:
        LLM tokens flow into PhraseStream which detects <|BREAK|> markers
        and yields complete clean phrases. Each phrase is synthesised by
        TTS immediately. Audio chunks flow into audio_out as generated.
        First audio plays before the LLM has finished responding.

    Rate adaptation:
        phrase_queue(maxsize=2) creates automatic backpressure. When TTS
        is busy, the splitter blocks on put(). The pipeline breathes at
        exactly the rate TTS can consume phrases.
    """
    def __init__(self, config: VoiceConfig):
        self.config  = config
        self.stt: STTProvider  = get_stt(config)
        self.tts: TTSProvider  = get_tts(config)
        self.llm: LLMProvider  = get_llm(config)
        self.vad: VADProcessor = VADProcessor(sensitivity=config.vad.sensitivity)
        self.history: list[dict]  = []
        self._loaded = False

    async def load(self) -> None:
        """
        Load STT and TTS models concurrently.
        Must be called before run_turn().
        LLM has no load step — it is a stateless API call.
        """
        await asyncio.gather(
            self.stt.load(),
            self.tts.load(),
        )
        self._loaded = True

    async def health(self) -> dict:
        return {
            "stt": await self.stt.health(),
            "tts": await self.tts.health(),
            "loaded": self._loaded,
        }

    async def run_turn(
        self,
        audio_in:  asyncio.Queue,
        audio_out: asyncio.Queue,
    ) -> PipelineMetrics:
        """
        Execute one complete voice turn.

        audio_in  — queue of np.ndarray chunks (float32, 16kHz, mono)
        audio_out — queue that receives bytes of WAV audio to play back

        Returns PipelineMetrics with per-stage timing.
        """
        if not self._loaded:
            raise RuntimeError(
                "Pipeline not loaded. Call await pipeline.load() first."
            )

        metrics = PipelineMetrics()
        turn_start = time.perf_counter()

        # phase 1 — STT
        stt_start = time.perf_counter()
        transcript = await self._run_stt(audio_in)
        metrics.stt_ms = (time.perf_counter() - stt_start) * 1000
        metrics.transcript = transcript

        if not transcript.strip():
            metrics.total_ms = (time.perf_counter() - turn_start) * 1000
            return metrics

        self.history.append({"role": "user", "content": transcript})

        # phase 2+3 — LLM + PhraseStream + TTS concurrently
        #
        # phrase_queue(maxsize=2) is the rate controller:
        #   when TTS is busy, split_tokens() blocks on put()
        #   the pipeline breathes at exactly the rate TTS can consume
        phrase_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)

        raw_response_parts: list[str]   = []
        llm_start = time.perf_counter()
        first_token_ref: list[float] = []
        tts_first_chunk_ref: list[float] = []

        async def split_tokens() -> None:
            splitter = PhraseStream()

            async for token in self.llm.stream(
                messages=self.history,
                system=self.config.system_prompt,
            ):
                if not first_token_ref:
                    first_token_ref.append(time.perf_counter())

                raw_response_parts.append(token)

                async for phrase in splitter.feed(token):
                    await phrase_queue.put(phrase)

            async for phrase in splitter.flush():
                await phrase_queue.put(phrase)

            await phrase_queue.put(None)

        async def synthesise_phrases() -> None:
            first_chunk = True

            while True:
                phrase = await phrase_queue.get()
                if phrase is None:
                    break

                async for audio_chunk in self.tts.synthesize(phrase):
                    if first_chunk:
                        tts_first_chunk_ref.append(time.perf_counter())
                        first_chunk = False
                    await audio_out.put(audio_chunk)

        await asyncio.gather(split_tokens(), synthesise_phrases())

        raw_response      = "".join(raw_response_parts)
        metrics.response  = raw_response.replace("<|BREAK|>", "").strip()

        metrics.llm_first_token_ms = (
            (first_token_ref[0] - llm_start) * 1000
            if first_token_ref else 0
        )
        metrics.llm_total_ms = (time.perf_counter() - llm_start) * 1000
        metrics.tts_first_chunk_ms = (
            (tts_first_chunk_ref[0] - llm_start) * 1000
            if tts_first_chunk_ref else 0
        )
        metrics.tts_total_ms = (time.perf_counter() - llm_start) * 1000
        metrics.total_ms     = (time.perf_counter() - turn_start) * 1000

        self.history.append({"role": "assistant", "content": metrics.response})

        return metrics

    async def _run_stt(self, audio_in: asyncio.Queue) -> str:
        """Drain the audio queue, apply VAD, transcribe what remains."""

        async def vad_filtered() -> AsyncIterator[np.ndarray]:
            while True:
                try:
                    chunk = audio_in.get_nowait()
                    if self.config.vad.enabled:
                        if self.vad.is_speech(chunk):
                            yield chunk
                    else:
                        yield chunk
                except asyncio.QueueEmpty:
                    break

        transcript_parts = []
        async for token in self.stt.transcribe(vad_filtered()):
            transcript_parts.append(token)

        return "".join(transcript_parts).strip()