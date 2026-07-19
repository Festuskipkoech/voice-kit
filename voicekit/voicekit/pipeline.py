import asyncio
import time
from typing import AsyncIterator
 
import numpy as np
 
from voicekit.config import VoiceConfig
from voicekit.providers.base import STTProvider, TTSProvider, LLMProvider
from voicekit.providers.registry import get_stt, get_tts, get_llm
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
        self.transcript: str = ""
        self.response: str = ""
    
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
        audio in VAD STT LLM TTS audio out
 
    The pipeline only knows about provider interfaces — never
    concrete implementations. Swapping a model changes the registry
    and the config file. Nothing here changes.
 
    Streaming cascade:
        LLM tokens flow into TTS as they arrive.
        TTS audio flows into the output queue as it generates.
        First audio plays before the LLM has finished responding.
        This is what makes the agent feel real-time.
    """

    def __init__(self, config: VoiceConfig):
        self.config = config
        self.stt: STTProvider = get_stt(config)        
        self.tts: TTSProvider = get_tts(config)
        self.llm: LLMProvider = get_llm(config)
        self.vad: VADProcessor = VADProcessor(
            sensitivity=config.vad.sensitivity
        )
        self.history: list[dict] = []
        self._loaded = False

    async def load(self) -> None:
        """
        Load STT and TTS models concurrently.
        Must be called before run_turn.
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
        audio_in: asyncio.Queue,
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

        # phase 1: STT
        # drain the audio queue through VAD then into STT

        stt_start = time.perf_counter()
        transcript = await self._run_stt(audio_in)
        metrics.stt_ms = (time.perf_counter() - stt_start) * 1000
        metrics.transcript = transcript

        if not transcript.strip():
            # nothing was said — end turn cleanly with no output
            metrics.total_ms = (time.perf_counter() - turn_start) * 1000
            return metrics
 
        self.history.append({"role": "user", "content": transcript})

        # phase 2+3: LLM streaming into TTS concurrently
        # this is the streaming cascade:
        # LLM tokens feed into a queue
        # TTS reads from that queue as tokens arrive
        # TTS audio feeds into audio_out as chunks are generated
        # all three happen simultaneously

        llm_token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_response_parts: list[str] = []

        llm_start = time.perf_counter()
        tts_start_ref: list[float] = []
        first_token_ref: list[float] = []

        async def feed_llm():
            async for token in self.llm.stream(
                messages=self.history,
                system=self.config.system_prompt,
            ):
                if not first_token_ref:
                    first_token_ref.append(time.perf_counter())
                full_response_parts.append(token)
                await llm_token_queue.put(token)
            await llm_token_queue.put(None)  # sentinel — stream is done
 
        async def token_stream() -> AsyncIterator[str]:
            while True:
                token = await llm_token_queue.get()
                if token is None:
                    break
                yield token
 
        async def feed_tts():
            first_chunk = True
            async for audio_chunk in self.tts.synthesize(token_stream()):
                if first_chunk:
                    tts_start_ref.append(time.perf_counter())
                    first_chunk = False
                await audio_out.put(audio_chunk)
        
        # run LLM feeding and TTS consuming concurrently
        await asyncio.gather(feed_llm(), feed_tts())

        full_response = "".join(full_response_parts)
        metrics.response = full_response
        metrics.llm_first_token_ms = (
            (first_token_ref[0] - llm_start) * 1000
            if first_token_ref else 0
        )
        metrics.llm_first_token_ms = (time.perf_counter() - llm_start) * 1000
        metrics.tts_first_chunk_ms = (
            (tts_start_ref[0] - llm_start) * 1000
            if tts_start_ref else 0
        )
        metrics.tts_total_ms = (time.perf_counter() - llm_start) * 1000
        metrics.total_ms = (time.perf_counter() - turn_start) * 1000

        self.history.append({"role": "assistant", "content": full_response})

        return metrics
    
    async def _run_stt(self, audio_in: asyncio.Queue) -> str:
        """
        Drain the audio queue, apply VAD, transcribe what remains.
        """
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

        transcript_parts =[]
        async for token in self.stt.transcribe(vad_filtered()):
            transcript_parts.append(token)
        
        return "".join(transcript_parts).strip()
 