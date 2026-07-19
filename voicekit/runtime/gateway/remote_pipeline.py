"""
Remote Pipeline

The gateway's pipeline implementation that calls STT and TTS
services over WebSocket rather than loading models locally.

True end-to-end streaming:
    TTS generates chunk → immediately placed in audio_out queue
    session.py drains audio_out concurrently while pipeline runs
    Client receives audio chunks as they are generated
    Transcript, response, metrics arrive after all audio is sent

    This means the client starts playing audio while TTS is still
    generating — exactly like a real phone call.

Architecture:
    Gateway (no models, lightweight)
        WebSocket
    STT service (Whisper loaded here)
        transcript
    LLM API (Claude/OpenAI)
        tokens stream
    TTS service (Chatterbox loaded here)
        audio chunks stream → audio_out queue → client
"""
import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator

import websockets

log = logging.getLogger(__name__)

STT_WS_URL = os.environ.get("STT_WS_URL", "ws://stt:8001/stt")
TTS_WS_URL = os.environ.get("TTS_WS_URL", "ws://tts:8002/tts")

# sentinel value placed in audio_out when pipeline is done
# signals the concurrent audio sender in session.py to stop
AUDIO_DONE = None

class PipelineMetrics:
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
            f"TTS_first={self.tts_first_chunk_ms:.0f}ms "
            f"total={self.total_ms:.0f}ms"
        )

class RemotePipeline:
    """
    Voice pipeline that delegates STT and TTS to remote services.
    The gateway only handles session management, LLM calls, and routing.

    True streaming: audio chunks flow into audio_out as TTS generates
    them. session.py drains audio_out concurrently so the client
    receives audio while synthesis is still in progress.
    """

    def __init__(self, config):
        self.config = config
        self.history: list[dict] = []
        self._loaded = False
        self.turn_in_progress = asyncio.Event()
        self._llm = self._init_llm(config)

    def _init_llm(self, config):
        if config.llm.provider == "anthropic":
            from voicekit.providers.llm.claude import ClaudeProvider
            return ClaudeProvider(model=config.llm.model)
        elif config.llm.provider == "openai":
            from voicekit.providers.llm.openai_provider import OpenAIProvider
            return OpenAIProvider(model=config.llm.model)
        elif config.llm.provider == "simulated":
            from voicekit.providers.llm.simulated import SimulatedLLM
            return SimulatedLLM(model=config.llm.model)
        else:
            raise ValueError(f"Unknown LLM provider: '{config.llm.provider}'")

    async def load(self) -> None:
        try:
            async with websockets.connect(STT_WS_URL, open_timeout=5) as ws:
                await ws.close()
        except Exception as e:
            raise RuntimeError(
                f"STT service not reachable at {STT_WS_URL}: {e}"
            )

        try:
            async with websockets.connect(TTS_WS_URL, open_timeout=5) as ws:
                await ws.close()
        except Exception as e:
            raise RuntimeError(
                f"TTS service not reachable at {TTS_WS_URL}: {e}"
            )

        self._loaded = True
        log.info(
            f"Remote pipeline ready — "
            f"STT: {STT_WS_URL} TTS: {TTS_WS_URL}"
        )

    async def health(self) -> dict:
        return {"stt": True, "tts": True, "loaded": self._loaded}

    async def run_turn(
        self,
        audio_in: asyncio.Queue,
        audio_out: asyncio.Queue,
    ) -> PipelineMetrics:
        """
        Execute one complete voice turn.

        audio_in  — queue of np.ndarray chunks from client
        audio_out — queue where audio chunks are placed as TTS generates them
                    session.py drains this concurrently while we run
                    a None sentinel is placed when all audio is done

        Returns PipelineMetrics after all audio has been queued.
        """
        if not self._loaded:
            raise RuntimeError(
                "Pipeline not loaded. Call await pipeline.load() first."
            )

        self.turn_in_progress.set()
        metrics = PipelineMetrics()
        turn_start = time.perf_counter()

        try:
            # phase 1 — STT
            stt_start = time.perf_counter()
            transcript = await self._transcribe(audio_in)
            metrics.stt_ms = (time.perf_counter() - stt_start) * 1000
            metrics.transcript = transcript

            if not transcript.strip():
                metrics.total_ms = (time.perf_counter() - turn_start) * 1000
                return metrics

            self.history.append({"role": "user", "content": transcript})

            # phase 2+3 — LLM streaming into TTS concurrently
            # audio chunks flow into audio_out as TTS generates them
            # session.py is draining audio_out simultaneously
            llm_token_queue: asyncio.Queue[str | None] = asyncio.Queue()
            full_response_parts: list[str] = []
            llm_start = time.perf_counter()
            first_token_ref: list[float] = []
            tts_first_chunk_ref: list[float] = []

            async def feed_llm():
                async for token in self._llm.stream(
                    messages=self.history,
                    system=self.config.system_prompt,
                ):
                    if not first_token_ref:
                        first_token_ref.append(time.perf_counter())
                    full_response_parts.append(token)
                    await llm_token_queue.put(token)
                await llm_token_queue.put(None)

            async def token_stream() -> AsyncIterator[str]:
                while True:
                    token = await llm_token_queue.get()
                    if token is None:
                        break
                    yield token

            async def feed_tts():
                first_chunk = True
                async for audio_chunk in self._synthesize(token_stream()):
                    if first_chunk:
                        tts_first_chunk_ref.append(time.perf_counter())
                        first_chunk = False
                    # place chunk in queue immediately as it arrives
                    # session.py receives and sends to client right away
                    await audio_out.put(audio_chunk)

            # LLM and TTS run concurrently
            # audio chunks flow to client while LLM is still generating
            await asyncio.gather(feed_llm(), feed_tts())

            full_response = "".join(full_response_parts)
            metrics.response = full_response
            metrics.llm_first_token_ms = (
                (first_token_ref[0] - llm_start) * 1000
                if first_token_ref else 0
            )
            metrics.llm_total_ms = (time.perf_counter() - llm_start) * 1000
            metrics.tts_first_chunk_ms = (
                (tts_first_chunk_ref[0] - llm_start) * 1000
                if tts_first_chunk_ref else 0
            )
            metrics.total_ms = (time.perf_counter() - turn_start) * 1000

            self.history.append({"role": "assistant", "content": full_response})
            return metrics

        finally:
            # always place sentinel — tells session.py audio is done
            # this runs even if an exception occurred
            await audio_out.put(AUDIO_DONE)
            self.turn_in_progress.clear()

    async def _transcribe(self, audio_in: asyncio.Queue) -> str:
        transcript = ""

        async with websockets.connect(STT_WS_URL) as ws:
            while True:
                try:
                    chunk = audio_in.get_nowait()
                    await ws.send(chunk.tobytes())
                except asyncio.QueueEmpty:
                    break

            await ws.send(json.dumps({"type": "end"}))

            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message["type"] == "done":
                    transcript = message.get("transcript", "")
                    break
                elif message["type"] == "error":
                    log.error(f"STT error: {message['message']}")
                    break

        return transcript

    async def _synthesize(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        """
        Stream text tokens to TTS service.
        Yield audio chunks immediately as TTS generates them.
        True streaming — no buffering.
        """
        async with websockets.connect(
            TTS_WS_URL,
            ping_interval=None,
            ping_timeout=None,
        ) as ws:

            async def send_tokens():
                async for token in text_stream:
                    await ws.send(json.dumps({
                        "type": "token",
                        "text": token,
                    }))
                await ws.send(json.dumps({"type": "end"}))

            send_task = asyncio.create_task(send_tokens())

            async for message in ws:
                if isinstance(message, bytes):
                    yield message
                elif isinstance(message, str):
                    data = json.loads(message)
                    if data["type"] == "done":
                        break
                    elif data["type"] == "error":
                        log.error(f"TTS error: {data['message']}")
                        break

            await send_task