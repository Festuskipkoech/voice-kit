"""
Remote Pipeline

The gateway's pipeline implementation. Calls STT and TTS services over
WebSocket rather than loading models locally. The gateway stays lightweight,
fast-starting, and memory-efficient — models load once in their dedicated
services.

Streaming architecture:
    Phase 1 — STT
        Audio chunks from client - WebSocket - Whisper service - transcript
    Phase 2 — LLM + Splitter (concurrent)
        LLM streams tokens - PhraseStream detects <|BREAK|> markers
        - clean complete phrases queued into phrase_queue
    Phase 3 — TTS (concurrent with Phase 2)
        synthesise_phrases() pulls one phrase at a time from phrase_queue
        - WebSocket - SS service - audio chunks - audio_out queue
    Phase 4 — Audio sender (concurrent with Phases 2+3, lives in session.py)
        audio_out queue drained by send_audio_stream() in session.py
        - WebSocket - client receives audio while LLM still generating

Rate adaptation via bounded queue:
    phrase_queue has maxsize=2. When STT is busy synthesising, the queue
    fills and split_tokens() blocks naturally on put(). LLM tokens accumulate
    in PhraseStream's internal buffer until STT finishes and pulls the next
    phrase. The entire pipeline breathes at exactly the rate STT can consume.
    No timers, no polling, no measurement needed.

Clean termination:
    AUDIO_DONE sentinel placed in audio_out in the finally block — always runs
    even if an exception occurs mid-synthesis. send_audio_stream() in session.py
    stops on the sentinel. No race condition, no hanging consumers.

Ping protection:
    turn_in_progress asyncio.Event set at start of run_turn(), cleared in
    finally. Ping loop in services/ping.py skips entirely during active turns.
    STT on CPU can take 5-10 seconds per phrase — the ping must never close
    a connection mid-synthesis.
"""
import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator

import websockets

from voicekit.utils.phrase_splitter import PhraseStream

log = logging.getLogger(__name__)

STT_WS_URL = os.environ.get("STT_WS_URL", "ws://stt:8001/stt")
TTS_WS_URL = os.environ.get("TTS_WS_URL", "ws://tts:8002/tts")

AUDIO_DONE = None

class PipelineMetrics:
    def __init__(self):
        self.stt_ms              = 0.0
        self.llm_first_token_ms  = 0.0
        self.llm_total_ms        = 0.0
        self.tts_first_chunk_ms  = 0.0
        self.tts_total_ms        = 0.0
        self.total_ms            = 0.0
        self.transcript          = ""
        self.response            = ""

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
    """
    def __init__(self, config):
        self.config          = config
        self.history         = []
        self._loaded         = False
        self.turn_in_progress = asyncio.Event()
        self._llm            = self._init_llm(config)

    def _init_llm(self, config):
        if config.llm.provider == "anthropic":
            from voicekit.providers.llm.claude import ClaudeProvider
            return ClaudeProvider(model=config.llm.model)
        # elif config.llm.provider == "openai":
        #     from voicekit.providers.llm.openai_provider import OpenAIProvider
        #     return OpenAIProvider(model=config.llm.model)
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
        log.info(f"Remote pipeline ready — STT: {STT_WS_URL} TTS: {TTS_WS_URL}")

    async def health(self) -> dict:
        return {"stt": True, "tts": True, "loaded": self._loaded}

    async def run_turn(
        self,
        audio_in: asyncio.Queue,
        audio_out: asyncio.Queue,
    ) -> PipelineMetrics:
        """
        Execute one complete voice turn.

        audio_in  — queue of np.ndarray chunks from the client
        audio_out — queue where audio chunks are placed as TTS generates them
                    session.py drains this concurrently
                    AUDIO_DONE sentinel placed in finally block when complete

        Returns PipelineMetrics after all audio has been queued.
        """
        if not self._loaded:
            raise RuntimeError(
                "Pipeline not loaded. Call await pipeline.load() first."
            )

        self.turn_in_progress.set()
        metrics    = PipelineMetrics()
        turn_start = time.perf_counter()

        try:
            # phase 1 — STT
            stt_start        = time.perf_counter()
            transcript       = await self._transcribe(audio_in)
            metrics.stt_ms   = (time.perf_counter() - stt_start) * 1000
            metrics.transcript = transcript

            if not transcript.strip():
                metrics.total_ms = (time.perf_counter() - turn_start) * 1000
                return metrics

            self.history.append({"role": "user", "content": transcript})

            # phrase_queue carries complete clean phrases from splitter to TTS
            # maxsize=2 creates automatic backpressure:
            #   when STT is busy, split_tokens() blocks on put()
            #   tokens accumulate in PhraseStream's buffer until space opens
            #   the pipeline breathes at exactly the rate STT can consume
            phrase_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)

            # raw token accumulation for metrics.response
            # clean_for_tts is not called here — PhraseStream handles cleaning
            # we strip <|BREAK|> once when assembling the final response string
            raw_response_parts: list[str] = []

            llm_start            = time.perf_counter()
            first_token_ref:      list[float] = []
            tts_first_chunk_ref:  list[float] = []

            async def split_tokens() -> None:
                """
                Stream LLM tokens through PhraseStream.
                PhraseStream detects <|BREAK|> markers, splits into phrases,
                cleans each phrase, and yields them one at a time.
                Each phrase is put into phrase_queue for synthesise_phrases().
                Sends None sentinel when the LLM stream is exhausted.
                """
                splitter = PhraseStream()

                async for token in self._llm.stream(
                    messages=self.history,
                    system=self.config.system_prompt,
                ):
                    if not first_token_ref:
                        first_token_ref.append(time.perf_counter())

                    raw_response_parts.append(token)

                    async for phrase in splitter.feed(token):
                        await phrase_queue.put(phrase)

                # flush any remaining buffer after LLM stream ends
                async for phrase in splitter.flush():
                    await phrase_queue.put(phrase)

                # sentinel tells synthesise_phrases() to stop
                await phrase_queue.put(None)

            async def synthesise_phrases() -> None:
                """
                Pull one phrase at a time from phrase_queue.
                Send each phrase to the TTS service and put audio chunks
                into audio_out as they arrive.

                Blocking on phrase_queue.get() when the queue is empty is
                intentional — this is where the adaptive rate happens.
                When STT is slow, phrases queue up behind it. When fast,
                they are consumed as fast as the LLM produces them.
                """
                first_chunk = True

                while True:
                    phrase = await phrase_queue.get()
                    if phrase is None:
                        break

                    async for audio_chunk in self._synthesize(phrase):
                        if first_chunk:
                            tts_first_chunk_ref.append(time.perf_counter())
                            first_chunk = False
                        await audio_out.put(audio_chunk)

            # LLM+splitter and TTS run concurrently
            # audio chunks flow into audio_out while the LLM is still generating
            await asyncio.gather(split_tokens(), synthesise_phrases())

            # assemble clean response for metrics — strip markers from raw parts
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
            metrics.total_ms = (time.perf_counter() - turn_start) * 1000

            self.history.append({"role": "assistant", "content": metrics.response})
            return metrics

        finally:
            # always place sentinel — tells session.py audio is done
            # runs even if an exception occurred mid-synthesis
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

    async def _synthesize(self, phrase: str) -> AsyncIterator[bytes]:
        """
        Send a single clean phrase to the TTS service over WebSocket.
        Yield audio chunks as they arrive.

        The phrase is already clean — PhraseStream stripped markers and
        markdown before putting it into phrase_queue.
        """
        async with websockets.connect(
            TTS_WS_URL,
            ping_interval=None,
            ping_timeout=None,
        ) as ws:
            await ws.send(json.dumps({"type": "phrase", "text": phrase}))
            await ws.send(json.dumps({"type": "end"}))

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