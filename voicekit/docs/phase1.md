# Phase 1 — Real Models, Silero VAD, Redis, Production Gateway

## Purpose

Complete technical implementation guide for Phase 1. Everything needed
to implement Phase 1 without prior context. Read alongside:

- `FLOW.md` — architectural principles, decisions, and why
- `prototype/PROTOTYPE.md` — interfaces and patterns to follow
- `prototype/voicekit/providers/base.py` — exact method signatures

Phase 1 is complete when all 29 integration tests pass including the
two previously skipped in simulation mode, and a real voice turn works
— user speaks, Whisper transcribes, Claude responds, Chatterbox speaks.

---

## What voicekit is in Phase 1

A Python package. Developers install it like any other package and
import classes directly into their projects. The models run inside their
application process. Their Dockerfile installs voicekit via requirements.
No separate voicekit server. No special deployment step.

```python
from voicekit.pipeline import VoicePipeline
from voicekit.config import load_config

pipeline = VoicePipeline(load_config())
await pipeline.load()
metrics = await pipeline.run_turn(audio_in, audio_out)
```

The Docker services in `runtime/` are local development tooling — not
the primary usage mode. They exist for developers who want to run models
as separate services. Most developers use direct import.

---

## Six things to build, in order

1. Redis — session state infrastructure
2. Silero VAD — replaces energy-based VAD
3. WhisperSTT — real speech-to-text
4. ClaudeProvider — real LLM
5. ChatterboxTTS — real text-to-speech
6. Production gateway — ping/pong, timeouts, guaranteed cleanup

Build and verify each step before moving to the next.

---

## Step 1 — Redis

### Why first

Redis is infrastructure with no model dependencies. Getting it working
first means every subsequent step has a persistence layer to build on.
It is also the simplest step — one container, a few lines of code.

### Add to docker-compose.yml

```yaml
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    mem_limit: 128m
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  gateway:
    depends_on:
      stt:
        condition: service_healthy
      tts:
        condition: service_healthy
      redis:
        condition: service_healthy    # add this
```

### Add to pyproject.toml

```toml
"redis[asyncio]>=5.0.0",
```

### Add to runtime/gateway/Dockerfile

```dockerfile
RUN pip install --no-cache-dir \
    fastapi uvicorn websockets numpy pyyaml redis
```

### What Redis stores

```
session:{session_id}    Hash    state, timestamps, turn count
history:{session_id}    String  JSON conversation history, TTL 1 hour
active_sessions         Set     all active session IDs
```

Audio never touches Redis. Audio travels through asyncio.Queue between
pipeline stages. Redis stores only persistent metadata.

### Verify

```bash
docker compose up -d redis
docker exec -it $(docker compose ps -q redis) redis-cli ping
# PONG
```

---

## Step 2 — Silero VAD

### Why Silero not energy-based

Energy-based VAD uses RMS amplitude. It fails in real environments —
background noise triggers false positives, quiet speakers get filtered
out, accents have different energy patterns. These failures cause Whisper
to receive silence (wasting compute) or miss actual speech (breaking
conversations).

Silero VAD is a 1.8MB neural network. It runs in ~12ms per 100ms audio
chunk on CPU. Accurate across noise levels, accents, and speaking styles.
It is the production standard for voice agents. No reason to ship the
inferior version first.

### Add to pyproject.toml

```toml
"torch>=2.0.0",
```

### Add to runtime Dockerfiles that load VAD

```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

CPU-only torch keeps image size manageable. Silero does not need GPU.

### Add volume for torch hub cache in docker-compose.yml

```yaml
  gateway:
    volumes:
      - torch_hub_cache:/root/.cache/torch/hub

volumes:
  torch_hub_cache:
```

Silero downloads from GitHub on first run (~1.8MB) and caches locally.
Without this volume, the download repeats on every container restart.

### File to replace — `voicekit/vad.py`

```python
# voicekit/vad.py

import numpy as np
import torch


class VADProcessor:
    """
    Voice Activity Detection using Silero VAD.

    Neural model — 1.8MB, CPU-only, ~12ms per 100ms chunk.
    Accurate across noise, accents, and speaking styles.

    Interface identical to prototype energy-based VAD.
    All callers continue working without modification.
    """

    _model = None
    _loaded = False

    def __init__(self, sensitivity: float = 0.5):
        if not 0.0 <= sensitivity <= 1.0:
            raise ValueError(
                f"VAD sensitivity must be between 0.0 and 1.0, got {sensitivity}"
            )
        self.sensitivity = sensitivity
        self._threshold = sensitivity
        self._ensure_loaded()

    @classmethod
    def _ensure_loaded(cls):
        if not cls._loaded:
            cls._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            cls._loaded = True

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Return True if the chunk contains speech.
        Audio must be float32, 16kHz mono, values in [-1.0, 1.0].
        """
        if len(audio_chunk) == 0:
            return False

        # Silero requires minimum 512 samples (32ms at 16kHz)
        if len(audio_chunk) < 512:
            audio_chunk = np.pad(
                audio_chunk, (0, 512 - len(audio_chunk))
            )

        tensor = torch.from_numpy(audio_chunk.astype(np.float32))

        with torch.no_grad():
            confidence = self._model(tensor, 16000).item()

        return confidence >= self._threshold

    def filter_stream(self, chunks: list[np.ndarray]) -> list[np.ndarray]:
        return [c for c in chunks if self.is_speech(c)]
```

### Verify

```bash
uv run pytest tests/unit/test_vad.py -v
# all 8 tests must pass
```

Quick sanity check in Python shell:
```python
import numpy as np
from voicekit.vad import VADProcessor

vad = VADProcessor(sensitivity=0.5)
silence = np.zeros(1600, dtype=np.float32)
speech = np.random.uniform(-0.5, 0.5, 1600).astype(np.float32)

print(vad.is_speech(silence))   # False
print(vad.is_speech(speech))    # True
```

---

## Step 3 — WhisperSTT

### Add to pyproject.toml

```toml
"faster-whisper>=1.0.0",
```

### Add to runtime/stt/Dockerfile

```dockerfile
RUN pip install --no-cache-dir \
    fastapi uvicorn websockets numpy pyyaml faster-whisper
```

### Model variant guide

| Variant | RAM | CPU latency | Recommendation |
|---|---|---|---|
| tiny | 390MB | ~20ms | development only |
| base | 740MB | ~30ms | low-resource servers |
| small | 1.5GB | ~150ms | recommended default |
| medium | 3GB | ~300ms | higher accuracy needs |
| large-v3 | 6GB | ~600ms | GPU recommended |

Start with `small`. Fits in 1.5GB RAM, runs on CPU, production-accurate.

### File to create — `voicekit/providers/stt/whisper.py`

```python
# voicekit/providers/stt/whisper.py

import asyncio
from typing import AsyncIterator

import numpy as np
from voicekit.providers.base import STTProvider


class WhisperSTT(STTProvider):
    """
    Whisper STT via faster-whisper (CTranslate2 reimplementation).
    Runs on CPU with int8 quantization. No GPU required for small variant.
    Model loads once at startup, serves all requests from memory.
    """

    def __init__(self, variant: str = "small"):
        self.variant = variant
        self.model = None

    async def load(self) -> None:
        from faster_whisper import WhisperModel

        # run in executor — model loading blocks 3-5 seconds
        # never block the event loop during startup
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.variant,
                device="cpu",         # change to "cuda" if GPU available
                compute_type="int8",  # int8 quantization — fast, low memory
            )
        )

    async def transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        chunks = []
        async for chunk in audio_stream:
            chunks.append(chunk)

        if not chunks:
            return

        audio = np.concatenate(chunks)

        # run inference in executor — transcribe() is synchronous
        # and CPU-intensive. Calling it on the event loop blocks
        # all other coroutines for the duration of inference.
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio,
                language="en",     # remove for auto-detection
                beam_size=5,
                vad_filter=True,   # faster-whisper internal VAD as backup
            )
        )

        for segment in segments:
            text = segment.text.strip()
            if text:
                yield text

    async def health(self) -> bool:
        return self.model is not None
```

### Critical — always use run_in_executor

Never call `self.model.transcribe()` directly on the event loop. It
blocks for 100-300ms. During that block TTS cannot stream, the gateway
cannot accept connections, ping/pong cannot respond. Always run in
executor.

### Register

```python
# voicekit/providers/registry.py
from voicekit.providers.stt.whisper import WhisperSTT

STT_REGISTRY = {
    "simulated": SimulatedSTT,
    "whisper": WhisperSTT,        # add
}
```

### Update config

```yaml
stt:
  model: whisper
  variant: small
```

### Verify

```bash
docker compose -f runtime/docker-compose.test.yml build stt-test
docker compose -f runtime/docker-compose.test.yml up -d stt-test
uv run pytest tests/integration/test_stt.py -v
# all 14 tests must pass including test_silence_produces_empty_transcript
```

---

## Step 4 — ClaudeProvider

### Add to pyproject.toml

```toml
"anthropic>=0.40.0",
```

### File to create — `voicekit/providers/llm/claude.py`

```python
# voicekit/providers/llm/claude.py

import asyncio
from typing import AsyncIterator

import anthropic
from voicekit.providers.base import LLMProvider


class ClaudeProvider(LLMProvider):
    """
    LLM provider via Anthropic Claude API.
    Reads ANTHROPIC_API_KEY from environment automatically.

    Uses executor + queue pattern because the Anthropic SDK is
    synchronous. Calling it directly on the event loop would block
    TTS audio streaming for the duration of the LLM response.
    """

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model
        self.client = anthropic.Anthropic()

    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 300
    ) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream_sync():
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                ) as stream:
                    for token in stream.text_stream:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, token
                        )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        executor_task = loop.run_in_executor(None, _stream_sync)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        await executor_task
```

### API key routing in gateway

The gateway receives `VOICEKIT_LLM_API_KEY` from the CLI. It must set
the provider-specific environment variable before the pipeline is
instantiated. Add to `runtime/gateway/server.py`:

```python
def _configure_llm_key(config) -> None:
    key = config.llm.api_key
    if not key:
        return
    if config.llm.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = key
    elif config.llm.provider == "openai":
        os.environ["OPENAI_API_KEY"] = key

# call this before creating VoicePipeline
_configure_llm_key(config)
```

### Model selection guide

- `claude-haiku-4-5` — fastest, cheapest, first token under 150ms, recommended
- `claude-sonnet-4-6` — better reasoning, use for complex agents
- `claude-opus-4-6` — best reasoning, rarely needed for voice

### Register

```python
from voicekit.providers.llm.claude import ClaudeProvider

LLM_REGISTRY = {
    "simulated": SimulatedLLM,
    "anthropic": ClaudeProvider,    # add
}
```

### Update config

```yaml
llm:
  provider: anthropic
  model: claude-haiku-4-5
  api_key: ${ANTHROPIC_API_KEY}
```

---

## Step 5 — ChatterboxTTS

### Add to pyproject.toml

```toml
"chatterbox-tts>=0.1.0",
"torchaudio>=2.0.0",
```

### Add to runtime/tts/Dockerfile

```dockerfile
RUN pip install --no-cache-dir \
    fastapi uvicorn websockets numpy pyyaml chatterbox-tts torchaudio
```

### Sentence boundary buffering — why and how

TTS models need sentence context to set intonation correctly. One token
at a time produces flat robotic speech. Waiting for the full LLM
response kills latency.

Sentence boundaries are the right compromise. Buffer tokens until a
sentence-ending punctuation mark, then synthesise immediately:

```
LLM generates: "I can help you with that. Here are three options."

Buffer fills: "I can help you with that"
Period arrives → synthesise "I can help you with that." → audio streams
Buffer resets

Buffer fills: " Here are three options"
Period arrives → synthesise "Here are three options." → audio streams
```

First audio plays before the LLM has finished generating the second
sentence. Sentence context gives TTS enough information for natural
intonation.

Trigger on: `.` `!` `?` `,`
Do not trigger on: `:` `;` (pause mid-thought, sounds wrong)

### File to create — `voicekit/providers/tts/chatterbox.py`

```python
# voicekit/providers/tts/chatterbox.py

import asyncio
import io
from typing import AsyncIterator

from voicekit.providers.base import TTSProvider

SAMPLE_RATE = 24000
SENTENCE_BOUNDARIES = (".", "!", "?", ",")


class ChatterboxTTS(TTSProvider):
    """
    Chatterbox Turbo TTS — 350M params, ~75ms first audio, MIT license.
    Supports voice cloning from 5-second reference audio.
    """

    def __init__(self, voice: str = "default"):
        self.voice = voice    # "default" or path to .wav reference audio
        self.model = None

    async def load(self) -> None:
        from chatterbox.tts import ChatterboxTTS as CBModel

        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: CBModel.from_pretrained(device="cpu")
        )

    async def synthesize(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        buffer = ""

        async for token in text_stream:
            buffer += token
            if buffer.rstrip().endswith(SENTENCE_BOUNDARIES):
                sentence = buffer.strip()
                if sentence:
                    async for chunk in self._synthesize_sentence(sentence):
                        yield chunk
                buffer = ""

        # flush remaining text after stream ends
        if buffer.strip():
            async for chunk in self._synthesize_sentence(buffer.strip()):
                yield chunk

    async def _synthesize_sentence(
        self, text: str
    ) -> AsyncIterator[bytes]:
        import torchaudio

        loop = asyncio.get_event_loop()

        # verify exact parameter names with: help(self.model.generate)
        # API may differ between Chatterbox versions
        wav_tensor = await loop.run_in_executor(
            None,
            lambda: self.model.generate(text)
        )

        buf = io.BytesIO()
        torchaudio.save(buf, wav_tensor, SAMPLE_RATE, format="wav")
        wav_bytes = buf.getvalue()

        chunk_size = SAMPLE_RATE * 2 // 10    # 100ms chunks
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i:i + chunk_size]
            await asyncio.sleep(0)    # yield control between chunks

    async def health(self) -> bool:
        return self.model is not None
```

### Verify Chatterbox API before using

```python
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="cpu")
help(model.generate)
# confirm exact parameter name for voice reference audio
```

The `generate()` signature changes between versions. Always verify
before using the voice cloning parameter.

### Register

```python
from voicekit.providers.tts.chatterbox import ChatterboxTTS

TTS_REGISTRY = {
    "simulated": SimulatedTTS,
    "chatterbox-turbo": ChatterboxTTS,    # add
}
```

### Update config

```yaml
tts:
  model: chatterbox-turbo
  voice: default    # or /path/to/reference.wav for voice cloning
```

### Verify

```bash
docker compose -f runtime/docker-compose.test.yml build tts-test
docker compose -f runtime/docker-compose.test.yml up -d tts-test
uv run pytest tests/integration/test_tts.py -v
# all 15 tests must pass including test_first_audio_arrives_before_synthesis_completes
```

---

## Step 6 — Production Gateway

The prototype gateway handled basic sessions. The production gateway
adds Redis persistence, ping/pong heartbeat, three independent timers,
and guaranteed cleanup via finally block.

### File to replace — `runtime/gateway/server.py`

```python
# runtime/gateway/server.py

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import redis.asyncio as aioredis
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, "/app")

from voicekit.config import VoiceConfig, STTConfig, TTSConfig, VADConfig, LLMConfig
from voicekit.pipeline import VoicePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GW] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

PING_INTERVAL = 30       # seconds between pings
PING_TIMEOUT = 10        # seconds to wait for pong before closing
IDLE_TIMEOUT = 1800      # seconds before closing idle session (30 min)
TURN_TIMEOUT = 30        # seconds before cancelling hung pipeline turn


def _config_from_env() -> VoiceConfig:
    return VoiceConfig(
        project=os.environ.get("VOICEKIT_PROJECT", "voicekit"),
        stt=STTConfig(
            model=os.environ.get("VOICEKIT_STT_MODEL", "simulated"),
            variant=os.environ.get("VOICEKIT_STT_VARIANT", "small"),
        ),
        tts=TTSConfig(
            model=os.environ.get("VOICEKIT_TTS_MODEL", "simulated"),
            voice=os.environ.get("VOICEKIT_TTS_VOICE", "default"),
        ),
        vad=VADConfig(
            enabled=os.environ.get("VOICEKIT_VAD_ENABLED", "true").lower() == "true",
            sensitivity=float(os.environ.get("VOICEKIT_VAD_SENSITIVITY", "0.5")),
        ),
        llm=LLMConfig(
            provider=os.environ.get("VOICEKIT_LLM_PROVIDER", "simulated"),
            model=os.environ.get("VOICEKIT_LLM_MODEL", "simulated"),
            api_key=os.environ.get("VOICEKIT_LLM_API_KEY", ""),
        ),
        system_prompt=os.environ.get(
            "VOICEKIT_SYSTEM_PROMPT",
            "You are a helpful voice assistant. Keep responses concise and natural.",
        ),
    )


def _configure_llm_key(config: VoiceConfig) -> None:
    key = config.llm.api_key
    if not key:
        return
    if config.llm.provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = key
    elif config.llm.provider == "openai":
        os.environ["OPENAI_API_KEY"] = key


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _config_from_env()
    _configure_llm_key(config)
    app.state.config = config
    app.state.redis = aioredis.Redis(
        host=os.environ.get("REDIS_HOST", "redis"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    log.info(
        f"Gateway ready — "
        f"STT: {config.stt.model}/{config.stt.variant} "
        f"TTS: {config.tts.model} "
        f"LLM: {config.llm.provider}/{config.llm.model}"
    )
    yield
    await app.state.redis.aclose()


app = FastAPI(title="voicekit-gateway", lifespan=lifespan)


@app.get("/health")
async def health():
    active = await app.state.redis.scard("active_sessions")
    return {"status": "ok", "active_sessions": active}


@app.websocket("/session")
async def session_endpoint(ws: WebSocket):
    await ws.accept()

    session_id = str(uuid.uuid4())[:8]
    config = app.state.config
    redis = app.state.redis

    log.info(f"[{session_id}] Session opened")

    await redis.hset(f"session:{session_id}", mapping={
        "state": "waiting",
        "created_at": str(time.time()),
        "last_active": str(time.time()),
    })
    await redis.sadd("active_sessions", session_id)

    pipeline = VoicePipeline(config)
    await pipeline.load()

    last_turn_time = time.time()
    last_pong_time = [time.time()]

    ping_task = asyncio.create_task(
        _ping_loop(ws, session_id, last_pong_time)
    )

    try:
        await ws.send_text(json.dumps({
            "type": "ready",
            "session_id": session_id,
        }))

        while True:
            if time.time() - last_turn_time > IDLE_TIMEOUT:
                log.info(f"[{session_id}] Idle timeout")
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Session idle timeout.",
                }))
                break

            audio_in: asyncio.Queue = asyncio.Queue()
            end_of_speech = asyncio.Event()

            async def receive_turn():
                while not end_of_speech.is_set():
                    try:
                        message = await asyncio.wait_for(
                            ws.receive(),
                            timeout=float(PING_INTERVAL + PING_TIMEOUT),
                        )
                    except asyncio.TimeoutError:
                        continue

                    if message["type"] == "websocket.disconnect":
                        end_of_speech.set()
                        return

                    if "bytes" in message and message["bytes"]:
                        chunk = np.frombuffer(
                            message["bytes"], dtype=np.float32
                        )
                        await audio_in.put(chunk)

                    elif "text" in message and message["text"]:
                        data = json.loads(message["text"])
                        if data.get("type") == "end_of_speech":
                            end_of_speech.set()
                        elif data.get("type") == "pong":
                            last_pong_time[0] = time.time()

            await asyncio.create_task(receive_turn())

            if audio_in.empty():
                continue

            await redis.hset(f"session:{session_id}", "state", "processing")

            audio_out: asyncio.Queue = asyncio.Queue()
            try:
                metrics = await asyncio.wait_for(
                    pipeline.run_turn(audio_in, audio_out),
                    timeout=float(TURN_TIMEOUT),
                )
            except asyncio.TimeoutError:
                log.warning(f"[{session_id}] Turn timeout")
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "Response timed out. Please try again.",
                }))
                await redis.hset(
                    f"session:{session_id}", "state", "waiting"
                )
                continue

            while not audio_out.empty():
                await ws.send_bytes(audio_out.get_nowait())

            await redis.set(
                f"history:{session_id}",
                json.dumps(pipeline.history),
                ex=3600,
            )

            last_turn_time = time.time()
            await redis.hset(f"session:{session_id}", mapping={
                "state": "waiting",
                "last_active": str(last_turn_time),
                "turn_count": str(len(pipeline.history) // 2),
            })

            await ws.send_text(json.dumps({
                "type": "transcript", "text": metrics.transcript,
            }))
            await ws.send_text(json.dumps({
                "type": "response", "text": metrics.response,
            }))
            await ws.send_text(json.dumps({
                "type": "metrics",
                "stt_ms": round(metrics.stt_ms),
                "llm_first_token_ms": round(metrics.llm_first_token_ms),
                "tts_first_chunk_ms": round(metrics.tts_first_chunk_ms),
                "total_ms": round(metrics.total_ms),
            }))

            log.info(f"[{session_id}] Turn complete — {metrics}")

    except WebSocketDisconnect:
        log.info(f"[{session_id}] Client disconnected")

    except Exception as e:
        log.exception(f"[{session_id}] Error: {e}")
        try:
            await ws.send_text(json.dumps({
                "type": "error", "message": str(e),
            }))
        except Exception:
            pass

    finally:
        # runs regardless of how session exits
        # no code path can skip this
        ping_task.cancel()
        await redis.delete(f"session:{session_id}")
        await redis.delete(f"history:{session_id}")
        await redis.srem("active_sessions", session_id)
        log.info(f"[{session_id}] Cleaned up")


async def _ping_loop(
    ws: WebSocket,
    session_id: str,
    last_pong_time: list[float],
) -> None:
    """
    Heartbeat — distinguishes idle-but-alive sessions from dead
    connections. A user thinking between turns still pongs.
    A dead connection does not.
    """
    while True:
        await asyncio.sleep(PING_INTERVAL)
        try:
            await ws.send_text(json.dumps({"type": "ping"}))
        except Exception:
            return

        deadline = time.time() + PING_TIMEOUT
        while time.time() < deadline:
            if last_pong_time[0] > time.time() - PING_INTERVAL:
                break
            await asyncio.sleep(1.0)
        else:
            log.warning(f"[{session_id}] Ping timeout — closing")
            try:
                await ws.close(code=1001, reason="Ping timeout")
            except Exception:
                pass
            return


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

---

## Latency targets

| Stage | Target | How to check |
|---|---|---|
| Silero VAD per chunk | under 15ms | manual timing in vad.py |
| Whisper small CPU | under 300ms | `metrics.stt_ms` |
| Claude Haiku first token | under 200ms | `metrics.llm_first_token_ms` |
| Chatterbox first audio | under 300ms | `metrics.tts_first_chunk_ms` |
| Total user-perceived | under 600ms | `metrics.total_ms` |

Diagnose stage by stage using metrics output. Do not guess — the numbers
tell you exactly where to optimise.

---

## Phase 1 complete when

```bash
# all 29 pass — zero skips
uv run pytest tests/unit/ tests/pipeline/ tests/integration/ -v

# real voice turn works end to end
voicekit dev
# speak → Whisper transcribes → Claude responds → Chatterbox speaks
```

Final test count:
- Unit: 15/15
- Pipeline: 10/10
- Integration STT: 14/14 (was 13+1 skipped)
- Integration TTS: 15/15 (was 14+1 skipped)

---

## Common problems

**`TypeError: Can't instantiate abstract class`**
A provider is missing one of its abstract methods. Check `load`,
`transcribe`/`synthesize`/`stream`, and `health` are all implemented.

**`RuntimeError: CUDA error` on CPU machine**
Change `device="cuda"` to `device="cpu"` in the provider's `load()`.

**TTS latency over 2 seconds on CPU**
Expected on CPU without GPU. Acceptable for development. For production
with Chatterbox Turbo, a GPU is strongly recommended. Kokoro (Phase 2)
is the CPU-friendly alternative.

**LLM tokens arrive all at once instead of streaming**
The executor pattern is not being used correctly. The synchronous
Anthropic SDK is blocking the event loop. Confirm `_stream_sync` runs
in `run_in_executor` and tokens flow through `asyncio.Queue`.

**Chatterbox `generate()` parameter error**
The API differs between versions. Run `help(model.generate)` to see
the exact signature for the installed version.

**Redis connection refused in gateway**
Gateway starts before Redis is healthy. Confirm `depends_on` with
`condition: service_healthy` is set for Redis in docker-compose.yml.