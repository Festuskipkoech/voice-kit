# voicekit — Engineering Flow

Master engineering document. Read this before reading any phase document
or touching any code. It contains everything needed to understand the
project, the decisions made, and why.

---

## What voicekit is

A Python package that provides voice agent infrastructure as importable
classes. Developers add `voicekit` to their dependencies and import
`VoicePipeline`, `WhisperSTT`, `ChatterboxTTS`, and related classes
directly into their projects — exactly as they would import LangChain,
FastAPI, or any other package.

Voicekit is not a platform. It is not a hosted service. It is not a
framework that owns your project structure. It is a package. It goes
where your project goes. When you deploy your project, voicekit deploys
with it via your requirements file, exactly like every other dependency.

---

## Repository structure

```
voicekit/
├── prototype/                  proven architecture, do not modify
│   ├── PROTOTYPE.md
│   └── ...
│
└── voicekit/                   the real project — Python package
    ├── README.md               product overview
    ├── FLOW.md                 this file
    ├── pyproject.toml
    ├── voicekit/               importable Python package
    │   ├── cli.py
    │   ├── config.py
    │   ├── pipeline.py
    │   ├── vad.py
    │   └── providers/
    │       ├── base.py
    │       ├── registry.py
    │       ├── stt/
    │       ├── tts/
    │       └── llm/
    ├── runtime/                Docker services for local development
    │   ├── docker-compose.yml
    │   ├── docker-compose.test.yml
    │   ├── stt/                Whisper STT service
    │   ├── tts/                Chatterbox TTS service
    │   └── gateway/            Session management, routing
    │       ├── main.py
    │       ├── config.py
    │       ├── dependencies.py
    │       ├── remote_pipeline.py
    │       ├── routes/
    │       │   ├── health.py
    │       │   └── session.py
    │       └── services/
    │           ├── ping.py
    │           └── session.py
    ├── tests/
    │   ├── unit/
    │   ├── pipeline/
    │   ├── integration/
    │   └── test_voice.py       end-to-end voice turn test
    ├── templates/
    │   └── voice.config.yaml
    └── docs/
        ├── phase-1.md
        └── phase-2.md
```

---

## Two usage modes

### Mode A — Direct import (recommended default)

The developer imports voicekit classes directly into their project.
Models run inside their application process. No separate services.
Deploys exactly like any other Python dependency.

```python
from voicekit.pipeline import VoicePipeline
from voicekit.config import load_config

pipeline = VoicePipeline(load_config())
await pipeline.load()
metrics = await pipeline.run_turn(audio_in, audio_out)
```

### Mode B — Docker services (local development and advanced use)

The CLI starts STT, TTS, Redis, and a gateway as separate Docker
containers. The gateway uses `RemotePipeline` which calls STT and TTS
over WebSocket rather than loading models locally.

```bash
voicekit dev     # starts all four services
```

Services:
- Redis `:6379` — session state and conversation history
- STT `:8001` — Whisper loaded once, serves all transcription requests
- TTS `:8002` — Chatterbox loaded once, serves all synthesis requests
- Gateway `:8000` — lightweight routing, LLM calls, session management

---

## The five architectural principles

### 1. The provider pattern

Every STT, TTS, and LLM model implements an abstract base class in
`voicekit/providers/base.py`. The pipeline only ever calls methods on
the base class.

```
base.py defines:    STTProvider, TTSProvider, LLMProvider
registry.py maps:   "whisper" → WhisperSTT, "chatterbox-turbo" → ChatterboxTTS
pipeline.py calls:  self.stt.transcribe(), self.tts.synthesize(), self.llm.stream()
```

Adding a new model: write one class, add one line to registry, change
one config value. Nothing else changes.

### 2. The streaming cascade

```
user stops speaking
    ↓ Silero VAD filtered silence throughout
    ↓ Whisper transcribes — tokens under 300ms
    ↓ LLM starts on first transcript token
    ↓ LLM tokens → token_queue
    ↓ TTS reads token_queue — buffers to sentence boundary
    ↓ First sentence complete — TTS generates audio chunk
    ↓ Chunk placed in audio_out queue immediately
    ↓ Concurrent audio sender sends chunk to client
    ↓ (LLM still generating, TTS still synthesising)
    ↓ More chunks arrive and are sent as generated
    ↓ All chunks sent → transcript, response, metrics sent
```

`asyncio.gather(feed_llm(), feed_tts())` in pipeline.py runs LLM and
TTS concurrently. `asyncio.gather(pipeline.run_turn(), send_audio_stream())`
in session.py streams chunks to the client as they are generated.

### 3. Remote pipeline — gateway loads no models

The gateway (`runtime/gateway/`) uses `RemotePipeline` not
`VoicePipeline`. This is a critical architectural decision:

**Why:** Loading Whisper and Chatterbox in the gateway duplicates the
models already loaded in the STT and TTS services. On a machine with
8GB RAM this causes OOM. Gateway startup takes 90+ seconds waiting for
models to load. Container restart means 90 seconds of downtime.

**How it works:** `RemotePipeline` calls STT and TTS services over
WebSocket. The gateway is lightweight — only the LLM client loads at
startup. Gateway restarts in seconds.

```python
# RemotePipeline in gateway — no model loading
async def load(self) -> None:
    # just verify services are reachable
    async with websockets.connect(STT_WS_URL): pass
    async with websockets.connect(TTS_WS_URL): pass

# delegates transcription over WebSocket
async def _transcribe(self, audio_in): ...

# delegates synthesis over WebSocket, yields chunks as they arrive
async def _synthesize(self, text_stream): ...
```

### 4. AUDIO_DONE sentinel for clean stream termination

`run_turn()` places `AUDIO_DONE = None` into `audio_out` in its
`finally` block when synthesis is complete. The concurrent
`send_audio_stream()` coroutine in `session.py` reads from `audio_out`
and stops when it receives the sentinel.

This ensures:
- All audio chunks are sent before text messages
- Clean termination even if an exception occurs mid-synthesis
- No race condition between audio sender and pipeline completion

```python
# in remote_pipeline.py finally block
finally:
    await audio_out.put(AUDIO_DONE)
    self.turn_in_progress.clear()

# in session.py send_audio_stream()
while True:
    chunk = await audio_out.get()
    if chunk is AUDIO_DONE:
        break
    await ws.send_bytes(chunk)
```

### 5. turn_in_progress — ping never kills active synthesis

The ping loop sends a ping every 30 seconds and closes dead connections
that do not respond. But on CPU, Chatterbox synthesis takes 90-150
seconds. Without protection, the ping would close the connection
mid-synthesis.

`turn_in_progress` is an `asyncio.Event` on `RemotePipeline`. Set at
the start of `run_turn()`, cleared in `finally`. The ping loop checks
this before sending a ping — if a turn is in progress it skips entirely.

```python
# in services/ping.py
if turn_in_progress.is_set():
    continue   # skip ping — turn in progress
```

---

## WebSocket lifecycle

```
WAITING     ready for audio, ping loop running (skips if turn active)
    ↓ audio arrives
RECEIVING   collecting chunks
    ↓ end_of_speech signal
PROCESSING  pipeline running
    ↓ (concurrently) audio chunks streaming to client
STREAMING   send_audio_stream() sending chunks as they arrive
    ↓ AUDIO_DONE sentinel received
TEXT        transcript, response, metrics sent
    ↓
WAITING     ready for next turn
    ↓ any exit condition
CLOSED      finally block runs, Redis cleanup
```

Three independent timers:
- **Ping/pong** (30s interval, 10s timeout) — skipped during active turns
- **Idle timeout** (30 minutes) — closes abandoned sessions
- **Turn timeout** (5 minutes) — cancels hung pipeline turns

---

## Audio format

**Client → Gateway (input):**
- Format: raw PCM, no WAV header
- Sample rate: 16000 Hz, mono, float32, values in [-1.0, 1.0]
- Chunk size: 1600 samples = 100ms

**Gateway → Client (output):**
- Chunk 1: complete WAV file (44-byte header + float32 PCM data)
- Chunks 2..N: raw float32 PCM bytes, no header
- Sample rate: 24000 Hz, mono
- Client must handle both formats — read chunk 1 with soundfile,
  read chunks 2..N as `np.frombuffer(chunk, dtype=np.float32)`

---

## Known issues and workarounds

### Chatterbox perth watermarker

`chatterbox-tts` depends on `perth` for audio watermarking. The
`PerthImplicitWatermarker` C extension fails to load with uv due to
missing `pkg_resources`. Fix applied in `ChatterboxTTS.load()`:

```python
import perth

class DummyWatermarker:
    def __init__(self, *args, **kwargs): pass
    def apply_watermark(self, wav, *args, **kwargs): return wav
    def __call__(self, *args, **kwargs): return args[0] if args else None

perth.PerthImplicitWatermarker = DummyWatermarker
perth.DummyWatermarker = DummyWatermarker

from chatterbox.tts_turbo import ChatterboxTurboTTS
```

Audio synthesis works correctly — the watermark is simply not embedded.

### Silero VAD window size

Silero requires exactly 512 samples per inference call. Audio arrives
in 1600-sample chunks. Implementation splits each chunk into 512-sample
windows and feeds them sequentially — state accumulates across windows
within a chunk, which is the correct streaming behaviour.

`reset_states()` is called after each complete utterance, not between
windows within a chunk. Calling it between windows destroys the context
the RNN needs for accurate detection.

### Chatterbox on CPU latency

Chatterbox Turbo on CPU takes 90-150 seconds for a typical response.
On GPU this drops to 2-5 seconds. The `turn_in_progress` flag and
5-minute turn timeout accommodate CPU inference. For production
deployment, a GPU is strongly recommended.

---

## Phase status

| Phase | Description | Status | Branch |
|---|---|---|---|
| Prototype | Architecture validation, simulated models | COMPLETE | main |
| Phase 1 | Real models, Redis, production WebSocket | COMPLETE | phase-1 → main |
| Phase 2 | Kokoro TTS, OpenAI LLM, observability | IN PROGRESS | - |
| Phase 3 | Prometheus metrics, Grafana dashboard | NOT STARTED | - |
| Phase 4 | CI, example projects, PyPI publish | NOT STARTED | - |

---

## Phase 1 — COMPLETE

See `docs/phase-1.md` for full implementation guide.

**What was built:**
- `WhisperSTT` — faster-whisper, CPU int8, lazy generator fix
- `ChatterboxTTS` — Chatterbox Turbo, sentence buffering, perth patch
- `ClaudeProvider` — Anthropic SDK, executor+queue streaming pattern
- `VADProcessor` — Silero VAD, 512-sample windows, reset_states pattern
- Redis — session state, conversation history TTL 1hr
- `RemotePipeline` — gateway calls STT/TTS over WebSocket
- `AUDIO_DONE` sentinel — clean audio stream termination
- `turn_in_progress` flag — ping never kills active synthesis
- Concurrent audio sender — true end-to-end streaming
- Production gateway — structured JSON logging, three timers

**Test results:**
```
Unit tests:        20/20 passing
Pipeline tests:     9/9  passing
Integration tests: 29/29 passing
End-to-end:        voice in → WAV out, confirmed working
```

---

## Phase 2 — IN PROGRESS

See `docs/phase-2.md` for implementation guide.

Kokoro TTS, OpenAI LLM provider, observability groundwork.

---

## Key decisions and why

**Remote pipeline pattern.**
Gateway loads no models. STT and TTS services each load their model
once and serve all requests. Gateway delegates over WebSocket. This
keeps gateway lightweight, fast-starting, and memory-efficient.

**AUDIO_DONE sentinel over queue.empty() drain.**
A simple `while not audio_out.empty(): send()` drain runs after
`run_turn()` completes — too late for true streaming. The sentinel
allows `send_audio_stream()` to run concurrently and terminate cleanly.

**turn_in_progress over increased ping interval.**
Increasing `PING_INTERVAL` to 120+ seconds to accommodate CPU
inference is a hack. The ping interval should reflect network
health checking needs (30 seconds), not model inference speed.
The flag decouples these concerns correctly.

**Sentence boundary buffering in TTS.**
One token at a time produces flat robotic delivery — no sentence
context for intonation. Full response before TTS kills latency.
Sentence boundaries give natural prosody with low latency.

**Package not platform.**
Voicekit installs like LangChain. Developers import classes. Their
deployment handles voicekit like every other dependency. No separate
voicekit server for the primary usage mode.

**No deploy command.**
Voicekit is a package. Packages do not deploy themselves.

---

## Environment variables

```
VOICEKIT_STT_MODEL       stt service     STT model identifier
VOICEKIT_STT_VARIANT     stt service     model size or variant
VOICEKIT_TTS_MODEL       tts service     TTS model identifier
VOICEKIT_TTS_VOICE       tts service     voice profile
VOICEKIT_LLM_PROVIDER    gateway         LLM provider identifier
VOICEKIT_LLM_MODEL       gateway         model name
VOICEKIT_LLM_API_KEY     gateway         routed to ANTHROPIC_API_KEY or OPENAI_API_KEY
VOICEKIT_VAD_ENABLED     gateway         whether VAD filters silence
VOICEKIT_VAD_SENSITIVITY gateway         VAD threshold 0.0-1.0
VOICEKIT_SYSTEM_PROMPT   gateway         LLM system prompt
REDIS_HOST               gateway         Redis hostname (default: redis)
REDIS_PORT               gateway         Redis port (default: 6379)
STT_WS_URL               gateway         STT WebSocket URL (default: ws://stt:8001/stt)
TTS_WS_URL               gateway         TTS WebSocket URL (default: ws://tts:8002/tts)
```

API keys go in `runtime/.env` for Docker services. Never commit this
file. It is in `.gitignore`.

---

## Running tests

```bash
# unit and pipeline — no Docker needed
uv run pytest tests/unit/ tests/pipeline/ -v

# integration — requires Docker test stack
docker compose -f runtime/docker-compose.test.yml up -d
uv run pytest tests/integration/ -v
docker compose -f runtime/docker-compose.test.yml down

# end-to-end voice test — requires full stack running
docker compose -f runtime/docker-compose.yml up -d
uv run python tests/test_voice.py
aplay tests/results/response.wav
```