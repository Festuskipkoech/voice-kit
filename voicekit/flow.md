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

The value it provides: the streaming voice pipeline — VAD filtering,
Whisper transcription, LLM routing, Chatterbox synthesis, and the
concurrent streaming cascade that makes it feel real-time — is pre-built,
pre-tuned, and tested. A developer imports it and focuses on agent logic
rather than infrastructure.

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
    │   ├── __init__.py
    │   ├── cli.py              local development CLI
    │   ├── config.py           voice.config.yaml loading
    │   ├── pipeline.py         streaming cascade
    │   ├── vad.py              Silero VAD
    │   └── providers/
    │       ├── base.py         STTProvider, TTSProvider, LLMProvider
    │       ├── registry.py     model registry
    │       ├── stt/
    │       ├── tts/
    │       └── llm/
    ├── runtime/                Docker services for local development
    │   ├── docker-compose.yml
    │   ├── docker-compose.test.yml
    │   ├── stt/
    │   ├── tts/
    │   └── gateway/
    ├── tests/
    │   ├── unit/
    │   ├── pipeline/
    │   └── integration/
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
No extra infrastructure. Deploys exactly like any other Python dependency.

```python
from voicekit.pipeline import VoicePipeline
from voicekit.config import load_config

pipeline = VoicePipeline(load_config())
await pipeline.load()
metrics = await pipeline.run_turn(audio_in, audio_out)
```

Their Dockerfile:
```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt   # voicekit installs here
COPY . .
CMD ["python", "main.py"]
```

Voicekit is just another line in requirements.txt. Nothing special about
how it deploys.

### Mode B — Docker services (local development and advanced use)

The CLI starts STT, TTS, and a gateway as separate Docker containers.
Useful during development for testing against real models. Also useful
for advanced deployments where developers want to scale services
independently or share models across multiple agent projects.

```bash
voicekit dev     # starts local Docker services
```

The Docker services expose WebSocket endpoints. The developer's project
connects to them over the network rather than importing models directly.

In production, a developer choosing Mode B includes voicekit's Docker
Compose files in their own deployment. Their choice, their infrastructure.
Voicekit supports both modes. Most developers start with Mode A.

---

## The four architectural principles

### 1. The provider pattern

Every STT, TTS, and LLM model implements an abstract base class defined
in `voicekit/providers/base.py`. The pipeline only ever calls methods
on the base class. It never imports a concrete provider.

```
base.py defines:    STTProvider, TTSProvider, LLMProvider
registry.py maps:   "whisper" → WhisperSTT, "chatterbox-turbo" → ChatterboxTTS
pipeline.py calls:  self.stt.transcribe(), self.tts.synthesize(), self.llm.stream()
```

Adding a new model: write one class, add one line to registry, change
one config value. Nothing else in the codebase changes.

### 2. The streaming cascade

The core of what makes voice feel real-time. LLM tokens flow directly
into TTS as they arrive. TTS generates audio before the LLM finishes.
First audio plays within 600ms of the user stopping speaking.

```
user stops speaking
    ↓ Silero VAD has filtered silence throughout
    ↓ Whisper transcribes — first tokens under 300ms
    ↓ LLM starts immediately on first transcript token
    ↓ LLM tokens → token_queue (asyncio.Queue)
    ↓ TTS reads token_queue — buffers to sentence boundary
    ↓ First sentence complete — TTS generates audio
    ↓ First audio chunk arrives under 600ms total
    ↓ (LLM still generating, more audio follows)
```

`asyncio.gather(feed_llm(), feed_tts())` in `pipeline.py` is the key
line. LLM generation and TTS synthesis run concurrently.

Target latency per stage:
- Silero VAD per chunk: under 15ms
- Whisper small (CPU): under 300ms
- Claude Haiku first token: under 150ms
- Chatterbox Turbo first audio: under 150ms
- Total user-perceived: under 600ms

### 3. One process per session

Python's GIL means only one thread executes Python bytecode at a time.
Running multiple voice sessions as threads in one process causes them
to interfere under load — latency spikes, audio delays, unpredictable
behaviour.

The correct pattern: one process per active voice session. Sessions
are fully isolated. A crash or memory leak in one session cannot
affect another. Each session's pipeline loads independently, runs
independently, and cleans up independently.

In Mode A this is the developer's responsibility — they should spawn
one process or use one async context per user session. In Mode B the
gateway enforces this with one pipeline instance per WebSocket connection.

### 4. Redis for session state

Audio never touches Redis. Redis stores:

```
session:{id}       Hash    state, timestamps, turn count
history:{id}       String  conversation history JSON, TTL 1 hour
active_sessions    Set     all active session IDs
```

Audio travels through `asyncio.Queue` between pipeline stages.
Redis stores persistent state that survives process restarts. In Mode A
the developer manages Redis themselves — voicekit provides the pipeline,
they manage the session. In Mode B the gateway manages Redis.

---

## WebSocket lifecycle (Mode B)

For developers using the Docker gateway, every session follows this
state machine:

```
WAITING     ready for audio, ping loop running
    ↓ audio arrives
RECEIVING   collecting chunks through VAD
    ↓ end_of_speech signal
PROCESSING  pipeline running (30s timeout)
    ↓ pipeline completes
STREAMING   sending audio chunks to client
    ↓ audio queue empty
WAITING     ready for next turn
    ↓ any exit condition
CLOSED      finally block runs, Redis cleanup
```

Three independent timers:

**Ping/pong (30s interval, 10s timeout)**
Server sends ping every 30 seconds. Client responds with pong. No pong
within 10 seconds — connection is dead, close it. This distinguishes
idle-but-alive sessions from dead connections without incorrectly
closing sessions where the user is just thinking.

**Idle timeout (30 minutes)**
No completed turn in 30 minutes — session is abandoned. Close cleanly.

**Turn timeout (30 seconds)**
Pipeline must complete within 30 seconds of receiving end_of_speech.
If it hangs, cancel the pipeline and reset to WAITING for the next
turn — do not close the session.

Cleanup is guaranteed by a `finally` block that always runs regardless
of how the session exits.

---

## Audio format contract

**Into the pipeline (from any transport):**
- Format: raw PCM, no WAV header
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Dtype: float32, values in [-1.0, 1.0]
- Chunk size: 1600 samples = 100ms per chunk

**Out of the pipeline (to any transport):**
- Format: WAV with header
- Sample rate: 24000 Hz
- Channels: 1 (mono)
- Bit depth: 16-bit signed integer

Different sample rates are intentional — Whisper expects 16kHz,
Chatterbox Turbo outputs 24kHz. Standard for voice pipelines.

---

## Phase status

| Phase | Description | Status |
|---|---|---|
| Prototype | Architecture validation, simulated models, full test suite | COMPLETE |
| Phase 1 | Real models, Silero VAD, Redis, production WebSocket | IN PROGRESS |
| Phase 2 | Kokoro TTS, OpenAI LLM, observability groundwork | NOT STARTED |
| Phase 3 | Prometheus metrics, Grafana dashboard | NOT STARTED |
| Phase 4 | CI, example projects, PyPI publish | NOT STARTED |

---

## Prototype — COMPLETE

See `prototype/PROTOTYPE.md` for full detail.

Proved: provider pattern, streaming cascade, session isolation, CLI,
WebSocket protocol, 27/29 tests passing (2 skipped in simulation mode
by design).

---

## Phase 1 — IN PROGRESS

See `docs/phase-1.md` for complete implementation guide.

Replaces simulated providers with real models. Adds Silero VAD, Redis,
and production WebSocket gateway with proper connection lifecycle.

Complete when all 29 integration tests pass including the two previously
skipped, and a real voice turn works end to end.

---

## Phase 2 — NOT STARTED

See `docs/phase-2.md` for complete implementation guide.

Adds Kokoro TTS as a second TTS option and OpenAI as a second LLM
provider — proving swappability is real. Adds observability groundwork
for Phase 3.

---

## Phase 3 — NOT STARTED

Prometheus metrics and Grafana dashboard. Every turn already captures
`PipelineMetrics`. This phase exposes those as Prometheus metrics and
builds a dashboard showing per-stage latency, active sessions, error
rates, and memory usage.

---

## Phase 4 — NOT STARTED

Open source release:
- GitHub Actions CI on every push
- Example projects — web agent, FreeSWITCH integration
- Load test results documented with hardware spec
- CONTRIBUTING.md explaining the provider pattern
- PyPI publish — `pip install voicekit` works from anywhere

---

## Key decisions and why

**Package not platform.**
Voicekit installs like LangChain and deploys like LangChain. Developers
do not run a separate voicekit server. They import classes. Their
deployment handles voicekit exactly like every other dependency.

**Silero VAD from day one.**
Energy-based VAD fails in real environments — noise, quiet speakers,
accents. Silero is a 1.8MB neural model, 12ms per chunk on CPU, accurate
across all conditions. No reason to ship the inferior version first.

**Redis from day one.**
In-process session state cannot survive restarts. Redis adds one
dependency and a few lines of code. The benefit is immediate.

**Chatterbox Turbo over ElevenLabs.**
Zero per-call cost, voice cloning, 75ms first audio, MIT license.

**Whisper over Deepgram.**
Zero per-minute cost after setup. Deepgram costs $0.0043 per minute —
significant at scale.

**Sentence boundary buffering in TTS.**
One word at a time produces flat robotic delivery — no sentence context
for intonation. Full response before TTS kills latency. Sentence
boundaries are the right compromise — natural prosody, audio starts
within one sentence of the LLM beginning to respond.

**One process per session.**
GIL prevents safe concurrent audio pipelines in one process. One process
per session eliminates an entire class of concurrency bugs.

**Docker Compose not Kubernetes.**
Local development tooling does not need Kubernetes. A single VPS handles
20+ concurrent sessions. When a project genuinely needs multi-machine
orchestration, that is the developer's decision to make for their
project, not voicekit's.

**No deploy command.**
Voicekit is a package. Packages do not deploy themselves. The developer's
own deployment process — Dockerfile, docker-compose, whatever they use
— installs voicekit as a dependency. No special handling needed.

---

## Environment variables

```
ANTHROPIC_API_KEY     set when using anthropic LLM provider
OPENAI_API_KEY        set when using openai LLM provider
```

Config resolves `${VAR_NAME}` from shell environment:

```yaml
llm:
  api_key: ${ANTHROPIC_API_KEY}
```

In Docker, pass via environment section in docker-compose.yml or
via `-e` flag. Standard Python package behaviour.

---

## Running tests

```bash
uv run pytest tests/unit/ tests/pipeline/ -v

docker compose -f runtime/docker-compose.test.yml up -d
uv run pytest tests/integration/ -v
docker compose -f runtime/docker-compose.test.yml down
```