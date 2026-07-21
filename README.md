# voicekit

Self-hosted voice agent infrastructure as a Python package. Import
Whisper STT, Chatterbox TTS, Silero VAD, and LLM routing directly
into your project. No separate services to manage. No per-minute fees.
Works with any Python voice agent regardless of transport.

---

## The problem it solves

Building a production voice agent means choosing between two bad
options. Pay a managed platform $0.15-0.25 per minute — hundreds of
dollars a month at scale — or spend weeks assembling frameworks
yourself. Voicekit is a third path. Import it. Models run locally.
The streaming pipeline is pre-built and pre-tuned.

---

## Install

```bash
pip install voicekit
# or
uv add voicekit
```

---

## Quick start

```python
import asyncio
from voicekit.pipeline import VoicePipeline
from voicekit.config import load_config

async def main():
    config = load_config()          # reads voice.config.yaml
    pipeline = VoicePipeline(config)
    await pipeline.load()           # loads Whisper and Chatterbox once

    audio_in = asyncio.Queue()      # put float32 PCM chunks here
    audio_out = asyncio.Queue()     # WAV chunks come out here

    metrics = await pipeline.run_turn(audio_in, audio_out)
    print(f"You said: {metrics.transcript}")
    print(f"Agent:    {metrics.response}")
    print(f"Total:    {metrics.total_ms:.0f}ms")

asyncio.run(main())
```

---

## Configuration

```yaml
# voice.config.yaml
project: my-agent

stt:
  model: whisper
  variant: small        # tiny / base / small / medium / large-v3

tts:
  model: chatterbox-turbo
  voice: default        # default or /path/to/reference.wav for cloning

vad:
  enabled: true
  sensitivity: 0.5

llm:
  provider: anthropic
  model: claude-haiku-4-5
  api_key: ${ANTHROPIC_API_KEY}

system_prompt: >
  You are a helpful voice assistant.
  Keep responses concise and natural.
  Never use markdown — you are speaking out loud.
```

---

## The pipeline

```
audio in (float32 PCM, 16kHz)
    ↓
Silero VAD          filters silence, ~1ms per chunk, CPU
    ↓
Whisper STT         transcription, <300ms on small/CPU
    ↓
LLM (streaming)     Claude / OpenAI, tokens stream out
    ↓ (concurrent)
Chatterbox TTS      synthesis, chunks stream as generated
    ↓
audio out (float32 WAV chunks, 24kHz)
```

LLM tokens stream directly into TTS. TTS generates audio before the
LLM finishes responding. Audio chunks flow to the caller immediately
as they are generated — true end-to-end streaming.

---

## Available models

| Type | Model ID | Description | Status |
|---|---|---|---|
| STT | `whisper` | OpenAI Whisper, CPU int8, faster-whisper | available |
| STT | `simulated` | Fake STT for development | available |
| TTS | `chatterbox-turbo` | Resemble AI, voice cloning, MIT | available |
| TTS | `kokoro` | 82M params, fast on CPU | Phase 2 |
| TTS | `simulated` | Fake TTS, real WAV output | available |
| LLM | `anthropic` | Claude Haiku, Sonnet, Opus | available |
| LLM | `openai` | GPT-4o-mini, GPT-4o | Phase 2 |
| LLM | `simulated` | Fake LLM, no API key | available |

---

## Audio format

**Input to pipeline:**
- Raw PCM, no WAV header
- 16000 Hz, mono, float32, values in [-1.0, 1.0]
- 1600 samples per chunk = 100ms

**Output from pipeline:**
- Chunk 1: complete WAV file (header + float32 PCM)
- Chunks 2..N: raw float32 PCM, no header
- 24000 Hz, mono
- Read chunk 1 with soundfile, read chunks 2..N as np.float32

---

## Adding a new model

```python
# voicekit/providers/stt/my_model.py
from voicekit.providers.base import STTProvider

class MySTT(STTProvider):
    async def load(self) -> None: ...
    async def transcribe(self, audio_stream) -> AsyncIterator[str]: ...
    async def health(self) -> bool: ...
```

```python
# voicekit/providers/registry.py
STT_REGISTRY = {
    "whisper": WhisperSTT,
    "my-model": MySTT,    # add this line
}
```

```yaml
stt:
  model: my-model
```

---

## Local development with Docker services

```bash
uv tool install voicekit
voicekit init my-agent
cd my-agent
voicekit dev
```

Starts four services:
- Redis `:6379` — session state
- STT `:8001` — Whisper, loaded once
- TTS `:8002` — Chatterbox, loaded once
- Gateway `:8000` — lightweight routing, no models

The gateway uses `RemotePipeline` — it calls STT and TTS over
WebSocket. Models load once in their services and are shared across
all sessions. The gateway restarts in seconds with no model reloading.

Connect to `ws://localhost:8000/session` for full voice turns.

---

## WebSocket protocol (gateway)

```
client → server    binary     float32 PCM audio, 16kHz, 100ms chunks
client → server    text       {"type": "end_of_speech"}
client → server    text       {"type": "pong"}
server → client    binary     float32 WAV chunks, 24kHz (chunk 1 has header)
server → client    text       {"type": "ready", "session_id": "..."}
server → client    text       {"type": "ping"}
server → client    text       {"type": "transcript", "text": "..."}
server → client    text       {"type": "response", "text": "..."}
server → client    text       {"type": "metrics", "total_ms": 580, ...}
server → client    text       {"type": "error", "message": "..."}
```

Audio chunks always arrive before transcript and metrics.
Respond to every ping with a pong to keep the session alive.

---

## Running tests

```bash
# unit and pipeline — no Docker needed
uv run pytest tests/unit/ tests/pipeline/ -v

# integration — Docker test stack
docker compose -f runtime/docker-compose.test.yml up -d
uv run pytest tests/integration/ -v

# end-to-end voice test
docker compose -f runtime/docker-compose.yml up -d
uv run python tests/test_voice.py
aplay tests/results/response.wav
```

---

## Documentation

- `FLOW.md` — master engineering document
- `prototype/PROTOTYPE.md` — prototype reference
- `docs/phase-1.md` — Phase 1 implementation (COMPLETE)
- `docs/phase-2.md` — Phase 2 implementation guide

---

## License

MIT