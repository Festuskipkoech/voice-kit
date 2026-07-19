# voicekit

Self-hosted voice agent infrastructure as a Python package. Import
Whisper STT, Chatterbox TTS, Silero VAD, and LLM routing directly into
your project. No separate services to manage. No per-minute API fees.
Works in any Python voice agent regardless of transport.

---

## The problem it solves

Building a production voice agent today means choosing between two bad
options. Pay a managed platform like Vapi or Retell $0.15–0.25 per
minute — which becomes hundreds of dollars a month at scale — or spend
weeks assembling frameworks yourself, writing Docker configuration,
tuning VAD, wiring models together, and debugging production issues
nobody warned you about.

Voicekit is a third path. Import it like any other Python package. Whisper
runs locally for STT. Chatterbox Turbo runs locally for TTS. Silero VAD
filters silence before it reaches the model. LLM providers are swappable
via config. The streaming cascade — where LLM tokens flow directly into
TTS as they arrive — is pre-built and pre-tuned. You write agent logic.
Voicekit handles the rest.

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

    # one voice turn
    audio_in = asyncio.Queue()      # put float32 PCM chunks here
    audio_out = asyncio.Queue()     # WAV chunks come out here

    # fill audio_in with audio from your transport
    # ...

    metrics = await pipeline.run_turn(audio_in, audio_out)
    print(f"Turn complete in {metrics.total_ms:.0f}ms")
    print(f"Transcript: {metrics.transcript}")
    print(f"Response:   {metrics.response}")

asyncio.run(main())
```

---

## Configuration

Create `voice.config.yaml` in your project root:

```yaml
project: my-agent

stt:
  model: whisper
  variant: small        # tiny / base / small / medium / large-v3

tts:
  model: chatterbox-turbo
  voice: default        # default or path to .wav for voice cloning

vad:
  enabled: true
  sensitivity: 0.5      # 0.0 = loud speech only / 1.0 = catch everything

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

## How it fits your project

Voicekit is transport-agnostic. It receives audio chunks in and returns
audio chunks out. Whatever your project uses to move audio — FreeSWITCH,
WebRTC, WebSocket, Twilio, microphone — connects to the same pipeline.

**FreeSWITCH / ESL project:**

```python
from voicekit.pipeline import VoicePipeline
from voicekit.config import load_config

pipeline = VoicePipeline(load_config())
await pipeline.load()

# in your ESL audio handler
async def handle_audio(rtp_chunk):
    await audio_in.put(rtp_chunk)

async def on_end_of_speech():
    metrics = await pipeline.run_turn(audio_in, audio_out)
    # inject audio_out back into FreeSWITCH via ESL
```

**WebSocket voice agent:**

```python
@app.websocket("/voice")
async def voice_endpoint(ws: WebSocket):
    await ws.accept()
    pipeline = VoicePipeline(load_config())
    await pipeline.load()

    while True:
        audio_in = asyncio.Queue()
        audio_out = asyncio.Queue()

        # receive audio from browser
        async for chunk in receive_audio(ws):
            await audio_in.put(chunk)

        await pipeline.run_turn(audio_in, audio_out)

        # send audio back to browser
        while not audio_out.empty():
            await ws.send_bytes(audio_out.get_nowait())
```

**Twilio media streams:**

```python
# in your Twilio webhook handler
async def handle_twilio_stream(websocket):
    pipeline = VoicePipeline(load_config())
    await pipeline.load()

    async for message in websocket:
        audio_chunk = decode_mulaw(message)
        await audio_in.put(audio_chunk)
```

---

## The pipeline

```
audio in
    ↓
Silero VAD          — filters silence, ~12ms per chunk, CPU
    ↓
Whisper STT         — transcription, <300ms, CPU or GPU
    ↓
LLM (streaming)     — Claude / OpenAI / Gemini, tokens stream out
    ↓
Chatterbox TTS      — synthesis, first audio <150ms, CPU or GPU
    ↓
audio out
```

LLM tokens stream directly into TTS as they arrive. TTS generates audio
before the LLM has finished responding. First audio plays within 600ms
of the user stopping speaking.

---

## Available models

| Type | Model ID | Description |
|---|---|---|
| STT | `whisper` | OpenAI Whisper via faster-whisper, CPU or GPU |
| STT | `simulated` | Fake STT for development, no models needed |
| TTS | `chatterbox-turbo` | 75ms first audio, voice cloning, MIT license |
| TTS | `kokoro` | 82M params, fast on CPU, no GPU required |
| TTS | `simulated` | Fake TTS for development, produces real WAV |
| LLM | `anthropic` | Claude Haiku, Sonnet, Opus |
| LLM | `openai` | GPT-4o-mini, GPT-4o |
| LLM | `simulated` | Fake LLM for development, no API key needed |

---

## Adding a new model

Write one class, register one line, change one config value.

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
    "my-model": MySTT,    # add this
}
```

```yaml
stt:
  model: my-model
```

Nothing else changes. The pipeline, config loader, and tests are all
model-agnostic.

---

## Audio format

**Input to pipeline:**
- Format: raw PCM, no WAV header
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Dtype: float32, values in [-1.0, 1.0]
- Chunk size: 1600 samples = 100ms per chunk

**Output from pipeline:**
- Format: WAV with header
- Sample rate: 24000 Hz
- Channels: 1 (mono)
- Bit depth: 16-bit signed integer

---

## Local development with Docker services

For local development, voicekit provides a CLI that starts STT, TTS,
and a gateway service as Docker containers. This is useful when you want
to test your agent against real models without embedding them directly
in your project.

```bash
uv tool install voicekit
voicekit init my-agent
cd my-agent
voicekit dev
```

This starts:
- STT service at `ws://localhost:8001/stt`
- TTS service at `ws://localhost:8002/tts`
- Gateway at `ws://localhost:8000/session`

Connect to the gateway for full pipeline turns, or connect directly to
STT and TTS for raw transcription and synthesis.

In production, your project's own Dockerfile installs voicekit as a
dependency and your application imports it directly. The Docker services
are development tooling only.

---

## PipelineMetrics

Every `run_turn()` call returns a `PipelineMetrics` object:

```python
metrics = await pipeline.run_turn(audio_in, audio_out)

metrics.transcript          # what the user said
metrics.response            # what the LLM responded
metrics.stt_ms              # time for transcription
metrics.llm_first_token_ms  # time to first LLM token
metrics.tts_first_chunk_ms  # time to first audio chunk
metrics.total_ms            # total turn time
```

Use these to monitor latency in production and identify which stage
to optimise if targets are not being met.

---

## Documentation

- `FLOW.md` — master engineering document
- `prototype/PROTOTYPE.md` — prototype reference
- `docs/phase-1.md` — Phase 1 implementation guide
- `docs/phase-2.md` — Phase 2 implementation guide

---

## License

MIT