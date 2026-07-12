# voicekit

Self-hosted voice agent infrastructure. STT, TTS, VAD, and LLM routing
in a single stack that runs on your own server. No per-minute fees for
transcription or synthesis. No black-box platform. One command to start.

---

## What it is

Most voice agent tools make you choose between two bad options: pay a
managed platform per minute of audio, or spend weeks assembling
frameworks like LiveKit and Pipecat yourself. Voicekit is a third path.

It runs Whisper for speech-to-text and Chatterbox Turbo for
text-to-speech on your own server. After setup, your STT and TTS cost
nothing per minute — only the server itself. The hard production
decisions are already made: one process per session, VAD before STT,
streaming cascade from LLM to TTS so audio starts before the response
finishes generating, health checks, graceful shutdown. You write agent
logic. Voicekit handles the rest.

It exposes two WebSocket endpoints. Your agent connects to them
regardless of whether it uses FreeSWITCH, WebRTC, WebSocket, or any
other transport. Voicekit does not care how audio arrives.

---

## Architecture

```
[your voice agent]
        |
        | ws://localhost:8000/session
        v
  [ gateway :8000 ]  -- holds session state, conversation history
        |
        +-- VAD -- [ STT service :8001 ]  -- Whisper (local)
        |
        +---------- [ LLM API ]           -- Claude / OpenAI / Gemini
        |
        +---------- [ TTS service :8002 ] -- Chatterbox Turbo (local)
```

Audio in flows through VAD, which filters silence before it reaches
Whisper. The transcript goes to the LLM. LLM tokens stream into TTS as
they arrive. TTS audio streams back before the LLM has finished
responding. That streaming cascade is what makes the agent feel
real-time rather than robotic.

All three services run as Docker containers. The gateway holds one
pipeline instance per session. Sessions are fully isolated.

---

## Quick start

```bash
uv tool install voicekit
voicekit init my-agent
cd my-agent
voicekit dev
```

That starts three services:

- STT at `http://localhost:8001`
- TTS at `http://localhost:8002`
- Gateway at `http://localhost:8000`

Connect to `ws://localhost:8000/session` from your agent to run voice
turns. Edit `voice.config.yaml` to configure models and system prompt.

---

## Configuration

All configuration lives in `voice.config.yaml` in your project directory.

```yaml
project: my-agent

stt:
  model: whisper          # which STT model to use
  variant: small          # tiny / base / small / medium / large-v3

tts:
  model: chatterbox-turbo # which TTS model to use
  voice: default          # built-in voice or path to reference audio

vad:
  enabled: true           # filter silence before sending to STT
  sensitivity: 0.5        # 0.0 = only loud speech / 1.0 = catch everything

llm:
  provider: anthropic     # anthropic / openai / simulated
  model: claude-haiku-4-5 # any model the provider supports
  api_key: ${ANTHROPIC_API_KEY}  # resolved from environment

system_prompt: >
  You are a helpful voice assistant.
  Keep responses concise and natural.
  Never use markdown — you are speaking out loud.
```

### Field reference

| Field | Description | Default |
|---|---|---|
| `project` | Name of this project | required |
| `stt.model` | STT model identifier | required |
| `stt.variant` | Model size or variant | `small` |
| `tts.model` | TTS model identifier | required |
| `tts.voice` | Voice profile or reference audio path | `default` |
| `vad.enabled` | Whether to filter silence before STT | `true` |
| `vad.sensitivity` | VAD threshold — higher catches quieter speech | `0.5` |
| `llm.provider` | LLM provider identifier | required |
| `llm.model` | Model name within that provider | required |
| `llm.api_key` | API key, supports `${ENV_VAR}` syntax | `""` |
| `system_prompt` | System prompt sent to the LLM each turn | built-in default |

---

## Available models

### STT

| Model ID | Description | Status |
|---|---|---|
| `simulated` | Fake STT for development and testing, no GPU needed | available |
| `whisper` | OpenAI Whisper via faster-whisper, runs on CPU or GPU | coming soon |
| `parakeet` | NVIDIA Parakeet, lowest latency available | planned |

### TTS

| Model ID | Description | Status |
|---|---|---|
| `simulated` | Fake TTS for development and testing, produces real WAV | available |
| `chatterbox-turbo` | Resemble AI Chatterbox Turbo, ~75ms first audio, voice cloning | coming soon |
| `kokoro` | Kokoro 82M, lightweight CPU-friendly alternative | planned |

### LLM providers

| Provider ID | Models | Status |
|---|---|---|
| `simulated` | Fake LLM for development and testing, no API key needed | available |
| `anthropic` | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6 | coming soon |
| `openai` | gpt-4o-mini, gpt-4o | planned |
| `gemini` | gemini-2.0-flash, gemini-2.5-pro | planned |

---

## CLI commands

```
voicekit init <name>           scaffold a new voice agent project
voicekit setup                 pull docker images and validate config
voicekit dev                   start the full stack locally
voicekit stop                  stop all running services
voicekit status                show status of running containers
voicekit models                list all available models
```

### voicekit init

Creates a new project directory with a `voice.config.yaml` and a
`prompt.txt`. Edit these two files — everything else is managed by
voicekit.

### voicekit dev

Starts all three services with logs streaming to your terminal.
Services restart automatically if they crash. Stop with Ctrl+C.

### voicekit setup

Pull Docker images for the models specified in your config. Run this
once after init, and again when you change models in the config.

---

## Adding a new model

Adding a new STT, TTS, or LLM takes three steps. Nothing else changes.

**Step 1 — write the provider class:**

```python
# voicekit/providers/stt/parakeet.py
from voicekit.providers.base import STTProvider

class ParakeetSTT(STTProvider):

    def __init__(self, variant: str = "0.6b"):
        self.variant = variant
        self.model = None

    async def load(self) -> None:
        # load model into memory once at startup
        ...

    async def transcribe(self, audio_stream) -> AsyncIterator[str]:
        # yield transcript tokens as they are recognised
        ...

    async def health(self) -> bool:
        return self.model is not None
```

**Step 2 — register it:**

```python
# voicekit/providers/registry.py
from voicekit.providers.stt.parakeet import ParakeetSTT

STT_REGISTRY = {
    "simulated": SimulatedSTT,
    "whisper": WhisperSTT,
    "parakeet": ParakeetSTT,   # add this line
}
```

**Step 3 — update your config:**

```yaml
stt:
  model: parakeet
  variant: 0.6b
```

Restart with `voicekit dev`. Nothing in the pipeline, gateway, or CLI
changes. The new model is live.

---

## WebSocket protocol

### Gateway — `/session`

Your agent connects here for full voice turns.

```
client → server  binary:  float32 PCM audio chunks (16kHz mono)
client → server  text:    {"type": "end_of_speech"}
server → client  binary:  WAV audio chunks (agent response)
server → client  text:    {"type": "transcript", "text": "..."}
server → client  text:    {"type": "response", "text": "..."}
server → client  text:    {"type": "metrics", "total_ms": 580, ...}
server → client  text:    {"type": "ready", "session_id": "..."}
```

### STT — `/stt`

Connect directly if you want raw transcription without the full pipeline.

```
client → server  binary:  float32 PCM audio chunks
client → server  text:    {"type": "end"}
server → client  text:    {"type": "token", "text": "word "}
server → client  text:    {"type": "done", "transcript": "..."}
```

### TTS — `/tts`

Connect directly if you want raw synthesis without the full pipeline.

```
client → server  text:    {"type": "token", "text": "word "}
client → server  text:    {"type": "end"}
server → client  binary:  WAV audio chunks
server → client  text:    {"type": "done"}
```

---

## Connecting your project

Voicekit does not care what transport your voice agent uses. The only
interface is the WebSocket endpoints. Here is how different project
types connect:

**FreeSWITCH-based agent:**

```python
# in your FastAPI ESL handler
async with websockets.connect("ws://localhost:8000/session") as ws:
    await ws.send(audio_chunk)           # send RTP audio
    response_audio = await ws.recv()     # get synthesis back
```

**Web browser agent:**

```javascript
// in your frontend
const ws = new WebSocket("ws://localhost:8000/session")
ws.send(audioChunk)                      // send microphone audio
ws.onmessage = (e) => playAudio(e.data) // play response
```

**Twilio media stream:**

```python
# in your Twilio webhook handler
async with websockets.connect("ws://localhost:8000/session") as ws:
    await ws.send(twilio_audio_chunk)
    response = await ws.recv()
```

The transport layer is entirely your concern. Voicekit sees audio bytes
in and sends audio bytes out.

---

## Running tests

**Unit tests — no Docker needed, runs in under 5 seconds:**

```bash
uv run pytest tests/unit/ -v
```

Tests config loading, VAD logic, environment variable resolution, and
error handling. No services required.

**Pipeline tests — no Docker needed, runs in under 60 seconds:**

```bash
uv run pytest tests/pipeline/ -v
```

Tests the full streaming cascade using simulated providers in-process.
Validates session isolation, conversation history, and that first audio
arrives before the LLM finishes responding.

**Integration tests — requires Docker:**

```bash
# start the test stack first
docker compose -f runtime/docker-compose.test.yml up -d

# run tests against live services
uv run pytest tests/integration/ -v
```

Tests the actual WebSocket servers running in Docker containers.
Validates health endpoints, streaming behaviour, latency budgets, and
concurrent session handling.

Test ports are 8011 (STT) and 8012 (TTS) so they never clash with your
dev stack on 8001 and 8002.

---

## Why voicekit

**vs managed platforms (Vapi, Retell, Bland)**

Managed platforms charge $0.15–0.25 per minute once you add STT, TTS,
LLM, and platform fees. At scale that is hundreds of dollars a month.
You also cannot customise the pipeline, cannot self-host for data
privacy, and are dependent on their uptime. Voicekit runs on a $24
Hetzner server. After setup, STT and TTS cost nothing per minute.

**vs frameworks (LiveKit Agents, Pipecat)**

LiveKit and Pipecat are excellent frameworks that give you building
blocks. They do not give you a running system. You still spend weeks
writing Docker configuration, tuning VAD, wiring WebRTC, handling
process isolation, setting up health checks, and figuring out deployment.
Voicekit is what you get when all of that work has already been done.
LiveKit is actually used by voicekit under the hood for WebRTC transport
when needed — the difference is that voicekit users never touch it.

**The honest summary**

Voicekit is not a framework you assemble. It is infrastructure you run.
The question is not which framework to use. The question is how much time
you want to spend on infrastructure before you can write your first line
of agent logic. With voicekit, that answer is three commands.

---

## Roadmap

- [ ] Whisper STT integration (faster-whisper, CPU and GPU)
- [ ] Chatterbox Turbo TTS integration
- [ ] Kokoro TTS as lightweight alternative
- [ ] Claude / OpenAI / Gemini LLM providers
- [ ] Silero VAD replacing energy-based VAD
- [ ] `voicekit deploy` command for one-command VPS deployment
- [ ] Prometheus metrics and Grafana dashboard
- [ ] Voice cloning support via Chatterbox reference audio
- [ ] Multi-language STT support

---

## Project structure

```
prototype/
├── voicekit/                  Python package — CLI and provider system
│   ├── cli.py                 all voicekit commands
│   ├── config.py              voice.config.yaml loading and validation
│   ├── pipeline.py            streaming cascade — the core of everything
│   ├── vad.py                 voice activity detection
│   └── providers/
│       ├── base.py            STTProvider, TTSProvider, LLMProvider interfaces
│       ├── registry.py        maps config strings to provider classes
│       ├── stt/               one file per STT model
│       ├── tts/               one file per TTS model
│       └── llm/               one file per LLM provider
├── runtime/                   Docker services
│   ├── docker-compose.yml     dev stack
│   ├── docker-compose.test.yml test stack on separate ports
│   ├── stt/                   STT WebSocket server
│   ├── tts/                   TTS WebSocket server
│   └── gateway/               session management and pipeline orchestration
├── tests/
│   ├── unit/                  pure logic, no services needed
│   ├── pipeline/              end-to-end pipeline, no Docker needed
│   └── integration/           live Docker services required
└── templates/
    └── voice.config.yaml      default config for new projects
```

---

## License

MIT