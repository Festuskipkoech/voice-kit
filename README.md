# Voicekit

A production-ready voice-agent engine built in Rust. Wire it to a microphone and a speaker; it handles everything in between - silence detection, speech recognition, language model inference, and speech synthesis - over a single WebSocket connection.

**Flexibility** - every model is behind a trait. Swap STT, LLM, or TTS by changing one line in `config.toml`. No recompile, no code change.

**Latency** - the LLM streams tokens directly into the TTS pipeline. The client hears the first audio chunk before the model finishes generating. Target: < 3 s end-to-end on CPU.

---

## Measured performance (Ubuntu 24.04, no GPU)

| Stage | Model | Avg latency |
|---|---|---|
| STT | Moonshine TINY (ONNX int8) | ~376 ms |
| LLM | Gemma 3 1B Q4_K_M (GGUF) | ~1 578 ms |
| TTS | Paradee int8 (ONNX) | ~708 ms |
| Total sequential | | ~2 661 ms |
| **First audio (streamed)** | | **~1 100 ms** |

*First audio = STT + first LLM sentence + TTS of that sentence. LLM and TTS overlap after the first sentence boundary.*

---

## I want to run Voicekit on my server

You only need Docker. No cloning required.

```bash
curl -O https://raw.githubusercontent.com/Festuskipkoech/voice-kit/main/docker-compose.yml
docker compose up
```

On first run:
- `config-init` writes `./voicekit-data/config.toml` with defaults
- `model-init` downloads and verifies all model weights into `./models/` (~5 GB)
- `voicekit-server` starts once models are ready

The server is ready when you see:

```
voicekit-server-1  | INFO voicekit_core::server: all engines loaded
voicekit-server-1  | INFO voicekit_core::server: listening addr=0.0.0.0:8080
```

On first run, before the server starts, review the generated config:

```bash
# first run - let config-init and model-init finish, then stop with Ctrl+C
docker compose up

# review and edit the config
nano ./voicekit-data/config.toml

# start the server
docker compose up voicekit-server
```

Subsequent runs skip downloading - models are verified by checksum and reused.

### Configuration

Edit `./voicekit-data/config.toml` then restart:

```bash
docker compose restart voicekit-server
```

#### `[server]`

| Field | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8080` | WebSocket port |
| `max_connections` | `4` | Maximum concurrent sessions |

#### `[session_policy]`

Controls what clients are permitted to override per session. All disabled by default.

| Field | Default | Description |
|---|---|---|
| `allow_system_prompt_override` | `false` | Clients may send a custom system prompt |
| `allow_voice_override` | `false` | Clients may select a different TTS voice |
| `allow_vad_override` | `false` | Clients may tune VAD sensitivity |

#### `[llm]`

| Field | Default | Description |
|---|---|---|
| `system_prompt` | see config | Assistant persona and instructions |
| `context_size` | `2048` | Token context window |

#### `[tts]`

| Field | Default | Description |
|---|---|---|
| `voice` | `af_heart` | Default TTS voice |
| `sample_rate` | `24000` | Output sample rate in Hz |

#### `[vad]`

| Field | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Speech probability threshold (0.0–1.0) |
| `silence_ms` | `800` | Silence duration before end-of-utterance |

### Port and log level

```bash
VOICEKIT_PORT=9000 docker compose up   # run on a different port
RUST_LOG=debug docker compose up       # verbose logging
```

### Updating

```bash
docker compose pull
docker compose up
```

Models are not re-downloaded unless checksums change.

### Requirements

- Docker 24+
- ~6 GB disk for model weights
- 8 GB RAM minimum (16 GB recommended)

---

## I want to modify or contribute to Voicekit

Clone the repo and build from source.

### Prerequisites

- Rust 1.92+
- CMake, libclang, libespeak-ng-dev (for building)
- Python 3.11+ with `uv` (for the Python SDK and scripts)

### Setup

```bash
git clone https://github.com/Festuskipkoech/voice-kit.git
cd voice-kit

# Download models
docker compose run --rm model-init

# Run all unit tests (no models required)
make test

# Run the server locally
cp config.template.toml config.local.toml
make run
```

### Common commands

```bash
make build          # release build
make test           # unit tests
make test-integration   # integration tests (models required)
make test-e2e           # full pipeline test against live server
make lint           # cargo clippy
make fmt            # cargo fmt
make py-test        # Python SDK tests (voicekit-py)
```

### Read these first

| Document | What it covers |
|---|---|
| [`Docs/Architecture.md`](Docs/ARCHITECTURE.md) | Package layers, engine traits, config schema, Docker deploy model, latency budget |
| [`Docs/FLOW.md`](Docs/FLOW.md) | One WebSocket session start to finish - VAD decisions, STT dispatch, LLM streaming, sentence buffer, TTS, edge cases |
| [`Docs/PROTOCOL.md`](Docs/PROTOCOL.md) | Wire protocol spec - frame types, audio format, handshake, close codes, SDK checklist |
| [`Docs/DIAGRAM.md`](Docs/DIAGRAM.md) | Visual architecture - pipeline diagram, layer stack, trait system, Docker diagram |

Start with [`Docs/Architecture.md`](Docs/ARCHITECTURE.md) before writing any Rust. Read [`Docs/FLOW.md`](Docs/FLOW.md) before touching the session loop.

### Packages

| Package | What it is |
|---|---|
| `voicekit-core` | Rust crate - all inference logic, traits, session loop |
| `voicekit-server` | Binary + Docker image - WebSocket server |
| `voicekit-py` | Python SDK - ready to use locally, not yet on PyPI |

`voicekit-cli` is paused and not part of the current release.

---

## License

MIT
