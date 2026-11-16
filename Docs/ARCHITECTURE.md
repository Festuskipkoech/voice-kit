# Architecture

Read this before writing any Rust. `FLOW.md` covers runtime behaviour; this document covers structure and decisions.

---

## Package layers

```
┌─────────────────────────────────────────────────────────┐
│  Python SDK        JS SDK            Go SDK              │
│  (voicekit-py)     (voicekit-js)     (voicekit-go)       │
│         │               │                │               │
│         └───────────────┴────────────────┘               │
│                         │  WebSocket ws://                │
└─────────────────────────┼─────────────────────────────────┘
                          ▼
          ┌───────────────────────────┐
          │      voicekit-server      │  ← binary / Docker image
          │  (thin WebSocket shell)   │
          └─────────────┬─────────────┘
                        │  Rust crate dep
          ┌─────────────▼─────────────┐
          │       voicekit-core       │  ← all inference logic lives here
          │  SttEngine  LlmEngine     │
          │  TtsEngine  VAD  Session  │
          └───────────────────────────┘
```

> **Rule:** `voicekit-server` is thin. All logic that touches models lives in `voicekit-core`. If you are adding inference code to the server, move it to core first.

> **Note:** `voicekit-cli` is paused and is not part of the current release.

---

## SDKs

The Python SDK (`voicekit-py`) is complete and ready to use locally. It is not yet published to PyPI — import directly from the package directory. JS and Go SDKs are planned.

---

## Engine traits

The traits are the flexibility contract. Everything swappable is behind one of these three.

### `SttEngine`

Receives a complete utterance — VAD-gated, 16 kHz f32 PCM — and returns the transcript as a string.

**Implementations:** `MoonshineEngine`, `WhisperEngine`

### `LlmEngine`

Takes a fully assembled prompt string and an `mpsc` channel sender. Streams tokens into the sender as they are generated. The session loop reads from the receiver and feeds the sentence buffer. Returns when generation is complete.

**Implementations:** `LlamaCppEngine`, `OpenAIEngine`

### `TtsEngine`

Receives a single cleaned sentence string and returns 24 kHz f32 mono PCM. Called once per sentence boundary — not once per full response. The sentence passes through `text_clean` before reaching the engine.

**Implementations:** `ParadeeEngine`, `KokoroEngine`

---

## How config selects the implementation

Each engine section in `config.toml` sets a `provider` string. At startup the server reads these strings and constructs the matching implementation. The server holds one boxed trait object per engine, shared across sessions via `Arc`. Adding a new model means implementing the relevant trait and registering a provider string in the factory — the session loop is unchanged.

---

## Model registry and path resolution

`models.toml` is a flat registry that maps short keys (such as `"silero-vad"` or `"gemma3-1b-q4"`) to repository paths and file lists. Each entry declares a directory path and the files inside it.

At startup, `main.rs` resolves the config model keys to filesystem paths in two distinct ways depending on the engine type:

- **STT and TTS** receive a directory path. These engines discover their own files internally by joining sub-paths onto the directory.
- **LLM and VAD** receive a specific file path. The resolver takes the first file listed in the registry entry and joins it onto the entry's directory to produce a single absolute path.

This distinction is intentional: ONNX-based engines (STT, TTS) load multiple artefacts from one directory; the LLM and VAD engines each have a single primary binary.

---

## Sentence boundary strategy

The LLM streams tokens. The session loop accumulates them in a string buffer and decides when to flush a sentence to TTS.

> **Rule:** flush when `sentencex` detects a sentence boundary, **or** when the buffer exceeds 200 characters (hard cap).

`sentencex` uses Unicode sentence boundary rules — not a naive deny list. It handles `"Dr. Smith said..."` and `"i.e. the following"` correctly without special-casing.

The 200-character cap is a safety net for LLMs that generate a wall of text without punctuation. It is not the primary mechanism.

**Why not instruct the LLM to insert markers?** Fragile. The model must follow the instruction on every turn, the instruction consumes prompt tokens, and it breaks when you swap the model. Sentence detection in the pipeline is model-agnostic.

---

## Text cleaning

Before any sentence is passed to the TTS engine it goes through a `text_clean` pass. This strips markdown formatting, removes tokens that would be spoken awkwardly (asterisks, URLs, bracket notation), and normalises whitespace. A sentence that is empty after cleaning is silently dropped — no TTS call is made.

---

## VAD behaviour

Voice Activity Detection runs on every 512-sample chunk (32 ms at 16 kHz) using Silero VAD (ONNX).

Each chunk is scored between 0.0 and 1.0:

| Score | Action |
|---|---|
| `≥ threshold` (0.5) | Speech — append chunk to utterance buffer, reset silence counter. |
| `< threshold` | Silence — increment silence counter. When the counter reaches the equivalent of `silence_ms` (default 800 ms), dispatch the utterance buffer to STT and clear it. |

The 800 ms trigger separates a breath pause (should not dispatch) from a true end-of-utterance. It is tunable via `config.toml`.

---

## Inference dependencies

| Component | Crate | Format |
|---|---|---|
| VAD | `ort` (ONNX Runtime) | `.onnx` |
| STT | `ort` (ONNX Runtime) | `.onnx` |
| TTS | `ort` (ONNX Runtime) | `.onnx` |
| LLM | `llama-cpp-2` | `.gguf` |

`ort` is the Rust binding to ONNX Runtime and handles CPU inference for VAD, STT, and TTS. `llama-cpp-2` wraps llama.cpp for GGUF model inference.

No GPU dependencies in the default build. GPU support is an optional feature flag.

---

## Docker deploy model

The server image is built in two stages:

1. **Builder** (`rust:1.92-slim`) — `cargo build --release`, produces a single binary.
2. **Runtime** (`debian:trixie-slim`) — copies the binary and runtime libraries. Final image is around 20 MB without models.

Models are never baked into the image. They mount from the host at runtime. Updating a model is a file copy on the host followed by a container restart — no image rebuild, no registry push.

### espeak-ng data path

The TTS engine (Paradee) uses espeak-ng for grapheme-to-phoneme (G2P) conversion via `misaki-rs`. The `espeak-rs-sys` crate builds a vendored espeak-ng at compile time and bakes its `OUT_DIR` as the data path into the binary. That build directory does not exist in the runtime image.

The environment variable `PIPER_ESPEAKNG_DATA_DIRECTORY` overrides the baked path at startup. `misaki-rs` checks this variable before calling `espeak_Initialize()`, so the binary finds the data without any recompile.

`docker-compose.yml` sets this to `/usr/lib/x86_64-linux-gnu/espeak-ng-data`, which is where `debian:trixie-slim` installs the `espeak-ng-data` package — a hard dependency of `libespeak-ng1`, already present in the runtime image. No image rebuild is required; a `docker compose restart voicekit-server` is sufficient.

**Developers building from source:** on Ubuntu/Debian with `libespeak-ng-dev` installed, `misaki-rs` resolves the system install automatically through its fallback chain and no manual configuration is needed. If your `espeak-ng-data` is in a non-standard location, set `PIPER_ESPEAKNG_DATA_DIRECTORY` before running the server.

---

## Latency budget

Measured on Ubuntu 24.04, no GPU, 4-core CPU.

| Stage | Model | Avg latency |
|---|---|---|
| VAD per chunk | Silero ONNX | < 10 ms |
| STT | Moonshine TINY ONNX | ~376 ms |
| LLM | Gemma 3 1B Q4_K_M | ~1 578 ms |
| TTS per sentence | Paradee int8 ONNX | ~708 ms |
| **Total sequential** | | **~2 661 ms** |
| **First audio (streamed)** | | **~1 100 ms** |

First audio = STT + time to first LLM sentence boundary + TTS of that sentence. All subsequent sentences overlap with LLM generation.

---

## Related

| Document | What it covers |
|---|---|
| `FLOW.md` | Runtime behaviour of a single WebSocket session |
| `DIAGRAM.md` | Visual reference for the pipeline and layers |
| `PROTOCOL.md` | Wire protocol between server and any client or SDK |
