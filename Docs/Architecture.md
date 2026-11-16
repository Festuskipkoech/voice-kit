# Architecture

> Read this before writing any Rust. FLOW.md covers runtime behaviour; this document covers structure and decisions.

---

## Package layers

```
┌─────────────────────────────────────────────────────────┐
│  Python SDK        JS SDK            Go SDK              │
│  (voicekit-py)     (voicekit-js)     (voicekit-go)       │
│         │               │                │               │
│         └───────────────┴────────────────┘               │
│                         │  WebSocket ws://                │
└─────────────────────────┼───────────────────────────────-┘
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
                        ▲
          ┌─────────────┘
          │  Rust crate dep
┌─────────┴─────────────┐
│    voicekit-cli        │  ← model pull, health, benchmark
└───────────────────────┘
```

**Rule:** voicekit-server and voicekit-cli are thin. All logic that touches models lives in voicekit-core. If you are adding inference code to the server, move it to core first.

---

## Engine traits

The traits are the flexibility contract. Everything swappable is behind one of these three.

### SttEngine

```rust
pub trait SttEngine: Send + Sync {
    fn transcribe(&self, audio: &[f32]) -> anyhow::Result<String>;
}
```

Receives a complete utterance (VAD-gated, 16 kHz f32 PCM). Returns the transcript.

Implementations: `MoonshineEngine`, `WhisperEngine`

### LlmEngine

```rust
pub trait LlmEngine: Send + Sync {
    fn generate(
        &self,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> anyhow::Result<()>;
}
```

Streams tokens into `tx`. The session loop reads from the receiver and feeds the sentence buffer. Returns when generation is complete.

Implementations: `LlamaCppEngine`, `OpenAIEngine`

### TtsEngine

```rust
pub trait TtsEngine: Send + Sync {
    fn synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>>;
}
```

Receives a single sentence. Returns 24 kHz f32 PCM. Called once per sentence boundary — not once per full response.

Implementations: `ParadeeEngine`, `KokoroEngine`

### How config selects the implementation

```toml
[stt]
provider = "moonshine"   # resolves to MoonshineEngine at startup

[llm]
provider = "llamacpp"    # resolves to LlamaCppEngine

[tts]
provider = "paradee"     # resolves to ParadeeEngine
```

The server holds `Box<dyn SttEngine>`, `Box<dyn LlmEngine>`, `Box<dyn TtsEngine>`. A new model ships as a new struct implementing the relevant trait plus a provider string registered in the factory. The session loop is unchanged.

---

## Sentence boundary strategy

The LLM streams tokens. The session loop accumulates them in a string buffer and decides when to flush a sentence to TTS.

**Rule:** flush when `sentencex` detects a sentence boundary, OR when the buffer exceeds 200 characters (hard cap).

`sentencex` is a Rust crate that uses Unicode sentence boundary rules — not a naive deny list. It handles "Dr. Smith said..." and "i.e. the following" correctly without special-casing.

The 200-character cap is a safety net for LLMs that generate a wall of text without punctuation. It is not the primary mechanism.

**Why not instruct the LLM to insert markers?** Fragile. The model must follow the instruction on every turn, the instruction consumes prompt tokens, and it breaks when you swap the model. Sentence detection in the pipeline is model-agnostic.

---

## VAD behaviour

Voice Activity Detection runs on every 512-sample chunk (32 ms at 16 kHz) using Silero VAD (ONNX).

```
chunk arrives → VAD scores chunk (0.0–1.0)
  score ≥ threshold (0.5) → speech; append chunk to utterance buffer
  score < threshold        → silence; start silence counter
    silence counter ≥ silence_ms (800 ms) → dispatch utterance buffer to STT; reset
```

The 800 ms silence trigger is the boundary between a breath pause (which should not trigger dispatch) and an end-of-utterance. It is tunable via `config.toml`.

---

## Inference dependencies

| Component | Crate | Format |
|-----------|-------|--------|
| VAD | `ort` (ONNX Runtime) | `.ort` |
| STT | `ort` (ONNX Runtime) | `.ort` |
| TTS | `ort` (ONNX Runtime) | `.onnx` |
| LLM | `llama-cpp-2` | `.gguf` |

`ort` is the Rust binding to ONNX Runtime. It handles CPU inference for VAD, STT, and TTS. `llama-cpp-2` wraps llama.cpp for GGUF model inference.

No GPU dependencies in the default build. GPU support is an optional feature flag.

---

## Docker deploy model

The server image is built in two stages:

1. **Builder** (`rust:1.82-slim`) — `cargo build --release`. Produces a single statically-linked binary.
2. **Runtime** (`debian:bookworm-slim`) — copies the binary plus `libgomp` (OpenMP, needed by ONNX Runtime). Final image is ~20 MB.

**Models are never baked into the image.** They mount from the host at runtime:

```
docker run -v $(pwd)/models:/models:ro ghcr.io/voicekit/voicekit-server:latest
```

This means updating a model is a file copy on the host followed by a container restart — no image rebuild, no push to a registry.

---

## Latency budget

Measured on Ubuntu 24.04, no GPU, 4-core CPU.

| Stage | Model | Avg |
|-------|-------|-----|
| VAD per chunk | Silero ONNX | < 10 ms |
| STT | Moonshine TINY ONNX | ~376 ms |
| LLM | Gemma 3 1B Q4_K_M | ~1 578 ms |
| TTS per sentence | Paradee int8 ONNX | ~708 ms |
| **Total sequential** | | **~2 661 ms** |
| **First audio (streamed)** | | **~1 100 ms** |

"First audio" = STT + time to first LLM sentence boundary + TTS of that sentence. All subsequent sentences overlap with LLM generation.

---

## Related

- [FLOW.md](FLOW.md) — runtime behaviour of a single WebSocket session
- [DIAGRAM.md](DIAGRAM.md) — visual reference for the pipeline and layers
