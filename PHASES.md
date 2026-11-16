# Voicekit — Build Phases

This document is the implementation guide. Read it alongside ARCHITECTURE.md and FLOW.md. Those two explain what to build; this explains in what order and how to know when each piece is done.

The rule across all phases: **each round must be independently runnable before the next round starts.** Do not wire STT to LLM before STT works on its own. Do not start the WebSocket server before the session loop works in a unit test. This is not a style preference — it is the only way to know which piece broke when something fails.

---

## Phase 1 — The working pipeline

Six rounds. At the end of round 6 you have a real server: a WAV file sent over WebSocket comes back as synthesised speech.

---

### Round 1 — Workspace scaffold

**What you build:**
- Cargo workspace with four members: `voicekit-core` (lib), `voicekit-server` (bin), `voicekit-cli` (bin), `tests/e2e` (bin)
- `Config` struct in `voicekit-core::config` with full serde deserialization from `config.toml`
- The three engine traits (`SttEngine`, `LlmEngine`, `TtsEngine`) defined in `voicekit-core::engines` with stub `todo!()` implementations so the crate compiles
- `voicekit-server::main` that loads config and prints it, then exits
- `voicekit-cli::main` that prints `"voicekit cli"` and exits

**Files touched:**
```
Cargo.toml                          ← workspace manifest
crates/voicekit-core/Cargo.toml
crates/voicekit-core/src/lib.rs
crates/voicekit-core/src/config.rs
crates/voicekit-core/src/engines/mod.rs
crates/voicekit-server/Cargo.toml
crates/voicekit-server/src/main.rs
crates/voicekit-cli/Cargo.toml
crates/voicekit-cli/src/main.rs
tests/e2e/Cargo.toml
config.toml                         ← template config
crates/voicekit-core/fixtures/config.toml
```

**Acceptance criteria:**
```bash
cargo build --workspace          # compiles with zero errors
cargo test --workspace           # config parsing test passes
```

The config test loads `fixtures/config.toml`, deserializes it into `Config`, and asserts that `config.stt.provider == "moonshine"`. That is the only test in this round.

---

### Round 2 — VAD

**What you build:**
- `voicekit-core::vad::Vad` struct that loads Silero ONNX at construction
- `Vad::process_chunk(&mut self, samples: &[f32]) -> Option<Vec<f32>>` — returns `Some(utterance)` when silence threshold is crossed, `None` otherwise
- Unit tests for the state machine: silence counter resets on speech, dispatch fires at exactly 25 consecutive silence chunks, empty buffer does not dispatch

**Files touched:**
```
crates/voicekit-core/src/vad.rs
crates/voicekit-core/tests/vad_test.rs
crates/voicekit-core/fixtures/hello.wav      ← 1-second 16kHz mono WAV
crates/voicekit-core/fixtures/silence.wav    ← 2 seconds of silence
```

**Acceptance criteria:**
```bash
cargo test -p voicekit-core vad   # all vad_test.rs tests pass
```

Manually: feed `hello.wav` chunk by chunk into `Vad::process_chunk` and assert that exactly one `Some(utterance)` is returned. Feed `silence.wav` and assert that `None` is always returned.

**Key detail:** `Vad` owns the ONNX session. It is constructed once at server startup and passed into the session task behind an `Arc`. It must be `Send + Sync`.

---

### Round 3 — STT

**What you build:**
- `MoonshineEngine` in `crates/voicekit-core/src/engines/stt/moonshine.rs` implementing `SttEngine`
- `SttEngine::transcribe(&self, audio: &[f32]) -> anyhow::Result<String>`
- Engine factory function in `engines/mod.rs` that reads `config.stt.provider` and returns `Box<dyn SttEngine>`
- Integration test: load `hello.wav`, transcribe it, assert the transcript contains the expected words

**Files touched:**
```
crates/voicekit-core/src/engines/stt/mod.rs
crates/voicekit-core/src/engines/stt/moonshine.rs
crates/voicekit-core/src/engines/mod.rs      ← factory fn
crates/voicekit-core/tests/integration/stt_test.rs
```

**Acceptance criteria:**
```bash
# unit tests (no model needed)
cargo test -p voicekit-core stt

# integration test (model must be present)
VOICEKIT_INTEGRATION=1 cargo test -p voicekit-core --features integration stt
```

The integration test loads the real Moonshine TINY model from `../../models/` (relative to the crate root) and transcribes `fixtures/hello.wav`. Assert the transcript is non-empty and matches the known content of the WAV.

---

### Round 4 — LLM

**What you build:**
- `LlamaCppEngine` in `crates/voicekit-core/src/engines/llm/llamacpp.rs` implementing `LlmEngine`
- `LlmEngine::generate(&self, prompt: &str, tx: Sender<String>) -> anyhow::Result<()>` — streams tokens into `tx`, returns when generation is complete
- Factory function updated for `config.llm.provider`
- Unit test: mock `LlmEngine` impl that sends three known tokens then closes — assert the receiver gets exactly those three tokens in order
- Integration test: load Gemma 3 1B, generate a short response, assert at least one token is received within 5 s

**Files touched:**
```
crates/voicekit-core/src/engines/llm/mod.rs
crates/voicekit-core/src/engines/llm/llamacpp.rs
crates/voicekit-core/src/engines/mod.rs
crates/voicekit-core/tests/integration/llm_test.rs
```

**Acceptance criteria:**
```bash
cargo test -p voicekit-core llm
VOICEKIT_INTEGRATION=1 cargo test -p voicekit-core --features integration llm
```

**Key detail:** the `generate` call must not block the Tokio runtime. Use `tokio::task::spawn_blocking` to wrap the llama.cpp call, which is synchronous and CPU-bound.

---

### Round 5 — Sentence buffer + TTS

**What you build:**
- `voicekit-core::sentence::SentenceBuffer` — accumulates tokens, flushes on `sentencex` boundary or 200-char hard cap
- `SentenceBuffer::push(&mut self, token: &str) -> Vec<String>` — returns zero or more sentences ready for TTS
- `SentenceBuffer::flush(&mut self) -> Option<String>` — drains any remaining text at end of generation
- `ParadeeEngine` in `crates/voicekit-core/src/engines/tts/paradee.rs` implementing `TtsEngine`
- `TtsEngine::synthesize(&self, text: &str) -> anyhow::Result<Vec<f32>>`
- Factory function updated for `config.tts.provider`
- Unit tests for `SentenceBuffer`: boundary on `.`, boundary on `?`, boundary on `!`, no boundary on `Dr.`, hard cap at 200 chars, flush returns remainder

**Files touched:**
```
crates/voicekit-core/src/sentence.rs
crates/voicekit-core/src/engines/tts/mod.rs
crates/voicekit-core/src/engines/tts/paradee.rs
crates/voicekit-core/src/engines/mod.rs
crates/voicekit-core/tests/sentence_test.rs
crates/voicekit-core/tests/integration/tts_test.rs
```

**Acceptance criteria:**
```bash
cargo test -p voicekit-core sentence
cargo test -p voicekit-core tts
VOICEKIT_INTEGRATION=1 cargo test -p voicekit-core --features integration tts
```

The sentence unit tests are the most important in the whole project — they are the only tests that run in CI without models and that protect the LLM→TTS boundary. Write them thoroughly.

---

### Round 6 — WebSocket server

**What you build:**
- `voicekit-core::session::run_session(socket, stt, llm, tts, vad, config)` — the Tokio task from FLOW.md, full implementation
- `voicekit-server::main` updated to: load config → build engines → bind TCP → `tokio::spawn(run_session(...))` per connection
- E2E test: spin up a server on a random port, connect a WebSocket, send `hello.wav` as binary frames, assert binary frames come back within 10 s

**Files touched:**
```
crates/voicekit-core/src/session.rs
crates/voicekit-server/src/main.rs
tests/e2e/src/pipeline_test.rs
```

**Acceptance criteria:**
```bash
# with real models present
VOICEKIT_INTEGRATION=1 cargo test -p e2e
```

Manual smoke test:
```bash
cargo run -p voicekit-server -- --config config.toml
# in another terminal:
python scripts/pipeline.py   # the Python pipeline from the experiment phase
```
The Python pipeline should produce a response. That is the end of Phase 1.

---

## Phase 2 — Packaging and SDKs

Start Phase 2 only after the Phase 1 smoke test passes. The WebSocket protocol between client and server is the contract the SDKs depend on. Changing it after SDKs exist is expensive.

### Round 7 — Protocol stabilisation

Before writing any SDK, write down the WebSocket protocol as a spec:
- What binary format audio frames use (PCM f32-LE, 16 kHz, 512 samples)
- What the server sends back (PCM f32-LE, 24 kHz, variable length)
- Any control messages (JSON text frames for transcript events, error events)
- Versioning strategy

Add this to `docs/PROTOCOL.md`. Do not start Round 8 until this document exists and matches the running server.

### Round 8 — Python SDK (`voicekit-py`)

First SDK because the Python experiment already has working mic capture and playback code to draw from. The SDK is a thin wrapper: open WebSocket, stream mic frames, receive audio frames, expose callbacks.

```python
client = VoicekitClient("ws://localhost:8080")
client.on_transcript(lambda text: print(text))
client.on_audio(lambda chunk: speaker.play(chunk))
client.stream_mic()      # blocks; Ctrl-C to stop
```

Publish to PyPI as `voicekit`.

### Round 9 — JavaScript SDK (`voicekit-js`)

Browser and Node.js. Uses the Web Audio API in the browser for mic capture and playback. Publish to npm as `voicekit`.

### Round 10 — Go SDK (`voicekit-go`)

Idiomatic Go: channels for audio in/out, context for cancellation. Publish as a Go module at `github.com/voicekit/go`.

### Round 11 — Docker image + registry

- Finalise the multi-stage Dockerfile
- GitHub Actions workflow to build and push to `ghcr.io/voicekit/voicekit-server` on every tag
- `voicekit-cli pull` downloads model files from a known URL (GitHub Releases or HuggingFace)

---

## Testing strategy

### What runs in CI (no models required)

- All unit tests in `voicekit-core`: config parsing, VAD state machine, sentence buffer
- Clippy (`cargo clippy --workspace -- -D warnings`)
- Format check (`cargo fmt --check`)

### What runs locally with models

Set `VOICEKIT_INTEGRATION=1` to enable:
- STT integration: transcribes `fixtures/hello.wav` with real Moonshine TINY
- LLM integration: generates one short response with real Gemma 3 1B
- TTS integration: synthesises one sentence with real Paradee int8
- E2E pipeline: full server round-trip

Gate these behind a Cargo feature:
```toml
[features]
integration = []
```

```rust
#[cfg(feature = "integration")]
#[test]
fn test_moonshine_transcribes_hello() { ... }
```

### Fixtures

| File | Content | Used by |
|------|---------|---------|
| `fixtures/hello.wav` | "Hello, how are you today?" 16kHz mono f32 1.2s | STT integration, E2E |
| `fixtures/silence.wav` | 2s of silence at 16kHz | VAD unit test |
| `fixtures/config.toml` | Minimal valid config pointing to `../../models/` | All integration tests |

Generate `hello.wav` and `silence.wav` using the existing `scripts/generate_test_audio.py` from the experiment phase. Commit them. They are small (< 100 KB each) and binary-stable.

### E2E test structure

```rust
// tests/e2e/src/pipeline_test.rs
#[tokio::test]
async fn full_pipeline_returns_audio() {
    let server = TestServer::start(config()).await;
    let mut client = WebSocketClient::connect(server.addr()).await;

    // send hello.wav as 512-sample binary frames
    let audio = load_wav("fixtures/hello.wav");
    for chunk in audio.chunks(512) {
        client.send_binary(pcm_to_bytes(chunk)).await;
    }
    // send 800ms of silence to trigger VAD dispatch
    client.send_silence(800).await;

    // assert audio frames arrive within 10s
    let response = timeout(Duration::from_secs(10), client.recv_audio()).await;
    assert!(response.is_ok(), "no audio received within 10s");
    assert!(!response.unwrap().is_empty());
}
```

---

## What is not in Phase 1 or Phase 2

These are closed decisions. Do not implement them in the first two phases:

| Feature | Why deferred |
|---------|--------------|
| Barge-in | Requires a parallel VAD task and an interrupt signal into the session loop — adds significant complexity before the basic loop is proven |
| GPU inference | Optional feature flag only; the default build is CPU |
| Persistent conversation history | In-memory history per session is sufficient; persistence requires a storage decision that is out of scope |
| Authentication | WebSocket auth (token, API key) is a Round 7+ concern after the protocol is stable |
| Multi-language STT | Moonshine TINY is English only; language selection is a config extension after the engine factory pattern is proven |
| Streaming TTS (chunk-level) | Paradee returns a complete buffer per sentence; true streaming (chunk-by-chunk within a sentence) is a TtsEngine API change for later |

---

## Decision log

Decisions made during the experiment phase. These are closed — do not re-litigate them when scaffolding.

**STT: Moonshine TINY over Whisper**
Moonshine TINY transcribes a complete utterance in ~376 ms on CPU. Whisper base is slower on the same hardware. Moonshine TINY_STREAMING exists for lower latency but introduces streaming state complexity; use it in a future round after the base pipeline is stable.

**LLM: Gemma 3 1B Q4_K_M over Qwen3 0.6B and Qwen3 1.7B**
Gemma 3 1B produced the best response quality in benchmarks with no factual errors. Qwen3 models spent tokens on `<think>` blocks even with `/no_think` prefix. Qwen3 0.6B is a valid speed fallback if Gemma 3 1B is too slow on the target hardware.

**TTS: Paradee int8 (distilled Kokoro) over Supertonic**
Paradee is 9 MB, MOS 4.41, voice `af_heart` (American female). Supertonic scored MOS 1.53 — robotic and unusable. No other model found with lower latency and comparable naturalness on CPU as of November 2026.

**Sentence boundaries: sentencex over naive deny list**
A deny list of abbreviations (`Dr.`, `Mr.`, `i.e.`) is fragile and language-specific. `sentencex` uses Unicode sentence boundary rules and handles these cases correctly without maintenance. The 200-character hard cap is a safety net only.

**Test fixtures: committed WAV files over skip-in-CI**
Skipping integration tests in CI means the loading path is never automatically tested. Small WAV fixtures (< 100 KB) are binary-stable and worth the repo size.

**Language: Rust over Python or Go**
Rust gives memory safety, predictable latency (no GC pauses), and direct access to `ort` and `llama-cpp-2` with no FFI overhead. Python is kept for scripting and SDKs only. Go was considered for the server layer but offers no inference library advantage over Rust.
