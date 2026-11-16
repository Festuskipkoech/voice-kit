# Diagrams

Visual reference for Voicekit's architecture. For prose explanations see [ARCHITECTURE.md](ARCHITECTURE.md) and [FLOW.md](FLOW.md).

---

## Package layers

```
┌──────────────────────────────────────────────────────────────┐
│          Python SDK          JS SDK           Go SDK          │
│          (voicekit-py)       (voicekit-js)    (voicekit-go)   │
│                 │                │                │            │
│                 └────────────────┴────────────────┘            │
│                              │                                  │
│                        WebSocket ws://                          │
└──────────────────────────────┼──────────────────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │        voicekit-server        │
               │   (binary / Docker image)     │
               └───────────────┬───────────────┘
                               │  Rust crate dep
               ┌───────────────▼───────────────┐
               │        voicekit-core          │
               │  SttEngine  LlmEngine         │
               │  TtsEngine  VAD  Session      │
               └───────────────────────────────┘
                               ▲
               ┌───────────────┘  Rust crate dep
               │
   ┌───────────┴───────────┐
   │    voicekit-cli        │
   │  pull · bench · health │
   └───────────────────────┘
```

SDKs connect over WebSocket. The server and CLI are both thin binaries that link voicekit-core. voicekit-core owns all inference logic.

---

## Real-time pipeline — one WebSocket session

```
Client (mic PCM 16kHz f32)
    │
    │  ws:// binary frames (512 samples / 32 ms)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  tokio::spawn — one task per session                             │
│                                                                  │
│  ┌─────────────┐    utterance   ┌─────────────┐    text         │
│  │    VAD      │ ─────────────► │    STT      │ ───────────►    │
│  │ Silero ONNX │                │ Moonshine   │                 │
│  │ 512-sample  │                │ TINY ONNX   │                 │
│  │ chunks      │                │ ~376 ms     │                 │
│  └─────────────┘                └─────────────┘                 │
│                                                      │           │
│                                              ┌───────▼────────┐ │
│                                              │      LLM       │ │
│                                              │  Gemma 3 1B    │ │
│                                              │  GGUF Q4_K_M   │ │
│                                              │  streams tokens │ │
│                                              └───────┬────────┘ │
│                                                      │ tokens   │
│                                                      ▼          │
│  ┌─────────────┐  sentence  ┌───────────────────────────────┐  │
│  │    TTS      │ ◄───────── │       Sentence Buffer         │  │
│  │  Paradee    │            │  sentencex · flush on . ? !   │  │
│  │  int8 ONNX  │            │  or 200-char hard cap         │  │
│  │  ~708 ms    │            └───────────────────────────────┘  │
│  └──────┬──────┘                                               │
│         │  audio PCM 24 kHz                                    │
└─────────┼────────────────────────────────────────────────────-─┘
          │
          │  ws:// binary frames
          ▼
     Client (speaker)
```

**Key:** LLM and TTS run in parallel after the first sentence boundary. The client hears audio before the LLM finishes generating.

---

## Sentence buffer flush logic

```
token arrives from LLM
    │
    ▼
append to buffer
    │
    ▼
sentencex detects boundary?
    │
    ├─ YES → flush up to boundary → TTS call → write audio to socket
    │         keep remainder in buffer
    │
    └─ NO  → buffer length ≥ 200 chars?
                 │
                 ├─ YES → flush whole buffer → TTS call → write audio to socket
                 │
                 └─ NO  → wait for next token

generation complete → flush any remaining buffer (even if no boundary)
```

---

## VAD state machine

```
                        ┌─────────────┐
                        │   SILENCE   │◄──────────────────────────┐
                        └──────┬──────┘                            │
                               │ score ≥ 0.5                       │
                               ▼                                   │
                        ┌─────────────┐                            │
                 ┌──────│   SPEECH    │──────┐                     │
                 │      └─────────────┘      │                     │
      score ≥ 0.5│                           │score < 0.5          │
         (stays) │                           ▼                     │
                 │                  ┌─────────────────┐            │
                 └──────────────────│ SILENCE COUNTER │            │
                                    │  < 25 chunks    │            │
                                    └────────┬────────┘            │
                                             │ ≥ 25 chunks         │
                                             │ (800 ms)            │
                                             ▼                     │
                                    ┌─────────────────┐            │
                                    │  DISPATCH STT   │────────────┘
                                    │  clear buffer   │
                                    └─────────────────┘
```

---

## Docker deploy

```
 ┌─────────────────────────────────┐
 │  Stage 1: rust:1.82-slim        │
 │  cargo build --release          │
 │  → /app/voicekit-server         │
 └────────────────┬────────────────┘
                  │ COPY binary
                  ▼
 ┌─────────────────────────────────┐
 │  Stage 2: debian:bookworm-slim  │
 │  voicekit-server binary         │
 │  libgomp (OpenMP / ONNX)        │
 │  ~20 MB total                   │
 └────────────────┬────────────────┘
                  │ docker run
                  ▼
 ┌─────────────────────────────────────────┐
 │  container                              │
 │  port 8080  ws://                       │
 │  /models   ← mounted from host (ro)    │  ← models never baked in
 │  /config.toml ← mounted from host (ro) │
 └─────────────────────────────────────────┘
          ▲
          │ -v $(pwd)/models:/models:ro
 ┌────────┴───────────┐
 │  host ./models/    │
 │  moonshine-tiny    │
 │  paradee-int8      │
 │  gemma3-1b-q4.gguf │
 └────────────────────┘
```

Updating a model: copy the new file to `./models/`, restart the container. No image rebuild.

---

## Engine trait wiring

```
config.toml
  [stt] provider = "moonshine"
  [llm] provider = "llamacpp"
  [tts] provider = "paradee"
         │
         ▼ startup
  EngineFactory::build(&config)
         │
         ├─ "moonshine" → Box<dyn SttEngine> (MoonshineEngine)
         ├─ "llamacpp"  → Box<dyn LlmEngine> (LlamaCppEngine)
         └─ "paradee"   → Box<dyn TtsEngine> (ParadeeEngine)
                 │
                 ▼ Arc-wrapped, shared across sessions
         Session { stt, llm, tts }
```

Adding a new model: implement the trait, register the provider string in the factory, ship the binary. Session loop unchanged.

---

## Related

- [ARCHITECTURE.md](Architecture.md) — full prose explanation of every decision shown here
- [FLOW.md](FLOW.md) — runtime behaviour, edge cases, what is not in v1
