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
```

SDKs connect over WebSocket. The server is a thin binary that links `voicekit-core`. `voicekit-core` owns all inference logic.

> **Note:** `voicekit-cli` is paused and not part of the current release.

---

## Real-time pipeline - one WebSocket session

```
Client (mic PCM 16kHz f32)
    │
    │  ws:// binary frames (512 samples / 32 ms)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  tokio::spawn - one task per session                             │
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
│  ┌──────────────────┐  sentence  ┌─────────────────────────┐   │
│  │   text_clean     │            │     Sentence Buffer     │   │
│  │       +          │ ◄───────── │  sentencex · . ? !      │   │
│  │      TTS         │            │  or 200-char hard cap   │   │
│  │  Paradee int8    │            └─────────────────────────┘   │
│  │  ~708 ms         │                                           │
│  └────────┬─────────┘                                           │
│           │  audio PCM 24 kHz                                   │
└───────────┼─────────────────────────────────────────────────────┘
            │
            │  ws:// binary frames
            ▼
       Client (speaker)
```

**Key:** LLM and TTS run in parallel after the first sentence boundary. The client hears audio before the LLM finishes generating. Each sentence is cleaned through `text_clean` before reaching TTS.

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
    ├─ YES → flush up to boundary → text_clean → TTS → write audio to socket
    │         keep remainder in buffer
    │
    └─ NO  → buffer length ≥ 200 chars?
                 │
                 ├─ YES → flush whole buffer → text_clean → TTS → write audio to socket
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
 │  Stage 1: rust:1.92-slim        │
 │  cargo build --release          │
 │  → /app/voicekit-server         │
 └────────────────┬────────────────┘
                  │ COPY binary
                  ▼
 ┌─────────────────────────────────┐
 │  Stage 2: debian:trixie-slim    │
 │  voicekit-server binary         │
 │  libgomp (OpenMP / ONNX)        │
 │  operator scripts baked in      │
 │  ~20 MB total (no models)       │
 └────────────────┬────────────────┘
                  │ docker run
                  ▼
 ┌─────────────────────────────────────────┐
 │  container                              │
 │  port 8080  ws://                       │
 │  /models        ← mounted from host    │  ← models never baked in
 │  /data/config.toml ← mounted from host │
 └─────────────────────────────────────────┘
          ▲
          │ volumes
 ┌────────┴───────────────┐
 │  host ./models/        │
 │  moonshine/tiny/       │
 │  paradee/              │
 │  llm/                  │
 │    gemma-3-1b-it.gguf  │
 │  silero/               │
 │    silero_vad.onnx     │
 └────────────────────────┘
```

Updating a model: copy the new files to `./models/`, restart the container. No image rebuild.

---

## Model path resolution

| Config key | Registry entry | Resolved to | Received by engine |
|---|---|---|---|
| `stt.model = "moonshine-tiny"` | `path = "moonshine/tiny"` | directory | STT engine |
| `tts.model = "paradee-int8"` | `path = "paradee"` | directory | TTS engine |
| `llm.model = "gemma3-1b-q4"` | `path = "llm"` + `files[0]` | single file | LLM engine |
| `vad.model = "silero-vad"` | `path = "silero"` + `files[0]` | single file | VAD engine |

**STT and TTS receive a directory.** ONNX-based engines load multiple artefacts internally.  
**LLM and VAD receive a specific file.** Each has a single primary binary.

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
         ├─ "moonshine" → Box<dyn SttEngine>  (MoonshineEngine)
         ├─ "llamacpp"  → Box<dyn LlmEngine>  (LlamaCppEngine)
         └─ "paradee"   → Box<dyn TtsEngine>  (ParadeeEngine)
                 │
                 ▼ Arc-wrapped, shared across sessions
         Session { stt, llm, tts }
```

Adding a new engine: implement the trait, register the provider string in the factory. The session loop is unchanged.

---

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) - full prose explanation of every decision shown here
- [FLOW.md](FLOW.md) - runtime behaviour, edge cases, what is not in v1
- [PROTOCOL.md](PROTOCOL.md) - wire protocol between server and any client or SDK
