# Session Flow

Read this before touching the session loop in `voicekit-core`. [ARCHITECTURE.md](ARCHITECTURE.md) covers structure; this document covers runtime behaviour.

---

## Overview

One WebSocket connection = one Tokio task = one session. Sessions are independent. The server spawns a task on accept and drops it on disconnect - no shared mutable state between sessions.

The session task owns: one VAD instance, one conversation history, and borrowed `Arc` references to the shared STT, LLM, and TTS engines.

```
client connects
  │
  ▼
tokio::spawn(session(socket, engines, config))
  │
  ├─► VAD loop        (runs on every incoming audio chunk)
  │      │
  │      └─► STT      (fires once per utterance)
  │             │
  │             └─► LLM        (streams tokens)
  │                    │
  │                    └─► Sentence buffer  (accumulates tokens)
  │                               │
  │                               └─► text_clean → TTS  (fires per sentence)
  │                                                 │
  │                                                 └─► write audio back to socket
  │
client disconnects → task exits
```

---

## Step by step

### 1. Client connects

The client opens a WebSocket connection and immediately starts the handshake. Expected audio format once the session is active: 16 kHz, mono, f32-LE, 512 samples per frame (32 ms). The server sends no audio until the client sends a `session.config` frame and receives `session.ready` back. See [PROTOCOL.md](PROTOCOL.md) for the full handshake spec.

### 2. VAD loop - every frame

Each incoming binary frame goes through Silero VAD:

- **Score ≥ threshold** - speech; append the frame to the utterance buffer and reset the silence counter.
- **Score < threshold** - silence; increment the counter. When the counter reaches 25 consecutive sub-threshold chunks (800 ms at 32 ms per chunk) and the utterance buffer is non-empty, dispatch the buffer to STT and clear it.

> **Edge case - VAD false trigger:** a short noise spike that scores above threshold for one or two chunks resets the silence counter. A true end-of-utterance still requires 25 consecutive silence chunks after the last real speech frame. STT on a very short non-speech buffer produces an empty or near-empty transcript, which the session drops silently.

> **Edge case - continuous speech beyond 30 s:** no hard cap by default. The buffer grows. If a cap is needed, add `max_utterance_ms` to the VAD config and dispatch mid-utterance when hit.

### 3. STT dispatch

The complete utterance buffer (f32 PCM of one utterance) is passed to the STT engine synchronously. This call blocks the session task for approximately 376 ms with Moonshine TINY. That is acceptable because the protocol is turn-based - the client is not expected to send new audio while the agent is processing.

> **Barge-in** (interrupting mid-response) requires a parallel VAD task and an interrupt signal. It is out of scope for v1.

If the transcript is empty or whitespace-only after trimming, the session discards it and returns to the VAD loop.

### 4. LLM generation - token streaming

The transcript is appended to the conversation history as a user turn. The full prompt (system prompt + history) is passed to the LLM engine, which runs in a background blocking task and streams tokens into an mpsc channel. The session loop reads tokens from the channel concurrently and passes each one to the sentence buffer.

### 5. Sentence buffer - when to flush to TTS

After each token the buffer checks two conditions:

- **Condition A - sentence boundary (primary):** `sentencex` detects a Unicode sentence boundary. If one exists, everything up to and including the last boundary is flushed as a single TTS call. The remainder stays in the buffer.
- **Condition B - 200-character hard cap (safety net):** if the buffer reaches 200 characters with no detected boundary, the whole buffer is flushed. This handles LLMs that generate run-on text. The TTS call may cover a grammatically incomplete sentence; this is a known trade-off.

After generation ends, any remaining text in the buffer is flushed as a final TTS call, even if it contains no sentence boundary.

### 6. Text cleaning and TTS - per sentence

Before each sentence reaches the TTS engine it passes through `text_clean`. This strips markdown formatting, removes awkwardly spoken tokens (asterisks, bracket notation, bare URLs), and normalises whitespace. A sentence that is empty after cleaning is silently dropped - no TTS call is made.

The cleaned sentence is passed to the TTS engine synchronously, which returns 24 kHz f32 mono PCM. The session immediately writes the PCM as a binary WebSocket frame back to the client.

Because TTS is called per sentence and each sentence is short, the client receives the first audio chunk approximately **1.1 s** after the user finishes speaking - before the LLM finishes generating the full response.

### 7. Client disconnect

The WebSocket read returns an error or a Close frame. The session task returns, dropping all held resources. Engines are behind `Arc` and are not dropped - just the session's references to them.

> **Edge case - disconnect mid-utterance:** the socket read fails. The utterance buffer is discarded. Any in-flight LLM generation is cancelled via a `CancellationToken`. The LLM task may still attempt to send on the channel after cancellation; this returns a `SendError` (receiver dropped) and must not panic.

> **Edge case - disconnect mid-TTS:** the socket write returns an error. The session propagates the error and exits, cancelling any in-flight LLM generation.

### 8. Conversation history

The session maintains a list of turns (role + content) for the full conversation. Each STT result appends a user turn. Each completed LLM generation (all sentences joined) appends an assistant turn. History is **in-memory and scoped to the session** - it is not persisted. When the WebSocket closes, history is gone.

---

## What is not in v1

- **Barge-in** - client interrupts agent mid-response
- **Persistent history** - conversation survives reconnect
- **Multi-language** - Moonshine TINY is English only
- **GPU inference** - CPU only in the default build
- **Authentication** - no token or API key on the WebSocket

---

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md) - package structure, traits, config schema
- [DIAGRAM.md](DIAGRAM.md) - visual pipeline reference
- [PROTOCOL.md](PROTOCOL.md) - wire protocol spec
