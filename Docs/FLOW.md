# Session Flow

> Read this before touching the session loop in `voicekit-core`. ARCHITECTURE.md covers structure; this document covers runtime behaviour.

---

## Overview

One WebSocket connection = one Tokio task = one session. Sessions are independent. The server spawns a task on accept and drops it on disconnect — no shared mutable state between sessions.

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
  │                               └─► TTS  (fires per sentence)
  │                                      │
  │                                      └─► write audio back to socket
  │
client disconnects → task exits
```

---

## Step by step

### 1. Client connects

The client opens a WebSocket connection and immediately starts sending binary frames of raw PCM audio.

**Expected format:** 16 kHz, mono, f32-LE, 512 samples per frame (32 ms).

The server sends no handshake. If the client sends a non-binary frame on connection, the session logs a warning and waits for binary.

---

### 2. VAD loop — every frame

Each incoming frame goes through Silero VAD synchronously.

```
frame arrives (512 samples, f32)
  │
  ▼
vad.score(frame)  → f32 in [0.0, 1.0]
  │
  ├─ score ≥ 0.5 (speech)
  │    └─► append frame to utterance_buffer
  │        reset silence_chunks counter
  │
  └─ score < 0.5 (silence)
       └─► increment silence_chunks
            │
            ├─ silence_chunks < threshold (800 ms / 32 ms = 25 chunks)
            │    └─► continue; do not dispatch yet
            │
            └─ silence_chunks ≥ 25
                 └─► if utterance_buffer is non-empty:
                       dispatch utterance_buffer to STT
                       clear utterance_buffer
                       reset silence_chunks
```

**Edge case — VAD false trigger:** A short noise spike scores ≥ 0.5 for one or two chunks, then falls back to silence. Because the silence counter resets only when speech is detected and counts back up from zero, a true end-of-utterance still requires 25 consecutive silence chunks after the last real speech frame. The buffer accumulates the noise, but STT on a very short non-speech buffer produces an empty or near-empty transcript, which the session drops before dispatching to the LLM.

**Edge case — continuous speech beyond 30 s:** No hard cap in the default config. The buffer grows. If you need a cap, add `max_utterance_ms` to the VAD config and dispatch+clear when hit, even mid-utterance. The LLM will receive a truncated transcript.

---

### 3. STT dispatch

The utterance buffer (complete f32 PCM of one utterance) is passed to `SttEngine::transcribe`.

```
transcript = stt.transcribe(&utterance_buffer)?
```

This call is synchronous and blocks the session task for ~376 ms (Moonshine TINY). That is acceptable because the client is not expected to send new audio while the agent is processing — it is a turn-based conversation. If you need barge-in (client can speak while agent is responding), the architecture needs a parallel VAD task and an interrupt signal; that is out of scope for v1.

If the transcript is empty or whitespace-only, the session discards it and returns to the VAD loop. Nothing is sent to the LLM.

---

### 4. LLM generation — token streaming

The transcript becomes the user turn in the conversation history. The session calls `LlmEngine::generate` with the full prompt (system prompt + history + new user turn) and a channel sender.

```rust
let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
tokio::spawn(async move { llm.generate(&prompt, tx) });

while let Some(token) = rx.recv().await {
    sentence_buffer.push_str(&token);
    try_flush(&sentence_buffer, &tts, &socket).await?;
}
// generation complete; flush any remaining text
force_flush(&sentence_buffer, &tts, &socket).await?;
```

The LLM task runs concurrently. The session loop reads tokens from `rx` and immediately passes them to the sentence buffer logic.

---

### 5. Sentence buffer — when to flush to TTS

The sentence buffer accumulates tokens. After each token, it checks for a flush condition.

**Condition A — sentence boundary (primary):**
`sentencex::get_sentence_boundaries(&buffer)` returns byte offsets of detected sentence ends. If any boundary exists, flush everything up to and including the last boundary as one TTS call. Keep the remainder in the buffer.

**Condition B — 200-character hard cap (safety net):**
If the buffer reaches 200 characters with no detected boundary, flush the whole buffer as a single TTS call. This handles LLMs that generate run-on text. The TTS call may cover a grammatically incomplete sentence; this is a known trade-off.

**What sentencex handles correctly:**
- `Dr. Smith said` — does not trigger a flush on the dot after "Dr"
- `i.e. the following` — does not trigger on the dot after "e"
- `Really? Yes, I think so.` — two boundaries, two TTS calls
- Trailing incomplete sentence — held in buffer until more tokens arrive or generation ends

**After generation ends:** any remaining text in the buffer is flushed as one final TTS call, even if it contains no sentence boundary.

---

### 6. TTS — per sentence

Each flush calls `TtsEngine::synthesize(sentence)` synchronously. This returns a `Vec<f32>` of 24 kHz mono PCM.

The session immediately writes the PCM as a binary WebSocket frame back to the client.

```
sentence ready
  │
  ▼
audio = tts.synthesize(&sentence)?   // ~708 ms for a typical sentence
  │
  ▼
socket.send(Message::Binary(pcm_to_bytes(&audio))).await?
```

Because TTS is called per sentence and each sentence is short, the client receives the first audio chunk ~1.1 s after the user finishes speaking — before the LLM finishes generating the full response.

---

### 7. Client disconnect

The WebSocket read returns an error or a Close frame. The session task returns, dropping all held resources (engines are behind `Arc` so they are not dropped — just the session's references to them).

**Edge case — disconnect mid-utterance:** The VAD loop is reading from the socket. If the socket closes, the read returns an error. The utterance buffer is discarded. Any in-flight LLM generation should be cancelled; use a `CancellationToken` passed into the `generate` call. If the LLM task is not cancelled, it will complete and attempt to send on the channel, which will return `SendError` (receiver dropped) — this must not panic; handle it with `let _ = tx.send(...)`.

**Edge case — disconnect mid-TTS:** The socket write in step 6 returns an error. The session should propagate this error and exit, cancelling any in-flight LLM generation.

---

### 8. Conversation history

The session maintains a `Vec<Message>` of `{role: user|assistant, content: String}` for the full conversation. Each STT result appends a user turn. Each completed LLM generation (all sentences joined) appends an assistant turn.

History is in-memory and scoped to the session. It is not persisted. When the WebSocket closes, history is gone.

---

## What is not in v1

- **Barge-in** — client interrupts agent mid-response
- **Persistent history** — conversation survives reconnect
- **Multi-language** — Moonshine TINY is English only
- **GPU inference** — CPU only in the default build
- **Authentication** — no token or API key on the WebSocket

---

## Related

- [ARCHITECTURE.md](Architecture.md) — package structure, traits, config schema
- [DIAGRAM.md](DIAGRAM.md) — visual pipeline reference
