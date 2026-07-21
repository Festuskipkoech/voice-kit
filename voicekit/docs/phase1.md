# Phase 1 — Real Models, Silero VAD, Redis, Production Gateway

## Status: COMPLETE

Branch: phase-1 → merged to main
All 29 tests passing. End-to-end voice turn confirmed working.
Speech in → Whisper → Claude → Chatterbox/Kokoro → audio out.

---

## Purpose

Complete technical implementation guide for Phase 1. Documents the
actual implementation as built — including architectural decisions made
during development that differ from the original plan.

Read alongside:
- `FLOW.md` — architectural principles and key decisions
- `prototype/PROTOTYPE.md` — interfaces and patterns
- `prototype/voicekit/providers/base.py` — exact method signatures

---

## What was built

Seven components in implementation order:

1. Redis — session state and conversation history
2. Silero VAD — neural voice activity detection
3. WhisperSTT — real speech-to-text via faster-whisper
4. ClaudeProvider — Anthropic SDK with executor+queue streaming
5. ChatterboxTTS — Resemble AI TTS with sentence buffering
6. KokoroTTS — 82M parameter TTS, true streaming on CPU
7. Production gateway — RemotePipeline, AUDIO_DONE sentinel,
   turn_in_progress, concurrent audio sender, three timers

---

## The key architectural decision — RemotePipeline

**Original plan:** gateway loads Whisper and Chatterbox directly.

**What actually happened:** loading both models in the gateway
duplicates what the STT and TTS services already do. On a machine
with 8GB RAM this causes OOM. Gateway startup takes 90+ seconds.
Container restart means 90 seconds of downtime.

**Decision:** gateway uses `RemotePipeline` — calls STT and TTS
services over WebSocket. No models in the gateway. Gateway starts
in seconds. Models load once in their dedicated services.

This is documented in `FLOW.md` under "The five architectural
principles". It is the single most important architectural decision
in Phase 1.

---

## Step 1 — Redis

Add to `runtime/docker-compose.yml`:

```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  restart: unless-stopped
  mem_limit: 128m
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 5s
    timeout: 3s
    retries: 5
    start_period: 5s
```

Add to gateway `depends_on`:

```yaml
gateway:
  depends_on:
    redis:
      condition: service_healthy
    stt:
      condition: service_healthy
    tts:
      condition: service_healthy
```

Add to `pyproject.toml` dependencies:

```toml
"redis>=5.0.0",
```

Redis session operations live in `runtime/gateway/services/session.py`.
All Redis calls are in one file — register, update, complete, cleanup,
count. The gateway never calls Redis directly in route handlers.

**Verify:**
```bash
docker compose -f runtime/docker-compose.yml up redis
redis-cli ping  # → PONG
```

---

## Step 2 — Silero VAD

**File:** `voicekit/vad.py`

**Install:**
```bash
uv add silero-vad
```

**Critical implementation details:**

Silero requires exactly 512 samples per inference call at 16kHz.
Audio arrives in 1600-sample chunks. Split into 512-sample windows
and feed sequentially — state accumulates across windows within a
chunk. This is correct streaming behaviour.

```python
WINDOW_SIZE = 512

for i in range(0, len(audio_chunk), WINDOW_SIZE):
    window = audio_chunk[i:i + WINDOW_SIZE]
    if len(window) < WINDOW_SIZE:
        window = np.pad(window, (0, WINDOW_SIZE - len(window)))
    tensor = torch.from_numpy(window.astype(np.float32))
    with torch.no_grad():
        confidence = self._model(tensor, 16000).item()
    if confidence >= self._threshold:
        return True
```

`reset_states()` is called after each complete utterance — NOT between
windows within a chunk. Calling it between windows destroys the RNN
context needed for accurate detection.

In `filter_stream()`, reset after each chunk decision:

```python
def filter_stream(self, chunks):
    result = []
    for chunk in chunks:
        if self.is_speech(chunk):
            result.append(chunk)
        self._model.reset_states()   # AFTER decision, not before
    return result
```

**Test audio for Silero:** random noise (`np.random.uniform`) is
correctly rejected as non-speech. Tests must use harmonic sine waves
(120Hz fundamental + harmonics to 3400Hz) to generate speech-like
signals Silero detects. Use 4800-sample chunks (300ms) for reliable
detection in tests.

**Verify:**
```bash
uv run pytest tests/unit/test_vad.py -v
# 10/10 passing
```

---

## Step 3 — WhisperSTT

**File:** `voicekit/providers/stt/whisper.py`

**Install:**
```bash
uv add faster-whisper
```

**Critical implementation details:**

**1. Lazy generator fix.** `faster_whisper.transcribe()` returns a
lazy generator. Iterating it outside `run_in_executor` runs inference
on the event loop, blocking all other coroutines. Force evaluation
inside the executor:

```python
segments = await loop.run_in_executor(
    None,
    lambda: list(self.model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )[0])   # [0] = segments, [1] = info — force list inside executor
)
```

**2. Hallucination suppression.** Whisper hallucinates on silence
("Thank you.", "you", "Bye."). Suppress with:
- `no_speech_threshold=0.6` — rejects low-confidence segments
- `log_prob_threshold=-1.0` — filters low probability outputs
- `vad_filter=True` — faster-whisper's internal VAD

**3. Lazy imports.** Import `faster_whisper` inside `load()` not at
module level. This allows the package to be imported without
faster-whisper installed (for gateway which uses RemotePipeline).

**Verify:**
```bash
uv run pytest tests/unit/ tests/pipeline/test_pipeline.py::test_silence_produces_no_transcript -v
```

---

## Step 4 — ClaudeProvider

**File:** `voicekit/providers/llm/claude.py`

**Install:**
```bash
uv add "anthropic>=0.40.0"
```

**Critical implementation details:**

The Anthropic SDK is synchronous. Streaming in an async context
requires the executor + asyncio.Queue bridge pattern:

```python
async def stream(self, messages, system):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def _stream_sync():
        with self.client.messages.stream(
            model=self.model,
            max_tokens=300,
            temperature=0.3,
            system=system,
            messages=messages,
        ) as stream:
            for token in stream.text_stream:
                loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, None)

    executor_task = loop.run_in_executor(None, _stream_sync)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token

    await executor_task
```

`call_soon_threadsafe` is the correct way to put items into an
asyncio.Queue from a thread. The sentinel `None` is always sent in
the thread's `finally` block to prevent the consumer from hanging.

**Temperature:** set to 0.3. Default (1.0) causes creative formatting
— emojis, asterisks, bullet points — that corrupt TTS output.

**Verify:**
```bash
ANTHROPIC_API_KEY=sk-ant-... uv run pytest tests/pipeline/test_pipeline.py -v
```

---

## Step 5 — ChatterboxTTS

**File:** `voicekit/providers/tts/chatterbox.py`

**Install:**
```bash
uv add "chatterbox-tts>=0.1.7"
```

**Critical implementation details:**

**1. perth DummyWatermarker patch.** `chatterbox-tts` depends on
`perth.PerthImplicitWatermarker` which fails to load with uv.
Patch before importing chatterbox:

```python
async def load(self) -> None:
    import perth

    class DummyWatermarker:
        def __init__(self, *args, **kwargs): pass
        def apply_watermark(self, wav, *args, **kwargs): return wav
        def __call__(self, *args, **kwargs): return args[0] if args else None

    perth.PerthImplicitWatermarker = DummyWatermarker
    perth.DummyWatermarker = DummyWatermarker

    from chatterbox.tts_turbo import ChatterboxTurboTTS
    ...
```

**2. Sentence boundary buffering.** Buffer tokens until `.!?,;:`
then synthesise. One token at a time produces flat robotic audio.
Full response before TTS kills latency. Sentence boundaries give
natural prosody with acceptable latency.

**3. CPU latency.** Chatterbox on CPU takes 90-150 seconds to first
audio chunk. This is hardware limitation, not a code issue. On GPU
it drops to 200-500ms. The `turn_in_progress` flag protects the
connection during synthesis. For CPU-only servers, use KokoroTTS.

**Audio output format:**
- Chunk 1: complete WAV file (header + float32 PCM)
- Chunks 2..N: raw float32 PCM bytes, no header
- Sample rate: 24000 Hz, mono

**Verify:**
```bash
uv run pytest tests/pipeline/test_pipeline.py::test_real_speech_produces_audio_output -v
```

---

## Step 6 — KokoroTTS

**File:** `voicekit/providers/tts/kokoro.py`

**Install:**
```bash
apt-get install -y espeak-ng    # system dependency — required
uv add "kokoro>=0.9.4"
```

**Why Kokoro over Chatterbox for CPU:**
- Chatterbox on CPU: 90-150 seconds to first audio
- Kokoro on CPU: 500-800ms to first audio
- Kokoro RTF on CPU: ~0.47 (generates 2x faster than real-time)
- True real-time streaming on CPU is achievable with Kokoro

**Tradeoff:** Kokoro does not support voice cloning. 54 fixed voices.
Chatterbox supports voice cloning from a reference audio clip.

**Streaming pattern:**
```python
SENTENCE_END = re.compile(r"[.!?,;:]")
MAX_BUFFER_TOKENS = 64

async for token in text_stream:
    buffer.append(token)
    if SENTENCE_END.search(token) or len(buffer) >= MAX_BUFFER_TOKENS:
        # synthesise immediately on sentence boundary
        async for chunk in synthesise_buffer("".join(buffer)):
            yield chunk
        buffer = []
```

**Use Kokoro by default for CPU servers:**
```yaml
# voice.config.yaml
tts:
  model: kokoro
  voice: af_bella
```

**Available voices (selected):**
```
af_bella    American female, warm
af_sarah    American female, clear
am_adam     American male, deep
am_michael  American male, natural
bf_emma     British female, formal
bm_george   British male, authoritative
```

---

## Step 7 — Production Gateway

**Files:**
```
runtime/gateway/
├── main.py              app creation, lifespan, Redis + pipeline init
├── config.py            reads env vars, builds VoiceConfig
├── dependencies.py      get_redis_ws(), get_pipeline_ws() via app.state
├── remote_pipeline.py   calls STT/TTS over WebSocket, no models loaded
├── routes/
│   ├── health.py        GET /health
│   └── session.py       WebSocket /session — full turn lifecycle
└── services/
    ├── ping.py          ping loop, respects turn_in_progress
    └── session.py       Redis session operations
```

### app.state pattern

Resources are stored on `app.state` during lifespan and injected via
`Depends()`. This is the FastAPI production standard.

```python
# in lifespan
app.state.redis = redis_client
app.state.pipeline = pipeline

# in dependencies.py
async def get_pipeline_ws(websocket: WebSocket):
    return websocket.app.state.pipeline
```

Never use module-level globals for shared resources. `app.state` is
explicitly scoped, testable by setting on a test app, and inspectable.

### AUDIO_DONE sentinel

`run_turn()` places `AUDIO_DONE = None` into `audio_out` in its
`finally` block. The concurrent `send_audio_stream()` in `session.py`
reads from `audio_out` and stops on the sentinel.

```python
# remote_pipeline.py
AUDIO_DONE = None

async def run_turn(self, audio_in, audio_out):
    ...
    try:
        await asyncio.gather(feed_llm(), feed_tts())
        ...
    finally:
        await audio_out.put(AUDIO_DONE)   # always runs
        self.turn_in_progress.clear()
```

```python
# session.py — runs concurrently with run_turn()
async def send_audio_stream():
    while True:
        chunk = await audio_out.get()
        if chunk is AUDIO_DONE:
            break
        await ws.send_bytes(chunk)
```

### turn_in_progress flag

`asyncio.Event` on `RemotePipeline`. Set at start of `run_turn()`,
cleared in `finally`. Ping loop checks before every ping:

```python
# services/ping.py
if turn_in_progress.is_set():
    continue   # skip ping — never close during active synthesis
```

### Concurrent execution

```python
# session.py
results = await asyncio.gather(
    pipeline.run_turn(audio_in, audio_out),
    send_audio_stream(),
    return_exceptions=True,
)
metrics = results[0]
```

Audio chunks flow to the client as TTS generates them. Text messages
(transcript, response, metrics) are sent after all audio completes.

### Three timers

| Timer | Value | Purpose |
|---|---|---|
| Ping interval | 30s | Send heartbeat when idle |
| Ping timeout | 10s | Close dead connections |
| Idle timeout | 30 min | Close abandoned sessions |
| Turn timeout | 5 min | Cancel hung pipeline turns |

Ping is skipped entirely during active turns via `turn_in_progress`.

### System prompt

```
You are a voice assistant. Your responses are converted to speech
and played as audio. The user cannot see text. Use plain spoken
English only. No emojis, no asterisks, no markdown, no bullet
points, no lists, no headers. Speak in short natural sentences
as if talking to someone on a phone call. Maximum 2 to 3 sentences
per response. Never use filler phrases.
```

Temperature: 0.3 — prevents creative formatting that corrupts TTS.

---

## Audio format reference

**Client → Gateway:**
```
format:      raw PCM, no WAV header
sample rate: 16000 Hz
channels:    1 (mono)
dtype:       float32
values:      -1.0 to 1.0
chunk size:  1600 samples = 100ms
```

**Gateway → Client:**
```
chunk 1:   complete WAV file (44-byte header + float32 PCM)
chunk 2..N: raw float32 PCM bytes, no header
sample rate: 24000 Hz
channels:   1 (mono)
dtype:      float32
```

Client reads chunk 1 with soundfile, chunks 2..N as:
```python
audio = np.frombuffer(chunk, dtype=np.float32)
```

---

## Environment variables

```
VOICEKIT_STT_MODEL        whisper
VOICEKIT_STT_VARIANT      small
VOICEKIT_TTS_MODEL        kokoro (or chatterbox-turbo)
VOICEKIT_TTS_VOICE        af_bella (or default for chatterbox)
VOICEKIT_LLM_PROVIDER     anthropic
VOICEKIT_LLM_MODEL        claude-haiku-4-5
VOICEKIT_LLM_API_KEY      sk-ant-... (routed to ANTHROPIC_API_KEY)
VOICEKIT_VAD_ENABLED      true
VOICEKIT_VAD_SENSITIVITY  0.5
REDIS_HOST                redis
REDIS_PORT                6379
STT_WS_URL                ws://stt:8001/stt
TTS_WS_URL                ws://tts:8002/tts
```

Store in `runtime/.env` — never commit this file.

---

## pyproject.toml

```toml
[project.optional-dependencies]
chatterbox = [
    "chatterbox-tts>=0.1.7",
    "resemble-perth>=1.0.1",
    "numba>=0.57.0",
    "llvmlite>=0.40.0",
    "setuptools",
]
kokoro = [
    "kokoro>=0.9.4",
]
dev = [
    "pytest",
    "pytest-asyncio",
    "anyio",
    "gtts",
]
```

Install:
```bash
uv sync --extra kokoro --extra dev
# or with chatterbox
uv sync --extra chatterbox --extra kokoro --extra dev
```

---

## Phase 1 complete checklist

- WhisperSTT transcribes real speech correctly
- Silence returns empty transcript (no hallucination)
- ClaudeProvider streams tokens via executor+queue pattern
- ChatterboxTTS or KokoroTTS synthesises audio
- Silero VAD filters silence — only speech reaches Whisper
- Redis stores session state and conversation history
- Gateway uses RemotePipeline — no models loaded in gateway
- AUDIO_DONE sentinel signals clean stream termination
- turn_in_progress prevents ping from killing active synthesis
- Concurrent audio sender streams chunks as TTS generates them
- Audio arrives at client before transcript and metrics
- All 29 tests passing
- End-to-end: `uv run python tests/test_voice.py` produces audible WAV