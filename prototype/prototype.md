# Prototype — Reference Documentation

The prototype lives at `voicekit/prototype/`. It is complete. Do not
modify it. It is reference material — the source of truth for every
interface, pattern, and convention the real package uses.

---

## Why the prototype exists

The prototype proved the architecture works before any real model was
involved. Every interface, the streaming cascade, session isolation, the
CLI, the WebSocket protocol, and the test infrastructure were validated
against simulated providers.

Debugging architecture with real models is slow — model bugs and
pipeline bugs look identical. The prototype separated those concerns.
When Phase 1 hits a problem, you know immediately whether it is a
pipeline problem (would have appeared in the prototype) or a model
problem (new in Phase 1).

---

## What was built

### Python package — `prototype/voicekit/`

**`providers/base.py`**
The three abstract interfaces every model must implement. These are
copied verbatim into the real project. Any confusion about what a
provider must implement — start here.

**`providers/registry.py`**
Maps config strings to provider classes. The entire model swapping
mechanism. One import, one dict entry per model. Nothing else changes.

**`providers/stt/simulated.py`**
Returns predictable transcripts based on audio duration. Short audio
returns "hello". Medium returns "hello how are you doing today". Long
returns a longer fixed string. Predictable output made pipeline tests
deterministic.

**`providers/tts/simulated.py`**
Generates real sine wave WAV audio (440Hz tone, 24kHz). Not silence —
actual playable bytes. Buffers tokens to sentence boundaries, same as
the real TTS will. Proved the sentence buffering logic works before
Chatterbox was involved.

**`providers/llm/simulated.py`**
Cycles through five canned responses. Streams tokens one word at a time
with configurable inter-token delay. Realistic enough to exercise the
TTS sentence-buffering logic under actual streaming conditions.

**`vad.py`**
Energy-based VAD. Interface is `is_speech(chunk) → bool`. Replaced by
Silero VAD in Phase 1. The interface is identical — callers see no
difference when Silero replaces it.

**`config.py`**
Loads and validates `voice.config.yaml`. Resolves `${VAR_NAME}` from
environment. Returns typed dataclasses. Clear errors for invalid config.

**`pipeline.py`**
The streaming cascade. VAD → STT → LLM → TTS, all streaming.
`asyncio.gather(feed_llm(), feed_tts())` is the key line — LLM and
TTS run concurrently. `PipelineMetrics` captures per-stage timing on
every turn.

**`cli.py`**
Local development CLI. Six commands with Typer: `init`, `setup`, `dev`,
`stop`, `status`, `models`. The `dev` command reads config and starts
Docker Compose with the right environment variables.

### Runtime services — `prototype/runtime/`

Local development Docker services. These are **not** part of the
importable package. They exist for developers who want to run STT, TTS,
and a gateway as separate services during development.

**`stt/server.py`** — FastAPI WebSocket server wrapping STT provider
**`tts/server.py`** — FastAPI WebSocket server wrapping TTS provider
**`gateway/server.py`** — session management, pipeline orchestration
**`docker-compose.yml`** — dev stack, ports 8000/8001/8002
**`docker-compose.test.yml`** — test stack, ports 8011/8012

All three servers use FastAPI lifespan (not deprecated `on_event`).

### Tests — `prototype/tests/`

**`unit/test_config.py`** — 7 tests
Config loading, validation, env var resolution, error handling.

**`unit/test_vad.py`** — 8 tests
Silence rejection, speech detection, sensitivity thresholds,
filter_stream batch processing.

**`pipeline/test_pipeline.py`** — 10 tests
Full streaming cascade, silence produces no output, history
accumulation, role assignment, metrics, concurrent session isolation,
and the critical test: first audio arrives before pipeline completes.

**`integration/test_stt.py`** — 14 tests
Health endpoint, transcription, silence handling, token streaming,
latency budget, concurrent sessions, protocol correctness.

**`integration/test_tts.py`** — 15 tests
Health endpoint, audio output, WAV validation, sample rate, progressive
chunking, latency, concurrent sessions, protocol correctness.

---

## Test results

```
Unit:           15/15 passing
Pipeline:       10/10 passing
Integration:    27/29 passing

Intentionally skipped (2):

test_silence_produces_empty_transcript (test_stt.py)
  Simulated STT returns text based on duration not content.
  Real Whisper correctly returns nothing for silence.
  Must pass in Phase 1.

test_first_audio_arrives_before_synthesis_completes (test_tts.py)
  Simulated TTS too slow to beat the token send time.
  Real Chatterbox Turbo starts audio within 75ms of first token.
  Must pass in Phase 1.
```

---

## Interfaces — source of truth

Copied from `prototype/voicekit/providers/base.py`. Every provider in
the real project must implement all methods exactly. Missing any method
raises `TypeError: Can't instantiate abstract class` at runtime.

```python
class STTProvider(ABC):

    async def load(self) -> None:
        # Load model into memory once at startup.
        # Blocks until ready. After return, health() returns True.
        ...

    async def transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        # Accept float32 PCM chunks at 16kHz mono.
        # Yield transcript tokens as recognised.
        # Must not wait for all audio before yielding.
        ...

    async def health(self) -> bool:
        # True if model loaded and ready.
        ...


class TTSProvider(ABC):

    async def load(self) -> None: ...

    async def synthesize(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        # Accept text tokens from LLM stream.
        # Yield WAV audio bytes as generated.
        # First audio chunk must arrive before text stream ends.
        ...

    async def health(self) -> bool: ...


class LLMProvider(ABC):

    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 300
    ) -> AsyncIterator[str]:
        # Accept conversation history and system prompt.
        # Yield response tokens as generated.
        # Never buffer the full response.
        ...
```

---

## How to run the prototype

```bash
cd voicekit/prototype
uv sync
uv tool install --editable .

# unit and pipeline — no Docker needed, runs in seconds
uv run pytest tests/unit/ tests/pipeline/ -v

# build and start test containers
docker compose -f runtime/docker-compose.test.yml build
docker compose -f runtime/docker-compose.test.yml up -d

# integration tests against live containers
uv run pytest tests/integration/ -v

# stop test containers
docker compose -f runtime/docker-compose.test.yml down

# try the CLI
voicekit --help
voicekit models
voicekit init test-project
cd test-project && voicekit dev
```

---

## What the prototype proved

**Provider pattern works.**
Config `stt.model: simulated` instantiates `SimulatedSTT`. Changing to
`stt.model: whisper` will instantiate `WhisperSTT`. The pipeline,
gateway, and tests are unaware of the change.

**Streaming cascade works.**
`test_first_audio_arrives_before_pipeline_completes` confirms audio
starts before the full pipeline turn completes — LLM and TTS are
genuinely concurrent, not sequential.

**Session isolation works.**
`test_concurrent_sessions_are_isolated` runs three pipeline instances
simultaneously and confirms each has independent history, independent
turn count, and independent state.

**CLI works.**
`voicekit dev` reads config, sets environment variables, and starts
Docker Compose. `voicekit init` creates real project files.

**WebSocket protocol works.**
Integration tests confirm binary/text message contract works for both
STT and TTS services, including done signals, concurrent sessions, and
protocol edge cases.