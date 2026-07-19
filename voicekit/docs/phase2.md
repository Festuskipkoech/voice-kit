# Phase 2 — Kokoro TTS, OpenAI LLM, Observability

## Purpose

Complete technical implementation guide for Phase 2. Assumes Phase 1
is fully complete — all 29 integration tests passing, real voice turns
working end to end, Redis, Silero VAD, Whisper, Chatterbox, and Claude
all in place.

Phase 2 proves the swappability story with real alternatives and lays
the observability groundwork for Phase 3.

---

## What Phase 1 already provides

Before starting Phase 2, confirm these all work:

- Silero VAD — neural VAD, accurate in real environments
- WhisperSTT — real transcription, silence correctly returns empty
- ChatterboxTTS — real synthesis, first audio before tokens finish
- ClaudeProvider — real LLM, executor pattern, streaming
- Redis — session state and conversation history
- Production gateway — ping/pong, three timers, finally cleanup
- All 29 integration tests passing

Phase 2 adds to this foundation without changing it.

---

## Step 1 — Kokoro TTS

### Why

Chatterbox Turbo runs slower than real time on CPU-only servers. A user
without a GPU cannot use it in production. Kokoro solves this.

Kokoro is an 82M parameter model:
- Generates audio faster than real time on a modern CPU core
- Fits in under 2GB RAM
- No GPU required
- 54 built-in voices across 8 languages
- Apache 2.0 license

This is the swappability story in action. The developer changes one line
in `voice.config.yaml` — `model: chatterbox-turbo` to `model: kokoro`
— and gets a CPU-friendly alternative. No code changes. The test suite
validates the same behaviour regardless of which TTS is active.

### Add to pyproject.toml

```toml
"kokoro>=0.9.0",
"soundfile>=0.12.0",
```

### Verify voice codes

Voice codes change between Kokoro versions. Always confirm against the
installed version before using:

```python
from kokoro import KPipeline
p = KPipeline(lang_code="a")
print(list(p.voices.keys()))
```

Common voices (verify these exist in your version):
```
American female:  af_heart, af_bella, af_sarah, af_nicole
American male:    am_adam, am_michael
British female:   bf_emma, bf_isabella
British male:     bm_george, bm_lewis
```

### File to create — `voicekit/providers/tts/kokoro.py`

```python
# voicekit/providers/tts/kokoro.py

import asyncio
import io
from typing import AsyncIterator

from voicekit.providers.base import TTSProvider

SAMPLE_RATE = 24000
SENTENCE_BOUNDARIES = (".", "!", "?", ",")


class KokoroTTS(TTSProvider):
    """
    Kokoro TTS — 82M params, Apache 2.0, fast on CPU.
    No GPU required. Good quality. 54 voices, 8 languages.
    CPU-friendly alternative to Chatterbox Turbo.
    """

    def __init__(self, voice: str = "af_heart"):
        self.voice = voice
        self.pipeline = None

    async def load(self) -> None:
        from kokoro import KPipeline

        loop = asyncio.get_event_loop()
        self.pipeline = await loop.run_in_executor(
            None,
            lambda: KPipeline(lang_code="a")   # "a" = American English
        )

    async def synthesize(
        self,
        text_stream: AsyncIterator[str]
    ) -> AsyncIterator[bytes]:
        buffer = ""

        async for token in text_stream:
            buffer += token
            if buffer.rstrip().endswith(SENTENCE_BOUNDARIES):
                sentence = buffer.strip()
                if sentence:
                    async for chunk in self._synthesize_sentence(sentence):
                        yield chunk
                buffer = ""

        if buffer.strip():
            async for chunk in self._synthesize_sentence(buffer.strip()):
                yield chunk

    async def _synthesize_sentence(
        self, text: str
    ) -> AsyncIterator[bytes]:
        import soundfile as sf

        loop = asyncio.get_event_loop()

        results = await loop.run_in_executor(
            None,
            lambda: list(
                self.pipeline(text, voice=self.voice, speed=1.0)
            )
        )

        for _, _, audio in results:
            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="WAV")
            yield buf.getvalue()
            await asyncio.sleep(0)

    async def health(self) -> bool:
        return self.pipeline is not None
```

### Register Kokoro

```python
# voicekit/providers/registry.py
from voicekit.providers.tts.kokoro import KokoroTTS

TTS_REGISTRY = {
    "simulated": SimulatedTTS,
    "chatterbox-turbo": ChatterboxTTS,
    "kokoro": KokoroTTS,               # add
}
```

### Add to runtime/tts/Dockerfile

```dockerfile
RUN pip install --no-cache-dir \
    fastapi uvicorn websockets numpy pyyaml \
    chatterbox-tts torchaudio \
    kokoro soundfile
```

### Use in config

```yaml
tts:
  model: kokoro
  voice: af_heart
```

### Verify

Switch config to Kokoro, rebuild TTS container, run all 15 TTS
integration tests. All must pass. The test suite is model-agnostic —
it tests streaming behaviour, WAV format, latency, and concurrency,
none of which depend on which specific TTS model is active.

---

## Step 2 — OpenAI LLM Provider

### Why

This confirms the LLM swappability story with a real alternative.
If swapping from Claude to GPT requires code changes, the registry
pattern has failed. This step proves it has not.

### Add to pyproject.toml

```toml
"openai>=1.40.0",
```

### File to create — `voicekit/providers/llm/openai_provider.py`

Name the file `openai_provider.py` not `openai.py`. Naming it `openai.py`
shadows the `openai` package and causes import errors throughout the
codebase.

```python
# voicekit/providers/llm/openai_provider.py

from typing import AsyncIterator
from voicekit.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    LLM provider via OpenAI API.
    Reads OPENAI_API_KEY from environment automatically.

    OpenAI SDK is async-native — no executor pattern needed.
    This is the key difference from ClaudeProvider.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI()

    async def stream(
        self,
        messages: list[dict],
        system: str,
        max_tokens: int = 300
    ) -> AsyncIterator[str]:
        # OpenAI uses system as first message, not separate parameter
        openai_messages = [
            {"role": "system", "content": system}
        ] + messages

        # async-native — stream directly, no executor needed
        async with self.client.chat.completions.stream(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
```

### Register OpenAI

```python
# voicekit/providers/registry.py
from voicekit.providers.llm.openai_provider import OpenAIProvider

LLM_REGISTRY = {
    "simulated": SimulatedLLM,
    "anthropic": ClaudeProvider,
    "openai": OpenAIProvider,        # add
}
```

### API key routing

The gateway's `_configure_llm_key()` from Phase 1 already handles this:

```python
elif config.llm.provider == "openai":
    os.environ["OPENAI_API_KEY"] = key
```

No gateway changes needed.

### Use in config

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
```

### Verify

Switch config to OpenAI, restart gateway, run a real voice turn. The
pipeline test suite is LLM-agnostic — it tests token streaming behaviour,
not which LLM produces the tokens.

---

## Step 3 — Structured observability logging

Phase 3 will add full Prometheus metrics and a Grafana dashboard. Phase 2
lays the groundwork — structured JSON logging on every session event so
Phase 3 has clean data to consume.

The gateway already logs turn metrics:
```python
log.info(f"[{session_id}] Turn complete — {metrics}")
```

Replace this with structured JSON logging throughout the gateway. Every
session event gets a machine-parseable log line:

```python
import json

# session opened
log.info(json.dumps({
    "event": "session_open",
    "session_id": session_id,
    "stt_model": config.stt.model,
    "tts_model": config.tts.model,
    "llm_provider": config.llm.provider,
    "ts": time.time(),
}))

# turn complete
log.info(json.dumps({
    "event": "turn_complete",
    "session_id": session_id,
    "stt_ms": round(metrics.stt_ms),
    "llm_first_token_ms": round(metrics.llm_first_token_ms),
    "tts_first_chunk_ms": round(metrics.tts_first_chunk_ms),
    "total_ms": round(metrics.total_ms),
    "transcript_length": len(metrics.transcript),
    "ts": time.time(),
}))

# session closed
log.info(json.dumps({
    "event": "session_close",
    "session_id": session_id,
    "reason": reason,            # "disconnect", "idle_timeout", "error"
    "turn_count": turn_count,
    "duration_s": round(time.time() - session_start),
    "ts": time.time(),
}))
```

This structured logging is the raw material Phase 3 consumes. A log
aggregator (Loki, CloudWatch, Datadog) can parse these JSON lines and
feed them into Prometheus counters and histograms.

---

## Phase 2 complete when

- Switching `tts.model: chatterbox-turbo` to `tts.model: kokoro`
  works without any code changes
- Switching `llm.provider: anthropic` to `llm.provider: openai`
  works without any code changes
- All 29 integration tests still pass with each combination
- Structured JSON logging is in every session lifecycle event

### Verification matrix

| STT | TTS | LLM | Status |
|---|---|---|---|
| whisper | chatterbox-turbo | anthropic | Phase 1 baseline |
| whisper | kokoro | anthropic | Phase 2 — verify |
| whisper | chatterbox-turbo | openai | Phase 2 — verify |
| whisper | kokoro | openai | Phase 2 — verify |

Run the full integration test suite for each combination. All 29 must
pass for all four combinations.

---

## Common problems

**Kokoro voice code not recognised**
Voice codes change between versions. Run
`list(KPipeline(lang_code="a").voices.keys())` against the installed
version to get the actual available voices.

**`import openai` fails after adding `openai_provider.py`**
A file named `openai.py` somewhere in the project is shadowing the
package. The provider file must be named `openai_provider.py`.

**OpenAI streaming `AttributeError`**
Ensure `openai>=1.40.0` is installed. The async streaming API changed
significantly in 1.0. Versions below 1.0 use a completely different
interface.

**Kokoro audio sounds robotic**
The sentence boundary buffering is not working — Kokoro is receiving
one word at a time. Check that the `synthesize()` method accumulates
tokens to sentence boundaries before calling `_synthesize_sentence()`.

**Structured logs not appearing as JSON**
The `log.info(json.dumps({...}))` pattern logs the JSON as a string
inside the log formatter's output. Configure the logger to output raw
JSON by replacing the formatter:

```python
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        try:
            return record.getMessage()    # already JSON
        except Exception:
            return super().format(record)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.getLogger().handlers = [handler]
```