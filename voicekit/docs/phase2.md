# Phase 2 — OpenAI LLM, Observability Groundwork

## Status: IN PROGRESS

Kokoro TTS is complete and integrated. OpenAI LLM provider and
observability are remaining.

---

## What Phase 1 provides

Before starting Phase 2, confirm these all work:

- WhisperSTT — real transcription, silence returns empty
- ChatterboxTTS — voice cloning, slower on CPU
- KokoroTTS — 82M params, true streaming on CPU (COMPLETE)
- ClaudeProvider — Anthropic SDK, executor pattern, streaming
- Silero VAD — neural VAD, accurate in real environments
- Redis — session state and conversation history
- RemotePipeline — gateway calls STT/TTS over WebSocket
- AUDIO_DONE sentinel — clean stream termination
- turn_in_progress — ping respects active synthesis
- All 29 tests passing

Phase 2 adds to this foundation without changing it.

---

## Step 1 — KokoroTTS (COMPLETE)

Kokoro is an 82M parameter TTS model, Apache 2.0, runs at 2x
real-time on CPU. First audio arrives within 500-800ms of the
first sentence boundary — true streaming on CPU.

**File:** `voicekit/providers/tts/kokoro.py`

**Why it matters:**
Chatterbox on CPU takes 90-150 seconds to first audio. Kokoro on
CPU takes 500-800ms. Any CPU-only server can now achieve true
real-time streaming.

**Config:**
```yaml
tts:
  model: kokoro
  voice: af_bella
```

**Install:**
```bash
# system dependency
apt-get install -y espeak-ng

# Python package
uv add "kokoro>=0.9.4"
```

**Verify:**
```bash
VOICEKIT_TTS_MODEL=kokoro uv run python tests/test_voice.py
aplay tests/results/response.wav
```

First audio chunk should arrive within 3 seconds. Total turn under
30 seconds on CPU. Compare with Chatterbox: 90-150 seconds.

---

## Step 2 — OpenAI LLM Provider

**File:** `voicekit/providers/llm/openai_provider.py`

**Why:** proves the swappability story. Developer changes one line
in `voice.config.yaml` from `provider: anthropic` to `provider: openai`
and gets GPT-4o-mini. No other code changes.

**Pattern:** identical to ClaudeProvider — synchronous SDK requires
executor+queue bridge for async streaming.

```python
from openai import OpenAI

class OpenAIProvider(LLMProvider):

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI()

    async def stream(
        self,
        messages: list[dict],
        system: str,
    ) -> AsyncIterator[str]:
        import asyncio
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _stream_sync():
            try:
                full_messages = [{"role": "system", "content": system}] + messages
                with self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    stream=True,
                    temperature=0.3,
                    max_tokens=300,
                ) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            loop.call_soon_threadsafe(queue.put_nowait, delta)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        executor_task = loop.run_in_executor(None, _stream_sync)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        await executor_task

    async def health(self) -> bool:
        return True
```

**Add to registry:**
```python
elif provider == "openai":
    from voicekit.providers.llm.openai_provider import OpenAIProvider
    return OpenAIProvider(model=config.llm.model)
```

**Add to pyproject.toml:**
```toml
[project.optional-dependencies]
openai = [
    "openai>=1.0.0",
]
```

**Config:**
```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
```

**Update configure_llm_key in config.py:**
```python
elif config.llm.provider == "openai":
    os.environ["OPENAI_API_KEY"] = key
```

**Verify:**
```bash
OPENAI_API_KEY=sk-... uv run pytest tests/pipeline/ -v
```

All pipeline tests must pass with OpenAI exactly as they do with
Anthropic — same test suite, different provider.

---

## Step 3 — Observability Groundwork

Phase 3 adds Prometheus + Grafana. Phase 2 lays the groundwork —
structured logging that Prometheus can scrape, and metric collection
in PipelineMetrics.

### Extend PipelineMetrics

Add fields to `remote_pipeline.py` PipelineMetrics:

```python
class PipelineMetrics:
    def __init__(self):
        self.stt_ms: float = 0
        self.llm_first_token_ms: float = 0
        self.llm_total_ms: float = 0
        self.tts_first_chunk_ms: float = 0
        self.tts_total_ms: float = 0
        self.total_ms: float = 0
        self.transcript: str = ""
        self.response: str = ""
        # Phase 2 additions
        self.transcript_chars: int = 0
        self.response_chars: int = 0
        self.audio_chunks: int = 0
        self.vad_filtered_ms: float = 0
        self.session_turn: int = 0
```

### Structured logging format

All gateway log events already use structured JSON. Phase 3 reads
these and exposes them as Prometheus counters and histograms.

Confirm every turn_complete log has these fields:
```json
{
  "event": "turn_complete",
  "session_id": "...",
  "stt_ms": 123,
  "llm_first_token_ms": 456,
  "tts_first_chunk_ms": 789,
  "total_ms": 1234,
  "transcript_length": 31,
  "turn_count": 1,
  "ts": 1234567890.123
}
```

Add `audio_chunks` and `response_length` to the log in `session.py`:
```python
log.info(json.dumps({
    "event": "turn_complete",
    "session_id": session_id,
    "stt_ms": round(metrics.stt_ms),
    "llm_first_token_ms": round(metrics.llm_first_token_ms),
    "tts_first_chunk_ms": round(metrics.tts_first_chunk_ms),
    "total_ms": round(metrics.total_ms),
    "transcript_length": len(metrics.transcript),
    "response_length": len(metrics.response),
    "audio_chunks": metrics.audio_chunks,
    "turn_count": len(pipeline.history) // 2,
    "ts": time.time(),
}))
```

---

## Phase 2 complete checklist

- KokoroTTS installed and producing audio — DONE
- True streaming on CPU confirmed — DONE
- OpenAI LLM provider implemented
- OpenAI passes all pipeline tests
- PipelineMetrics extended with response_length and audio_chunks
- turn_complete log includes audio_chunks and response_length
- All 29 tests still passing with both Anthropic and OpenAI