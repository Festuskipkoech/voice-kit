# voicekit — True Streaming Architecture

## Status: APPROVED FOR IMPLEMENTATION
## Session: Implement in a fresh conversation window

---

## Problem Statement

### What we built

The pipeline runs LLM and TTS concurrently via `asyncio.gather(feed_llm(), feed_tts())`.
LLM tokens flow into `llm_token_queue`. TTS reads from the queue, buffers tokens until
a sentence boundary, then synthesises a phrase and yields audio chunks.

The architecture is correct. The implementation has a timing bug.

### Why it fails

Claude Haiku completes a short response in ~1 second total. All tokens flood
`llm_token_queue` simultaneously before TTS has consumed any of them.

By the time `synthesize()` starts iterating the token stream, all tokens are already
queued. The buffer fills to MAX_BUFFER_TOKENS immediately and flushes the entire
response as one giant phrase — one Kokoro synthesis call taking 10-15 seconds.

The user waits 13 seconds for audio instead of hearing the first phrase in 2-3 seconds.

### What we tried that did not work

- `asyncio.sleep(0)` in `feed_llm()` — LLM still faster than event loop switches
- `Queue(maxsize=1)` for `llm_token_queue` — synthesis blocks the consumer,
  queue backs up anyway
- Smaller `MAX_BUFFER_TOKENS` — all tokens arrive before first check fires
- Removing `MIN_BUFFER_CHARS` — same root cause, buffer absorbs everything

### Root cause in one sentence

Token buffering and TTS synthesis run in the same coroutine. When Kokoro blocks
on synthesis, no tokens are consumed. When synthesis finishes, all remaining tokens
are waiting and the next flush takes everything at once.

---

## Solution

### Core insight

Separate token buffering and TTS synthesis into independent concurrent tasks.
Use LLM-inserted breakpoint markers for reliable phrase detection.
Use bounded queues for automatic rate adaptation.

### Three concurrent stages

```
Stage 1 — SPLITTER
Reads LLM token stream
Detects <|BREAK|> markers (or falls back to punctuation)
Puts complete phrases into phrase_queue
Runs independently — never blocks on TTS

Stage 2 — SYNTHESISER
Pulls one phrase from phrase_queue
Calls Kokoro create_stream() on that phrase
Puts audio chunks into audio_out queue
When busy synthesising, phrase_queue fills naturally (backpressure)

Stage 3 — SENDER (already exists as send_audio_stream() in session.py)
Pulls audio chunks from audio_out queue
Sends to WebSocket client immediately
```

### The <|BREAK|> marker system

The LLM is instructed to insert `<|BREAK|>` at every natural speech boundary.
This is more reliable than regex on a token stream because:
- Tokens split unpredictably across punctuation
- The LLM knows where natural speech pauses belong
- Detection is O(1) per token — just check for the marker string

Example LLM output:
```
Sure,<|BREAK|> I am doing well and ready to help.<|BREAK|> What can I assist you with?<|BREAK|>
```

### How rate adaptation works — no measurement needed

```python
phrase_queue = asyncio.Queue(maxsize=2)
```

When Kokoro is slow:
- phrase_queue fills after 2 phrases
- Splitter blocks on `await phrase_queue.put()`
- LLM tokens accumulate in splitter buffer
- No overflow — natural backpressure

When Kokoro finishes a phrase:
- It pulls the next phrase immediately — already waiting in queue
- No gap between phrases for the user
- Splitter unblocks and queues the next phrase

The bounded queue IS the rate controller. No timers, no measurement needed.

### User experience

```
t=0s     user stops speaking
t=1.5s   Whisper transcribes
t=2.4s   LLM first token arrives, splitter starts
t=2.6s   LLM emits "Sure,<|BREAK|>" — splitter flushes phrase 1
t=4.1s   Kokoro synthesises "Sure," — FIRST AUDIO PLAYS
t=4.1s   phrase 2 already queued — "I am doing well."
t=6.5s   Kokoro synthesises phrase 2 — plays immediately, no gap
t=6.5s   phrase 3 already queued
t=8.0s   phrase 3 plays
t=8.1s   transcript, response, metrics sent
```

True streaming — user hears audio in 2-4 seconds on CPU.

---

## PhraseStream — The Splitter Utility

### File: `voicekit/utils/phrase_splitter.py`

Three operating modes:

```
MODE_DETECTING   watching for first <|BREAK|> marker
                 if marker found within DETECTION_THRESHOLD tokens → MODE_MARKER
                 if threshold exceeded with no marker → MODE_FALLBACK

MODE_MARKER      LLM correctly inserting markers
                 split on <|BREAK|>
                 reliable, fast, natural boundaries

MODE_FALLBACK    LLM failed to insert markers (rare — <1% with correct prompt)
                 split on strong sentence boundaries only (.!?)
                 minimum MIN_PHRASE_CHARS before splitting
                 never interferes with MODE_MARKER
```

### Detection logic

```python
DETECTION_THRESHOLD = 30   # tokens before giving up on markers
MIN_PHRASE_CHARS = 20      # minimum chars before fallback split
SENTENCE_END = re.compile(r"[.!?](\s|$)")   # strong boundaries only

mode = MODE_DETECTING
detection_buffer = []

async for token in token_stream:
    if mode == MODE_DETECTING:
        detection_buffer.append(token)
        joined = "".join(detection_buffer)

        if "<|BREAK|>" in joined:
            mode = MODE_MARKER
            # split detection buffer on marker and flush complete phrases
            ...
        elif len(detection_buffer) >= DETECTION_THRESHOLD:
            mode = MODE_FALLBACK
            # apply punctuation splitting to detection buffer
            ...

    elif mode == MODE_MARKER:
        # fast path — check for marker in each token
        ...

    elif mode == MODE_FALLBACK:
        # punctuation splitting
        ...
```

### Why fallback never interferes with good output

- Mode is set once and never reversed
- `<|BREAK|>` markers are stripped from response text before sending to client
- Fallback uses only `.!?` — no commas, no semicolons — prevents mid-thought splits
- Fallback is a safety net, not a primary path

---

## System Prompt

Temperature: **0.1** — deterministic marker insertion, natural response content.

```
You are a voice assistant. Everything you say is immediately converted
to speech and played as audio. The user cannot see text.

You MUST follow these rules without any exception:

MARKER RULE — CRITICAL:
After every sentence or natural speech pause, you MUST insert the exact
marker <|BREAK|> with no spaces around it.
Example: "Sure,<|BREAK|> I am doing well.<|BREAK|> How can I help?<|BREAK|>"
Every response MUST contain at least one <|BREAK|> marker.
This is a system requirement. Missing markers will cause audio failure.

CONTENT RULES:
- Plain spoken English only
- No emojis, asterisks, markdown, bullet points, lists, or headers
- Maximum 2-3 sentences per response
- Never repeat yourself
```

---

## Files That Change

### Modified files

| File | What changes |
|---|---|
| `voicekit/providers/tts/kokoro.py` | `synthesize()` accepts a single phrase string, not a token stream. Remove all buffer logic, constants, and flush conditions. Keep executor+queue bridge for `create_stream()`. |
| `voicekit/providers/base.py` | Update `TTSProvider.synthesize()` signature to accept a phrase string |
| `runtime/gateway/remote_pipeline.py` | Replace `feed_llm + token_stream + feed_tts` with `split_tokens() + synthesise_phrases()` as concurrent tasks. Remove `asyncio.sleep(0)` and `Queue(maxsize=1)`. Keep `AUDIO_DONE` sentinel and `turn_in_progress` flag. |
| `runtime/gateway/config.py` | Updated system prompt with `<|BREAK|>` instruction |
| `voicekit/providers/llm/claude.py` | Change temperature from 0.3 to 0.1 |
| `voicekit/utils/text_cleaner.py` | Add `<|BREAK|>` stripping from response text |

### New files

| File | Purpose |
|---|---|
| `voicekit/utils/phrase_splitter.py` | `PhraseStream` class — three-mode splitter with detection, marker, and fallback |

### Files with no changes

```
runtime/gateway/routes/session.py       — unchanged
runtime/gateway/routes/health.py        — unchanged
runtime/gateway/services/ping.py        — unchanged
runtime/gateway/services/session.py     — unchanged
runtime/gateway/dependencies.py         — unchanged
runtime/gateway/main.py                 — unchanged
voicekit/providers/stt/whisper.py       — unchanged
voicekit/providers/llm/claude.py        — temperature only
runtime/tts/server.py                   — unchanged
runtime/stt/server.py                   — unchanged
runtime/docker-compose.yml              — unchanged
tests/                                  — minor updates after implementation
```

---

## What Gets Removed From Current Code

### From `remote_pipeline.py`

Remove:
```python
await asyncio.sleep(0)          # did not fix the timing bug
llm_token_queue = asyncio.Queue(maxsize=1)   # replace with phrase_queue
feed_llm()                      # replaced by split_tokens()
token_stream()                  # replaced by phrase_queue
feed_tts()                      # replaced by synthesise_phrases()
```

Keep:
```python
AUDIO_DONE = None               # sentinel still needed
self.turn_in_progress           # ping protection still needed
self._transcribe()              # STT unchanged
self._synthesise()              # renamed, now accepts phrase string
```

### From `kokoro.py`

Remove:
```python
FLUSH_CHARS = re.compile(...)   # replaced by <|BREAK|> detection in splitter
MAX_BUFFER_TOKENS = 3           # no longer needed
MIN_BUFFER_CHARS = 1            # no longer needed
buffer = []                     # no longer needed
on_punctuation / on_max_tokens  # no longer needed
the entire token buffering loop  # replaced by single phrase input
import logging / debug log line  # remove before shipping
```

Keep:
```python
from kokoro_onnx import Kokoro  # correct package
create_stream()                 # correct streaming API
executor+queue bridge pattern   # correct async bridge
first_chunk WAV header logic    # correct audio format handling
```

---

## New `remote_pipeline.py` Architecture

```python
async def run_turn(self, audio_in, audio_out):

    # phase 1 — STT (unchanged)
    transcript = await self._transcribe(audio_in)
    if not transcript.strip():
        return metrics

    self.history.append({"role": "user", "content": transcript})

    # phase 2+3 — LLM → Splitter → TTS (new)
    phrase_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)
    full_response_parts = []

    async def split_tokens():
        # stream LLM tokens through PhraseStream
        # put complete phrases into phrase_queue
        # put None sentinel when done
        splitter = PhraseStream()
        async for token in self._llm.stream(...):
            full_response_parts.append(clean_for_tts(token))
            async for phrase in splitter.feed(token):
                await phrase_queue.put(phrase)
        async for phrase in splitter.flush():
            await phrase_queue.put(phrase)
        await phrase_queue.put(None)

    async def synthesise_phrases():
        # pull phrases one at a time
        # synthesise each with Kokoro
        # put audio chunks into audio_out
        while True:
            phrase = await phrase_queue.get()
            if phrase is None:
                break
            async for chunk in self._synthesise(phrase):
                await audio_out.put(chunk)

    await asyncio.gather(split_tokens(), synthesise_phrases())

    # AUDIO_DONE sentinel (unchanged)
    # metrics collection (unchanged)
```

---

## New `kokoro.py` synthesize() signature

```python
async def synthesize(self, phrase: str) -> AsyncIterator[bytes]:
    """
    Synthesise a single complete phrase.
    Called once per phrase from synthesise_phrases() in remote_pipeline.
    Phrase is a complete sentence or speech unit — no buffering needed here.
    """
    phrase = phrase.strip()
    if not phrase:
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run():
        try:
            stream = self._kokoro.create_stream(
                phrase, voice=self.voice, speed=1.0, lang=self.lang
            )
            # create_stream is async — run in executor via sync wrapper
            import asyncio as _asyncio
            loop2 = _asyncio.new_event_loop()
            async def _collect():
                async for samples, sr in stream:
                    if samples is not None and len(samples) > 0:
                        loop.call_soon_threadsafe(queue.put_nowait, samples)
            loop2.run_until_complete(_collect())
            loop2.close()
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    executor_task = loop.run_in_executor(None, _run)

    first_chunk = True
    while True:
        audio = await queue.get()
        if audio is None:
            break
        if first_chunk:
            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            yield buf.getvalue()
            first_chunk = False
        else:
            yield audio.astype(np.float32).tobytes()

    await executor_task
```

---

## New `phrase_splitter.py` sketch

```python
import re
from typing import AsyncIterator

MODE_DETECTING = "detecting"
MODE_MARKER = "marker"
MODE_FALLBACK = "fallback"

DETECTION_THRESHOLD = 30
MIN_PHRASE_CHARS = 20
SENTENCE_END = re.compile(r"[.!?](\s|$)")
MARKER = "<|BREAK|>"


class PhraseStream:
    """
    Splits an LLM token stream into complete phrases.

    Primary: splits on <|BREAK|> markers inserted by the LLM.
    Fallback: splits on sentence boundaries if LLM fails to insert markers.
    Detection: watches first DETECTION_THRESHOLD tokens to determine mode.
    """

    def __init__(self):
        self.mode = MODE_DETECTING
        self.buffer = ""
        self.token_count = 0

    async def feed(self, token: str) -> AsyncIterator[str]:
        """Feed one token. Yields complete phrases when ready."""
        self.token_count += 1
        self.buffer += token

        if self.mode == MODE_DETECTING:
            if MARKER in self.buffer:
                self.mode = MODE_MARKER
                async for phrase in self._split_on_marker():
                    yield phrase
            elif self.token_count >= DETECTION_THRESHOLD:
                self.mode = MODE_FALLBACK
                async for phrase in self._split_on_punctuation():
                    yield phrase

        elif self.mode == MODE_MARKER:
            async for phrase in self._split_on_marker():
                yield phrase

        elif self.mode == MODE_FALLBACK:
            async for phrase in self._split_on_punctuation():
                yield phrase

    async def flush(self) -> AsyncIterator[str]:
        """Flush any remaining buffer at end of stream."""
        remaining = self.buffer.replace(MARKER, "").strip()
        if remaining:
            yield remaining
        self.buffer = ""

    async def _split_on_marker(self) -> AsyncIterator[str]:
        while MARKER in self.buffer:
            phrase, self.buffer = self.buffer.split(MARKER, 1)
            phrase = phrase.strip()
            if phrase:
                yield phrase

    async def _split_on_punctuation(self) -> AsyncIterator[str]:
        while True:
            match = SENTENCE_END.search(self.buffer)
            if not match:
                break
            end = match.end()
            phrase = self.buffer[:end].strip()
            self.buffer = self.buffer[end:]
            if phrase and len(phrase) >= MIN_PHRASE_CHARS:
                yield phrase
```

---

## Implementation Checklist for New Session

```
1. Write voicekit/utils/phrase_splitter.py
2. Update voicekit/utils/text_cleaner.py — add <|BREAK|> stripping
3. Rewrite voicekit/providers/tts/kokoro.py — phrase input, no buffer logic
4. Update voicekit/providers/base.py — new TTSProvider.synthesize() signature
5. Rewrite runtime/gateway/remote_pipeline.py — split_tokens + synthesise_phrases
6. Update runtime/gateway/config.py — new system prompt, temperature 0.1
7. Update voicekit/providers/llm/claude.py — temperature 0.1
8. Rebuild gateway and TTS containers
9. Run uv run python tests/test_voice.py
10. Verify multiple chunks arriving at different times
11. Run uv run pytest tests/integration/test_streaming.py -v
12. All streaming assertions should pass
13. Commit
```

---

## Success Criteria

```
test_voice.py output:
  [audio] chunk 1: XXXX bytes (+2-4s from end_of_speech)   ← first phrase
  [audio] chunk 2: XXXX bytes (+4-6s from end_of_speech)   ← second phrase
  [audio] chunk 3: XXXX bytes (+6-8s from end_of_speech)   ← third phrase

  PASS — First audio before metrics
  PASS — First chunk within 10.0s
  PASS — Chunks spread over Xs (not a single burst)

test_streaming.py:
  test_audio_arrives_before_metrics     PASSED
  test_first_audio_latency              PASSED
  test_chunks_arrive_progressively      PASSED   ← currently SKIPPED
  test_response_is_plain_text           PASSED
  test_metrics_contain_timing_fields    PASSED
  test_concurrent_sessions_both_stream  PASSED
```

---

## Complete Patch Accounting — Keep or Remove

### `remote_pipeline.py`

| Patch | Action |
|---|---|
| `AUDIO_DONE = None` sentinel | KEEP |
| `self.turn_in_progress = asyncio.Event()` | KEEP |
| `await audio_out.put(AUDIO_DONE)` in finally | KEEP |
| `self.turn_in_progress.set/clear()` | KEEP |
| `from voicekit.utils.text_cleaner import clean_for_tts` | KEEP |
| `await asyncio.sleep(0)` in `feed_llm()` | REMOVE |
| `asyncio.Queue(maxsize=1)` for `llm_token_queue` | REMOVE |
| `feed_llm()`, `token_stream()`, `feed_tts()` | REMOVE — replaced by `split_tokens()` and `synthesise_phrases()` |

---

### `voicekit/providers/tts/kokoro.py`

| Patch | Action |
|---|---|
| `kokoro-onnx` + `Kokoro` class | KEEP |
| `create_stream()` call | KEEP |
| Executor+queue bridge (`_run`, `call_soon_threadsafe`) | KEEP |
| First chunk WAV header logic | KEEP |
| `FLUSH_CHARS`, `MAX_BUFFER_TOKENS`, `MIN_BUFFER_CHARS` constants | REMOVE |
| Token buffer loop (`buffer`, `on_punctuation`, `on_max_tokens`) | REMOVE |
| `synthesize()` accepting `text_stream: AsyncIterator[str]` | REMOVE — new signature accepts `phrase: str` |
| `import logging` debug line | REMOVE |
| `logging.getLogger(__name__).info(f"TTS flush...")` | REMOVE |

---

### `voicekit/utils/text_cleaner.py`

| Patch | Action |
|---|---|
| `clean_for_tts()` function | KEEP |
| Emoji removal | KEEP |
| Markdown stripping | KEEP |
| `<|BREAK|>` stripping | ADD — not yet implemented |

---

### `runtime/gateway/config.py`

| Patch | Action |
|---|---|
| Strict no-markdown system prompt | KEEP — extend with `<|BREAK|>` instruction |
| `VOICEKIT_SYSTEM_PROMPT` env var reading | KEEP |

---

### `runtime/.env`

| Patch | Action |
|---|---|
| `VOICEKIT_LLM_API_KEY` | KEEP |
| `VOICEKIT_TTS_MODEL=kokoro` | KEEP |
| `VOICEKIT_STT_VARIANT=tiny` | KEEP |
| `VOICEKIT_SYSTEM_PROMPT=...` | UPDATE — replace with new `<|BREAK|>` prompt |

---

### `voicekit/providers/llm/claude.py`

| Patch | Action |
|---|---|
| `temperature=0.3` | UPDATE — change to `0.1` |
| `max_tokens=300` | KEEP |

---

### `pyproject.toml`

| Patch | Action |
|---|---|
| `[tool.uv] conflicts` section | KEEP |
| `kokoro = ["kokoro-onnx>=0.5.0"]` | KEEP |
| `chatterbox` extra | KEEP |
| `requires-python = ">=3.11,<3.14"` | KEEP |

---

### New files to create in new session

| File | Action |
|---|---|
| `voicekit/utils/phrase_splitter.py` | CREATE |
| `voicekit/utils/__init__.py` | KEEP — already exists |