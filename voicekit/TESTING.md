# voicekit — Testing Guide

Complete reference for running, understanding, and troubleshooting
every test in the voicekit test suite. Covers developer onboarding,
test suite structure, expected outputs, and solutions to every failure
encountered during Phase 1 development.

---

## Test suite structure

```
tests/
├── unit/
│   ├── test_config.py       config loading, env variable resolution
│   └── test_vad.py          Silero VAD — silence rejection, speech detection
├── pipeline/
│   └── test_pipeline.py     full VoicePipeline with real models, no Docker
├── integration/
│   ├── conftest.py          Docker lifecycle fixtures
│   ├── test_stt.py          Whisper STT service over WebSocket
│   └── test_tts.py          Chatterbox/Kokoro TTS service over WebSocket
├── fixtures/
│   └── speech.raw           float32 PCM speech audio, 16kHz
├── results/
│   └── response.wav         end-to-end test output (gitignored)
└── test_voice.py            end-to-end voice turn — the real proof
```

---

## Developer onboarding — first run

Follow this sequence exactly when setting up voicekit for the first time.

### 1. Clone and install

```bash
git clone https://github.com/Festuskipkoech/voice-kit.git
cd voice-kit/voicekit
uv sync --extra kokoro --extra dev
```

### 2. Set API key

```bash
cp .env.example .env           # if it exists
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

The `.env` file is gitignored — never commit it.

### 3. Generate speech fixture

```bash
mkdir -p tests/fixtures
uv pip install gtts
uv run python -c "
from gtts import gTTS
import subprocess
tts = gTTS('hello how are you doing today', lang='en')
tts.save('/tmp/speech.mp3')
subprocess.run([
    'ffmpeg', '-i', '/tmp/speech.mp3',
    '-ar', '16000', '-ac', '1', '-f', 'f32le',
    'tests/fixtures/speech.raw'
], check=True)
print('Speech fixture created')
"
```

Requires: `ffmpeg` installed (`apt install ffmpeg` or `brew install ffmpeg`)

### 4. Run unit tests

```bash
uv run pytest tests/unit/ -v
```

Expected: 20/20 passing. No Docker, no API key, no models.

### 5. Run pipeline tests

```bash
uv run pytest tests/pipeline/ -v
```

Expected: 9/9 passing. Requires `ANTHROPIC_API_KEY`. Downloads Whisper
tiny and Kokoro models on first run (~400MB, ~327MB). Subsequent runs
use local cache.

### 6. Start Docker stack

```bash
# create runtime/.env with API key
echo "VOICEKIT_LLM_API_KEY=sk-ant-..." > runtime/.env

docker compose -f runtime/docker-compose.yml build
docker compose -f runtime/docker-compose.yml up -d
```

### 7. Verify services

```bash
curl http://localhost:8001/health   # STT
curl http://localhost:8002/health   # TTS
curl http://localhost:8000/health   # Gateway
```

Expected:
```json
{"status":"ok","model":"whisper","variant":"small"}
{"status":"ok","model":"kokoro","voice":"af_bella"}
{"status":"ok","active_sessions":0}
```

### 8. Run end-to-end test

```bash
uv run python tests/test_voice.py
aplay tests/results/response.wav
```

Expected output:
```
Connecting to ws://localhost:8000/session...
Gateway: {'type': 'ready', 'session_id': '...'}
Sending 2.5s of speech audio...
End of speech sent. Waiting for response...
  [audio] chunk 1: 4800 bytes
  [audio] chunk 2: 4800 bytes
  ...
  [audio] chunk N: XXXX bytes

You said:    'Hello, how are you doing today?'
Agent said:  'Hello. I am doing well. How can I help you?'

Latency breakdown:
  STT:             ~12000ms
  LLM first token: ~800ms
  TTS first chunk: ~600ms     ← with Kokoro on CPU
  Total:           ~15000ms

Total audio chunks received: N
Saving N audio chunks to WAV...
Saved to:  tests/results/response.wav
Play with: aplay tests/results/response.wav
```

---

## Running individual test suites

### Unit tests — no dependencies

```bash
uv run pytest tests/unit/ -v
# or specific file
uv run pytest tests/unit/test_vad.py -v
uv run pytest tests/unit/test_config.py -v
```

No Docker. No API key. No models. Should complete in under 10 seconds.

### Pipeline tests — real models, no Docker

```bash
uv run pytest tests/pipeline/ -v
```

Requires:
- `ANTHROPIC_API_KEY` in environment or `.env`
- `faster-whisper` installed (`uv sync --extra dev`)
- `kokoro>=0.9.4` installed (`uv sync --extra kokoro`)
- `espeak-ng` system package (`apt install espeak-ng`)
- `tests/fixtures/speech.raw` generated (see onboarding step 3)

First run downloads model weights:
- Whisper tiny: ~75MB, ~10 seconds
- Kokoro: ~327MB, ~30 seconds

Subsequent runs use cache — loads in ~5 seconds.

Models load once via module-level cache:
```python
_shared_pipeline = None

@pytest.fixture
async def pipeline():
    global _shared_pipeline
    if _shared_pipeline is None:
        p = VoicePipeline(make_config())
        await p.load()
        _shared_pipeline = p
    _shared_pipeline.vad.reset_states()
    return _shared_pipeline
```

### Integration tests — full Docker stack

```bash
# start test stack on separate ports
docker compose -f runtime/docker-compose.test.yml build
docker compose -f runtime/docker-compose.test.yml up -d

# run integration tests
uv run pytest tests/integration/ -v

# tear down
docker compose -f runtime/docker-compose.test.yml down
```

Test ports:
- STT: 8011 (not 8001)
- TTS: 8012 (not 8002)

The test stack runs on different ports so it does not conflict with
the development stack.

### End-to-end test

```bash
# requires full dev stack running
docker compose -f runtime/docker-compose.yml up -d

uv run python tests/test_voice.py
aplay tests/results/response.wav
```

---

## Test descriptions

### Unit — test_config.py (7 tests)

| Test | What it checks |
|---|---|
| test_valid_config_loads | YAML config loads without error |
| test_missing_config_file_raises_clear_error | useful error message |
| test_missing_required_key_raises_error | catches missing stt.model etc |
| test_invalid_vad_sensitivity_raises_error | 0.0-1.0 range enforced |
| test_env_variable_in_api_key_resolved | ${ANTHROPIC_API_KEY} resolved |
| test_missing_env_variable_raises_clear_error | useful error when key absent |
| test_default_system_prompt_applied_when_not_set | default prompt exists |

### Unit — test_vad.py (10 tests)

| Test | What it checks |
|---|---|
| test_silence_is_rejected | zeros return False |
| test_empty_chunk_is_rejected | empty array returns False |
| test_silence_rejected_at_various_lengths | 512/1024/1600/3200/4800 samples |
| test_speech_like_audio_is_accepted | harmonic sine wave detected |
| test_louder_speech_detected | higher amplitude also detected |
| test_invalid_sensitivity_raises_error | <0.0 and >1.0 raise ValueError |
| test_is_speech_returns_bool | return type is bool, not tensor |
| test_reset_states_does_not_crash | reset_states() callable without error |
| test_filter_stream_removes_silence | 3 silence + 2 speech → 2 returned |
| test_filter_stream_empty_input | empty list → empty list |

### Pipeline — test_pipeline.py (9 tests)

| Test | What it checks |
|---|---|
| test_pipeline_loads_successfully | Whisper + Kokoro load, health True |
| test_run_turn_before_load_raises_error | RuntimeError with "load()" in message |
| test_silence_produces_no_transcript | silence → empty transcript (VAD disabled) |
| test_real_speech_produces_transcript | fixture audio → non-empty transcript |
| test_real_speech_produces_audio_output | transcript + response + audio all present |
| test_metrics_populated | stt_ms, llm_first_token_ms, tts_first_chunk_ms > 0 |
| test_conversation_history_accumulates | history grows by 2 per turn |
| test_history_has_correct_roles | user/assistant roles in correct order |
| test_first_audio_arrives_before_pipeline_completes | streaming cascade works |

### Integration — test_stt.py (13 tests)

| Group | Tests |
|---|---|
| Health | health 200, model info in response |
| Transcription | silence → empty, speech → transcript, short/long handled |
| Latency | under 5s for 1.5s audio, consistent across sequential requests |
| Concurrency | 3 concurrent sessions all complete, independent results |
| Protocol | done always sent, end signal closes session |

### Integration — test_tts.py (12 tests)

| Group | Tests |
|---|---|
| Health | health 200, model info in response |
| Audio | text → audio, valid WAV, 24000Hz, longer text → more audio |
| Streaming | multiple chunks, first audio under 5s, first chunk before send done |
| Concurrency | 3 concurrent sessions all produce audio, independent |
| Protocol | done always sent, sentence boundary triggers audio |

---

## Troubleshooting

### Unit test failures

**`test_filter_stream_removes_silence` — AssertionError: 1 == 2**

Silero is an RNN — state accumulates. `reset_states()` must be called
AFTER each chunk decision, not before. If called before, the RNN starts
cold with no context and gives unreliable confidence scores.

Fix in `voicekit/vad.py`:
```python
def filter_stream(self, chunks):
    result = []
    for chunk in chunks:
        if self.is_speech(chunk):
            result.append(chunk)
        self._model.reset_states()   # AFTER decision
    return result
```

Use 4800-sample chunks (300ms) in tests, not 1600-sample (100ms).
More windows give Silero more context for reliable detection.

**`test_speech_like_audio_is_accepted` — fails**

Random noise (`np.random.uniform`) is correctly rejected by Silero.
Use harmonic sine waves:

```python
def make_speech_like(samples=4800, fundamental_hz=120.0, amplitude=0.5):
    t = np.linspace(0, samples / 16000, samples, endpoint=False)
    signal = np.zeros(samples, dtype=np.float32)
    harmonic = fundamental_hz
    harmonic_amplitude = amplitude
    while harmonic < 3400:
        signal += harmonic_amplitude * np.sin(2 * np.pi * harmonic * t)
        harmonic += fundamental_hz
        harmonic_amplitude *= 0.7
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * amplitude
    return signal.astype(np.float32)
```

---

### Pipeline test failures

**`ModuleNotFoundError: No module named 'providers'`**

`voicekit/providers/registry.py` has wrong imports. Fix:
```python
# wrong
from providers.base import STTProvider

# correct
from voicekit.providers.base import STTProvider
```

This happens because inside Docker containers `/app` is on sys.path,
but when running locally the package must use full package paths.

**`ModuleNotFoundError: No module named 'chatterbox'`**

Chatterbox is an optional dependency. Install it:
```bash
uv sync --extra chatterbox
```

Or switch to Kokoro which is lighter:
```bash
uv sync --extra kokoro
```

**`AttributeError: module 'perth' has no attribute 'PerthImplicitWatermarker'`**

Apply the DummyWatermarker patch before importing chatterbox:
```python
import perth

class DummyWatermarker:
    def __init__(self, *args, **kwargs): pass
    def apply_watermark(self, wav, *args, **kwargs): return wav
    def __call__(self, *args, **kwargs): return args[0] if args else None

perth.PerthImplicitWatermarker = DummyWatermarker
perth.DummyWatermarker = DummyWatermarker

from chatterbox.tts_turbo import ChatterboxTurboTTS
```

**`test_silence_produces_no_transcript` — Whisper hallucinated on silence**

Whisper produces filler phrases on silence. Set these parameters in
`WhisperSTT.transcribe()`:
```python
no_speech_threshold=0.6
log_prob_threshold=-1.0
vad_filter=True
```

**Pipeline tests hanging for 10+ minutes**

Chatterbox on CPU takes 90-150 seconds per synthesis. Switch to Kokoro:
```python
def make_config():
    return VoiceConfig(
        tts=TTSConfig(model="kokoro", voice="af_bella"),
        ...
    )
```

Or disable VAD and use Whisper tiny to speed up tests:
```python
stt=STTConfig(model="whisper", variant="tiny"),
vad=VADConfig(enabled=False, ...),
```

---

### Docker failures

**`failed to bind host port 0.0.0.0:6379: address already in use`**

System Redis is running. Stop it:
```bash
sudo systemctl stop redis-server
sudo systemctl disable redis-server
```

**`failed to set up container networking: network not found`**

Docker network got into a broken state:
```bash
docker compose -f runtime/docker-compose.yml down --remove-orphans
docker network prune -f
docker compose -f runtime/docker-compose.yml up
```

**`Unknown TTS model: 'kokoro'`**

`runtime/tts/server.py` does not have kokoro in `_load_provider()`.
Add the branch:
```python
elif model == "kokoro":
    from voicekit.providers.tts.kokoro import KokoroTTS
    return KokoroTTS(voice=voice)
```

Then rebuild:
```bash
docker compose -f runtime/docker-compose.yml build tts
```

**Gateway crashes: `ModuleNotFoundError: No module named 'config'`**

`main.py` imports happen before `sys.path.insert(0, "/app")`. Move the
sys.path insert to the very first line:
```python
import sys
sys.path.insert(0, "/app")   # must be before all other imports

import logging
...
```

**TTS container keeps restarting — health check failing**

Model is downloading from HuggingFace. Monitor:
```bash
docker compose -f runtime/docker-compose.yml logs -f tts
```

Watch for download progress. Kokoro: ~327MB. Chatterbox: ~2GB.
Once `TTS model ready` appears, the health check passes.

**Disk full during Docker build**

```bash
# free build cache
docker builder prune -af

# remove unused images
docker image prune -af

# check space
df -h /
```

If TTS image is too large (Chatterbox + Kokoro = 7.5GB), remove
Chatterbox from the Dockerfile and use Kokoro only.

---

### End-to-end test failures

**`No audio received` — test script exits before audio arrives**

The websockets library's own ping mechanism is closing the connection.
Disable it in the test client:
```python
async with websockets.connect(
    GATEWAY_URL,
    ping_interval=None,   # disable — gateway handles ping/pong
    ping_timeout=None,
) as ws:
```

**`wave.Error: unknown format: 3`**

Python's `wave` module does not support float32 WAV (format 3).
Use `soundfile` instead:
```python
import soundfile as sf
audio, sample_rate = sf.read(io.BytesIO(chunk), dtype="float32")
```

**WAV file is 0.05 seconds long despite many chunks received**

Chunks 2..N are raw float32 PCM, not WAV files. Read them as:
```python
audio = np.frombuffer(chunk, dtype=np.float32)  # NOT int16
```

Using `dtype=np.int16` for float32 data produces garbage audio.

**LLM response contains emojis and markdown**

Claude temperature is too high (default 1.0). Set to 0.3.
System prompt must explicitly prohibit formatting:
```
No emojis, no asterisks, no markdown, no bullet points.
Speak in plain sentences as if talking on a phone call.
```

**Ping timeout killing connection during TTS synthesis**

`turn_in_progress` flag not set or ping loop not checking it.
Verify `remote_pipeline.py` sets `self.turn_in_progress.set()` at
the start of `run_turn()` and clears it in `finally`.
Verify `services/ping.py` checks `turn_in_progress.is_set()` before
sending ping.

**`tts_first_chunk_ms: 120000` — 2 minutes to first audio**

This is Chatterbox on CPU. It is not a bug — it is hardware limitation.
Switch to Kokoro for CPU servers:
```yaml
tts:
  model: kokoro
  voice: af_bella
```

---

## CI checklist

Before merging to main, all of these must pass:

```bash
# 1. unit tests
uv run pytest tests/unit/ -v
# expected: 20/20

# 2. pipeline tests
uv run pytest tests/pipeline/ -v
# expected: 9/9

# 3. integration tests (requires Docker)
docker compose -f runtime/docker-compose.test.yml up -d
uv run pytest tests/integration/ -v
docker compose -f runtime/docker-compose.test.yml down
# expected: 29/29

# 4. end-to-end test
docker compose -f runtime/docker-compose.yml up -d
uv run python tests/test_voice.py
# expected: audio chunks received, WAV saved, no error
```

Total: 58/58 automated tests + end-to-end voice confirmation.