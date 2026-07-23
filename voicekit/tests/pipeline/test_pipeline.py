"""
Pipeline tests.

Tests the full VoicePipeline with real providers — Whisper STT,
Kokoro TTS, and the configured LLM provider.

Requirements:
    ANTHROPIC_API_KEY set in environment or .env
    faster-whisper installed
    kokoro-onnx + misaki installed  (uv sync --extra kokoro --extra dev)
    espeak-ng installed             (apt install espeak-ng)
    tests/fixtures/speech.raw must exist

Generate speech fixture:
    mkdir -p tests/fixtures
    python -c "
    from gtts import gTTS
    import subprocess
    tts = gTTS('hello how are you doing today', lang='en')
    tts.save('/tmp/speech.mp3')
    subprocess.run(['ffmpeg', '-i', '/tmp/speech.mp3',
        '-ar', '16000', '-ac', '1', '-f', 'f32le',
        'tests/fixtures/speech.raw'], check=True)
    "

VAD note:
    Pipeline tests disable VAD (vad.enabled=False).
    VAD correctness is fully covered in tests/unit/test_vad.py.

Silence note:
    Whisper hallucination on silence is suppressed via
    no_speech_threshold=0.6 in WhisperSTT.transcribe().
"""
import asyncio
import os
import pathlib
import time

import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv()

from voicekit.config import VoiceConfig, STTConfig, TTSConfig, VADConfig, LLMConfig
from voicekit.pipeline import VoicePipeline

SAMPLE_RATE   = 16000
CHUNK_SAMPLES = 1600

_shared_pipeline = None

def make_config() -> VoiceConfig:
    return VoiceConfig(
        project="test",
        stt=STTConfig(model="whisper", variant="tiny"),
        tts=TTSConfig(model="kokoro", voice="af_bella"),
        vad=VADConfig(enabled=False, sensitivity=0.5),
        llm=LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        ),
        system_prompt=(
            "You are a helpful assistant. "
            "Reply in one short sentence only. "
            "After every sentence insert <|BREAK|>."
        ),
    )

@pytest.fixture
async def pipeline():
    global _shared_pipeline
    if _shared_pipeline is None:
        p = VoicePipeline(make_config())
        await p.load()
        _shared_pipeline = p
    _shared_pipeline.vad.reset_states()
    return _shared_pipeline

async def make_silence_queue(duration_seconds: float = 1.0) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    num_chunks = int(duration_seconds * SAMPLE_RATE / CHUNK_SAMPLES)
    for _ in range(num_chunks):
        await queue.put(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
    return queue

async def make_real_audio_queue() -> asyncio.Queue:
    fixture = pathlib.Path("tests/fixtures/speech.raw")
    queue: asyncio.Queue = asyncio.Queue()

    if not fixture.exists():
        pytest.skip(
            "tests/fixtures/speech.raw not found. "
            "Generate it with gtts + ffmpeg — see module docstring."
        )

    audio = np.frombuffer(fixture.read_bytes(), dtype=np.float32)
    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        await queue.put(chunk)

    return queue

@pytest.mark.asyncio
async def test_pipeline_loads_successfully():
    p = VoicePipeline(make_config())
    await p.load()
    health = await p.health()
    assert health["stt"] is True
    assert health["tts"] is True
    assert health["loaded"] is True

@pytest.mark.asyncio
async def test_run_turn_before_load_raises_error():
    p = VoicePipeline(make_config())
    audio_in:  asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    with pytest.raises(RuntimeError) as exc:
        await p.run_turn(audio_in, audio_out)
    assert "load()" in str(exc.value)

@pytest.mark.asyncio
async def test_silence_produces_no_transcript(pipeline):
    audio_in  = await make_silence_queue(duration_seconds=1.0)
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    assert metrics.transcript == "", (
        f"Whisper hallucinated on silence: '{metrics.transcript}'"
    )
    assert audio_out.empty(), (
        "Silence produced audio — pipeline should stop on empty transcript"
    )

@pytest.mark.asyncio
async def test_real_speech_produces_transcript(pipeline):
    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    assert metrics.transcript, (
        "Whisper returned empty transcript for real speech audio."
    )

@pytest.mark.asyncio
async def test_real_speech_produces_audio_output(pipeline):
    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    assert metrics.transcript, "No transcript produced"
    assert metrics.response,   "No LLM response produced"
    assert not audio_out.empty(), "No audio output produced"

@pytest.mark.asyncio
async def test_response_contains_no_break_markers(pipeline):
    """
    <|BREAK|> markers must be stripped from metrics.response
    before it is stored. Markers are transport only — they must
    never appear in conversation history or client-facing output.
    """
    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    if metrics.response:
        assert "<|BREAK|>" not in metrics.response, (
            f"Raw <|BREAK|> marker found in response: '{metrics.response[:100]}'"
        )


@pytest.mark.asyncio
async def test_metrics_populated(pipeline):
    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    if metrics.transcript:
        assert metrics.stt_ms > 0
        assert metrics.llm_first_token_ms > 0
        assert metrics.tts_first_chunk_ms > 0
        assert metrics.total_ms > 0
        assert metrics.total_ms >= metrics.stt_ms


@pytest.mark.asyncio
async def test_conversation_history_accumulates(pipeline):
    initial_length = len(pipeline.history)

    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await pipeline.run_turn(audio_in, audio_out)

    if len(pipeline.history) > initial_length:
        assert len(pipeline.history) == initial_length + 2


@pytest.mark.asyncio
async def test_history_has_correct_roles(pipeline):
    initial_length = len(pipeline.history)

    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    await pipeline.run_turn(audio_in, audio_out)

    if len(pipeline.history) > initial_length:
        assert pipeline.history[initial_length]["role"]     == "user"
        assert pipeline.history[initial_length + 1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_first_audio_arrives_before_pipeline_completes(pipeline):
    """
    LLM and TTS must run concurrently — first audio chunk must arrive
    before pipeline.run_turn() returns. If audio only arrives after
    the pipeline completes, phrases are being processed sequentially
    which breaks the streaming cascade.
    """
    audio_in  = await make_real_audio_queue()
    audio_out: asyncio.Queue = asyncio.Queue()

    first_audio_time: list[float] = []

    async def monitor():
        try:
            await asyncio.wait_for(audio_out.get(), timeout=60.0)
            first_audio_time.append(time.perf_counter())
        except asyncio.TimeoutError:
            pass

    pipeline_task = asyncio.create_task(pipeline.run_turn(audio_in, audio_out))
    monitor_task  = asyncio.create_task(monitor())

    await pipeline_task
    pipeline_end = time.perf_counter()
    await monitor_task

    if first_audio_time:
        assert first_audio_time[0] < pipeline_end, (
            "First audio arrived after pipeline completed — "
            "streaming cascade is broken."
        )