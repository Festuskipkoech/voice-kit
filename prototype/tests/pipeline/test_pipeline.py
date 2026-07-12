import asyncio
import time

import numpy as np
import pytest

from voicekit.config import VoiceConfig, STTConfig, TTSConfig, VADConfig, LLMConfig
from voicekit.pipeline import VoicePipeline

def make_config() -> VoiceConfig:
    return VoiceConfig(
        project="test",
        stt=STTConfig(model="simulated", variant="small"),
        tts=TTSConfig(model="simulated", voice="default"),
        vad=VADConfig(enabled=True, sensitivity=0.5),
        llm=LLMConfig(provider="simulated", model="simulated", api_key=""),
        system_prompt="You are a helpful assistant.",
    )

async def make_audio_queue(
    duration_seconds: float = 1.5,
    amplitude: float = 0.5,
) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    chunk_size = 1600
    sample_rate = 16000
    num_chunks = int(duration_seconds * sample_rate / chunk_size)
    for _ in range(num_chunks):
        chunk = np.random.uniform(-amplitude, amplitude, chunk_size).astype(np.float32)
        await queue.put(chunk)
    return queue

async def make_silence_queue(duration_seconds: float = 1.0) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    chunk_size = 1600
    sample_rate = 16000
    num_chunks = int(duration_seconds * sample_rate / chunk_size)
    for _ in range(num_chunks):
        chunk = np.zeros(chunk_size, dtype=np.float32)
        await queue.put(chunk)
    return queue

@pytest.fixture
async def pipeline():
    p = VoicePipeline(make_config())
    await p.load()
    return p

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
    audio_in: asyncio.Queue = asyncio.Queue()
    audio_out: asyncio.Queue = asyncio.Queue()
    with pytest.raises(RuntimeError) as exc:
        await p.run_turn(audio_in, audio_out)
    assert "load()" in str(exc.value)

@pytest.mark.asyncio
async def test_pipeline_produces_audio_output(pipeline):
    audio_in = await make_audio_queue(duration_seconds=1.5)
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    assert not audio_out.empty(), "Pipeline produced no audio output"
    assert metrics.transcript, "Pipeline produced no transcript"
    assert metrics.response, "Pipeline produced no LLM response"

@pytest.mark.asyncio
async def test_silence_produces_no_output(pipeline):
    audio_in = await make_silence_queue(duration_seconds=1.0)
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    assert audio_out.empty(), "Silence should produce no audio output"
    assert metrics.transcript == "", "Silence should produce empty transcript"

@pytest.mark.asyncio
async def test_pipeline_completes_in_reasonable_time(pipeline):
    audio_in = await make_audio_queue(duration_seconds=1.0)
    audio_out: asyncio.Queue = asyncio.Queue()

    start = time.perf_counter()
    await pipeline.run_turn(audio_in, audio_out)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 5000, (
        f"Simulated pipeline took {elapsed_ms:.0f}ms. "
        f"If simulation is this slow, real models will be worse."    )


@pytest.mark.asyncio
async def test_conversation_history_accumulates(pipeline):
    for _ in range(3):
        audio_in = await make_audio_queue(1.0)
        audio_out: asyncio.Queue = asyncio.Queue()
        await pipeline.run_turn(audio_in, audio_out)

    # 3 user turns + 3 assistant turns
    assert len(pipeline.history) == 6, (
        f"Expected 6 history entries after 3 turns, got {len(pipeline.history)}"
    )

@pytest.mark.asyncio
async def test_history_has_correct_roles(pipeline):
    audio_in = await make_audio_queue(1.0)
    audio_out: asyncio.Queue = asyncio.Queue()
    await pipeline.run_turn(audio_in, audio_out)

    assert pipeline.history[0]["role"] == "user"
    assert pipeline.history[1]["role"] == "assistant"
    assert pipeline.history[0]["content"]
    assert pipeline.history[1]["content"]

@pytest.mark.asyncio
async def test_metrics_populated(pipeline):
    audio_in = await make_audio_queue(1.0)
    audio_out: asyncio.Queue = asyncio.Queue()

    metrics = await pipeline.run_turn(audio_in, audio_out)

    assert metrics.stt_ms > 0
    assert metrics.llm_first_token_ms > 0
    assert metrics.tts_first_chunk_ms > 0
    assert metrics.total_ms > 0
    assert metrics.total_ms >= metrics.stt_ms

@pytest.mark.asyncio
async def test_concurrent_sessions_are_isolated():
    """
    The most important test in the suite.
    Two pipeline instances running simultaneously must not
    share state — history, turn count, or any other per-session data.
    """
    async def run_session(turns: int) -> list[dict]:
        p = VoicePipeline(make_config())
        await p.load()
        for _ in range(turns):
            audio_in = await make_audio_queue(1.0)
            audio_out: asyncio.Queue = asyncio.Queue()
            await p.run_turn(audio_in, audio_out)
        return p.history

    # run three independent sessions concurrently
    results = await asyncio.gather(
        run_session(2),
        run_session(3),
        run_session(1),
    )

    # each session has its own independent history
    assert len(results[0]) == 4, f"Session 0: expected 4 entries, got {len(results[0])}"
    assert len(results[1]) == 6, f"Session 1: expected 6 entries, got {len(results[1])}"
    assert len(results[2]) == 2, f"Session 2: expected 2 entries, got {len(results[2])}"

@pytest.mark.asyncio
async def test_first_audio_arrives_before_pipeline_completes(pipeline):
    """
    Validates the streaming cascade.
    Audio output must begin arriving before the full pipeline
    turn completes — otherwise streaming is broken and the
    agent will feel unresponsive.
    """
    audio_in = await make_audio_queue(1.5)
    audio_out: asyncio.Queue = asyncio.Queue()

    first_audio_time: list[float] = []

    async def monitor():
        while True:
            try:
                await asyncio.wait_for(audio_out.get(), timeout=10.0)
                if not first_audio_time:
                    first_audio_time.append(time.perf_counter())
                    return
            except asyncio.TimeoutError:
                return

    pipeline_task = asyncio.create_task(pipeline.run_turn(audio_in, audio_out))
    monitor_task = asyncio.create_task(monitor())

    await pipeline_task
    pipeline_end = time.perf_counter()

    await monitor_task

    assert first_audio_time, "No audio was produced at all"
    assert first_audio_time[0] < pipeline_end, (
        "First audio chunk arrived after pipeline completed. "
        "The streaming cascade is not working — "
        "LLM and TTS are not running concurrently."
    )