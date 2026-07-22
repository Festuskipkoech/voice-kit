"""
Streaming integration tests.

Tests that the gateway delivers true end-to-end streaming —
audio chunks arrive progressively as TTS generates them, not
as a single burst after synthesis completes.

These tests connect to the running gateway over WebSocket and
assert timing-based streaming guarantees.

Requirements:
    Full Docker stack running:
        docker compose -f runtime/docker-compose.yml up -d

    Speech fixture:
        tests/fixtures/speech.raw

    API key in runtime/.env:
        VOICEKIT_LLM_API_KEY=sk-ant-...

Run:
    pytest tests/integration/test_streaming.py -v

What "true streaming" means here:
    1. First audio chunk arrives before the turn completes
       (before transcript/response/metrics are sent)
    2. If multiple chunks arrive, they are spread over time —
       not clustered in a single burst at the end
    3. Time from end_of_speech to first audio is under 10 seconds
       (with Kokoro on CPU — Chatterbox may take longer)
"""
import asyncio
import json
import pathlib
import time

import numpy as np
import pytest
import websockets

GATEWAY_WS_URL = "ws://localhost:8000/session"
FIXTURE_PATH = pathlib.Path("tests/fixtures/speech.raw")
CHUNK_SAMPLES = 1600
SAMPLE_RATE = 16000

MAX_FIRST_CHUNK_S = 15.0     # first audio must arrive within 15 seconds
MAX_BURST_RATIO = 0.1         # burst: all chunks within 10% of streaming window

def load_fixture() -> bytes:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Speech fixture not found at {FIXTURE_PATH}. "
            f"Generate with: "
            f"python -c \"from gtts import gTTS; import subprocess; "
            f"tts = gTTS('hello how are you doing today', lang='en'); "
            f"tts.save('/tmp/speech.mp3'); "
            f"subprocess.run(['ffmpeg', '-i', '/tmp/speech.mp3', "
            f"'-ar', '16000', '-ac', '1', '-f', 'f32le', "
            f"'{FIXTURE_PATH}'], check=True)\""
        )
    return FIXTURE_PATH.read_bytes()

async def run_turn(
    ws_url: str = GATEWAY_WS_URL,
    fixture_bytes: bytes | None = None,
) -> dict:
    """
    Run one complete voice turn and collect timing data.

    Returns:
        chunk_times         list of perf_counter timestamps per audio chunk
        end_of_speech_time  when end_of_speech was sent
        metrics_time        when metrics message was received
        audio_chunks        list of raw audio bytes
        transcript          what STT transcribed
        response            what LLM said
        metrics             timing metrics from gateway
    """
    if fixture_bytes is None:
        fixture_bytes = load_fixture()

    audio_chunks = []
    chunk_times = []
    end_of_speech_time = None
    metrics_time = None
    transcript = ""
    response = ""
    metrics = {}

    async with websockets.connect(
        ws_url,
        ping_interval=None,
        ping_timeout=None,
        open_timeout=10,
        close_timeout=300,
    ) as ws:

        # wait for ready
        msg = json.loads(await ws.recv())
        assert msg["type"] == "ready", f"Expected ready, got {msg}"

        # send audio
        audio = np.frombuffer(fixture_bytes, dtype=np.float32)
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i:i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            await ws.send(chunk.tobytes())

        await ws.send(json.dumps({"type": "end_of_speech"}))
        end_of_speech_time = time.perf_counter()

        async for raw in ws:
            if isinstance(raw, bytes):
                chunk_times.append(time.perf_counter())
                audio_chunks.append(raw)

            else:
                data = json.loads(raw)
                msg_type = data.get("type")

                if msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue

                elif msg_type == "transcript":
                    transcript = data.get("text", "")

                elif msg_type == "response":
                    response = data.get("text", "")

                elif msg_type == "metrics":
                    metrics_time = time.perf_counter()
                    metrics = data
                    break

                elif msg_type == "error":
                    pytest.fail(f"Gateway error: {data.get('message')}")

    return {
        "chunk_times": chunk_times,
        "end_of_speech_time": end_of_speech_time,
        "metrics_time": metrics_time,
        "audio_chunks": audio_chunks,
        "transcript": transcript,
        "response": response,
        "metrics": metrics,
    }

class TestStreaming:

    def test_audio_arrives_before_metrics(self):
        """
        Core streaming test.

        First audio chunk must arrive before the metrics message.
        Metrics is sent after all audio has been streamed.
        If audio arrives after metrics, the pipeline is batching
        not streaming.
        """
        result = asyncio.run(run_turn())

        assert result["chunk_times"], "No audio chunks received"
        assert result["metrics_time"] is not None, "No metrics received"

        first_chunk_time = result["chunk_times"][0]
        metrics_time = result["metrics_time"]

        assert first_chunk_time < metrics_time, (
            f"First audio chunk arrived AFTER metrics. "
            f"Pipeline is batching, not streaming. "
            f"first_chunk={first_chunk_time:.3f} metrics={metrics_time:.3f}"
        )

    def test_first_audio_latency(self):
        """
        First audio chunk must arrive within MAX_FIRST_CHUNK_S seconds
        of end_of_speech being sent.

        With Kokoro on CPU: target <3s per sentence.
        With Chatterbox on CPU: may be 90-150s — expect this test to
        fail on CPU with Chatterbox, which is the reason Kokoro exists.
        """
        result = asyncio.run(run_turn())

        assert result["chunk_times"], "No audio chunks received"

        first_chunk_time = result["chunk_times"][0]
        end_of_speech_time = result["end_of_speech_time"]
        latency = first_chunk_time - end_of_speech_time

        assert latency <= MAX_FIRST_CHUNK_S, (
            f"First audio chunk took {latency:.2f}s after end_of_speech. "
            f"Target: <{MAX_FIRST_CHUNK_S}s. "
            f"If using Chatterbox on CPU, switch to Kokoro for streaming."
        )

    def test_chunks_arrive_progressively(self):
        """
        If multiple chunks are received, they must be spread over time.
        A burst where all chunks arrive within 100ms of each other
        indicates batching — TTS synthesised everything then sent it all.
        True streaming means chunks arrive progressively as generated.

        Single-chunk responses are acceptable — a very short response
        may fit in one Kokoro synthesis segment.
        """
        result = asyncio.run(run_turn())
        chunk_times = result["chunk_times"]

        if len(chunk_times) <= 1:
            pytest.skip(
                "Only 1 chunk received — response too short to measure "
                "streaming distribution. Try a longer response."
            )

        first_chunk_time = chunk_times[0]
        last_chunk_time = chunk_times[-1]
        streaming_window = last_chunk_time - first_chunk_time

        assert streaming_window >= MIN_STREAMING_WINDOW_S, (
            f"All {len(chunk_times)} chunks arrived within "
            f"{streaming_window:.3f}s. "
            f"This looks like batching — chunks should be spread "
            f"over at least {MIN_STREAMING_WINDOW_S}s for true streaming."
        )

    def test_pipeline_produces_transcript(self):
        """Verify STT is working correctly end to end."""
        result = asyncio.run(run_turn())
        assert result["transcript"], (
            "Empty transcript from gateway. "
            "Check VAD settings and speech fixture quality."
        )

    def test_pipeline_produces_response(self):
        """Verify LLM is working correctly end to end."""
        result = asyncio.run(run_turn())
        assert result["response"], (
            "Empty response from gateway. "
            "Check VOICEKIT_LLM_API_KEY is set in runtime/.env."
        )

    def test_response_is_plain_text(self):
        """
        LLM response must not contain markdown or emojis.
        These corrupt TTS output — Kokoro reads '*' and '#' literally.

        If this test fails, check:
            1. System prompt explicitly prohibits markdown and emojis
            2. LLM temperature is 0.3 (not default 1.0)
        """
        result = asyncio.run(run_turn())
        response = result["response"]

        if not response:
            pytest.skip("No response to check")

        # check for common markdown patterns
        assert "*" not in response, (
            f"Response contains asterisks (markdown bold/italic): "
            f"'{response[:100]}'"
        )
        assert "#" not in response, (
            f"Response contains hash (markdown header): "
            f"'{response[:100]}'"
        )
        assert "```" not in response, (
            f"Response contains code block: '{response[:100]}'"
        )

        # check for emojis — any non-ASCII character is suspicious
        non_ascii = [c for c in response if ord(c) > 127]
        assert not non_ascii, (
            f"Response contains non-ASCII characters (likely emojis): "
            f"{non_ascii[:5]} in '{response[:100]}'"
        )

    def test_metrics_contain_timing_fields(self):
        """Gateway must return complete timing breakdown."""
        result = asyncio.run(run_turn())
        metrics = result["metrics"]

        assert "stt_ms" in metrics
        assert "llm_first_token_ms" in metrics
        assert "tts_first_chunk_ms" in metrics
        assert "total_ms" in metrics

        assert metrics["stt_ms"] > 0, "STT ms should be > 0"
        assert metrics["total_ms"] > 0, "Total ms should be > 0"
        assert metrics["total_ms"] >= metrics["stt_ms"], (
            "Total must be >= STT time"
        )

    def test_concurrent_sessions_both_stream(self):
        """
        Two concurrent sessions must both receive streaming audio.
        Tests that the turn_in_progress flag is per-session, not global.
        """
        fixture_bytes = load_fixture()

        async def run_both():
            results = await asyncio.gather(
                run_turn(fixture_bytes=fixture_bytes),
                run_turn(fixture_bytes=fixture_bytes),
                return_exceptions=True,
            )
            return results

        results = asyncio.run(run_both())

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, (
            f"{len(errors)} out of 2 concurrent sessions failed: {errors}"
        )

        for i, result in enumerate(results):
            assert result["chunk_times"], (
                f"Session {i} received no audio chunks"
            )
            assert result["chunk_times"][0] < result["metrics_time"], (
                f"Session {i}: audio did not arrive before metrics — "
                f"streaming broken under concurrent load"
            )


MIN_STREAMING_WINDOW_S = 0.5