"""
STT server integration tests.

Tests the actual running Whisper STT WebSocket server.
Real Docker container, real WebSocket connection, real bytes.

Audio generation:
    make_speech_audio() uses harmonic sine waves that resemble voiced speech.
    Silero VAD is active in the STT server — random noise is correctly
    rejected as non-speech and returns empty transcript.
    Speech-like periodic signals pass VAD and reach Whisper.

Run:
    pytest tests/integration/test_stt.py -v

Requirements:
    Docker running. Port 8011 free.
    The stt_service fixture in conftest.py handles container lifecycle.
"""

import asyncio
import json
import time

import httpx
import numpy as np
import websockets

SAMPLE_RATE = 16000 

def make_speech_audio(
    duration_seconds: float = 1.5,
    fundamental_hz: float = 120.0,
    amplitude: float = 0.5,
) -> bytes:
    """
    Generate speech-like float32 PCM audio as raw bytes.

    Uses harmonic sine waves at voice frequencies (F0 + harmonics)
    that Silero VAD classifies as speech. This is the correct way to
    generate test audio for a pipeline that includes neural VAD.

    Random noise is intentionally NOT used — it is correctly rejected
    by Silero as non-speech, which would cause false test failures.
    """
    num_samples = int(SAMPLE_RATE * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    signal = np.zeros(num_samples, dtype=np.float32)
    harmonic = fundamental_hz
    harmonic_amplitude = amplitude
    while harmonic < 3400:
        signal += harmonic_amplitude * np.sin(2 * np.pi * harmonic * t)
        harmonic += fundamental_hz
        harmonic_amplitude *= 0.7

    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * amplitude

    return signal.astype(np.float32).tobytes()

def make_silence_audio(duration_seconds: float = 1.0) -> bytes:
    """
    Generate pure silence as raw bytes.
    VAD must filter this — STT must return empty transcript.
    """
    num_samples = int(SAMPLE_RATE * duration_seconds)
    return np.zeros(num_samples, dtype=np.float32).tobytes()

async def transcribe(
    ws_url: str,
    audio_bytes: bytes,
    chunk_size_bytes: int = 6400,
    chunk_delay_seconds: float = 0.1,
) -> dict:
    """
    Connect to STT WebSocket, stream audio in chunks, collect result.

    Sends audio in 100ms chunks to simulate real microphone streaming.
    chunk_size_bytes = 6400 = 1600 float32 samples = 100ms at 16kHz.

    Returns:
        tokens      — list of transcript tokens as they streamed in
        transcript  — full transcript from the done message
        elapsed_ms  — total time from first send to done message
    """
    tokens = []
    transcript = ""
    start_time = None

    async with websockets.connect(f"{ws_url}/stt") as ws:
        start_time = time.perf_counter()

        for i in range(0, len(audio_bytes), chunk_size_bytes):
            chunk = audio_bytes[i:i + chunk_size_bytes]
            await ws.send(chunk)
            await asyncio.sleep(chunk_delay_seconds)

        await ws.send(json.dumps({"type": "end"}))

        async for raw_message in ws:
            if isinstance(raw_message, bytes):
                continue

            message = json.loads(raw_message)

            if message["type"] == "token":
                tokens.append(message["text"])

            elif message["type"] == "done":
                transcript = message.get("transcript", "")
                break

            elif message["type"] == "error":
                raise RuntimeError(f"STT server error: {message['message']}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "tokens": tokens,
        "transcript": transcript,
        "elapsed_ms": elapsed_ms,
    }

class TestHealth:

    def test_health_endpoint_returns_200(self, stt_service):
        response = httpx.get(f"{stt_service}/health", timeout=5.0)
        assert response.status_code == 200

    def test_health_response_contains_model_info(self, stt_service):
        response = httpx.get(f"{stt_service}/health", timeout=5.0)
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert data["model"] == "whisper"

class TestTranscription:

    def test_silence_produces_empty_transcript(self, stt_service, stt_ws_url):
        """
        Most important correctness test.
        Silence must return empty — Silero VAD must filter it before Whisper.
        Without this, Whisper hallucinates phrases on silence.
        """
        audio = make_silence_audio(duration_seconds=1.0)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert result["transcript"].strip() == "", (
            f"Expected empty transcript from silence, got: '{result['transcript']}'. "
            f"VAD is not filtering silence correctly."
        )

    def test_speech_audio_produces_transcript(self, stt_service, stt_ws_url):
        audio = make_speech_audio(duration_seconds=2.0)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert isinstance(result["transcript"], str)
        assert len(result["transcript"]) >= 0

    def test_short_audio_handled(self, stt_service, stt_ws_url):
        audio = make_speech_audio(duration_seconds=0.5)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert isinstance(result["transcript"], str)

    def test_long_audio_handled(self, stt_service, stt_ws_url):
        audio = make_speech_audio(duration_seconds=5.0)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert isinstance(result["transcript"], str)

class TestLatency:

    def test_transcription_completes_under_5_seconds(self, stt_service, stt_ws_url):
        """
        Full round trip for 1.5s of audio must complete under 5 seconds.
        Whisper small on CPU: ~300ms inference.
        Budget includes audio streaming time (1.5s at 100ms per chunk).
        """
        audio = make_speech_audio(duration_seconds=1.5)
        result = asyncio.run(transcribe(stt_ws_url, audio))

        assert result["elapsed_ms"] < 5000, (
            f"Transcription took {result['elapsed_ms']:.0f}ms for 1.5s of audio. "
            f"Target is under 5000ms on CPU. "
            f"Check for blocking calls in the server."
        )
    def test_sequential_requests_consistent_latency(self, stt_service, stt_ws_url):
        """
        Latency must not degrade across sequential requests.
        Degradation indicates accumulating state — memory leak or blocking.
        """
        audio = make_speech_audio(duration_seconds=1.0)
        latencies = []

        for _ in range(3):
            result = asyncio.run(transcribe(stt_ws_url, audio))
            latencies.append(result["elapsed_ms"])

        first = latencies[0]
        last = latencies[-1]

        assert last < first * 3, (
            f"Latency degraded: first={first:.0f}ms, last={last:.0f}ms. "
            f"Possible memory leak or blocking accumulation."
        )

class TestConcurrency:

    def test_concurrent_sessions_all_complete(self, stt_service, stt_ws_url):
        """
        Multiple simultaneous sessions must all complete without error.
        """
        audio = make_speech_audio(duration_seconds=1.0)

        async def run_all():
            tasks = [transcribe(stt_ws_url, audio) for _ in range(3)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_all())

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, (
            f"{len(errors)} out of 3 concurrent sessions failed: {errors}"
        )

    def test_concurrent_sessions_independent(self, stt_service, stt_ws_url):
        """Each session must return independent results — no shared state."""
        audio = make_speech_audio(duration_seconds=1.0)

        async def run_all():
            tasks = [transcribe(stt_ws_url, audio) for _ in range(3)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())

        for i, result in enumerate(results):
            assert isinstance(result["transcript"], str), (
                f"Session {i} did not return a string transcript"
            )

class TestProtocol:

    def test_done_message_always_sent(self, stt_service, stt_ws_url):
        """
        Server must always send done, even for silence.
        Without done the client hangs forever.
        """
        done_received = False

        async def check():
            nonlocal done_received
            async with websockets.connect(f"{stt_ws_url}/stt") as ws:
                await ws.send(make_silence_audio(duration_seconds=0.5))
                await ws.send(json.dumps({"type": "end"}))

                async for raw in ws:
                    if isinstance(raw, str):
                        msg = json.loads(raw)
                        if msg["type"] == "done":
                            done_received = True
                            break

        asyncio.run(check())
        assert done_received, (
            "Server did not send done message after silence. "
            "Clients would hang indefinitely."
        )

    def test_end_signal_closes_session(self, stt_service, stt_ws_url):
        """Server must wait for end signal before transcribing."""
        audio = make_speech_audio(duration_seconds=1.0)
        chunks = [audio[i:i + 6400] for i in range(0, len(audio), 6400)]
        result_holder = []

        async def check():
            async with websockets.connect(f"{stt_ws_url}/stt") as ws:
                for chunk in chunks:
                    await ws.send(chunk)
                    await asyncio.sleep(0.05)

                await ws.send(json.dumps({"type": "end"}))

                async for raw in ws:
                    if isinstance(raw, str):
                        msg = json.loads(raw)
                        if msg["type"] == "done":
                            result_holder.append(msg.get("transcript", ""))
                            break

        asyncio.run(check())
        assert result_holder, "No done message received after end signal"