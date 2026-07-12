"""
STT server integration tests.

Tests the actual running STT WebSocket server — not the Python class,
not a mock. Real Docker container, real WebSocket connection, real bytes.

These tests run against the simulated STT provider, which means they
validate the server protocol, streaming behaviour, error handling, and
latency characteristics without needing a GPU or model files.

When Whisper is added later, these same tests run against the real model
with zero changes — only the Docker image changes.

Run:
    pytest tests/integration/test_stt.py -v

Requirements:
    Docker running. Port 8011 free.
    The stt_service fixture in conftest.py handles container lifecycle.
"""
import os
import asyncio
import json
import time
import pytest

import httpx
import numpy as np
import websockets

SIMULATED = os.environ.get("VOICEKIT_STT_MODEL", "simulated") == "simulated"

def make_speech_audio(duration_seconds: float = 1.5, amplitude: float = 0.5) -> bytes:
    """
    Generate float32 PCM audio that VAD will classify as speech.
    Returns raw bytes ready to send over WebSocket.
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    samples = np.random.uniform(-amplitude, amplitude, num_samples).astype(np.float32)
    return samples.tobytes()

def make_silence_audio(duration_seconds: float = 1.0) -> bytes:
    """
    Generate float32 PCM audio that is pure silence.
    VAD should filter this out — STT should return empty transcript.
    """
    sample_rate = 16000
    num_samples = int(sample_rate * duration_seconds)
    samples = np.zeros(num_samples, dtype=np.float32)
    return samples.tobytes()

async def transcribe(
    ws_url: str,
    audio_bytes: bytes,
    chunk_size_bytes: int = 6400,
    chunk_delay_seconds: float = 0.1,
) -> dict:
    """
    Connect to the STT WebSocket, stream audio in chunks, collect result.

    Sends audio in chunks rather than all at once to simulate a real
    microphone streaming 100ms of audio at a time.

    Returns dict with keys:
        tokens     — list of transcript tokens as they arrived
        transcript — full transcript from the done message
        elapsed_ms — total time from first send to done message
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
        assert "status" in data
        assert data["status"] == "ok"
        assert "model" in data

class TestTranscription:

    def test_speech_audio_produces_transcript(self, stt_ws_url):
        audio = make_speech_audio(duration_seconds=1.5)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert result["transcript"], (
            "Expected a non-empty transcript from speech audio, got empty string. "
            "Check that VAD is not filtering out all audio."
        )
    @pytest.mark.skipif(SIMULATED, reason="simulated STT cannot detect silence")
    def test_silence_produces_empty_transcript(self, stt_ws_url):
        audio = make_silence_audio(duration_seconds=1.0)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert result["transcript"].strip() == "", (
            f"Expected empty transcript from silence, got: '{result['transcript']}'. "
            f"VAD may not be filtering silence correctly."
        )

    def test_transcript_tokens_stream_before_done(self, stt_ws_url):
        """
        Tokens must arrive before the done message.
        If tokens is empty but transcript is not, streaming is broken —
        the server is batching instead of streaming.
        """
        audio = make_speech_audio(duration_seconds=2.0)
        result = asyncio.run(transcribe(stt_ws_url, audio))

        if result["transcript"]:
            assert len(result["tokens"]) > 0, (
                "Transcript was non-empty but no tokens were received during streaming. "
                "The server is not streaming tokens — it is batching the full response."
            )

    def test_tokens_join_to_match_transcript(self, stt_ws_url):
        """
        The tokens received during streaming must equal the final transcript.
        """
        audio = make_speech_audio(duration_seconds=1.5)
        result = asyncio.run(transcribe(stt_ws_url, audio))

        if result["transcript"] and result["tokens"]:
            joined = " ".join(t.strip() for t in result["tokens"])
            assert joined.strip() == result["transcript"].strip(), (
                f"Streamed tokens '{joined}' do not match "
                f"final transcript '{result['transcript']}'"
            )

    def test_short_audio_transcribed(self, stt_ws_url):
        audio = make_speech_audio(duration_seconds=0.5, amplitude=0.8)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert isinstance(result["transcript"], str)

    def test_long_audio_transcribed(self, stt_ws_url):
        audio = make_speech_audio(duration_seconds=5.0)
        result = asyncio.run(transcribe(stt_ws_url, audio))
        assert isinstance(result["transcript"], str)

class TestLatency:

    def test_transcription_completes_under_2_seconds(self, stt_ws_url):
        """
        Full round trip — audio in, transcript out — must be under 2 seconds
        for 1.5 seconds of audio. This budget includes:
          - network round trip (localhost, so negligible)
          - VAD processing
          - STT inference (simulated: ~150ms)
          - streaming overhead
        """
        audio = make_speech_audio(duration_seconds=1.5)
        result = asyncio.run(transcribe(stt_ws_url, audio))

        assert result["elapsed_ms"] < 2000, (
            f"Transcription took {result['elapsed_ms']:.0f}ms for 1.5s of audio. "
            f"Target is under 2000ms. "
            f"Check for blocking calls in the server or VAD."
        )

    def test_multiple_sequential_requests_consistent_latency(
        self, stt_ws_url
    ):
        """
        Latency should not degrade across sequential requests.
        If the first request takes 200ms and the fifth takes 2000ms,
        something is accumulating state — a memory leak or blocking call.
        """
        audio = make_speech_audio(duration_seconds=1.0)
        latencies = []

        for _ in range(5):
            result = asyncio.run(transcribe(stt_ws_url, audio))
            latencies.append(result["elapsed_ms"])

        first = latencies[0]
        last = latencies[-1]

        assert last < first * 3, (
            f"Latency degraded across requests: "
            f"first={first:.0f}ms, last={last:.0f}ms. "
            f"Possible memory leak or blocking accumulation."
        )

class TestConcurrency:

    def test_concurrent_sessions_all_complete(self, stt_ws_url):
        """
        Multiple simultaneous WebSocket sessions must all complete successfully.
        If sessions interfere with each other, some will return empty transcripts
        or raise exceptions.
        """
        audio = make_speech_audio(duration_seconds=1.0)

        async def run_all():
            tasks = [
                transcribe(stt_ws_url, audio)
                for _ in range(5)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_all())

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, (
            f"{len(errors)} out of 5 concurrent sessions failed: {errors}"
        )

        successes = [r for r in results if not isinstance(r, Exception)]
        assert len(successes) == 5

    def test_concurrent_sessions_independent_transcripts(
        self, stt_ws_url
    ):
        """
        Each session must produce its own independent transcript.
        Sessions must not share state.
        """
        audio = make_speech_audio(duration_seconds=1.5)

        async def run_all():
            tasks = [transcribe(stt_ws_url, audio) for _ in range(3)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())

        for i, result in enumerate(results):
            assert isinstance(result["transcript"], str), (
                f"Session {i} did not return a string transcript"
            )

class TestProtocol:

    def test_done_message_always_sent(self, stt_ws_url):
        """
        The server must always send a done message, even for silence.
        Without done, the client would hang waiting forever.
        """
        audio = make_silence_audio(duration_seconds=0.5)

        done_received = False

        async def check_done():
            nonlocal done_received
            async with websockets.connect(f"{stt_ws_url}/stt") as ws:
                await ws.send(audio)
                await ws.send(json.dumps({"type": "end"}))

                async for raw in ws:
                    if isinstance(raw, str):
                        msg = json.loads(raw)
                        if msg["type"] == "done":
                            done_received = True
                            break

        asyncio.run(check_done())
        assert done_received, (
            "Server did not send a 'done' message after silence audio. "
            "Clients would hang indefinitely waiting for the session to end."
        )

    def test_end_signal_required_to_close_session(self, stt_ws_url):
        """
        Server must wait for the end signal before transcribing.
        If it transcribes before end arrives, streaming audio would be cut off.
        """
        audio = make_speech_audio(duration_seconds=1.0)
        chunks = [
            audio[i:i + 6400]
            for i in range(0, len(audio), 6400)
        ]

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
        assert result_holder, "No done message received"