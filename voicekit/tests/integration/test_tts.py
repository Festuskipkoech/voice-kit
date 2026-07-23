"""
TTS server integration tests.

Tests the Kokoro TTS WebSocket server running in Docker.
Real container, real WebSocket connection, real audio bytes.

Protocol under test:
    client -> server  text: {"type": "phrase", "text": "complete phrase"}
    client -> server  text: {"type": "end"}
    server -> client  binary: WAV chunk (first has header)
    server -> client  binary: raw float32 PCM chunks (subsequent)
    server -> client  text:   {"type": "done"}
    server -> client  text:   {"type": "error", "message": "..."}

Run:
    pytest tests/integration/test_tts.py -v

Requirements:
    Docker running. Port 8012 free.
    tts_service fixture in conftest.py handles container lifecycle.
"""
import asyncio
import io
import json
import time

import httpx
import pytest
import soundfile as sf
import websockets

def read_wav_info(audio_bytes: bytes) -> dict:
    """Read WAV header info from first chunk using soundfile."""
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    return {
        "sample_rate": sample_rate,
        "channels":    1 if data.ndim == 1 else data.shape[1],
        "frames":      len(data),
        "duration_s":  len(data) / sample_rate,
    }

async def synthesize(ws_url: str, phrase: str) -> dict:
    """
    Connect to TTS WebSocket, send one phrase, collect audio chunks.

    Returns:
        chunks          list of raw bytes as received
        audio           all chunks concatenated
        first_chunk_ms  ms from send to first audio chunk
        total_ms        total time from send to done message
        chunk_count     number of audio chunks received
    """
    chunks           = []
    first_chunk_time = None
    start_time       = None

    async with websockets.connect(f"{ws_url}/tts") as ws:
        start_time = time.perf_counter()

        await ws.send(json.dumps({"type": "phrase", "text": phrase}))
        await ws.send(json.dumps({"type": "end"}))

        async for message in ws:
            if isinstance(message, bytes):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                chunks.append(message)

            elif isinstance(message, str):
                data = json.loads(message)
                if data["type"] == "done":
                    break
                elif data["type"] == "error":
                    raise RuntimeError(f"TTS server error: {data['message']}")

    total_ms = (time.perf_counter() - start_time) * 1000
    first_chunk_ms = (
        (first_chunk_time - start_time) * 1000
        if first_chunk_time else None
    )

    return {
        "chunks":        chunks,
        "audio":         b"".join(chunks),
        "first_chunk_ms": first_chunk_ms,
        "total_ms":      total_ms,
        "chunk_count":   len(chunks),
    }

class TestHealth:
    def test_health_endpoint_returns_200(self, tts_service):
        response = httpx.get(f"{tts_service}/health", timeout=5.0)
        assert response.status_code == 200

    def test_health_response_contains_model_info(self, tts_service):
        response = httpx.get(f"{tts_service}/health", timeout=5.0)
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data
        assert data["model"] == "kokoro"
        assert "voice" in data


class TestAudioOutput:
    @pytest.mark.asyncio
    async def test_phrase_produces_audio(self, tts_service, tts_ws_url):
        result = await synthesize(tts_ws_url, "Hello, how can I help you today?")
        assert len(result["audio"]) > 0, "TTS produced no audio for valid phrase"

    @pytest.mark.asyncio
    async def test_first_chunk_is_valid_wav(self, tts_service, tts_ws_url):
        result = await synthesize(tts_ws_url, "This is a test of the audio output format.")
        assert result["chunks"], "No audio chunks received"
        info = read_wav_info(result["chunks"][0])
        assert info["sample_rate"] == 24000, (
            f"Expected 24000Hz, got {info['sample_rate']}Hz"
        )
        assert info["channels"] == 1, (
            f"Expected mono, got {info['channels']} channels"
        )

    @pytest.mark.asyncio
    async def test_longer_phrase_produces_more_audio(self, tts_service, tts_ws_url):
        short  = await synthesize(tts_ws_url, "Hello.")
        long   = await synthesize(
            tts_ws_url,
            "Hello there. This is a longer sentence with more words "
            "that should produce significantly more audio than the short one."
        )
        assert len(long["audio"]) > len(short["audio"]), (
            "Longer phrase should produce more audio"
        )

    @pytest.mark.asyncio
    async def test_empty_phrase_returns_error(self, tts_service, tts_ws_url):
        """
        Empty phrase must return an error message, not hang.
        The server validates the phrase is non-empty before synthesising.
        """
        error_received = False
        async with websockets.connect(f"{tts_ws_url}/tts") as ws:
            await ws.send(json.dumps({"type": "phrase", "text": ""}))
            await ws.send(json.dumps({"type": "end"}))
            async for message in ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data["type"] == "error":
                        error_received = True
                        break
                    elif data["type"] == "done":
                        break
        assert error_received, "Empty phrase should return an error"

    @pytest.mark.asyncio
    async def test_audio_arrives_in_chunks(self, tts_service, tts_ws_url):
        """Audio must arrive in multiple chunks, not one blob."""
        result = await synthesize(
            tts_ws_url,
            "This sentence is long enough to produce multiple audio "
            "chunks during streaming synthesis with Kokoro."
        )
        assert result["chunk_count"] > 1, (
            f"Expected multiple chunks, got {result['chunk_count']}. "
            f"Server may be buffering the full response before sending."
        )

class TestLatency:
    @pytest.mark.asyncio
    async def test_first_audio_chunk_under_5_seconds(self, tts_service, tts_ws_url):
        """
        Kokoro on CPU: 500-800ms to first audio chunk.
        5 second budget is generous — if this fails, the model is not loaded.
        """
        result = await synthesize(tts_ws_url, "Hello there. How can I help you?")
        assert result["first_chunk_ms"] is not None, "No audio chunks received"
        assert result["first_chunk_ms"] < 5000, (
            f"First audio chunk at {result['first_chunk_ms']:.0f}ms. "
            f"Target under 5000ms."
        )

class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_sessions_all_produce_audio(self, tts_service, tts_ws_url):
        phrase = "Hello, this is a concurrent synthesis test."
        results = await asyncio.gather(
            *[synthesize(tts_ws_url, phrase) for _ in range(3)],
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, (
            f"{len(errors)} of 3 concurrent sessions failed: {errors}"
        )
        for i, result in enumerate(results):
            assert len(result["audio"]) > 0, (
                f"Session {i} produced no audio in concurrent test"
            )

    @pytest.mark.asyncio
    async def test_concurrent_sessions_independent(self, tts_service, tts_ws_url):
        phrase  = "Each session should produce the same audio independently."
        results = await asyncio.gather(
            *[synthesize(tts_ws_url, phrase) for _ in range(3)]
        )
        sizes = [len(r["audio"]) for r in results]
        assert max(sizes) < min(sizes) * 3, (
            f"Audio sizes vary too much across sessions: {sizes}"
        )

class TestProtocol:
    @pytest.mark.asyncio
    async def test_done_always_sent_after_phrase(self, tts_service, tts_ws_url):
        """Server must always send done after successful synthesis."""
        done_received = False
        async with websockets.connect(f"{tts_ws_url}/tts") as ws:
            await ws.send(json.dumps({"type": "phrase", "text": "Hello."}))
            await ws.send(json.dumps({"type": "end"}))
            async for message in ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data["type"] == "done":
                        done_received = True
                        break
        assert done_received, "Server did not send done after synthesis"

    @pytest.mark.asyncio
    async def test_missing_phrase_returns_error(self, tts_service, tts_ws_url):
        """
        Sending only end with no phrase must return an error.
        Clients should not hang waiting for audio that will never come.
        """
        error_received = False
        async with websockets.connect(f"{tts_ws_url}/tts") as ws:
            await ws.send(json.dumps({"type": "end"}))
            async for message in ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    if data["type"] == "error":
                        error_received = True
                        break
                    elif data["type"] == "done":
                        break
        assert error_received, (
            "Server did not return error when no phrase was sent"
        )