"""
TTS server integration tests.

Tests the actual running Chatterbox Turbo TTS WebSocket server.
Real Docker container, real WebSocket connection, real bytes.

These tests send text tokens and assert on audio output.
No audio generation on input side — TTS tests are straightforward.

Note on test_first_audio_arrives_before_synthesis_completes:
    This test was skipped in simulation mode because the simulated TTS
    was too slow to beat the token send time. Real Chatterbox Turbo
    starts audio within ~200ms of the first sentence boundary.
    With real models this test must pass.

Run:
    pytest tests/integration/test_tts.py -v

Requirements:
    Docker running. Port 8012 free.
    The tts_service fixture in conftest.py handles container lifecycle.
"""
import asyncio
import io
import json
import time
import wave

import httpx
import websockets

def parse_wav_header(audio_bytes: bytes) -> dict:
    """Parse WAV header and return audio properties."""
    if len(audio_bytes) < 44:
        raise ValueError(
            f"Audio too short to be a WAV file: {len(audio_bytes)} bytes"
        )
    with wave.open(io.BytesIO(audio_bytes)) as wav:
        return {
            "channels": wav.getnchannels(),
            "sample_rate": wav.getframerate(),
            "sample_width": wav.getsampwidth(),
            "num_frames": wav.getnframes(),
            "duration_seconds": wav.getnframes() / wav.getframerate(),
        }

async def synthesize(
    ws_url: str,
    text: str,
    token_delay_seconds: float = 0.05,
) -> dict:
    """
    Connect to TTS WebSocket, stream text tokens, collect audio.

    Sends tokens one word at a time — same as the real pipeline feeds TTS
    from the LLM token stream.

    Returns:
        chunks          — list of raw WAV bytes chunks as received
        audio           — all chunks concatenated
        first_chunk_ms  — ms from first token send to first audio chunk
        total_ms        — total time from first send to done message
        chunk_count     — number of audio chunks received
    """
    chunks = []
    first_chunk_time = None
    start_time = None

    async with websockets.connect(f"{ws_url}/tts") as ws:
        words = text.split()
        start_time = time.perf_counter()

        for word in words:
            await ws.send(json.dumps({
                "type": "token",
                "text": word + " ",
            }))
            await asyncio.sleep(token_delay_seconds)

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
        "chunks": chunks,
        "audio": b"".join(chunks),
        "first_chunk_ms": first_chunk_ms,
        "total_ms": total_ms,
        "chunk_count": len(chunks),
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
        assert data["model"] == "chatterbox-turbo"

class TestAudioOutput:

    def test_text_produces_audio(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(
            tts_ws_url,
            "Hello, how can I help you today?"
        ))
        assert len(result["audio"]) > 0, (
            "TTS produced no audio bytes for valid text input."
        )

    def test_output_is_valid_wav(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(
            tts_ws_url,
            "This is a test of the audio output format."
        ))
        assert len(result["audio"]) > 0, "No audio produced"
        wav_info = parse_wav_header(result["audio"])
        assert wav_info["channels"] == 1, (
            f"Expected mono audio, got {wav_info['channels']} channels."
        )

    def test_output_sample_rate_is_24000(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(
            tts_ws_url,
            "Checking the sample rate of this output."
        ))
        wav_info = parse_wav_header(result["audio"])
        assert wav_info["sample_rate"] == 24000, (
            f"Expected 24000Hz, got {wav_info['sample_rate']}Hz."
        )

    def test_longer_text_produces_more_audio(self, tts_service, tts_ws_url):
        short_result = asyncio.run(synthesize(tts_ws_url, "Hello."))
        long_result = asyncio.run(synthesize(
            tts_ws_url,
            "Hello there. This is a longer sentence with more words "
            "that should produce significantly more audio than the short one."
        ))
        assert len(long_result["audio"]) > len(short_result["audio"]), (
            "Longer text should produce more audio."
        )

    def test_empty_text_produces_no_audio(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(tts_ws_url, ""))
        assert len(result["audio"]) == 0, (
            f"Empty text should produce no audio, got {len(result['audio'])} bytes."
        )

    def test_audio_streams_progressively(self, tts_service, tts_ws_url):
        """
        Audio must arrive in multiple chunks, not one blob.
        This is what allows playback to start before synthesis completes.
        """
        result = asyncio.run(synthesize(
            tts_ws_url,
            "This sentence is long enough to produce multiple audio "
            "chunks during streaming synthesis."
        ))
        assert result["chunk_count"] > 1, (
            f"Expected multiple audio chunks, got {result['chunk_count']}. "
            f"Server may be buffering the full response before sending."
        )

class TestLatency:

    def test_first_audio_chunk_under_5_seconds(self, tts_service, tts_ws_url):
        """
        Time to first audio chunk must be under 5 seconds on CPU.
        Chatterbox Turbo on GPU: ~200ms.
        Chatterbox Turbo on CPU: up to 2-3 seconds per sentence.
        5 second budget accommodates CPU-only servers.
        """
        result = asyncio.run(synthesize(
            tts_ws_url,
            "Hello there. How can I help you?"
        ))
        assert result["first_chunk_ms"] is not None, "No audio chunks received"
        assert result["first_chunk_ms"] < 5000, (
            f"First audio chunk at {result['first_chunk_ms']:.0f}ms. "
            f"Target under 5000ms on CPU."
        )

    def test_first_audio_arrives_before_synthesis_completes(
        self, tts_service, tts_ws_url
    ):
        """
        This test was previously skipped in simulation mode.
        With real Chatterbox Turbo it must pass.

        First audio chunk must arrive before all tokens have been sent.
        This is the definition of streaming TTS — audio generation starts
        on the first sentence boundary while the rest of the text is
        still being streamed in.
        """
        text = (
            "Hello there. This gives Chatterbox time to start streaming "
            "audio before all the remaining tokens have been sent "
            "from the token sender to the TTS server."
        )
        words = text.split()
        # time it takes to send all tokens at 50ms per word
        token_send_time_ms = len(words) * 50

        result = asyncio.run(synthesize(tts_ws_url, text, token_delay_seconds=0.05))

        assert result["first_chunk_ms"] is not None, "No audio produced"
        assert result["first_chunk_ms"] < token_send_time_ms, (
            f"First audio at {result['first_chunk_ms']:.0f}ms, "
            f"all tokens sent by {token_send_time_ms:.0f}ms. "
            f"TTS is buffering — not streaming. "
            f"Check sentence boundary buffering in ChatterboxTTS."
        )

class TestConcurrency:

    def test_concurrent_sessions_all_produce_audio(self, tts_service, tts_ws_url):
        text = "Hello, this is a concurrent synthesis test."

        async def run_all():
            tasks = [synthesize(tts_ws_url, text) for _ in range(3)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_all())

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, (
            f"{len(errors)} out of 3 concurrent sessions failed: {errors}"
        )

        for i, result in enumerate(results):
            assert len(result["audio"]) > 0, (
                f"Session {i} produced no audio in concurrent test"
            )

    def test_concurrent_sessions_independent(self, tts_service, tts_ws_url):
        text = "Each session should produce the same audio independently."

        async def run_all():
            tasks = [synthesize(tts_ws_url, text) for _ in range(3)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())
        sizes = [len(r["audio"]) for r in results]

        assert max(sizes) < min(sizes) * 3, (
            f"Audio sizes vary too much across sessions: {sizes}. "
            f"Sessions may be sharing synthesis state."
        )

class TestProtocol:

    def test_done_message_always_sent(self, tts_service, tts_ws_url):
        """Server must always send done, even for empty input."""
        done_received = False

        async def check():
            nonlocal done_received
            async with websockets.connect(f"{tts_ws_url}/tts") as ws:
                await ws.send(json.dumps({"type": "end"}))
                async for message in ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data["type"] == "done":
                            done_received = True
                            break

        asyncio.run(check())
        assert done_received, (
            "Server did not send done after empty input. "
            "Clients would hang indefinitely."
        )

    def test_sentence_boundary_triggers_audio(self, tts_service, tts_ws_url):
        """
        Audio must start after a sentence boundary (period).
        Validates the sentence-buffering logic in ChatterboxTTS.
        """
        first_chunk_ms = None

        async def check():
            nonlocal first_chunk_ms
            async with websockets.connect(f"{tts_ws_url}/tts") as ws:
                send_time = time.perf_counter()
                await ws.send(json.dumps({"type": "token", "text": "Hello there."}))
                await asyncio.sleep(0.3)
                await ws.send(json.dumps({"type": "end"}))

                async for message in ws:
                    if isinstance(message, bytes) and first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - send_time) * 1000
                    elif isinstance(message, str):
                        data = json.loads(message)
                        if data["type"] == "done":
                            break

        asyncio.run(check())

        assert first_chunk_ms is not None, (
            "No audio produced after sentence with period."
        )
        assert first_chunk_ms < 5000, (
            f"Audio took {first_chunk_ms:.0f}ms after sentence boundary. "
            f"Sentence buffering may not be triggering on periods."
        )