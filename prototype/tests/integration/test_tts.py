"""
TTS server integration tests.

Tests the actual running TTS WebSocket server — not the Python class,
not a mock. Real Docker container, real WebSocket connection, real bytes.

These tests run against the simulated TTS provider, which means they
validate the server protocol, streaming behaviour, WAV output format,
and latency characteristics without needing a GPU or model files.

When Chatterbox Turbo is added later, these same tests run against the
real model with zero changes — only the Docker image changes.

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
    """
    Parse a WAV file header and return audio properties.
    Raises if the bytes are not a valid WAV file.
    """
    if len(audio_bytes) < 44:
        raise ValueError(
            f"Audio too short to be a WAV file: {len(audio_bytes)} bytes"
        )

    try:
        with wave.open(io.BytesIO(audio_bytes)) as wav:
            return {
                "channels": wav.getnchannels(),
                "sample_rate": wav.getframerate(),
                "sample_width": wav.getsampwidth(),
                "num_frames": wav.getnframes(),
                "duration_seconds": wav.getnframes() / wav.getframerate(),
            }
    except wave.Error as e:
        raise ValueError(f"Invalid WAV data: {e}")

async def synthesize(
    ws_url: str,
    text: str,
    token_delay_seconds: float = 0.05,
) -> dict:
    """
    Connect to the TTS WebSocket, stream text tokens, collect audio chunks.

    Sends tokens one word at a time to simulate LLM streaming behaviour.
    This is how the real pipeline feeds TTS — not a full sentence at once.

    Returns dict with keys:
        chunks         — list of raw audio bytes chunks as they arrived
        audio          — all chunks concatenated
        first_chunk_ms — time from first token send to first audio chunk
        total_ms       — total time from first send to done message
        chunk_count    — number of audio chunks received
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
            f"Expected mono audio (1 channel), got {wav_info['channels']} channels."
        )

    def test_output_sample_rate_is_correct(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(
            tts_ws_url,
            "Checking the sample rate of this audio output."
        ))

        wav_info = parse_wav_header(result["audio"])
        assert wav_info["sample_rate"] == 24000, (
            f"Expected 24000Hz sample rate, got {wav_info['sample_rate']}Hz. "
            f"Downstream audio playback depends on this being correct."
        )

    def test_longer_text_produces_more_audio(self, tts_service, tts_ws_url):
        short_result = asyncio.run(synthesize(tts_ws_url, "Hello."))
        long_result = asyncio.run(synthesize(
            tts_ws_url,
            "Hello there. This is a much longer sentence that should produce "
            "significantly more audio than the short one did."
        ))

        assert len(long_result["audio"]) > len(short_result["audio"]), (
            "Longer text should produce more audio bytes than shorter text."
        )

    def test_empty_text_produces_no_audio(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(tts_ws_url, ""))
        assert len(result["audio"]) == 0, (
            f"Empty text should produce no audio, "
            f"got {len(result['audio'])} bytes."
        )

    def test_audio_chunks_stream_progressively(self, tts_service, tts_ws_url):
        """
        Audio must arrive in multiple chunks, not one large blob.
        Progressive streaming is what allows audio playback to start
        before synthesis is complete.
        """
        result = asyncio.run(synthesize(
            tts_ws_url,
            "This sentence is long enough to produce multiple audio chunks "
            "during streaming synthesis."
        ))

        assert result["chunk_count"] > 1, (
            f"Expected multiple audio chunks for streaming, "
            f"got {result['chunk_count']}. "
            f"Server may be buffering entire response before sending."
        )

class TestLatency:

    def test_first_audio_chunk_under_500ms(self, tts_service, tts_ws_url):
        """
        Time from sending the first token to receiving the first audio chunk
        must be under 500ms. This is the latency the user perceives as
        'how long before the agent starts speaking'.

        Simulated TTS targets ~80ms. Real Chatterbox Turbo targets ~75ms.
        We use 500ms here to give Docker networking and startup overhead
        reasonable headroom while still catching regressions.
        """
        result = asyncio.run(synthesize(
            tts_ws_url,
            "Hello there. How can I help you?"
        ))

        assert result["first_chunk_ms"] is not None, "No audio chunks received"
        assert result["first_chunk_ms"] < 500, (
            f"First audio chunk arrived at {result['first_chunk_ms']:.0f}ms. "
            f"Target is under 500ms. "
            f"This is the latency the user hears as response delay."
        )

    def test_synthesis_completes_under_3_seconds(self, tts_service, tts_ws_url):
        result = asyncio.run(synthesize(
            tts_ws_url,
            "This is a moderately long response that tests total synthesis time."
        ))

        assert result["total_ms"] < 3000, (
            f"TTS took {result['total_ms']:.0f}ms total. "
            f"Target is under 3000ms."
        )

    def test_first_audio_arrives_before_synthesis_completes(
        self, tts_ws_url
    ):
        """
        This is the most important latency test.

        First audio chunk must arrive before all text tokens have been sent.
        That is what 'streaming TTS' means — audio starts while text is
        still arriving. If first_chunk_ms > total_token_send_time,
        the server is buffering everything before generating audio.
        """
        text = (
            "This is a longer response to give TTS time to start streaming "
            "audio before all the text tokens have been sent to the server."
        )
        words = text.split()
        token_send_time_ms = len(words) * 50

        result = asyncio.run(synthesize(tts_ws_url, text, token_delay_seconds=0.05))

        assert result["first_chunk_ms"] is not None, "No audio produced"
        assert result["first_chunk_ms"] < token_send_time_ms, (
            f"First audio chunk arrived at {result['first_chunk_ms']:.0f}ms, "
            f"but all tokens were sent by {token_send_time_ms:.0f}ms. "
            f"TTS is not streaming — it is waiting for all text before generating audio."
        )

class TestConcurrency:

    def test_concurrent_sessions_all_produce_audio(self, tts_ws_url):
        """
        Multiple simultaneous TTS sessions must all complete successfully
        and all produce audio. Sessions must not interfere with each other.
        """
        text = "Hello, this is a concurrent synthesis test."

        async def run_all():
            tasks = [synthesize(tts_ws_url, text) for _ in range(5)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_all())

        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, (
            f"{len(errors)} out of 5 concurrent TTS sessions failed: {errors}"
        )

        for i, result in enumerate(results):
            assert len(result["audio"]) > 0, (
                f"Session {i} produced no audio in concurrent test"
            )

    def test_concurrent_sessions_independent_audio(self, tts_ws_url):
        """
        Each session must produce roughly the same amount of audio for
        the same text — proving sessions are not sharing synthesis state.
        """
        text = "Each session should produce the same audio independently."

        async def run_all():
            tasks = [synthesize(tts_ws_url, text) for _ in range(3)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())

        sizes = [len(r["audio"]) for r in results]
        min_size = min(sizes)
        max_size = max(sizes)

        assert max_size < min_size * 2, (
            f"Audio sizes vary too much across sessions: {sizes}. "
            f"Sessions may be sharing synthesis state."
        )

class TestProtocol:

    def test_done_message_always_sent(self, tts_ws_url):
        """
        The server must always send a done message, even for empty input.
        Without done, the client would hang waiting forever.
        """
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
            "Server did not send a 'done' message after empty input. "
            "Clients would hang indefinitely."
        )

    def test_sentence_boundary_triggers_audio(self, tts_ws_url):
        """
        Audio should start arriving after a sentence-ending punctuation mark.
        This validates the sentence-buffering logic in the TTS provider.
        """
        first_chunk_after_period = None

        async def check():
            nonlocal first_chunk_after_period
            async with websockets.connect(f"{tts_ws_url}/tts") as ws:
                send_time = time.perf_counter()

                await ws.send(json.dumps({"type": "token", "text": "Hello there."}))
                await asyncio.sleep(0.3)

                await ws.send(json.dumps({"type": "end"}))

                async for message in ws:
                    if isinstance(message, bytes) and first_chunk_after_period is None:
                        first_chunk_after_period = (
                            time.perf_counter() - send_time
                        ) * 1000
                    elif isinstance(message, str):
                        data = json.loads(message)
                        if data["type"] == "done":
                            break

        asyncio.run(check())

        assert first_chunk_after_period is not None, (
            "No audio produced after sentence with period."
        )
        assert first_chunk_after_period < 1000, (
            f"Audio took {first_chunk_after_period:.0f}ms after sentence boundary. "
            f"Sentence-buffering logic may not be triggering on periods."
        )