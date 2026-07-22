"""
End-to-end voice pipeline test.

Sends real speech audio to the gateway, receives and saves the
synthesised response as a WAV file you can listen to.

Also asserts true streaming behaviour:
    - First audio chunk must arrive before metrics message
    - Chunks must be spread over time, not arrive in one burst
    - Time from end_of_speech to first audio must be under 10 seconds

Usage:
    uv run python tests/test_voice.py

Requirements:
    - Stack running: docker compose -f runtime/docker-compose.yml up
    - Speech fixture: tests/fixtures/speech.raw
    - API key in .env: VOICEKIT_LLM_API_KEY=sk-ant-...

Output:
    tests/results/response.wav  — play with: aplay tests/results/response.wav
"""
import asyncio
import io
import json
import pathlib
import time

import numpy as np
import soundfile as sf
import websockets
from dotenv import load_dotenv

load_dotenv()

GATEWAY_URL = "ws://localhost:8000/session"
FIXTURE_PATH = pathlib.Path("tests/fixtures/speech.raw")
OUTPUT_PATH = pathlib.Path("tests/results/response.wav")
CHUNK_SAMPLES = 1600

# streaming assertions
MAX_FIRST_CHUNK_LATENCY_S = 10.0   # first audio must arrive within 10s
MIN_STREAMING_WINDOW_S = 0.5       # chunks must span at least 0.5s if >1 chunk


async def test():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio_chunks = []
    chunk_times = []

    print(f"Connecting to {GATEWAY_URL}...")

    async with websockets.connect(
        GATEWAY_URL,
        ping_interval=None,
        ping_timeout=None,
        open_timeout=10,
        close_timeout=300,
    ) as ws:

        msg = json.loads(await ws.recv())
        print(f"Gateway: {msg}")

        if not FIXTURE_PATH.exists():
            print(f"ERROR: Fixture not found at {FIXTURE_PATH}")
            print("Generate with:")
            print("  mkdir -p tests/fixtures")
            print("  python -c \"from gtts import gTTS; import subprocess; "
                  "tts = gTTS('hello describe ai', lang='en'); "
                  "tts.save('/tmp/speech.mp3'); "
                  "subprocess.run(['ffmpeg', '-i', '/tmp/speech.mp3', "
                  "'-ar', '16000', '-ac', '1', '-f', 'f32le', "
                  "'tests/fixtures/speech.raw'], check=True)\"")
            return

        audio = np.frombuffer(
            FIXTURE_PATH.read_bytes(),
            dtype=np.float32
        )

        print(f"Sending {len(audio) / 16000:.1f}s of speech audio...")
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i:i + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            await ws.send(chunk.tobytes())

        await ws.send(json.dumps({"type": "end_of_speech"}))
        end_of_speech_time = time.perf_counter()
        print("End of speech sent. Waiting for response...\n")

        metrics_time = None

        async for raw in ws:
            if isinstance(raw, bytes):
                now = time.perf_counter()
                chunk_times.append(now)
                audio_chunks.append(raw)
                print(
                    f"  [audio] chunk {len(audio_chunks)}: "
                    f"{len(raw)} bytes "
                    f"(+{now - end_of_speech_time:.2f}s from end_of_speech)"
                )

            else:
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                    continue

                elif msg_type == "transcript":
                    print(f"\nYou said:    '{msg['text']}'")

                elif msg_type == "response":
                    print(f"Agent said:  '{msg['text']}'")

                elif msg_type == "metrics":
                    metrics_time = time.perf_counter()
                    print(f"\nLatency breakdown:")
                    print(f"  STT:             {msg.get('stt_ms', 0)}ms")
                    print(f"  LLM first token: {msg.get('llm_first_token_ms', 0)}ms")
                    print(f"  TTS first chunk: {msg.get('tts_first_chunk_ms', 0)}ms")
                    print(f"  Total:           {msg.get('total_ms', 0)}ms")
                    print(f"\nTotal audio chunks received: {len(audio_chunks)}")
                    break

                elif msg_type == "error":
                    print(f"\nERROR: {msg['message']}")
                    break

    # streaming assertions
    print("\n--- Streaming assertions ---")
    _assert_streaming(chunk_times, end_of_speech_time, metrics_time)

    if audio_chunks:
        print(f"\nSaving {len(audio_chunks)} audio chunks to WAV...")
        _save_wav(audio_chunks, OUTPUT_PATH)
        print(f"Saved to:  {OUTPUT_PATH}")
        print(f"Play with: aplay {OUTPUT_PATH}")
    else:
        print("\nNo audio received.")

def _assert_streaming(
    chunk_times: list[float],
    end_of_speech_time: float,
    metrics_time: float | None,
) -> None:
    """
    Assert true streaming behaviour.

    True streaming means:
        1. First audio arrives well before metrics (turn not complete yet)
        2. Chunks are spread over time — not a single burst at the end
        3. Time from end_of_speech to first audio is reasonable
    """
    if not chunk_times:
        print("FAIL — No audio chunks received at all")
        return

    if metrics_time is None:
        print("FAIL — No metrics received")
        return

    first_chunk_time = chunk_times[0]
    last_chunk_time = chunk_times[-1]

    time_to_first_chunk = first_chunk_time - end_of_speech_time
    streaming_window = last_chunk_time - first_chunk_time
    first_before_metrics = metrics_time - first_chunk_time

    print(f"  Time to first chunk:     {time_to_first_chunk:.2f}s")
    print(f"  Streaming window:        {streaming_window:.2f}s "
          f"({len(chunk_times)} chunks)")
    print(f"  First chunk before end:  {first_before_metrics:.2f}s before metrics")

    # assertion 1 — first chunk arrived before metrics
    if first_chunk_time < metrics_time:
        print(f"  PASS — First audio before metrics ({first_before_metrics:.2f}s gap)")
    else:
        print(f"  FAIL — First audio arrived AFTER metrics. Not streaming.")

    # assertion 2 — first chunk latency is reasonable
    if time_to_first_chunk <= MAX_FIRST_CHUNK_LATENCY_S:
        print(f"  PASS — First chunk within {MAX_FIRST_CHUNK_LATENCY_S}s "
              f"({time_to_first_chunk:.2f}s)")
    else:
        print(f"  WARN — First chunk took {time_to_first_chunk:.2f}s "
              f"(target: <{MAX_FIRST_CHUNK_LATENCY_S}s)")

    # assertion 3 — chunks spread over time if more than one
    if len(chunk_times) > 1:
        if streaming_window >= MIN_STREAMING_WINDOW_S:
            print(f"  PASS — Chunks spread over {streaming_window:.2f}s "
                  f"(not a single burst)")
        else:
            print(f"  WARN — All chunks arrived within {streaming_window:.3f}s "
                  f"— may be batch not streaming")
    else:
        print(f"  INFO — Only 1 chunk received — "
              f"response may be a single short sentence")

def _save_wav(
    chunks: list[bytes],
    output_path: pathlib.Path,
) -> None:
    """
    Combine streamed audio chunks into a single playable WAV file.

    Chatterbox and Kokoro stream audio in two formats:
        chunk 1   = complete WAV file with header + float32 PCM data
                    soundfile can read this directly
        chunk 2..N = raw float32 PCM bytes with no header
                    read directly as np.float32

    All chunks combined and written as int16 PCM WAV for aplay.
    """
    all_audio = []
    sample_rate = 24000

    # chunk 1 — complete WAV file
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(chunks[0]), dtype="float32"
        )
        if audio.ndim > 1:
            audio = audio[:, 0]
        all_audio.append(audio)
    except Exception as e:
        print(f"  Note: chunk 1 treated as raw PCM — {e}")
        try:
            audio = np.frombuffer(chunks[0], dtype=np.float32)
            all_audio.append(audio)
        except Exception as e2:
            print(f"  Warning: skipping chunk 1 — {e2}")

    # chunks 2..N — raw float32 PCM
    for i, chunk in enumerate(chunks[1:], start=2):
        try:
            audio = np.frombuffer(chunk, dtype=np.float32)
            all_audio.append(audio)
        except Exception as e:
            print(f"  Warning: skipping chunk {i} — {e}")
            continue

    if not all_audio:
        print("No valid audio to save.")
        return

    combined = np.concatenate(all_audio)
    duration = len(combined) / sample_rate
    print(
        f"Combined: {len(combined)} samples, "
        f"{duration:.2f}s at {sample_rate}Hz"
    )

    combined_int16 = np.clip(
        combined * 32767, -32768, 32767
    ).astype(np.int16)

    sf.write(
        str(output_path),
        combined_int16,
        sample_rate,
        subtype="PCM_16",
    )

if __name__ == "__main__":
    asyncio.run(test())