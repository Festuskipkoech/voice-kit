"""
End-to-end voice pipeline test.

Sends real speech audio to the gateway, receives and saves the
synthesised response as a WAV file you can listen to.

True streaming behaviour:
    Audio chunks arrive progressively as TTS generates them.
    The client collects chunks as they stream in.
    Transcript, response, and metrics arrive after all audio.
    The test breaks on metrics — by then all audio is received.

Usage:
    uv run python tests/test_voice.py

Requirements:
    - Stack running: docker compose -f runtime/docker-compose.yml up
    - Speech fixture: tests/fixtures/speech.raw
    - API key in .env: VOICEKIT_LLM_API_KEY=sk-ant-...

Output:
    tests/results/response.wav  — play with: aplay tests/results/response.wav

Audio format notes:
    Chatterbox streams:
        chunk 1   = complete WAV file (header + float32 PCM data)
        chunk 2..N = raw float32 PCM bytes (no header)
    All chunks are assembled into a single int16 PCM WAV for playback.
"""
import asyncio
import io
import json
import pathlib

import numpy as np
import soundfile as sf
import websockets
from dotenv import load_dotenv

load_dotenv()

GATEWAY_URL = "ws://localhost:8000/session"
FIXTURE_PATH = pathlib.Path("tests/fixtures/speech.raw")
OUTPUT_PATH = pathlib.Path("tests/results/response.wav")
CHUNK_SAMPLES = 1600

async def test():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audio_chunks = []

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
                  "tts = gTTS('hello decsribe, ai', lang='en'); "
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
        print("End of speech sent. Waiting for response...\n")

        async for raw in ws:
            if isinstance(raw, bytes):
                audio_chunks.append(raw)
                print(
                    f"  [audio] chunk {len(audio_chunks)}: "
                    f"{len(raw)} bytes"
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

    if audio_chunks:
        print(f"\nSaving {len(audio_chunks)} audio chunks to WAV...")
        _save_wav(audio_chunks, OUTPUT_PATH)
        print(f"Saved to:  {OUTPUT_PATH}")
        print(f"Play with: aplay {OUTPUT_PATH}")
    else:
        print("\nNo audio received.")

def _save_wav(
    chunks: list[bytes],
    output_path: pathlib.Path,
) -> None:
    """
    Combine streamed audio chunks into a single playable WAV file.

    Chatterbox streams audio in two formats:
        chunk 1   = complete WAV file with header + float32 PCM data
                    soundfile can read this directly
        chunk 2..N = raw float32 PCM bytes with no header
                    read directly as np.float32

    All chunks are combined and written as int16 PCM WAV
    for maximum compatibility with aplay and other players.
    """
    all_audio = []
    sample_rate = 24000

    # chunk 1 — complete WAV file, read with soundfile
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(chunks[0]), dtype="float32"
        )
        if audio.ndim > 1:
            audio = audio[:, 0]   # mono — take first channel
        all_audio.append(audio)
    except Exception as e:
        # if chunk 1 is also raw PCM, fall through to raw handling
        print(f"  Note: chunk 1 treated as raw PCM — {e}")
        try:
            audio = np.frombuffer(chunks[0], dtype=np.float32)
            all_audio.append(audio)
        except Exception as e2:
            print(f"  Warning: skipping chunk 1 — {e2}")

    # chunks 2..N — raw float32 PCM, no WAV header
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

    # convert float32 [-1.0, 1.0] to int16 for aplay compatibility
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