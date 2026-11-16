# voicekit-py

Python SDK for the Voicekit voice-agent server.

Voicekit is a low-latency voice pipeline: microphone audio goes in, synthesised speech comes out.
This SDK handles the WebSocket protocol, audio capture, and playback so you can build
voice-enabled applications in a few lines of Python.

## Requirements

- Python 3.12+
- A running Voicekit server (`docker compose up` or `cargo run`)

## Installation

```bash
pip install voicekit-py
# or
uv add voicekit-py
```

## Quickstart

```python
import asyncio
from voicekit import VoicekitClient, SessionConfig
from voicekit.audio import AudioCapture, AudioPlayer

async def main():
    async with VoicekitClient("ws://localhost:8080") as client:
        await client.start_session(SessionConfig())

        async with AudioCapture() as cap, AudioPlayer() as player:
            await client.run(on_audio=player.play, capture=cap)

asyncio.run(main())
```

Speak into your microphone. The server transcribes your words, generates a response, and streams
synthesised audio back to your speakers.

## API

### `VoicekitClient(url, *, open_timeout=10.0)`

Async context manager that owns the WebSocket connection.

```python
async with VoicekitClient("ws://localhost:8080") as client:
    ...
```

| Method | Description |
|---|---|
| `start_session(config)` | Send session config and wait for server acknowledgement. Must be called before sending audio. |
| `send_audio(samples)` | Send one 512-sample float32 frame to the server. |
| `receive_audio()` | Async generator that yields float32 audio frames as they arrive. |
| `run(on_audio, capture)` | Convenience method: runs send and receive loops concurrently. |
| `connected` | `True` if the WebSocket is currently open. |

### `SessionConfig`

Optional overrides sent to the server at session start. Fields left as `None` use the server's
configured defaults.

```python
SessionConfig(
    voice="af_heart",        # TTS voice ID
    system_prompt="...",     # override the LLM system prompt
    vad_threshold=0.5,       # VAD sensitivity (0-1)
    silence_ms=800,          # ms of silence before VAD fires
)
```

The server enforces a policy on which fields can be overridden. Attempting to override a locked
field raises `PolicyError`.

### `AudioCapture` / `AudioPlayer`

Thin wrappers around sounddevice that produce and consume 512-sample float32 frames at the correct
sample rates (16 kHz mic, 24 kHz TTS).

```python
async with AudioCapture() as cap, AudioPlayer() as player:
    async for frame in cap:
        await client.send_audio(frame)
```

## Exceptions

| Exception | When raised |
|---|---|
| `HandshakeError` | Server rejected the session config |
| `PolicyError` | Server policy forbids the requested override (subclass of `HandshakeError`) |
| `ProtocolError` | Server sent an unexpected frame |
| `NotConnectedError` | Method called before connecting |

## Wire protocol

- **Mic audio** (client to server): binary frames, 512 x float32-LE, 16 kHz
- **TTS audio** (server to client): binary frames, float32-LE, 24 kHz
- **Control frames**: JSON text (`session.config`, `session.ready`, `session.error`)

## License

MIT
