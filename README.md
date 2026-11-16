# Voicekit

A production-ready voice-agent engine built in Rust. You wire it to a microphone and a speaker; it handles everything in between — silence detection, speech recognition, language model inference, and speech synthesis — over a single WebSocket connection.

Two non-negotiable design values:

**Flexibility** — every model is behind a trait. Swap STT, LLM, or TTS by changing one line in `config.toml`. No recompile, no code change.

**Latency** — the LLM streams tokens directly into the TTS pipeline. The client hears the first audio chunk before the model finishes generating. Target: < 3 s end-to-end on CPU.

## Measured performance (Ubuntu 24.04, no GPU)

| Stage | Model | Avg latency |
|---|---|---|
| STT | Moonshine TINY (ONNX int8) | ~376 ms |
| LLM | Gemma 3 1B Q4_K_M (GGUF) | ~1 578 ms |
| TTS | Paradee int8 (ONNX) | ~708 ms |
| Total sequential | | ~2 661 ms |
| First audio (streamed) | | ~1 100 ms |

"First audio" = STT + first LLM sentence + TTS of that sentence. LLM and TTS overlap after the first sentence boundary.

## Quick start

```bash
# 1. Pull models (cached to ./models/)
voicekit pull moonshine-tiny paradee-int8 gemma3-1b-q4

# 2. Run the server
docker run -p 8080:8080 \
  -v $(pwd)/models:/models:ro \
  -v $(pwd)/config.toml:/config.toml:ro \
  ghcr.io/voicekit/voicekit-server:latest

# 3. Install the Python SDK
uv add voicekit
# or: pip install voicekit
```

```python
import asyncio
from voicekit import VoicekitClient, SessionConfig
from voicekit.audio import AudioCapture, AudioPlayer

async def main():
    async with VoicekitClient("ws://localhost:8080") as client:
        await client.start_session(SessionConfig(voice="af_heart"))
        async with AudioCapture() as cap, AudioPlayer() as player:
            await client.run(on_audio=player.play, capture=cap)

asyncio.run(main())
```

## Packages

| Package | What it is |
|---|---|
| `voicekit-core` | Rust crate — all inference logic, traits, session loop |
| `voicekit-server` | Binary + Docker image — WebSocket server |
| `voicekit-cli` | Model manager, health check, benchmark runner |
| `voicekit-py` | Python SDK |
| `voicekit-js` | JavaScript/TypeScript SDK |
| `voicekit-go` | Go SDK |

## Documentation

| Document | What it covers |
|---|---|
| `ARCHITECTURE.md` | Package layers, engine traits, config schema, Docker deploy model, latency budget |
| `FLOW.md` | One WebSocket session start to finish — VAD decisions, STT dispatch, LLM streaming, sentence buffer, TTS, edge cases |
| `PROTOCOL.md` | Wire protocol spec — frame types, audio format, handshake, close codes, SDK checklist |
| `DIAGRAM.md` | Visual architecture — pipeline diagram, layer stack, trait system, Docker diagram |

Start with `ARCHITECTURE.md` before writing any Rust. Read `FLOW.md` before touching the session loop.

## Configuration

```toml
[server]
host = "0.0.0.0"
port = 8080

[vad]
threshold       = 0.5
silence_ms      = 800
chunk_samples   = 512

[stt]
provider   = "moonshine"
model_path = "/models/moonshine/tiny"

[llm]
provider   = "llamacpp"
model_path = "/models/gemma3-1b-q4.gguf"
system_prompt = "You are a helpful voice assistant. Keep answers short and conversational."

[tts]
provider    = "paradee"
model_path  = "/models/paradee/onnx/paradee_int8.onnx"
voice       = "af_heart"
sample_rate = 24000
```

## License

MIT
