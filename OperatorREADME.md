# Voicekit

A self-hosted, real-time voice assistant server. Clients connect over WebSocket, stream raw PCM audio, and receive synthesised speech back — all processing happens on-device with no third-party API calls.

**Stack:** Silero VAD → Moonshine STT → Gemma 3 1B LLM → Paradee TTS

---

## Quickstart

You only need Docker. No cloning required.

```bash
curl -O https://raw.githubusercontent.com/Festuskipkoech/voice-kit/main/docker-compose.yml
docker compose up
```

On first run:
- `config-init` writes `./voicekit-data/config.toml` with defaults
- `model-init` downloads and verifies all model weights into `./models/` (~5 GB)
- `voicekit-server` starts once models are ready

The server is ready when you see:

```
voicekit-server-1  | INFO voicekit_core::server: all engines loaded
voicekit-server-1  | INFO voicekit_core::server: listening addr=0.0.0.0:8080
```

Subsequent runs skip downloading — models are verified by checksum and reused.

---

## Configuration

Edit `./voicekit-data/config.toml` then restart:

```bash
docker compose restart voicekit-server
```

### `[server]`

| Field | Default | Description |
|---|---|---|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8080` | WebSocket port |
| `max_connections` | `4` | Maximum concurrent sessions |

### `[session_policy]`

Controls what clients are permitted to override per session. All disabled by default.

| Field | Default | Description |
|---|---|---|
| `allow_system_prompt_override` | `false` | Clients may send a custom system prompt |
| `allow_voice_override` | `false` | Clients may select a different TTS voice |
| `allow_vad_override` | `false` | Clients may tune VAD sensitivity |

### `[llm]`

| Field | Default | Description |
|---|---|---|
| `system_prompt` | see config | Assistant persona and instructions |
| `context_size` | `2048` | Token context window |

### `[tts]`

| Field | Default | Description |
|---|---|---|
| `voice` | `af_heart` | Default TTS voice |
| `sample_rate` | `24000` | Output sample rate in Hz |

### `[vad]`

| Field | Default | Description |
|---|---|---|
| `threshold` | `0.5` | Speech probability threshold (0.0–1.0) |
| `silence_ms` | `800` | Silence duration before end-of-utterance |

---

## WebSocket Protocol

Connect to `ws://<host>:8080`.

### Handshake

Send within 5 seconds of connecting or the server closes the connection:

```json
{"type": "session.config"}
```

With optional overrides (only honoured if `session_policy` permits them):

```json
{
  "type": "session.config",
  "system_prompt": "You are a pirate assistant.",
  "voice": "af_sky",
  "vad_threshold": 0.6,
  "vad_silence_ms": 600
}
```

Server responds:

```json
{"type": "session.ready"}
```

### Audio streaming

After `session.ready`, send raw 16-bit PCM audio at **16 kHz mono** as binary WebSocket frames. VAD runs continuously — the server detects speech boundaries automatically.

### Server messages

| Message | Description |
|---|---|
| `{"type":"session.ready"}` | Handshake accepted |
| `{"type":"transcript","text":"..."}` | STT result for the utterance |
| `{"type":"llm_token","token":"..."}` | Streamed LLM token |
| binary frame | TTS audio chunk (16-bit PCM, 24 kHz mono) |
| `{"type":"turn.end"}` | Server finished this response |
| `{"type":"session.error","message":"..."}` | Fatal error — connection closing |

### End session

Close the WebSocket with code `1000` when done.

---

## Port and log level

```bash
VOICEKIT_PORT=9000 docker compose up          # run on a different port
RUST_LOG=debug docker compose up              # verbose logging
```

---

## Updating

```bash
docker compose pull
docker compose up
```

Models are not re-downloaded unless checksums change.

---

## Requirements

- Docker 24+
- ~6 GB disk for model weights
- 8 GB RAM minimum (16 GB recommended)
