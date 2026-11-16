# Voicekit Wire Protocol - v1

This document is the complete contract between a Voicekit server and any client
or SDK. Implement against this spec; do not read server source code to infer
behaviour.

---

## Overview

Voicekit uses a single persistent WebSocket connection per session. The
protocol has two frame types:

| Frame type | Direction | Content |
|---|---|---|
| Text | both | JSON control frames |
| Binary | both | Raw PCM audio |

Binary is **only** audio. Text is **only** control. Never send audio as text or
control as binary.

---

## Connection

```
ws://host:port
```

No path, no subprotocol header required. Connect and immediately send the
handshake - do not send audio before `session.ready` is received.

TLS: use `wss://` in production. The server accepts both; TLS termination is
handled by a reverse proxy in front of the server.

---

## Session Lifecycle

```
Client                            Server
  |                                 |
  |──── TCP + WebSocket upgrade ───►|
  |                                 |
  |──── session.config (text) ─────►|  must arrive within 5 s
  |                                 |
  |◄─── session.ready  (text) ──────|  or session.error → connection closed
  |                                 |
  |──── binary audio frames ───────►|  512 samples per frame, continuously
  |◄─── binary audio frames ────────|  TTS output, one or more per utterance
  |                                 |
  |     (repeat for each turn)      |
  |                                 |
  |──── Close frame ───────────────►|  client ends session
  |◄─── Close frame ────────────────|
```

Idle timeout: if no audio frame arrives for **45 seconds** after `session.ready`,
the server closes the connection with code `1001`.

Heartbeat: the server sends a Ping frame every **15 seconds**. The client must
reply with a Pong. No Pong within the next Ping interval = server closes with
`1001`.

---

## Handshake

### Client → Server: `session.config`

Must be the first frame sent after the WebSocket upgrade. Must be a JSON text
frame.

```json
{
  "type": "session.config",
  "system_prompt": "You are a helpful assistant.",
  "voice": "af_heart",
  "vad_threshold": 0.5,
  "silence_ms": 600
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `type` | string | yes | Must be `"session.config"` |
| `system_prompt` | string | no | Overrides server default system prompt |
| `voice` | string | no | TTS voice ID to use for this session |
| `vad_threshold` | float | no | VAD probability threshold (0.0–1.0) |
| `silence_ms` | integer | no | Silence duration before utterance dispatch (ms) |

Omitted optional fields fall back to server config defaults.

Override fields that the server policy forbids (see `voicekit.toml`
`[session_policy]`) cause a `session.error` and the connection is closed.

### Server → Client: `session.ready`

```json
{
  "type": "session.ready"
}
```

The session is now active. Begin streaming audio.

### Server → Client: `session.error`

```json
{
  "type": "session.error",
  "message": "human-readable reason"
}
```

Sent before the server closes the connection. The connection is always closed
immediately after this frame. Do not send audio after receiving `session.error`.

---

## Audio Format

The two directions use different sample rates.

| Direction | Encoding | Sample rate | Channels | Frame size |
|---|---|---|---|---|
| Client → Server (mic) | f32-LE | 16 000 Hz | 1 (mono) | exactly 512 samples = 2 048 bytes |
| Server → Client (TTS) | f32-LE | 24 000 Hz | 1 (mono) | variable length |

**Client → Server:** stream frames continuously while the user speaks. Send
exactly 512 samples per frame at 16 kHz. Frames with the wrong number of
samples are discarded with a server-side warning.

**Server → Client:** one or more binary frames per utterance, sent as TTS
finishes each sentence. Each frame is 24 kHz f32-LE PCM. Length varies by
sentence. The client plays them in the order received.

### Byte layout

Each frame is a flat array of 32-bit floats in little-endian byte order:

```
bytes 0–3   → sample[0]  (f32-LE)
bytes 4–7   → sample[1]  (f32-LE)
...
bytes 2044–2047 → sample[511] (f32-LE)
```

Python example:

```python
import struct
samples = struct.pack('<512f', *pcm_floats)   # send
pcm_floats = struct.unpack('<512f', frame)    # receive
```

---

## Control Frame Reference

All control frames are JSON text frames with a `type` field.

### Frames sent by the client

| `type` | When | Notes |
|---|---|---|
| `session.config` | First frame only | Required; see Handshake |

### Frames sent by the server

| `type` | When | Notes |
|---|---|---|
| `session.ready` | After valid handshake | Session is now active |
| `session.error` | On any failure | Connection closed immediately after |

---

## Close Codes

| Code | Meaning |
|---|---|
| `1000` | Normal closure - client sent Close frame |
| `1001` | Going away - idle timeout or server shutdown |
| `1002` | Protocol error - malformed frame or wrong frame type |
| `1008` | Policy violation - override denied by session policy |
| `4000`–`4999` | Reserved for future Voicekit application codes |

Always handle `1001` with a reconnect attempt. Treat `1008` as a configuration
error; do not reconnect without fixing the override field.

---

## Available Voices

Available voices (configure via `voice` in `session.config`):

| ID | Description |
|---|---|
| `af_heart` | Female, American English |
| `bf_isabella` | Female, British English |
| `am_puck` | Male, American English |
| `bm_fable` | Male, British English |

Pass the ID in `session.config` as `"voice"`. The server rejects unknown voice
IDs with `session.error`.

---

## Worked Example

A complete single-turn session showing every frame in order.

```
1. TCP connect to ws://localhost:8080

2. CLIENT TEXT →
   {"type":"session.config","voice":"af_heart"}

3. SERVER TEXT ←
   {"type":"session.ready"}

4. CLIENT BINARY → [2048 bytes: 512 × f32-LE samples]
   CLIENT BINARY → [2048 bytes]
   CLIENT BINARY → [2048 bytes]
   ...  (user speaking; VAD accumulates until silence detected)

5. SERVER BINARY ← [N bytes: TTS audio for first sentence]
   SERVER BINARY ← [N bytes: TTS audio for second sentence]
   ...  (streamed as each sentence is synthesised)

6. CLIENT BINARY → [more audio frames if user continues speaking]

7. CLIENT CLOSE → code 1000
   SERVER CLOSE ← code 1000
```

---

## Error Scenarios

### Handshake timeout

Client connects but sends nothing for 5 seconds:

```
SERVER TEXT ← {"type":"session.error","message":"handshake timeout: no session.config received"}
SERVER CLOSE ← code 1002
```

### Wrong first frame type

Client sends binary before `session.config`:

```
SERVER TEXT ← {"type":"session.error","message":"expected a JSON text frame for session.config, got binary or other"}
SERVER CLOSE ← code 1002
```

### Policy violation

Server has `allow_voice_override = false` and client sends `"voice"`:

```
SERVER TEXT ← {"type":"session.error","message":"voice override is not permitted on this server"}
SERVER CLOSE ← code 1008
```

### Unknown voice

```
SERVER TEXT ← {"type":"session.error","message":"unknown voice: xyz"}
SERVER CLOSE ← code 1002
```

---

## SDK Implementation Checklist

- [ ] Connect, send `session.config`, wait for `session.ready` before streaming
- [ ] Send exactly 512 samples per binary frame
- [ ] Handle `session.error` - log `message`, do not retry immediately
- [ ] Reply to Ping frames with Pong (most WebSocket libraries do this automatically)
- [ ] Handle close code `1001` with exponential-backoff reconnect
- [ ] Handle close code `1008` as a configuration error - do not reconnect
- [ ] Play received binary audio frames in order without buffering delays
- [ ] Send Close frame when the user ends the session
