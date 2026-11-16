"""
Unit tests for the protocol codec - no server required.
"""

import json

import numpy as np
import pytest

from voicekit.protocol import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SessionConfigFrame,
    SessionErrorFrame,
    SessionReadyFrame,
    decode_audio,
    decode_server_text,
    encode_audio,
)


class TestSessionConfigFrame:
    def test_minimal_encode(self):
        raw = SessionConfigFrame().encode()
        obj = json.loads(raw)
        assert obj == {"type": "session.config"}

    def test_full_encode(self):
        cfg = SessionConfigFrame(
            system_prompt="You are a pirate.",
            voice="bf_isabella",
            vad_threshold=0.4,
            silence_ms=700,
        )
        obj = json.loads(cfg.encode())
        assert obj["type"] == "session.config"
        assert obj["system_prompt"] == "You are a pirate."
        assert obj["voice"] == "bf_isabella"
        assert obj["vad_threshold"] == pytest.approx(0.4)
        assert obj["silence_ms"] == 700

    def test_partial_encode_omits_none(self):
        obj = json.loads(SessionConfigFrame(voice="am_puck").encode())
        assert "voice" in obj
        assert "system_prompt" not in obj
        assert "vad_threshold" not in obj
        assert "silence_ms" not in obj


class TestDecodeServerText:
    def test_session_ready(self):
        frame = decode_server_text('{"type":"session.ready"}')
        assert isinstance(frame, SessionReadyFrame)

    def test_session_error(self):
        frame = decode_server_text(
            '{"type":"session.error","message":"voice override not permitted"}'
        )
        assert isinstance(frame, SessionErrorFrame)
        assert frame.message == "voice override not permitted"

    def test_session_error_no_message(self):
        frame = decode_server_text('{"type":"session.error"}')
        assert isinstance(frame, SessionErrorFrame)
        assert "(no message)" in frame.message

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown server frame type"):
            decode_server_text('{"type":"unknown.thing"}')

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="malformed JSON"):
            decode_server_text("{bad json}")


class TestAudioCodec:
    def test_encode_roundtrip(self):
        samples = np.random.uniform(-1.0, 1.0, FRAME_SAMPLES).astype(np.float32)
        wire = encode_audio(samples)
        assert len(wire) == FRAME_BYTES
        decoded = decode_audio(wire)
        np.testing.assert_array_almost_equal(samples, decoded)

    def test_encode_wrong_size_raises(self):
        with pytest.raises(ValueError, match="exactly 512"):
            encode_audio(np.zeros(100, dtype=np.float32))

    def test_decode_multi_frame(self):
        # server may send multiple frames concatenated
        n = 3
        wire = bytes(FRAME_BYTES * n)
        decoded = decode_audio(wire)
        assert decoded.shape == (FRAME_SAMPLES * n,)

    def test_encode_accepts_list(self):
        samples = [0.0] * FRAME_SAMPLES
        wire = encode_audio(samples)
        assert len(wire) == FRAME_BYTES
