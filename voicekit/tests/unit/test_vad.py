"""
VAD unit tests.

Tests VADProcessor against Silero VAD.

Key facts about Silero that shape these tests:

1. It is a streaming RNN — NOT a stateless per-chunk classifier.
   It builds context across sequential 512-sample windows.
   State must be reset between independent utterances.

2. It does NOT detect speech by amplitude alone.
   Random noise is correctly rejected even at high amplitude.
   Tests must use periodic speech-like signals (harmonics).

3. A single 512-sample window (32ms) may not always be enough context
   for confident detection — longer signals with multiple windows are
   more reliable for testing purposes.

4. The most critical production test is silence rejection.
   Whisper hallucinates on silence — VAD must filter it.
"""

import numpy as np
import pytest

from voicekit.vad import VADProcessor

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1600   # 100ms at 16kHz


def make_silence(samples: int = CHUNK_SAMPLES) -> np.ndarray:
    """Pure silence — all zeros."""
    return np.zeros(samples, dtype=np.float32)


def make_speech_like(
    samples: int = CHUNK_SAMPLES,
    fundamental_hz: float = 120.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a voiced speech-like signal using harmonic sine waves.

    120 Hz fundamental with harmonics up to telephone bandwidth (3400 Hz)
    mimics the periodic structure of voiced speech. Silero was trained on
    real speech and recognises this harmonic structure as speech-like.

    Random noise does NOT have this structure and is correctly rejected.
    """
    t = np.linspace(0, samples / SAMPLE_RATE, samples, endpoint=False)
    signal = np.zeros(samples, dtype=np.float32)

    harmonic = fundamental_hz
    harmonic_amplitude = amplitude
    while harmonic < 3400:
        signal += harmonic_amplitude * np.sin(2 * np.pi * harmonic * t)
        harmonic += fundamental_hz
        harmonic_amplitude *= 0.7

    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * amplitude

    return signal.astype(np.float32)


# ── silence tests ────────────────────────────────────────────────────────────

def test_silence_is_rejected():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.is_speech(make_silence()) is False


def test_empty_chunk_is_rejected():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.is_speech(np.array([], dtype=np.float32)) is False


def test_silence_rejected_at_various_lengths():
    """
    Silence must be rejected at all chunk lengths.
    This is the most critical production test — silence reaching Whisper
    causes hallucinations ("Thank you for watching", etc).
    """
    vad = VADProcessor(sensitivity=0.5)
    for length in [512, 1024, 1600, 3200, 4800]:
        vad.reset_states()
        silence = np.zeros(length, dtype=np.float32)
        assert vad.is_speech(silence) is False, (
            f"Silence of {length} samples incorrectly classified as speech"
        )


# ── speech detection tests ───────────────────────────────────────────────────

def test_speech_like_audio_is_accepted():
    """
    Speech-like audio (harmonic sine waves) must be detected.
    Uses a longer chunk for more reliable RNN context.
    """
    vad = VADProcessor(sensitivity=0.5)
    # use 4800 samples (300ms) for reliable detection
    speech = make_speech_like(samples=4800, amplitude=0.5)
    assert vad.is_speech(speech) is True


def test_louder_speech_detected():
    vad = VADProcessor(sensitivity=0.5)
    speech = make_speech_like(samples=4800, amplitude=0.8)
    assert vad.is_speech(speech) is True


# ── configuration tests ──────────────────────────────────────────────────────

def test_invalid_sensitivity_raises_error():
    with pytest.raises(ValueError):
        VADProcessor(sensitivity=1.5)

    with pytest.raises(ValueError):
        VADProcessor(sensitivity=-0.1)


def test_is_speech_returns_bool():
    """Return type must be bool, never a tensor or float."""
    vad = VADProcessor(sensitivity=0.5)
    result = vad.is_speech(make_silence())
    assert type(result) is bool


# ── reset_states tests ───────────────────────────────────────────────────────

def test_reset_states_does_not_crash():
    """reset_states must be callable without error."""
    vad = VADProcessor(sensitivity=0.5)
    vad.reset_states()   # should not raise


def test_silence_rejected_after_reset():
    """After reset, silence must still be rejected."""
    vad = VADProcessor(sensitivity=0.5)
    vad.reset_states()
    assert vad.is_speech(make_silence()) is False


# ── filter_stream tests ──────────────────────────────────────────────────────

def test_filter_stream_removes_silence():
    vad = VADProcessor(sensitivity=0.5)
    # use very clear silence (zeros) and loud speech-like signal
    chunks = [
        make_silence(samples=4800),
        make_speech_like(samples=4800, amplitude=0.8),
        make_silence(samples=4800),
        make_speech_like(samples=4800, amplitude=0.8),
        make_silence(samples=4800),
    ]
    filtered = vad.filter_stream(chunks)
    vad.reset_states()   # clean up after

    assert len(filtered) >= 1, "No speech chunks detected"
    assert len(filtered) <= 2, "Silence passed through filter"


def test_filter_stream_all_silence():
    vad = VADProcessor(sensitivity=0.5)
    chunks = [make_silence(samples=4800) for _ in range(3)]
    filtered = vad.filter_stream(chunks)
    assert len(filtered) == 0


def test_filter_stream_empty_input():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.filter_stream([]) == []


# ── edge case tests ──────────────────────────────────────────────────────────

def test_chunk_shorter_than_window_handled():
    """
    Chunks shorter than 512 samples must be padded and handled
    without raising an error.
    """
    vad = VADProcessor(sensitivity=0.5)
    short = np.zeros(256, dtype=np.float32)
    result = vad.is_speech(short)
    assert isinstance(result, bool)