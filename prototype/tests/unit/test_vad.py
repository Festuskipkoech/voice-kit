import numpy as np
import pytest
from voicekit.vad import VADProcessor

def make_silence(samples: int = 1600) -> np.ndarray:
    return np.zeros(samples, dtype=np.float32)

def make_speech(samples: int = 1600, amplitude: float = 0.5) -> np.ndarray:
    return np.random.uniform(-amplitude, amplitude, samples).astype(np.float32)

def make_quiet_speech(samples: int = 1600) -> np.ndarray:
    return np.random.uniform(-0.02, 0.02, samples).astype(np.float32)

def test_silence_is_rejected():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.is_speech(make_silence()) is False

def test_loud_speech_is_accepted():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.is_speech(make_speech(amplitude=0.5)) is True

def test_empty_chunk_is_rejected():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.is_speech(np.array([], dtype=np.float32)) is False

def test_high_sensitivity_accepts_quiet_speech():
    vad = VADProcessor(sensitivity=0.9)
    quiet = make_quiet_speech()
    assert vad.is_speech(quiet) is True

def test_low_sensitivity_rejects_quiet_speech():
    vad = VADProcessor(sensitivity=0.1)
    quiet = make_quiet_speech()
    assert vad.is_speech(quiet) is False

def test_invalid_sensitivity_raises_error():
    with pytest.raises(ValueError):
        VADProcessor(sensitivity=1.5)

    with pytest.raises(ValueError):
        VADProcessor(sensitivity=-0.1)

def test_filter_stream_removes_silence():
    vad = VADProcessor(sensitivity=0.5)
    chunks = [
        make_silence(),
        make_speech(),
        make_silence(),
        make_speech(),
        make_silence(),
    ]
    filtered = vad.filter_stream(chunks)
    assert len(filtered) == 2

def test_filter_stream_empty_input():
    vad = VADProcessor(sensitivity=0.5)
    assert vad.filter_stream([]) == []