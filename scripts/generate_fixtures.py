#!/usr/bin/env python3
"""
Generate WAV fixtures for Voicekit unit and integration tests.

Mirrors the Rust ParadeeEngine exactly:
  - G2P via misaki (same library misaki_rs wraps)
  - vocab from models/models/paradee/config.json
  - inference via onnxruntime on models/models/paradee/onnx/paradee_int8.onnx
  - 24kHz output resampled to 16kHz for Moonshine STT

Outputs:
  <out-dir>/hello.wav   - Paradee TTS speech (16kHz mono PCM)
  <out-dir>/silence.wav - pure silence (16kHz mono PCM)

Run from the repo root:
    python3 scripts/generate_fixtures.py

Requires:
    pip install onnxruntime misaki numpy scipy
    make models   (to populate models/models/paradee/)
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

PARADEE_DIR = Path("models/models/paradee")
PARADEE_RATE = 24_000  # model native sample rate
PAD_ID = 0  # BOS/EOS token (matches Rust const PAD_ID)
MAX_LEN = 512  # maximum token sequence length

VOICE_IDS = {
    "af_heart": 0,
    "bf_isabella": 1,
    "am_puck": 2,
    "bm_fable": 3,
}

DEFAULT_VOICE = "af_heart"
DEFAULT_TEXT = "hello world"


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    brate = sample_rate * 2  # 16-bit mono
    data_sz = n * 2

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_sz))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, brate, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_sz))
        clipped = np.clip(samples, -1.0, 1.0)
        f.write((clipped * 32767).astype("<i2").tobytes())

    print(f"wrote {path}  ({n} samples, {n / sample_rate:.3f}s)")


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)
    except ImportError:
        # fallback: linear interpolation
        ratio = src_rate / dst_rate
        out_len = int(len(audio) / ratio)
        idx_f = np.arange(out_len, dtype=np.float64) * ratio
        lo = idx_f.astype(np.int64)
        hi = np.minimum(lo + 1, len(audio) - 1)
        frac = (idx_f - lo).astype(np.float32)
        return audio[lo] * (1.0 - frac) + audio[hi] * frac


def g2p(text: str) -> str:
    try:
        from misaki import en
    except ImportError:
        print("ERROR: misaki not installed.  Run: pip install misaki")
        sys.exit(1)

    g2p_en = en.G2P(trf=False, british=False)
    phonemes, _ = g2p_en(text)
    return phonemes


def load_vocab(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    raw = cfg.get("vocab")
    if not raw:
        raise RuntimeError(f"'vocab' key missing from {config_path}")
    # keys are single-char phoneme symbols; values are integer IDs
    return {k: int(v) for k, v in raw.items() if len(k) == 1}


def text_to_ids(phonemes: str, vocab: dict) -> list:
    ids = [PAD_ID]
    for ch in phonemes:
        if ch in vocab:
            ids.append(vocab[ch])
    ids.append(PAD_ID)
    if len(ids) > MAX_LEN:
        ids = ids[: MAX_LEN - 1] + [PAD_ID]
    return ids


def synthesize(text: str, voice: str, speed: float = 1.0) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.  Run: pip install onnxruntime")
        sys.exit(1)

    onnx_path = PARADEE_DIR / "onnx" / "paradee_int8.onnx"
    config_path = PARADEE_DIR / "config.json"

    for p in (onnx_path, config_path):
        if not p.exists():
            print(f"ERROR: {p} not found — run `make models` first.")
            sys.exit(1)

    voice_id = VOICE_IDS.get(voice)
    if voice_id is None:
        print(f"ERROR: unknown voice '{voice}'.  Valid: {list(VOICE_IDS)}")
        sys.exit(1)

    vocab = load_vocab(config_path)

    phonemes = g2p(text)
    ids = text_to_ids(phonemes, vocab)

    if len(ids) <= 2:  # only BOS+EOS, nothing mapped
        print(f"WARNING: G2P produced no known phonemes for '{text}'")
        print(f"raw phonemes: {phonemes!r}")
        sys.exit(1)

    t = len(ids)
    input_ids = np.array(ids, dtype=np.int64).reshape(1, t)
    spd = np.array([speed], dtype=np.float32)
    vid = np.array([voice_id], dtype=np.int64)

    sess = ort.InferenceSession(str(onnx_path))
    out = sess.run(
        None,
        {
            "input_ids": input_ids,
            "speed": spd,
            "voice": vid,
        },
    )

    # waveform: [1, samples] f32 at PARADEE_RATE
    waveform = out[0].reshape(-1).astype(np.float32)

    # normalise amplitude to 0.9 peak
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.9

    return waveform


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Voicekit test fixtures using the Paradee TTS ONNX model"
    )
    parser.add_argument(
        "--out-dir",
        default="crates/voicekit-core/fixtures",
        help="Output directory (default: crates/voicekit-core/fixtures)",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help=f"Text to synthesise (default: '{DEFAULT_TEXT}')",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=list(VOICE_IDS),
        help=f"Paradee voice (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Output sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Silence duration in seconds (default: 2.0)",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    # --- hello.wav ---
    print(f"Synthesising '{args.text}' with voice '{args.voice}' via Paradee ONNX ...")
    audio_24k = synthesize(args.text, args.voice, args.speed)
    print(
        f"  raw output: {len(audio_24k)} samples @ {PARADEE_RATE}Hz "
        f"({len(audio_24k) / PARADEE_RATE:.3f}s)"
    )

    audio_out = resample(audio_24k, PARADEE_RATE, args.sample_rate)
    audio_out = np.concatenate(
        [audio_out, np.zeros(int(args.sample_rate * args.duration), dtype=np.float32)]
    )
    write_wav(out_dir / "hello.wav", audio_out, args.sample_rate)

    # --- silence.wav ---
    silence = np.zeros(int(args.sample_rate * args.duration), dtype=np.float32)
    write_wav(out_dir / "silence.wav", silence, args.sample_rate)

    print("\nDone.")


if __name__ == "__main__":
    main()
