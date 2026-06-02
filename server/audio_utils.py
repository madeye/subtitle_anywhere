from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Convert PCM16 little-endian bytes to float32 array in [-1.0, 1.0]."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def validate_audio(data: bytes) -> bool:
    """Check that *data* is valid PCM16 (even number of bytes, non-empty)."""
    if not data:
        return False
    return len(data) % 2 == 0


def load_wav(path: str) -> np.ndarray:
    """Load a WAV file as a float32 numpy array.

    Expects 16 kHz mono PCM16 as produced by `extract_audio`.
    """
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    return samples


def ensure_ffmpeg() -> None:
    """Raise a clear error if FFmpeg is not available on PATH."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to extract audio from local videos")


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract mono 16 kHz PCM WAV audio from a local media file."""
    ensure_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        os.fspath(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        os.fspath(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path
