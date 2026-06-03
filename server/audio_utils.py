from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".sub", ".vtt"}

# Maps variant ISO 639 codes to a canonical form for comparison.
_LANG_CANONICAL: dict[str, str] = {
    "zh": "zho", "chi": "zho", "zho": "zho",
    "en": "eng", "eng": "eng",
    "ja": "jpn", "jpn": "jpn",
    "ko": "kor", "kor": "kor",
    "fr": "fra", "fre": "fra", "fra": "fra",
    "de": "deu", "ger": "deu", "deu": "deu",
    "es": "spa", "spa": "spa",
    "pt": "por", "por": "por",
    "ru": "rus", "rus": "rus",
    "ar": "ara", "ara": "ara",
    "it": "ita", "ita": "ita",
    "nl": "nld", "dut": "nld", "nld": "nld",
    "pl": "pol", "pol": "pol",
    "tr": "tur", "tur": "tur",
    "vi": "vie", "vie": "vie",
    "th": "tha", "tha": "tha",
    "sv": "swe", "swe": "swe",
    "da": "dan", "dan": "dan",
    "fi": "fin", "fin": "fin",
    "no": "nor", "nob": "nor", "nno": "nor", "nor": "nor",
    "uk": "ukr", "ukr": "ukr",
    "cs": "ces", "cze": "ces", "ces": "ces",
    "el": "ell", "gre": "ell", "ell": "ell",
    "he": "heb", "heb": "heb",
    "hi": "hin", "hin": "hin",
    "hu": "hun", "hun": "hun",
    "id": "ind", "ind": "ind",
    "ms": "msa", "may": "msa", "msa": "msa",
    "ro": "ron", "rum": "ron", "ron": "ron",
    "sk": "slk", "slo": "slk", "slk": "slk",
    "bg": "bul", "bul": "bul",
    "hr": "hrv", "hrv": "hrv",
    "sr": "srp", "srp": "srp",
    "lt": "lit", "lit": "lit",
    "lv": "lav", "lav": "lav",
    "et": "est", "est": "est",
    "ca": "cat", "cat": "cat",
    "fa": "fas", "per": "fas", "fas": "fas",
}


def normalize_lang(code: str) -> str:
    return _LANG_CANONICAL.get(code.lower().strip(), code.lower().strip())


def langs_match(a: str, b: str) -> bool:
    return normalize_lang(a) == normalize_lang(b)


def probe_subtitle_languages(video_path: Path) -> list[str]:
    """Return language codes of embedded subtitle streams via ffprobe."""
    if not shutil.which("ffprobe"):
        return []
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-select_streams", "s",
        os.fspath(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("ffprobe failed for %s: %s", video_path, exc)
        return []
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    langs: list[str] = []
    for stream in data.get("streams", []):
        lang = stream.get("tags", {}).get("language", "")
        if lang:
            langs.append(lang)
    return langs


def find_external_subtitles(video_path: Path, target_lang: str) -> Path | None:
    """Return the first sidecar subtitle file matching *target_lang*, or None."""
    stem = video_path.stem
    parent = video_path.parent
    target_norm = normalize_lang(target_lang)
    lang_variants = [c for c, canon in _LANG_CANONICAL.items() if canon == target_norm]
    for ext in SUBTITLE_EXTENSIONS:
        for lang_code in lang_variants:
            candidate = parent / f"{stem}.{lang_code}{ext}"
            if candidate.is_file():
                return candidate
        bare = parent / f"{stem}{ext}"
        if bare.is_file():
            return bare
    return None


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
