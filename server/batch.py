"""Batch-processing business logic: input collection, engine factory,
validation, output filtering, and collision detection.

Shared by both the CLI (cli.py) and the web UI (run_manager.py).
"""

from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path

from audio_utils import filename_contains_lang, find_external_subtitles, langs_match, probe_subtitle_languages
from config import BatchConfig
from mlx_whisper_engine import MLXWhisperEngine, normalize_whisper_language_code
from pipeline import output_path_for
from seamless import SeamlessM4TEngine

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".flv",
    ".wmv",
    ".mp3",
    ".m4a",
    ".wav",
    ".aac",
    ".flac",
}


def create_engine(config: BatchConfig):
    if config.backend == "mlx-whisper":
        if config.device != "auto":
            logger.warning("--device is ignored by the MLX Whisper backend; MLX uses Apple GPU when available")
        return MLXWhisperEngine(
            config.model_id,
            translator_model_id=config.translator_model_id,
            local_only=config.local_only,
        )
    return SeamlessM4TEngine(config.model_id, device=config.device, local_only=config.local_only)


def collect_inputs(patterns: list[str], progress: bool = False) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    scanned = 0
    last_report = time.monotonic() if progress else 0.0
    for pattern in patterns:
        matches = glob.glob(os.path.expanduser(pattern), recursive=True)
        if not matches:
            fallback = Path(pattern).expanduser()
            if not fallback.exists():
                logger.warning("No matches for pattern: %s", pattern)
            candidates = [fallback]
        else:
            candidates = [Path(item) for item in matches]
        for candidate in candidates:
            if candidate.is_dir():
                if progress:
                    print(f"progress\tcollecting\t{len(files)}\tscanning {candidate}", flush=True)
                for child in candidate.rglob("*"):
                    _append_media(files, seen, child)
                    scanned += 1
                    if progress:
                        now = time.monotonic()
                        if now - last_report >= 5:
                            print(f"progress\tcollecting\t{len(files)}\t{scanned} files scanned", flush=True)
                            last_report = now
            else:
                _append_media(files, seen, candidate)
    if progress:
        print(f"progress\tcollecting\t{len(files)}\t{scanned} files scanned", flush=True)
    files.sort(key=lambda p: p.name)
    return files


def _append_media(files: list[Path], seen: set[Path], candidate: Path) -> None:
    if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        return
    resolved = candidate.resolve()
    if resolved not in seen:
        seen.add(resolved)
        files.append(resolved)


def validate_batch_config(config: BatchConfig) -> list[str]:
    errors: list[str] = []
    if config.backend == "seamless" and config.source_lang.lower() in ("auto", ""):
        errors.append("--backend seamless does not support --source-lang auto; specify a language code explicitly")
    if config.backend == "mlx-whisper":
        source_code = normalize_whisper_language_code(config.source_lang)
        target_code = normalize_whisper_language_code(config.target_lang)
        if config.translate and target_code != "en" and not config.translator_model_id:
            if source_code is None or target_code != source_code:
                errors.append("--backend mlx-whisper needs --translator-model for non-English translation")
    if config.chunk_seconds <= 0:
        errors.append("--chunk-seconds must be greater than 0")
    if config.cue_seconds <= 0:
        errors.append("--cue-seconds must be greater than 0")
    if config.max_cue_chars <= 0:
        errors.append("--max-cue-chars must be greater than 0")
    if config.max_cue_sentences < 0:
        errors.append("--max-cue-sentences must be greater than or equal to 0")
    if config.min_chunk_seconds <= 0:
        errors.append("--min-chunk-seconds must be greater than 0")
    if config.min_chunk_seconds > config.chunk_seconds:
        errors.append("--min-chunk-seconds must be less than or equal to --chunk-seconds")
    if config.silence_threshold < 0:
        errors.append("--silence-threshold must be greater than or equal to 0")
    return errors


def filter_existing_outputs(inputs: list[Path], config: BatchConfig, progress: bool = False) -> tuple[list[Path], list[tuple[Path, str]]]:
    if config.overwrite or not config.skip_existing:
        return inputs, []
    total = len(inputs)
    remaining: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for i, input_path in enumerate(inputs):
        if progress:
            print(f"progress\t{i + 1}\t{total}", flush=True)
        reason = _skip_reason(input_path, config)
        if reason:
            skipped.append((input_path, reason))
        else:
            remaining.append(input_path)
    return remaining, skipped


def _skip_reason(input_path: Path, config: BatchConfig) -> str | None:
    output_path = output_path_for(input_path, config)
    if output_path.exists():
        return f"output exists: {output_path}"
    fname_lang = filename_contains_lang(input_path, config.target_lang)
    if fname_lang is not None:
        return f"filename contains target lang: {fname_lang}"
    ext_sub = find_external_subtitles(input_path, config.target_lang)
    if ext_sub is not None:
        return f"external subtitle: {ext_sub.name}"
    embedded_langs = probe_subtitle_languages(input_path)
    for lang in embedded_langs:
        if langs_match(lang, config.target_lang):
            return f"embedded subtitle: {lang}"
    return None


def find_output_collisions(inputs: list[Path], config: BatchConfig) -> dict[Path, list[Path]]:
    outputs: dict[Path, list[Path]] = {}
    for input_path in inputs:
        output_path = output_path_for(input_path, config).resolve()
        outputs.setdefault(output_path, []).append(input_path)
    return {
        output_path: input_paths
        for output_path, input_paths in outputs.items()
        if len(input_paths) > 1
    }
