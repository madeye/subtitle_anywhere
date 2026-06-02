"""Batch local-video subtitle pipeline."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_utils import extract_audio, load_wav
from config import BatchConfig
from seamless import SeamlessM4TEngine, Segment

logger = logging.getLogger(__name__)


@dataclass
class SubtitleResult:
    input_path: Path
    output_path: Path
    segment_count: int


class SubtitlePipeline:
    """Generate bilingual external subtitles for local media files."""

    def __init__(
        self,
        engine: SeamlessM4TEngine,
        config: BatchConfig,
    ) -> None:
        self.engine = engine
        self.config = config

    def process_file(self, input_path: Path) -> SubtitleResult:
        input_path = input_path.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

        output_path = output_path_for(input_path, self.config)
        if output_path.exists() and not self.config.overwrite:
            raise FileExistsError(f"Subtitle already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="subtitle_anywhere_", dir=self.config.work_dir
        ) as temp_dir:
            audio_path = Path(temp_dir) / f"{input_path.stem}.wav"
            logger.info("Extracting audio from %s", input_path)
            extract_audio(input_path, audio_path)

            audio = load_wav(str(audio_path))
            segments = self._process_chunks(audio)

            if self.config.keep_audio:
                kept_audio = output_path.with_suffix(".wav")
                os.replace(audio_path, kept_audio)
                logger.info("Kept extracted audio: %s", kept_audio)

        output_path.write_text(render_srt(segments), encoding="utf-8")
        logger.info("Wrote %s (%d segments)", output_path, len(segments))
        return SubtitleResult(input_path=input_path, output_path=output_path, segment_count=len(segments))

    def _process_chunks(self, audio: np.ndarray) -> list[Segment]:
        segments: list[Segment] = []
        for start, end, chunk in iter_audio_chunks(
            audio,
            chunk_seconds=self.config.chunk_seconds,
            min_chunk_seconds=self.config.min_chunk_seconds,
            silence_threshold=self.config.silence_threshold,
        ):
            logger.info("Processing %.1fs-%.1fs", start, end)
            source_text, translated_text = self.engine.transcribe_translate(
                chunk,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                translate=self.config.translate,
            )
            if not source_text and not translated_text:
                continue
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=source_text,
                    language=self.config.source_lang,
                    translated_text=translated_text,
                )
            )
        return segments


def render_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines = []
        if segment.text:
            lines.append(segment.text)
        if segment.translated_text and segment.translated_text != segment.text:
            lines.append(segment.translated_text)
        if not lines:
            continue
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}",
                    *lines,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def output_path_for(input_path: Path, config: BatchConfig) -> Path:
    suffix = f".{config.target_lang}.srt" if config.translate else ".srt"
    if config.output_dir:
        return config.output_dir / f"{input_path.stem}{suffix}"
    return input_path.with_name(f"{input_path.stem}{suffix}")


def iter_audio_chunks(
    audio: np.ndarray,
    chunk_seconds: float,
    min_chunk_seconds: float,
    silence_threshold: float,
    sample_rate: int = 16_000,
):
    chunk_size = max(1, int(chunk_seconds * sample_rate))
    min_size = max(1, int(min_chunk_seconds * sample_rate))
    start_sample = 0
    while start_sample < len(audio):
        nominal_end = min(len(audio), start_sample + chunk_size)
        end_sample = _find_split_point(
            audio,
            start_sample=start_sample,
            nominal_end=nominal_end,
            min_size=min_size,
            silence_threshold=silence_threshold,
            sample_rate=sample_rate,
        )
        chunk = audio[start_sample:end_sample]
        if len(chunk) < min_size:
            break
        if _rms(chunk) >= silence_threshold:
            yield start_sample / sample_rate, end_sample / sample_rate, chunk
        start_sample = end_sample


def _find_split_point(
    audio: np.ndarray,
    start_sample: int,
    nominal_end: int,
    min_size: int,
    silence_threshold: float,
    sample_rate: int,
) -> int:
    if nominal_end >= len(audio):
        return len(audio)

    search_start = max(start_sample + min_size, nominal_end - int(5 * sample_rate))
    search_end = min(len(audio), nominal_end + int(3 * sample_rate))
    window = max(1, int(0.25 * sample_rate))

    for pos in range(search_start, search_end, window):
        value = _rms(audio[pos:min(len(audio), pos + window)])
        if value < silence_threshold:
            return pos

    return nominal_end


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
