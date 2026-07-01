"""Batch local-video subtitle pipeline."""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
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
            extract_source = input_path
            staged_video: Path | None = None
            if self.config.stage_input:
                staged_video = Path(temp_dir) / f"src_{input_path.name}"
                logger.info("Copying %s to local temp before extraction", input_path)
                shutil.copyfile(input_path, staged_video)
                extract_source = staged_video

            logger.info("Extracting audio from %s", extract_source)
            extract_audio(extract_source, audio_path)

            if staged_video is not None:
                staged_video.unlink(missing_ok=True)
                logger.info("Removed local copy %s", staged_video)

            audio = load_wav(str(audio_path))
            logger.info("Audio duration %.1fs", len(audio) / 16_000)
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
        supports_lines = hasattr(self.engine, "transcribe_translate_lines")
        for start, end, chunk in iter_audio_chunks(
            audio,
            chunk_seconds=self.config.chunk_seconds,
            min_chunk_seconds=self.config.min_chunk_seconds,
            silence_threshold=self.config.silence_threshold,
        ):
            logger.info("Processing %.1fs-%.1fs", start, end)
            if supports_lines:
                chunk_lines = self.engine.transcribe_translate_lines(
                    chunk,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                    translate=self.config.translate,
                )
                for line in chunk_lines:
                    absolute_line = Segment(
                        start=start + line.start,
                        end=min(end, start + line.end),
                        text=line.text,
                        language=line.language or self.config.source_lang,
                        translated_text=line.translated_text,
                    )
                    segments.extend(
                        split_segment_for_display(
                            absolute_line,
                            cue_seconds=self.config.cue_seconds,
                            max_cue_chars=self.config.max_cue_chars,
                            max_cue_sentences=self.config.max_cue_sentences,
                        )
                    )
                continue

            source_text, translated_text = self.engine.transcribe_translate(
                chunk,
                source_lang=self.config.source_lang,
                target_lang=self.config.target_lang,
                translate=self.config.translate,
            )
            if not source_text and not translated_text:
                continue
            segments.extend(
                split_segment_for_display(
                    Segment(
                        start=start,
                        end=end,
                        text=source_text,
                        language=self.config.source_lang,
                        translated_text=translated_text,
                    ),
                    cue_seconds=self.config.cue_seconds,
                    max_cue_chars=self.config.max_cue_chars,
                    max_cue_sentences=self.config.max_cue_sentences,
                )
            )
        return segments


def split_segment_for_display(
    segment: Segment,
    cue_seconds: float,
    max_cue_chars: int,
    max_cue_sentences: int = 0,
) -> list[Segment]:
    if segment.translated_text is None:
        segment = Segment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            language=segment.language,
            translated_text="",
        )
    duration = max(0.0, segment.end - segment.start)
    sentence_count = (
        len(_split_sentence_units(segment.text)) if segment.text else 0
    )
    sentence_cap_ok = (
        max_cue_sentences <= 0 or sentence_count <= max_cue_sentences
    )
    if (
        duration <= cue_seconds
        and max(len(segment.text), len(segment.translated_text)) <= max_cue_chars
        and sentence_cap_ok
    ):
        return [segment]

    sentence_parts = (
        math.ceil(sentence_count / max_cue_sentences)
        if max_cue_sentences > 0 and sentence_count > 0
        else 1
    )
    part_count = max(
        1,
        math.ceil(duration / cue_seconds),
        math.ceil(len(segment.text) / max_cue_chars) if segment.text else 1,
        math.ceil(len(segment.translated_text) / max_cue_chars) if segment.translated_text else 1,
        sentence_parts,
    )
    source_parts = split_text_balanced(segment.text, part_count)
    translated_parts = split_text_balanced(segment.translated_text, part_count)
    part_count = max(len(source_parts), len(translated_parts))
    source_parts.extend([""] * (part_count - len(source_parts)))
    translated_parts.extend([""] * (part_count - len(translated_parts)))

    cue_duration = duration / part_count if part_count else duration
    display_segments: list[Segment] = []
    for index in range(part_count):
        part_start = segment.start + cue_duration * index
        part_end = segment.end if index == part_count - 1 else segment.start + cue_duration * (index + 1)
        if not source_parts[index] and not translated_parts[index]:
            continue
        display_segments.append(
            Segment(
                start=part_start,
                end=part_end,
                text=source_parts[index],
                language=segment.language,
                translated_text=translated_parts[index],
            )
        )
    return display_segments


def split_text_balanced(text: str, part_count: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if part_count <= 1:
        return [text]

    separator = " " if " " in text else ""
    units = _split_sentence_units(text)
    target_length = max(1, math.ceil(len(text) / part_count))
    if len(units) < part_count or any(len(unit) > target_length * 1.5 for unit in units):
        units = _split_word_or_char_units(text)
    parts = _pack_units(units, part_count, separator)
    if len(parts) < part_count:
        parts = _pack_units(_split_word_or_char_units(text), part_count, separator)
    return parts


def _split_sentence_units(text: str) -> list[str]:
    units = [unit.strip() for unit in re.findall(r"[^.!?。！？；;]+[.!?。！？；;]*", text) if unit.strip()]
    return units or [text]


def _split_word_or_char_units(text: str) -> list[str]:
    if " " in text:
        return [unit for unit in text.split(" ") if unit]
    return list(text)


def _pack_units(units: list[str], part_count: int, separator: str) -> list[str]:
    remaining_units = units[:]
    remaining_parts = part_count
    parts: list[str] = []
    while remaining_parts > 0 and remaining_units:
        target = max(1, math.ceil(sum(len(unit) for unit in remaining_units) / remaining_parts))
        current: list[str] = []
        current_length = 0
        while remaining_units and (not current or current_length + len(remaining_units[0]) <= target):
            unit = remaining_units.pop(0)
            current.append(unit)
            current_length += len(unit)
        parts.append(separator.join(current).strip())
        remaining_parts -= 1
    if remaining_units:
        tail = separator.join(remaining_units)
        if parts:
            parts[-1] = f"{parts[-1]}{separator}{tail}".strip()
        else:
            parts.append(tail)
    return [part for part in parts if part]


def render_srt(segments: list[Segment]) -> str:
    blocks: list[str] = []
    written = 0
    for segment in segments:
        lines = []
        if segment.text:
            lines.append(segment.text)
        if segment.translated_text and segment.translated_text != segment.text:
            lines.append(segment.translated_text)
        if not lines:
            continue
        written += 1
        blocks.append(
            "\n".join(
                [
                    str(written),
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
            if _rms(chunk) >= silence_threshold:
                logger.debug("Processing short final chunk %.1fs-%.1fs", start_sample / sample_rate, end_sample / sample_rate)
                yield start_sample / sample_rate, end_sample / sample_rate, chunk
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
