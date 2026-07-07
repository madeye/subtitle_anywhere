import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BatchConfig:
    """Runtime configuration for local batch subtitle generation."""

    backend: str = "mlx-whisper"
    model_id: str = "mlx-community/whisper-tiny"
    translator_model_id: str | None = "mlx-community/Qwen3-1.7B-4bit"
    local_only: bool = False
    device: str = "auto"
    source_lang: str = "auto"
    target_lang: str = "zho"
    translate: bool = True
    output_dir: Path | None = None
    overwrite: bool = False
    skip_existing: bool = False
    keep_audio: bool = False
    stage_input: bool = False
    embed_subtitles: bool = False
    chunk_seconds: float = 28.0
    cue_seconds: float = 4.0
    max_cue_chars: int = 90
    max_cue_sentences: int = 1
    min_chunk_seconds: float = 1.0
    silence_threshold: float = 0.003
    work_dir: str = field(
        default_factory=lambda: os.path.join(tempfile.gettempdir(), "subtitle_anywhere")
    )
