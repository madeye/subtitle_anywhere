"""Batch subtitle CLI for local video files."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

from config import BatchConfig
from hf_utils import (
    check_proxy_connect,
    check_network,
    format_bytes,
    get_local_model_status,
    get_model_info,
    load_env_file,
    PROXY_ENV_KEYS,
    PROXY_OVERRIDE_ENV,
    resolve_model_dir,
)
from mlx_whisper_engine import (
    DEFAULT_MLX_TRANSLATOR_MODEL,
    DEFAULT_MLX_WHISPER_MODEL,
    MLXWhisperEngine,
    normalize_whisper_language_code,
    resolve_mlx_model_path,
)
from audio_utils import find_external_subtitles, langs_match, probe_subtitle_languages
from pipeline import SubtitlePipeline, output_path_for
from seamless import SeamlessM4TEngine

logger = logging.getLogger("subtitle_anywhere")
DEFAULT_SEAMLESS_MODEL = "facebook/seamless-m4t-medium"

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate bilingual external SRT subtitles for local media files."
    )
    parser.add_argument("inputs", nargs="*", help="Input files, directories, or glob patterns")
    parser.add_argument("-o", "--output-dir", type=Path, help="Directory for generated .srt files")
    parser.add_argument("--source-lang", default="eng", help="Source language code, e.g. eng, zho, jpn")
    parser.add_argument("--target-lang", default="zho", help="Target language code, e.g. zho, eng, jpn")
    parser.add_argument(
        "--backend",
        choices=["seamless", "mlx-whisper"],
        default="mlx-whisper",
        help="Inference backend: MLX Whisper for Apple GPU ASR, or SeamlessM4T for direct S2TT",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id or local model directory for the selected backend",
    )
    parser.add_argument(
        "--translator-model",
        default=None,
        help="MLX text model for translating mlx-whisper transcripts, e.g. mlx-community/Qwen3-1.7B-4bit",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for --backend seamless; ignored by the MLX backend",
    )
    parser.add_argument("--no-translate", action="store_true", help="Write source-language SRT only")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing subtitle files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip inputs whose target subtitle already exists")
    parser.add_argument("--keep-audio", action="store_true", help="Keep extracted WAV files beside output")
    parser.add_argument("--dry-run", action="store_true", help="Preview batch inputs and output paths without loading the model")
    parser.add_argument("--chunk-seconds", type=float, default=28.0)
    parser.add_argument("--cue-seconds", type=float, default=4.0, help="Maximum display cue duration after transcription")
    parser.add_argument("--max-cue-chars", type=int, default=90, help="Maximum characters per displayed subtitle line")
    parser.add_argument(
        "--max-cue-sentences",
        type=int,
        default=1,
        help="Maximum sentences per displayed cue; 0 disables the cap",
    )
    parser.add_argument("--min-chunk-seconds", type=float, default=1.0)
    parser.add_argument("--silence-threshold", type=float, default=0.003)
    parser.add_argument("--check-model", action="store_true", help="Check model metadata and exit")
    parser.add_argument("--check-proxy", action="store_true", help="Check HTTP CONNECT support for the configured proxy and exit")
    parser.add_argument("--check-network", action="store_true", help="Check Hugging Face endpoint/proxy connectivity and exit")
    parser.add_argument("--model-status", action="store_true", help="Show local model cache status and exit")
    parser.add_argument("--download-model", action="store_true", help="Download the configured model and exit")
    parser.add_argument("--local-only", action="store_true", help="Require an existing local model snapshot; do not download")
    parser.add_argument("--proxy", help="Override HTTP/HTTPS/ALL proxy for this invocation")
    parser.add_argument("--no-proxy", action="store_true", help="Disable proxy for this invocation")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()
    if args.overwrite and args.skip_existing:
        parser.error("--overwrite and --skip-existing are mutually exclusive")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.log_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)

    load_env_file()
    apply_proxy_args(args.proxy, args.no_proxy)

    if args.check_proxy:
        status = check_proxy_connect()
        print(f"proxy={status.proxy} target={status.target} ok={status.ok} response={status.response}")
        return 0 if status.ok else 1

    if args.check_network:
        try:
            print(check_network())
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    if args.check_model:
        if args.backend == "mlx-whisper":
            print(
                f"{selected_model(args)} backend=mlx-whisper "
                f"translator={selected_translator_model(args) or 'none'} "
                "metadata checks use mlx-whisper/mlx-lm on first load"
            )
            return 0
        try:
            info = get_model_info(selected_model(args))
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        print(
            f"{info.model_id} revision={info.revision or 'unknown'} "
            f"config={info.has_config} files={info.file_count} "
            f"size={format_bytes(info.total_size_bytes)}"
        )
        return 0

    if args.model_status:
        if args.backend == "mlx-whisper":
            print("MLX Whisper uses the Hugging Face Hub cache managed by mlx-whisper")
            return 0
        statuses = get_local_model_status(selected_model(args))
        if not statuses:
            print("no local snapshots")
            return 0
        for status in statuses:
            print(
                f"{status.model_id} revision={status.revision} "
                f"config={status.has_config} files={status.file_count} "
                f"size={format_bytes(status.total_size_bytes)} "
                f"partial_files={status.temp_file_count} "
                f"partial_size={format_bytes(status.temp_size_bytes)} "
                f"path={status.path}"
            )
        return 0

    if args.download_model:
        if args.backend == "mlx-whisper":
            try:
                print(resolve_mlx_model_path(selected_model(args), local_only=args.local_only))
                translator_model = selected_translator_model(args)
                if translator_model:
                    print(resolve_mlx_model_path(translator_model, local_only=args.local_only))
            except RuntimeError as exc:
                logger.error("%s", exc)
                return 1
            return 0
        try:
            print(resolve_model_dir(selected_model(args), local_only=args.local_only))
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        return 0

    if not args.inputs:
        parser.error(
            "inputs are required unless --check-proxy, --check-network, --check-model, "
            "--model-status, or --download-model is used"
        )

    inputs = collect_inputs(args.inputs)
    if not inputs:
        logger.error("No supported input media files found")
        return 2

    config = BatchConfig(
        backend=args.backend,
        model_id=selected_model(args),
        translator_model_id=selected_translator_model(args),
        local_only=args.local_only,
        device=args.device,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        translate=not args.no_translate,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        keep_audio=args.keep_audio,
        chunk_seconds=args.chunk_seconds,
        cue_seconds=args.cue_seconds,
        max_cue_chars=args.max_cue_chars,
        max_cue_sentences=args.max_cue_sentences,
        min_chunk_seconds=args.min_chunk_seconds,
        silence_threshold=args.silence_threshold,
    )
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    validation_errors = validate_batch_config(config)
    if validation_errors:
        for error in validation_errors:
            logger.error("%s", error)
        return 2
    inputs, skipped_existing = filter_existing_outputs(inputs, config)
    collisions = find_output_collisions(inputs, config)
    if collisions:
        for output_path, input_paths in collisions.items():
            logger.error(
                "Output path collision for %s from: %s",
                output_path,
                ", ".join(str(path) for path in input_paths),
            )
        return 2
    if args.dry_run:
        print_dry_run(inputs, config, skipped_existing)
        return 0
    for input_path, reason in skipped_existing:
        print(f"skip: {input_path} ({reason})")
    if not inputs:
        logger.info("No inputs to process after skipping existing outputs")
        return 0

    engine = create_engine(config)
    pipeline = SubtitlePipeline(engine, config)

    failed = 0
    for index, input_path in enumerate(inputs, start=1):
        logger.info("[%d/%d] Processing %s", index, len(inputs), input_path)
        try:
            result = pipeline.process_file(input_path)
        except Exception:
            failed += 1
            logger.exception("Failed: %s", input_path)
            continue
        print(f"{result.output_path} ({result.segment_count} segments)")

    if failed:
        logger.error("%d of %d files failed", failed, len(inputs))
        return 1
    return 0


def selected_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.backend == "mlx-whisper":
        return DEFAULT_MLX_WHISPER_MODEL
    return DEFAULT_SEAMLESS_MODEL


def selected_translator_model(args: argparse.Namespace) -> str | None:
    if args.backend != "mlx-whisper":
        return None
    if args.translator_model:
        return args.translator_model
    target_code = normalize_whisper_language_code(args.target_lang)
    source_code = normalize_whisper_language_code(args.source_lang)
    if not args.no_translate and target_code != source_code and target_code != "en":
        return DEFAULT_MLX_TRANSLATOR_MODEL
    return None


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


def collect_inputs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
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
                for child in sorted(candidate.rglob("*")):
                    _append_media(files, seen, child)
            else:
                _append_media(files, seen, candidate)
    return files


def _append_media(files: list[Path], seen: set[Path], candidate: Path) -> None:
    if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
        return
    resolved = candidate.resolve()
    if resolved not in seen:
        seen.add(resolved)
        files.append(resolved)


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


def validate_batch_config(config: BatchConfig) -> list[str]:
    errors: list[str] = []
    if config.backend == "mlx-whisper":
        source_code = normalize_whisper_language_code(config.source_lang)
        target_code = normalize_whisper_language_code(config.target_lang)
        if config.translate and target_code != source_code and target_code != "en" and not config.translator_model_id:
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


def filter_existing_outputs(inputs: list[Path], config: BatchConfig) -> tuple[list[Path], list[tuple[Path, str]]]:
    if config.overwrite or not config.skip_existing:
        return inputs, []
    remaining: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for input_path in inputs:
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
    ext_sub = find_external_subtitles(input_path, config.target_lang)
    if ext_sub is not None:
        return f"external subtitle: {ext_sub.name}"
    embedded_langs = probe_subtitle_languages(input_path)
    for lang in embedded_langs:
        if langs_match(lang, config.target_lang):
            return f"embedded subtitle: {lang}"
    return None


def print_dry_run(inputs: list[Path], config: BatchConfig, skipped_existing: list[tuple[Path, str]] | None = None) -> None:
    skipped_existing = skipped_existing or []
    print(
        f"inputs={len(inputs)} skipped={len(skipped_existing)} "
        f"backend={config.backend} model={config.model_id} "
        f"translator={config.translator_model_id or 'none'} "
        f"translate={config.translate} source={config.source_lang} target={config.target_lang} "
        f"chunk_seconds={config.chunk_seconds} cue_seconds={config.cue_seconds}"
    )
    for input_path, reason in skipped_existing:
        print(f"skip: {input_path} ({reason})")
    for input_path in inputs:
        print(f"{input_path} -> {output_path_for(input_path, config)}")


def apply_proxy_args(proxy: str | None, no_proxy: bool) -> None:
    if no_proxy:
        os.environ[PROXY_OVERRIDE_ENV] = "none"
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        return
    if proxy:
        os.environ[PROXY_OVERRIDE_ENV] = "override"
        for key in PROXY_ENV_KEYS:
            os.environ[key] = proxy


if __name__ == "__main__":
    sys.exit(main())
