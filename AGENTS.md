# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python CLI for batch-generating bilingual external subtitles from local media.

- `server/server.py` is the command-line entry point and batch coordinator.
- `server/pipeline.py` extracts audio, chunks it, runs inference, and writes `.srt` output.
- `server/mlx_whisper_engine.py` runs the default MLX Whisper ASR plus optional MLX LLM text translation.
- `server/seamless.py` contains optional SeamlessM4T support; do not make it the default path.
- `server/hf_utils.py` handles Hugging Face metadata, cache status, downloads, and `.env` proxy loading.
- `server/audio_utils.py` contains FFmpeg extraction and WAV loading helpers.
- `docs/index.html` is a small static usage page. Do not reintroduce browser extension code.

## Build, Test, and Development Commands

- `cd server && pip install -r requirements.txt`: install runtime dependencies.
- `python server/server.py --help`: verify CLI argument wiring.
- `python server/server.py --check-proxy`: verify the configured proxy.
- `python server/server.py --check-network`: verify Hugging Face connectivity.
- `python server/server.py --check-model`: verify default model metadata.
- `python server/server.py --model-status`: inspect local model cache state.
- `python server/server.py sample.mp4 --output-dir subtitles --dry-run`: preview input/output paths without loading the model.
- `python -m py_compile server/*.py`: syntax-check all Python modules.
- `python -m unittest discover -s tests`: run lightweight regression tests.
- `python server/server.py sample.mp4 --backend mlx-whisper --source-lang eng --target-lang zho --output-dir subtitles`: run an MLX smoke test.

FFmpeg must be available on `PATH`; install with `brew install ffmpeg` on macOS.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints for public helpers, dataclasses for structured results/config, and module-level loggers. Keep CLI parsing in `server.py`, media orchestration in `pipeline.py`, and model-specific code in backend modules. Prefer `Path` objects and avoid hard-coded user directories.

## Testing Guidelines

Use `unittest` for lightweight coverage. Keep tests under `tests/`, and mock model downloads or inference unless a change specifically requires local weights. For behavioral changes, run `py_compile`, `--help`, unit tests, and a short-media smoke test when weights are available. Verify timestamps, source text, translated lines, and names like `movie.zho.srt`.

## Commit & Pull Request Guidelines

The history uses short imperative subjects. Keep commits focused by subsystem. Pull requests should describe the CLI change, list validation commands, note device/model assumptions, and include a small SRT excerpt for subtitle-output changes.

## Security & Configuration Tips

Do not commit model snapshots, temporary audio, private generated subtitles, `.env`, or local caches. Keep Hugging Face tokens and proxy settings local. Use `--proxy` or `--no-proxy` for one-shot overrides.
