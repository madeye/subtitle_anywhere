# Subtitle Anywhere

Batch-generate bilingual external subtitles for local video or audio files.

The project is now a local CLI pipeline: it extracts audio with FFmpeg, chunks
the waveform, runs local speech and translation models, and writes `.srt` files
for players such as IINA, VLC, mpv, or Plex. The default model downloads use
official Hugging Face, with proxy settings loaded from `.env`.

## Features

- Batch input from files, directories, or shell globs.
- Local subtitle generation with an MLX backend on Apple GPU.
- Source-language text plus translated text in one external SRT file.
- Hugging Face downloads can use proxy settings from `.env`.
- No browser extension, WebSocket server, API key, or cloud inference service.

## Requirements

- Python 3.10+.
- FFmpeg on `PATH`.
- Apple silicon GPU for the MLX backend.
- Enough disk space for the MLX Whisper and MLX translator checkpoints.

Install dependencies:

```bash
cd server
pip install -r requirements.txt
```

Install FFmpeg if needed:

```bash
brew install ffmpeg
```

## Usage

Run from the repository root:

```bash
python server/server.py ~/Movies/*.mp4 --source-lang eng --target-lang zho --output-dir subtitles
```

Process a whole directory recursively:

```bash
python server/server.py ~/Videos --target-lang jpn --overwrite
```

Resume a batch without touching completed subtitles:

```bash
python server/server.py ~/Videos --output-dir subtitles --skip-existing
```

Preview a batch without loading the model:

```bash
python server/server.py ~/Videos --output-dir subtitles --dry-run
```

Generate source-language subtitles only:

```bash
python server/server.py lecture.mov --source-lang eng --no-translate
```

Run ASR and translation on the default MLX backend:

```bash
python server/server.py movie.mp4 --source-lang eng --target-lang zho --output-dir subtitles
```

By default, `mlx-whisper` transcribes audio on MLX and `mlx-lm` translates the
transcript when the target is not English. The default
translator model is `mlx-community/Qwen3-1.7B-4bit`; override it with
`--translator-model`. For ASR-only runs, add `--no-translate` to skip loading
the translator model.

By default, output files are named like `video.zho.srt` beside the input file,
or inside `--output-dir` when provided.
If multiple inputs would write the same path inside `--output-dir`, the CLI
fails before loading the model; rename the media files or run separate output
directories to avoid collisions.
Chunking options are validated before model load: `--chunk-seconds` and
`--min-chunk-seconds` must be positive, `--min-chunk-seconds` cannot exceed
`--chunk-seconds`, and `--silence-threshold` cannot be negative.
The default inference chunk window is 10 seconds to preserve ASR and translation
context. Display cues are split afterward with `--cue-seconds` and
`--max-cue-chars` so one screen does not contain a dense multi-sentence block.

## Model Downloads

Default models:

```text
mlx-community/whisper-tiny
mlx-community/Qwen3-1.7B-4bit
```

Downloads use `https://huggingface.co`.
For proxied access, set `HTTP_PROXY` and `HTTPS_PROXY` in `.env`, for example
`http://127.0.0.1:8080`. You can check metadata or download the checkpoint
explicitly before a batch run:

```bash
python server/server.py --check-proxy
python server/server.py --check-network
python server/server.py --check-model
python server/server.py --model-status
python server/server.py --download-model
python server/server.py --local-only ~/Videos/*.mp4 --output-dir subtitles
```

`--check-proxy` validates HTTP CONNECT support on the configured proxy.
`--check-network` validates the configured Hugging Face endpoint and proxy.
`--check-model` reports the selected MLX speech and translator models.
`--download-model` preloads both default MLX models into the Hugging Face cache.
Use `--local-only` to require existing cached model snapshots and fail fast
instead of downloading during a batch run.

Override proxy settings for one command without editing `.env`:

```bash
python server/server.py --proxy http://127.0.0.1:8080 --check-proxy
python server/server.py --no-proxy --check-network
```

You can also pass local MLX model snapshot directories:

```bash
python server/server.py movie.mkv --model ~/.cache/huggingface/hub/models--mlx-community--whisper-tiny/snapshots/<revision>
```

The MLX Whisper backend accepts the same common aliases and maps them to
Whisper language codes such as `en`, `zh`, `ja`, and `ko`.

The optional `--backend seamless` path is retained for direct SeamlessM4T S2TT,
but it requires installing PyTorch, Transformers, and SentencePiece separately.

## Output Format

Each SRT cue contains the source text first and the translated text second:

```srt
1
00:00:01,230 --> 00:00:03,900
Hello, welcome back.
你好，欢迎回来。
```

Generated lines that are mostly punctuation or spacing are suppressed to avoid
writing obvious model artifacts into the SRT.

## Web UI

A zero-dependency local web frontend lets you set the source and destination
folders without remembering CLI flags.

```bash
python server/web.py
# open http://127.0.0.1:8765
```

Source folder is the only required field; the subtitle destination defaults to
the same folder (toggle off to point somewhere else). Saved settings live in
`~/.config/subtitle_anywhere/config.json` (override with
`SUBTITLE_ANYWHERE_CONFIG`). The page also previews the exact CLI command for
the current selection so you can copy-paste it into a terminal run.

## Development Checks

```bash
python -m unittest discover -s tests
python -m py_compile server/*.py
python server/server.py --help
```

For an end-to-end smoke test, run the CLI on a short local video and verify that
the generated `.srt` opens in a media player.

## License

[MIT](LICENSE)
