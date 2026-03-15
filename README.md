# Subtitle Anywhere

Real-time speech-to-text subtitles for any browser tab with local translation.

## Features

- Real-time speech recognition using [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) (auto language detection)
- Local translation using [NLLB-200](https://ai.meta.com/research/no-language-left-behind/) (200 languages, INT8 quantized via CTranslate2)
- Chrome extension with subtitle overlay on any video
- No API keys needed -- everything runs locally
- Preserves original audio quality (stereo passthrough)

## Architecture

```
Chrome Extension                    Python Server (WebSocket)
+-----------------+                +------------------------+
| Tab Audio       |  PCM 16kHz    | SenseVoiceSmall (ASR)  |
| Capture --------+--------------->  Language Detection     |
|                 |                |         |              |
| Subtitle        |  JSON         | NLLB-200 (Translation) |
| Overlay <-------+--------------+  CTranslate2 INT8      |
+-----------------+                +------------------------+
```

## Quick Start

### Server

```bash
cd server
pip install -r requirements.txt
python server.py --device mps    # macOS Apple Silicon
python server.py --device cuda   # NVIDIA GPU
python server.py --device cpu    # CPU only
```

Models are downloaded automatically on first run (~300MB for translation, ~200MB for ASR).

### Extension

1. Open `chrome://extensions`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `extension/` directory
4. Open any tab with a video, click the extension icon
5. Select a target language and click "Start Capture"

## Configuration

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda` | Compute device (`cuda`, `cpu`, `mps`) |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `10095` | WebSocket port |
| `--model-id` | `iic/SenseVoiceSmall` | ASR model |
| `--log-level` | `INFO` | Logging level |

### Extension Settings

- **Target Language**: Select from 14 languages in the popup
- **Font size, color, opacity**: Customizable in Advanced Settings
- **WebSocket URL**: Default `ws://127.0.0.1:10095`

## Supported Languages

Source (auto-detected): English, Chinese, Japanese, Korean, Cantonese

Translation targets: English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic, Hindi, Thai, Vietnamese, Italian

## Requirements

- Python 3.10+
- Chrome/Chromium browser
- macOS (MPS), Linux/Windows (CUDA), or CPU

## License

[MIT](LICENSE)
