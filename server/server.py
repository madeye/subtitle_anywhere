"""Subtitle Anywhere -- real-time transcription WebSocket server.

Protocol
--------
1. Client connects and sends a JSON config message::

       {"mode": "2pass", "chunk_size": [5, 10, 5],
        "audio_fs": 16000, "wav_format": "pcm",
        "is_speaking": true,
        "target_lang": "zh"}

   ``target_lang`` is optional.  When present, final results will
   include a ``translated_text`` field using a local Opus-MT model.

2. Client streams binary PCM16-LE mono 16 kHz audio chunks.
3. Client sends ``{"is_speaking": false}`` to signal end-of-speech.
4. Server replies with JSON::

       {"text": "...", "mode": "2pass-online",  "is_final": false}
       {"text": "...", "translated_text": "...", "mode": "2pass-offline", "is_final": true}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from typing import Any, Optional

import numpy as np
import websockets
from websockets.server import ServerConnection

from audio_utils import pcm16_to_float32, validate_audio
from config import ServerConfig
from transcriber import Transcriber
from translator import Translator

logger = logging.getLogger("subtitle_anywhere")

# ---------------------------------------------------------------------------
# Per-connection session
# ---------------------------------------------------------------------------

class Session:
    """Holds per-connection state."""

    def __init__(
        self,
        config: ServerConfig,
        transcriber: Transcriber,
        translator: Translator,
    ) -> None:
        self.config = config
        self.transcriber = transcriber
        self.translator = translator

        self.buffer: list[np.ndarray] = []
        self.buffer_samples: int = 0

        self.audio_fs: int = 16000
        self.mode: str = "2pass"
        self.is_speaking: bool = True

        # Target language for translation (set from client init message)
        self.target_lang: str = ""

        self._inference_task: asyncio.Task[None] | None = None

    # -- buffer management ---------------------------------------------------

    def append_audio(self, data: bytes) -> None:
        if not validate_audio(data):
            logger.warning("Received invalid audio chunk (%d bytes)", len(data))
            return
        samples = pcm16_to_float32(data)
        self.buffer.append(samples)
        self.buffer_samples += len(samples)

    def get_audio(self) -> np.ndarray:
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.buffer)

    def clear_buffer(self) -> None:
        self.buffer.clear()
        self.buffer_samples = 0

    @property
    def buffer_duration_s(self) -> float:
        return self.buffer_samples / self.audio_fs if self.audio_fs else 0.0

    # -- lifecycle -----------------------------------------------------------

    def start_inference_loop(self, ws: ServerConnection) -> None:
        if self._inference_task is None:
            self._inference_task = asyncio.create_task(
                self._periodic_inference(ws)
            )

    async def stop_inference_loop(self) -> None:
        if self._inference_task is not None:
            self._inference_task.cancel()
            try:
                await self._inference_task
            except asyncio.CancelledError:
                pass
            self._inference_task = None

    # -- periodic inference --------------------------------------------------

    async def _periodic_inference(self, ws: ServerConnection) -> None:
        interval = self.config.inference_interval_s
        try:
            while True:
                await asyncio.sleep(interval)
                if not self.is_speaking:
                    break
                await self._run_partial(ws)
        except asyncio.CancelledError:
            return

    async def _run_partial(self, ws: ServerConnection) -> None:
        audio = self.get_audio()
        if audio.size == 0:
            return

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.transcriber.transcribe, audio
        )

        msg = json.dumps(
            {"text": result.text, "mode": "2pass-online", "is_final": False}
        )
        logger.debug("Partial result: %s", result.text[:120] if result.text else "(empty)")
        try:
            await ws.send(msg)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def run_final(self, ws: ServerConnection) -> None:
        audio = self.get_audio()
        if audio.size == 0:
            resp: dict[str, Any] = {
                "text": "", "mode": "2pass-offline", "is_final": True
            }
            try:
                await ws.send(json.dumps(resp))
            except websockets.exceptions.ConnectionClosed:
                pass
            self.clear_buffer()
            return

        # Clear buffer immediately so new audio can accumulate during
        # transcription and translation
        self.clear_buffer()

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self.transcriber.transcribe, audio
        )

        text = result.text
        detected_lang = result.language

        logger.info("Final result (%s): %s", detected_lang or "?", text[:200] if text else "(empty)")

        resp: dict[str, Any] = {
            "text": text, "mode": "2pass-offline", "is_final": True
        }

        # Translate and send together with original text
        if text and self.target_lang and detected_lang:
            try:
                translated = await loop.run_in_executor(
                    None,
                    self.translator.translate,
                    text, detected_lang, self.target_lang,
                )
                if translated and translated != text:
                    resp["translated_text"] = translated
                    logger.info("Translated (%s→%s): %s",
                                detected_lang, self.target_lang, translated[:200])
            except Exception:
                logger.exception("Translation failed")

        try:
            await ws.send(json.dumps(resp))
        except websockets.exceptions.ConnectionClosed:
            pass


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------

async def handle_connection(
    ws: ServerConnection,
    config: ServerConfig,
    transcriber: Transcriber,
    translator: Translator,
) -> None:
    remote = ws.remote_address
    logger.info("Client connected: %s", remote)

    session = Session(config, transcriber, translator)

    try:
        async for message in ws:
            if isinstance(message, str):
                try:
                    payload: dict[str, Any] = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from %s: %r", remote, message[:120])
                    continue

                if "audio_fs" in payload:
                    session.audio_fs = int(payload["audio_fs"])
                if "mode" in payload:
                    session.mode = str(payload["mode"])

                # Target language from client
                if "target_lang" in payload:
                    session.target_lang = str(payload["target_lang"])
                    if session.target_lang and session.target_lang != "none":
                        logger.info("Translation target: %s", session.target_lang)
                    else:
                        session.target_lang = ""

                is_speaking = payload.get("is_speaking")

                if is_speaking is True:
                    session.is_speaking = True
                    session.start_inference_loop(ws)
                elif is_speaking is False:
                    session.is_speaking = False
                    await session.stop_inference_loop()
                    await session.run_final(ws)

                continue

            if isinstance(message, (bytes, bytearray)):
                session.append_audio(bytes(message))

                if session.buffer_duration_s >= config.max_buffer_s:
                    logger.info(
                        "Buffer reached %.1fs -- forcing inference for %s",
                        session.buffer_duration_s, remote,
                    )
                    await session.stop_inference_loop()
                    await session.run_final(ws)
                    if session.is_speaking:
                        session.start_inference_loop(ws)

                continue

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected: %s", remote)
    except Exception:
        logger.exception("Unexpected error for %s", remote)
    finally:
        await session.stop_inference_loop()
        logger.info("Session cleaned up for %s", remote)


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

async def run_server(
    config: ServerConfig,
    transcriber: Transcriber,
    translator: Translator,
) -> None:
    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    async def _handler(ws: ServerConnection) -> None:
        await handle_connection(ws, config, transcriber, translator)

    async with websockets.serve(
        _handler, config.host, config.port, max_size=2**22,
    ):
        logger.info("Server listening on ws://%s:%d", config.host, config.port)
        await stop_event.wait()

    logger.info("Server shut down gracefully")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Subtitle Anywhere -- real-time transcription server"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=10095, help="Listen port")
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu", "mps"], help="Compute device",
    )
    parser.add_argument("--model-id", type=str, default="iic/SenseVoiceSmall")
    parser.add_argument("--vad-model", type=str, default="fsmn-vad")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = ServerConfig(
        host=args.host, port=args.port, device=args.device,
        model_id=args.model_id, vad_model=args.vad_model,
    )

    logger.info("Initialising transcriber ...")
    transcriber = Transcriber(
        model_id=config.model_id, vad_model=config.vad_model,
        device=config.device,
    )

    logger.info("Initialising translator (models loaded on demand) ...")
    translator = Translator(device=config.device)

    asyncio.run(run_server(config, transcriber, translator))


if __name__ == "__main__":
    main()
