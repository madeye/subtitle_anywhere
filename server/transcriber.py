import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# SenseVoice prepends language/emotion/event tags like <|zh|><|NEUTRAL|><|Speech|><|woitn|>
_TAG_RE = re.compile(r"<\|[^|]*\|>")
_LANG_TAG_RE = re.compile(r"<\|(zh|en|ja|ko|yue)\|>")


class TranscriptionResult:
    __slots__ = ("text", "language")

    def __init__(self, text: str, language: str) -> None:
        self.text = text
        self.language = language


class Transcriber:
    """Thin wrapper around FunASR's SenseVoice model."""

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        vad_model: str = "fsmn-vad",
        device: str = "cuda",
    ) -> None:
        from funasr import AutoModel  # lazy import to keep startup flexible

        logger.info(
            "Loading model=%s  vad=%s  device=%s", model_id, vad_model, device
        )
        self._model = AutoModel(
            model=model_id,
            vad_model=vad_model,
            device=device,
            trust_remote_code=True,
        )
        logger.info("Model loaded successfully")

    def transcribe(
        self, audio: np.ndarray, language: str = "auto"
    ) -> TranscriptionResult:
        """Run inference on an audio buffer.

        Returns:
            TranscriptionResult with recognised text and detected language code.
        """
        if audio.size == 0:
            return TranscriptionResult("", "")

        try:
            results = self._model.generate(
                input=audio,
                language=language,
                use_itn=True,
                batch_size_s=0,
            )
        except Exception:
            logger.exception("Inference failed")
            return TranscriptionResult("", "")

        if not results:
            return TranscriptionResult("", "")

        detected_lang = ""
        text_parts: list[str] = []

        for item in results:
            if isinstance(item, dict):
                raw = item.get("text", "")
            else:
                raw = getattr(item, "text", str(item))

            if not raw:
                continue

            # Extract language from the first tag (e.g. <|zh|>)
            if not detected_lang:
                lang_match = _LANG_TAG_RE.search(raw)
                if lang_match:
                    detected_lang = lang_match.group(1)

            # Strip all SenseVoice tags
            text = _TAG_RE.sub("", raw).strip()
            if text:
                text_parts.append(text)

        return TranscriptionResult(" ".join(text_parts), detected_lang)
