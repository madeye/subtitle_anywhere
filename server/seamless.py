"""SeamlessM4T speech-to-text translation engine."""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hf_utils import resolve_model_dir

logger = logging.getLogger(__name__)

LANGUAGE_ALIASES = {
    "zh": "cmn",
    "zho": "cmn",
    "chi": "cmn",
    "cn": "cmn",
    "jp": "jpn",
    "ja": "jpn",
    "kr": "kor",
    "ko": "kor",
}


@dataclass
class Segment:
    """A timed bilingual subtitle segment."""

    start: float
    end: float
    text: str
    language: str = ""
    translated_text: str = ""


class SeamlessM4TEngine:
    """Run ASR/S2TT with a local SeamlessM4T model snapshot."""

    def __init__(self, model_id_or_path: str, device: str = "auto", local_only: bool = False) -> None:
        self.model_dir = resolve_model_dir(model_id_or_path, local_only=local_only)
        self.device = self._select_device(device)

        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, SeamlessM4TModel, SeamlessM4Tv2Model
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for SeamlessM4T. "
                "Install dependencies with `pip install -r server/requirements.txt`."
            ) from exc

        self._torch = torch
        logger.info("Loading SeamlessM4T model from %s", self.model_dir)
        self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
        config = AutoConfig.from_pretrained(str(self.model_dir))
        model_class = SeamlessM4Tv2Model if config.model_type == "seamless_m4t_v2" else SeamlessM4TModel
        self.model = model_class.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        logger.info("SeamlessM4T ready on %s", self.device)

    def transcribe_translate(
        self,
        audio: np.ndarray,
        source_lang: str,
        target_lang: str,
        translate: bool,
    ) -> tuple[str, str]:
        source_text = self.generate_text(audio, source_lang)
        translated_text = ""
        if translate and target_lang != source_lang:
            translated_text = self.generate_text(audio, target_lang)
        return source_text, translated_text

    def generate_text(self, audio: np.ndarray, target_lang: str) -> str:
        if audio.size == 0:
            return ""
        inputs = self.processor(
            audio=audio,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs,
                tgt_lang=normalize_language_code(target_lang),
                generate_speech=False,
            )
        tokens = _first_token_tensor(output)
        text = self.processor.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return filter_low_content_text(text)

    def _select_device(self, requested: str) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if requested != "auto":
            return requested
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


def _first_token_tensor(output):
    if hasattr(output, "sequences"):
        return output.sequences[0].tolist()
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, list):
        output = output[0]
    if hasattr(output, "ndim") and output.ndim > 1:
        output = output[0]
    if hasattr(output, "tolist"):
        return output.tolist()
    return output


def normalize_language_code(language: str) -> str:
    code = language.strip()
    return LANGUAGE_ALIASES.get(code.lower(), code)


def filter_low_content_text(text: str) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    content = [char.lower() for char in text if _is_content_char(char)]
    content_chars = len(content)
    visible_chars = sum(1 for char in text if not char.isspace())
    if visible_chars == 0:
        return ""
    if content_chars < 2 or content_chars / visible_chars < 0.2:
        return ""
    if content_chars >= 12 and len(set(content)) / content_chars < 0.18:
        return ""
    if _has_repeated_content(content):
        return ""
    return text


def _has_repeated_content(content: list[str]) -> bool:
    content_chars = len(content)
    if content_chars < 12:
        return False
    content_text = "".join(content)
    max_width = min(8, content_chars // 2)
    for width in range(2, max_width + 1):
        grams = [content_text[index : index + width] for index in range(content_chars - width + 1)]
        if not grams:
            continue
        _, count = Counter(grams).most_common(1)[0]
        if count >= 4 and count * width / content_chars >= 0.5:
            return True
    return False


def _is_content_char(char: str) -> bool:
    if char.isalnum():
        return True
    return unicodedata.category(char).startswith(("L", "N"))
