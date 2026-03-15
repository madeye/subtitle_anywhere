"""Local translation using NLLB-200 via CTranslate2 (INT8 quantized).

Uses the pre-converted model ``JustFrederik/nllb-200-distilled-600M-ct2-int8``
(~300 MB), downloaded automatically from HuggingFace on first use.
"""

import logging
import os
from collections import OrderedDict
from typing import Optional

import ctranslate2
import sentencepiece as spm
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

_CT2_MODEL = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
_SP_MODEL = "facebook/nllb-200-distilled-600M"

# Map SenseVoice / UI language codes → NLLB FLORES-200 codes.
_LANG_TO_FLORES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "yue": "yue_Hant",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "it": "ita_Latn",
}

# CTranslate2 device mapping
_DEVICE_MAP = {
    "cuda": "cuda",
    "mps": "cpu",  # CTranslate2 doesn't support MPS; CPU is fast with INT8
    "cpu": "cpu",
}


class Translator:
    """Translates text using NLLB-200 (CTranslate2 INT8) with LRU cache."""

    def __init__(self, device: str = "cpu", cache_size: int = 500) -> None:
        ct2_device = _DEVICE_MAP.get(device, "cpu")

        logger.info("Downloading CTranslate2 model: %s", _CT2_MODEL)
        ct2_path = snapshot_download(_CT2_MODEL)

        logger.info("Loading CTranslate2 translator on %s", ct2_device)
        self._translator = ctranslate2.Translator(
            ct2_path, device=ct2_device, compute_type="int8",
            inter_threads=1, intra_threads=8,
        )

        # SentencePiece tokenizer — download only the .model file, not the full repo
        logger.info("Downloading SentencePiece tokenizer from %s", _SP_MODEL)
        sp_file = hf_hub_download(_SP_MODEL, filename="sentencepiece.bpe.model")
        logger.info("Loading SentencePiece tokenizer: %s", sp_file)
        self._sp = spm.SentencePieceProcessor(sp_file)

        logger.info("Translation model loaded (INT8)")

        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate *text* from *source_lang* to *target_lang*.

        Language codes are SenseVoice-style (en, zh, ja, ko, yue).
        Returns the original text if translation is not possible.
        """
        if not text or not text.strip():
            return text

        src_flores = _LANG_TO_FLORES.get(source_lang)
        tgt_flores = _LANG_TO_FLORES.get(target_lang)

        if not src_flores or not tgt_flores:
            logger.warning("Unsupported language pair: %s → %s", source_lang, target_lang)
            return text

        if src_flores == tgt_flores:
            return text

        cache_key = f"{src_flores}|{tgt_flores}|{text}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            # Tokenize with SentencePiece, prepend source language token, append EOS
            tokens = self._sp.encode(text, out_type=str)
            tokens = [src_flores] + tokens + ["</s>"]

            results = self._translator.translate_batch(
                [tokens],
                target_prefix=[[tgt_flores]],
                beam_size=2,
            )

            # Decode, skipping the leading language token
            output_tokens = results[0].hypotheses[0][1:]
            result = self._sp.decode(output_tokens)
        except Exception:
            logger.exception("Translation failed for %s→%s: %s",
                             source_lang, target_lang, text[:80])
            return text

        self._cache_put(cache_key, result)
        return result

    # -- cache helpers -------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
        self._cache[key] = value
