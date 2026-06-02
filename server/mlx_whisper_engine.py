"""MLX Whisper speech-to-text backend for Apple silicon."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from seamless import filter_low_content_text

logger = logging.getLogger(__name__)

DEFAULT_MLX_WHISPER_MODEL = "mlx-community/whisper-tiny"
DEFAULT_MLX_TRANSLATOR_MODEL = "mlx-community/Qwen3-1.7B-4bit"

WHISPER_LANGUAGE_ALIASES = {
    "eng": "en",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "jpn": "ja",
    "kor": "ko",
    "fra": "fr",
    "fre": "fr",
    "deu": "de",
    "ger": "de",
    "spa": "es",
}

LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


class MLXWhisperEngine:
    """Run Whisper ASR on MLX.

    Whisper can transcribe multilingual speech and translate speech to English.
    It cannot directly translate English speech to Chinese or other arbitrary
    target languages.
    """

    def __init__(
        self,
        model_id_or_path: str = DEFAULT_MLX_WHISPER_MODEL,
        translator_model_id: str | None = None,
        local_only: bool = False,
    ) -> None:
        try:
            import mlx.core as mx
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is required for the MLX backend. "
                "Install dependencies with `pip install -r server/requirements.txt`."
            ) from exc

        self.model_id_or_path = resolve_mlx_model_path(
            model_id_or_path or DEFAULT_MLX_WHISPER_MODEL,
            local_only=local_only,
        )
        self.translator_model_id = translator_model_id
        self.local_only = local_only
        self._translator_model = None
        self._translator_tokenizer = None
        self._mx = mx
        self._mlx_whisper = mlx_whisper
        logger.info("MLX Whisper ready on %s with model %s", mx.default_device(), self.model_id_or_path)

    def transcribe_translate(
        self,
        audio: np.ndarray,
        source_lang: str,
        target_lang: str,
        translate: bool,
    ) -> tuple[str, str]:
        if audio.size == 0:
            return "", ""

        source_code = normalize_whisper_language_code(source_lang)
        target_code = normalize_whisper_language_code(target_lang)
        if not translate or target_code == source_code:
            return self._run_whisper(audio, source_code, task="transcribe"), ""

        source_text = self._run_whisper(audio, source_code, task="transcribe")
        if not source_text:
            return "", ""
        if self.translator_model_id:
            return source_text, self._translate_text(source_text, source_code, target_code)
        if target_code != "en":
            raise RuntimeError(
                "The MLX Whisper backend needs --translator-model for non-English translation. "
                f"Example: --translator-model {DEFAULT_MLX_TRANSLATOR_MODEL}"
            )
        translated_text = self._run_whisper(audio, source_code, task="translate")
        return source_text, translated_text

    def _run_whisper(self, audio: np.ndarray, language: str | None, task: str) -> str:
        options: dict[str, object] = {
            "path_or_hf_repo": self.model_id_or_path,
            "verbose": None,
            "task": task,
            "condition_on_previous_text": False,
        }
        if language:
            options["language"] = language
        result = self._mlx_whisper.transcribe(audio.astype(np.float32, copy=False), **options)
        return filter_low_content_text(str(result.get("text", "")).strip())

    def _translate_text(self, text: str, source_lang: str | None, target_lang: str | None) -> str:
        model, tokenizer = self._load_translator()
        source_name = language_name(source_lang)
        target_name = language_name(target_lang)
        user_prompt = (
            f"Translate this subtitle text from {source_name} to {target_name}. "
            "Translate every source-language fragment, even if the final phrase is incomplete. "
            "Return only the translation, with no explanation, quotes, or copied source text "
            "unless it is a name.\n\n"
            f"{text}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are a subtitle translator. Preserve meaning and natural subtitle phrasing.",
            },
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            prompt = f"{messages[0]['content']}\n\n{user_prompt}\n"
        from mlx_lm import generate

        translation = generate(model, tokenizer, prompt, verbose=False, max_tokens=256).strip()
        return clean_translation(translation)

    def _load_translator(self):
        if self._translator_model is None or self._translator_tokenizer is None:
            if not self.translator_model_id:
                raise RuntimeError("translator_model_id is required for MLX text translation")
            from mlx_lm import load

            translator_path = resolve_mlx_model_path(
                self.translator_model_id,
                local_only=self.local_only,
            )
            logger.info("Loading MLX translator model %s", translator_path)
            self._translator_model, self._translator_tokenizer = load(translator_path)
        return self._translator_model, self._translator_tokenizer


def resolve_mlx_model_path(model_id_or_path: str, local_only: bool = False) -> str:
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return str(path.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve MLX models. "
            "Install dependencies with `pip install -r server/requirements.txt`."
        ) from exc
    try:
        return snapshot_download(repo_id=model_id_or_path, local_files_only=True)
    except Exception:
        if local_only:
            raise RuntimeError(f"No complete cached MLX model snapshot found for {model_id_or_path}")
    try:
        return snapshot_download(
            repo_id=model_id_or_path,
            local_files_only=False,
        )
    except Exception as exc:
        mode = "local cache" if local_only else "Hugging Face"
        raise RuntimeError(f"Failed to resolve MLX model {model_id_or_path} from {mode}: {exc}") from exc


def normalize_whisper_language_code(language: str) -> str | None:
    code = language.strip().lower()
    if not code or code == "auto":
        return None
    return WHISPER_LANGUAGE_ALIASES.get(code, code)


def language_name(language: str | None) -> str:
    if not language:
        return "the detected language"
    return LANGUAGE_NAMES.get(language, language)


def clean_translation(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    markers = ("Translation:", "Translated text:", "Output:")
    for marker in markers:
        if text.startswith(marker):
            text = text[len(marker) :].strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return filter_low_content_text(text)
