from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from seamless import SeamlessM4TEngine, filter_low_content_text, normalize_language_code


class FakeTensor:
    def to(self, device: str):
        return self


class FakeProcessor:
    def __call__(self, *, audio, sampling_rate: int, return_tensors: str):
        self.audio = audio
        self.sampling_rate = sampling_rate
        self.return_tensors = return_tensors
        return {"input_features": FakeTensor()}

    def decode(self, tokens, skip_special_tokens: bool, clean_up_tokenization_spaces: bool) -> str:
        self.tokens = tokens
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        return " hello "


class FakeModel:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[1, 2, 3]]


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    def inference_mode(self):
        return FakeInferenceMode()


class SeamlessEngineTests(unittest.TestCase):
    def test_generate_text_uses_audio_processor_argument(self) -> None:
        engine = SeamlessM4TEngine.__new__(SeamlessM4TEngine)
        engine.device = "cpu"
        engine.processor = FakeProcessor()
        engine.model = FakeModel()
        engine._torch = FakeTorch()

        text = engine.generate_text(np.zeros(16_000, dtype=np.float32), "zho")

        self.assertEqual(text, "hello")
        self.assertEqual(engine.processor.sampling_rate, 16_000)
        self.assertEqual(engine.processor.return_tensors, "pt")
        self.assertEqual(engine.model.kwargs["tgt_lang"], "cmn")
        self.assertFalse(engine.model.kwargs["generate_speech"])
        self.assertFalse(engine.processor.clean_up_tokenization_spaces)

    def test_normalize_language_code_keeps_supported_codes(self) -> None:
        self.assertEqual(normalize_language_code("eng"), "eng")
        self.assertEqual(normalize_language_code("zho"), "cmn")

    def test_filter_low_content_text_drops_punctuation_heavy_output(self) -> None:
        self.assertEqual(filter_low_content_text(", , , , , , , ,"), "")
        self.assertEqual(filter_low_content_text("Hello world."), "Hello world.")
        self.assertEqual(filter_low_content_text("世界好,这是字幕测试。"), "世界好,这是字幕测试。")

    def test_filter_low_content_text_drops_repeated_token_output(self) -> None:
        self.assertEqual(filter_low_content_text("什么是什么是什么是什么是什么是什么是什么是什么"), "")
        self.assertEqual(filter_low_content_text("警官布莱森, 警官布莱森, 警官布莱森, 警官布莱森"), "")


if __name__ == "__main__":
    unittest.main()
