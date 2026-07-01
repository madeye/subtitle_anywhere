from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from config import BatchConfig
from pipeline import SubtitlePipeline, iter_audio_chunks, render_srt, split_segment_for_display, split_text_balanced
from seamless import Segment


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe_translate(
        self,
        audio: np.ndarray,
        source_lang: str,
        target_lang: str,
        translate: bool,
    ) -> tuple[str, str]:
        self.calls += 1
        return f"source {self.calls}", f"target {self.calls}" if translate else ""


class PipelineTests(unittest.TestCase):
    def test_render_srt_writes_bilingual_lines(self) -> None:
        text = render_srt([Segment(1.23, 3.9, "Hello", "eng", "你好")])
        self.assertEqual(
            text,
            "1\n00:00:01,230 --> 00:00:03,900\nHello\n你好\n",
        )

    def test_split_segment_for_display_preserves_context_but_shortens_cues(self) -> None:
        segment = Segment(
            10.0,
            20.0,
            "First sentence. Second sentence. Third sentence.",
            "eng",
            "第一句。第二句。第三句。",
        )

        parts = split_segment_for_display(segment, cue_seconds=4.0, max_cue_chars=90)

        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0].start, 10.0)
        self.assertAlmostEqual(parts[0].end, 13.333, places=3)
        self.assertEqual(parts[-1].end, 20.0)
        self.assertIn("First sentence", parts[0].text)
        self.assertIn("第二句", "".join(part.translated_text for part in parts))

    def test_split_segment_for_display_enforces_sentence_cap(self) -> None:
        segment = Segment(
            0.0,
            2.0,
            "First. Second. Third.",
            "eng",
            "甲。乙。丙。",
        )

        parts = split_segment_for_display(
            segment, cue_seconds=10.0, max_cue_chars=90, max_cue_sentences=1
        )

        self.assertEqual(len(parts), 3)
        self.assertIn("First", parts[0].text)
        self.assertIn("Second", parts[1].text)
        self.assertIn("Third", parts[2].text)

    def test_split_segment_for_display_sentence_cap_disabled(self) -> None:
        segment = Segment(
            0.0,
            2.0,
            "First. Second. Third.",
            "eng",
            "甲。乙。丙。",
        )

        parts = split_segment_for_display(
            segment, cue_seconds=10.0, max_cue_chars=90, max_cue_sentences=0
        )

        self.assertEqual(len(parts), 1)

    def test_split_text_balanced_breaks_long_words_by_words(self) -> None:
        parts = split_text_balanced("one two three four five six", 3)

        self.assertEqual(len(parts), 3)
        self.assertEqual(" ".join(parts), "one two three four five six")

    def test_split_text_balanced_breaks_oversized_sentence(self) -> None:
        text = "one two three four five six seven eight nine ten."
        parts = split_text_balanced(text, 3)

        self.assertEqual(len(parts), 3)
        self.assertLess(max(len(part) for part in parts), len(text))

    def test_continuous_audio_uses_configured_chunk_window(self) -> None:
        audio = np.ones(16_000 * 2, dtype=np.float32) * 0.02
        chunks = list(
            iter_audio_chunks(
                audio,
                chunk_seconds=1.0,
                min_chunk_seconds=0.5,
                silence_threshold=0.003,
            )
        )
        self.assertEqual([(start, end) for start, end, _ in chunks], [(0.0, 1.0), (1.0, 2.0)])

    def test_process_file_extracts_from_source_without_staging_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_file = root / "remote.mp4"
            input_file.write_bytes(b"media")
            config = BatchConfig(output_dir=root / "out", work_dir=str(root / "work"))
            Path(config.work_dir).mkdir()
            pipeline = SubtitlePipeline(FakeEngine(), config)

            with patch("pipeline.shutil.copyfile") as mock_copy:
                with patch("pipeline.extract_audio") as mock_extract:
                    with patch("pipeline.load_wav", return_value=np.ones(16_000, dtype=np.float32)):
                        result = pipeline.process_file(input_file)

        mock_copy.assert_not_called()
        self.assertEqual(mock_extract.call_args.args[0], input_file.resolve())
        self.assertEqual(result.segment_count, 1)

    def test_process_file_can_stage_input_before_extracting_audio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_file = root / "remote.mp4"
            input_file.write_bytes(b"media")
            config = BatchConfig(
                output_dir=root / "out",
                work_dir=str(root / "work"),
                stage_input=True,
            )
            Path(config.work_dir).mkdir()
            pipeline = SubtitlePipeline(FakeEngine(), config)

            def fake_copy(src: Path, dst: Path) -> None:
                Path(dst).write_bytes(Path(src).read_bytes())

            with patch("pipeline.shutil.copyfile", side_effect=fake_copy) as mock_copy:
                with patch("pipeline.extract_audio") as mock_extract:
                    with patch("pipeline.load_wav", return_value=np.ones(16_000, dtype=np.float32)):
                        pipeline.process_file(input_file)

        mock_copy.assert_called_once()
        extract_source = mock_extract.call_args.args[0]
        self.assertNotEqual(extract_source, input_file.resolve())
        self.assertEqual(extract_source.name, f"src_{input_file.name}")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for media extraction")
    def test_process_file_writes_bilingual_srt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_wav = root / "sample.wav"
            self._write_tone_wav(input_wav, seconds=2)

            config = BatchConfig(
                output_dir=root / "out",
                work_dir=str(root / "work"),
                chunk_seconds=1.0,
                min_chunk_seconds=0.5,
                silence_threshold=0.001,
                source_lang="eng",
                target_lang="zho",
            )
            Path(config.work_dir).mkdir()

            result = SubtitlePipeline(FakeEngine(), config).process_file(input_wav)

            self.assertEqual(result.output_path.name, "sample.zho.srt")
            self.assertEqual(result.segment_count, 2)
            output = result.output_path.read_text(encoding="utf-8")
            self.assertIn("source 1\ntarget 1", output)
            self.assertIn("source 2\ntarget 2", output)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for media extraction")
    def test_existing_output_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_wav = root / "sample.wav"
            self._write_tone_wav(input_wav, seconds=1)
            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "sample.zho.srt").write_text("existing", encoding="utf-8")

            config = BatchConfig(output_dir=out_dir, work_dir=str(root / "work"))
            Path(config.work_dir).mkdir()

            with self.assertRaises(FileExistsError):
                SubtitlePipeline(FakeEngine(), config).process_file(input_wav)

    @staticmethod
    def _write_tone_wav(path: Path, seconds: int) -> None:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16_000)
            samples = bytearray()
            for i in range(16_000 * seconds):
                value = int(12_000 * math.sin(2 * math.pi * 440 * i / 16_000))
                samples += value.to_bytes(2, "little", signed=True)
            wf.writeframes(samples)


if __name__ == "__main__":
    unittest.main()
