from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import batch
import server
from config import BatchConfig
from hf_utils import LocalSnapshotStatus, ModelInfo, ProxyConnectStatus
from mlx_whisper_engine import prefer_mlx_gpu


class BatchTests(unittest.TestCase):
    def test_collect_inputs_expands_directories_globs_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "a.mp4"
            nested = root / "nested"
            nested.mkdir()
            audio = nested / "b.wav"
            ignored = root / "notes.txt"
            for path in (video, audio, ignored):
                path.write_text("x", encoding="utf-8")

            result = batch.collect_inputs([str(root), str(root / "*.mp4")])

            self.assertEqual(result, [video.resolve(), audio.resolve()])

    def test_find_output_collisions_detects_same_stem_in_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "a" / "clip.mp4"
            second = root / "b" / "clip.mkv"
            for path in (first, second):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            config = BatchConfig(output_dir=root / "out", target_lang="zho")
            collisions = batch.find_output_collisions([first, second], config)

        self.assertEqual(len(collisions), 1)
        self.assertEqual(next(iter(collisions.values())), [first, second])

    def test_find_output_collisions_allows_distinct_stems(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "a" / "clip-one.mp4"
            second = root / "b" / "clip-two.mkv"
            config = BatchConfig(output_dir=root / "out", target_lang="zho")

            self.assertEqual(batch.find_output_collisions([first, second], config), {})

    def test_filter_existing_outputs_skips_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "clip.wav"
            media.write_text("x", encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            output_path = out_dir / "clip.zho.srt"
            output_path.write_text("existing", encoding="utf-8")
            config = BatchConfig(output_dir=out_dir, skip_existing=True)

            remaining, skipped = batch.filter_existing_outputs([media], config)

        self.assertEqual(remaining, [])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][0], media)
        self.assertIn("output exists", skipped[0][1])

    def test_filter_existing_outputs_ignores_existing_when_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "clip.wav"
            media.write_text("x", encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "clip.zho.srt").write_text("existing", encoding="utf-8")
            config = BatchConfig(output_dir=out_dir, skip_existing=True, overwrite=True)

            remaining, skipped = batch.filter_existing_outputs([media], config)

        self.assertEqual(remaining, [media])
        self.assertEqual(skipped, [])

    def test_validate_batch_config_accepts_defaults(self) -> None:
        config = BatchConfig()

        self.assertEqual(config.chunk_seconds, 28.0)
        self.assertEqual(config.cue_seconds, 4.0)
        self.assertEqual(config.max_cue_chars, 90)
        self.assertEqual(config.max_cue_sentences, 1)
        self.assertEqual(batch.validate_batch_config(config), [])

    def test_validate_batch_config_rejects_invalid_timing(self) -> None:
        config = BatchConfig(
            chunk_seconds=0,
            min_chunk_seconds=2,
            silence_threshold=-0.1,
            cue_seconds=0,
            max_cue_chars=0,
        )
        errors = batch.validate_batch_config(config)

        self.assertIn("--chunk-seconds must be greater than 0", errors)
        self.assertIn("--cue-seconds must be greater than 0", errors)
        self.assertIn("--max-cue-chars must be greater than 0", errors)
        self.assertIn("--min-chunk-seconds must be less than or equal to --chunk-seconds", errors)
        self.assertIn("--silence-threshold must be greater than or equal to 0", errors)

    def test_validate_batch_config_rejects_non_positive_min_chunk(self) -> None:
        config = BatchConfig(min_chunk_seconds=0)

        self.assertIn(
            "--min-chunk-seconds must be greater than 0",
            batch.validate_batch_config(config),
        )

    def test_create_engine_passes_mlx_translator_model(self) -> None:
        config = BatchConfig(
            backend="mlx-whisper",
            model_id="mlx-community/whisper-tiny",
            translator_model_id="mlx-community/Qwen3-1.7B-4bit",
        )

        with patch.object(batch, "MLXWhisperEngine") as mock_engine:
            batch.create_engine(config)

        mock_engine.assert_called_once_with(
            "mlx-community/whisper-tiny",
            translator_model_id="mlx-community/Qwen3-1.7B-4bit",
            local_only=False,
        )

    def test_filter_skips_external_sidecar_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "movie.mp4"
            media.write_text("x", encoding="utf-8")
            sidecar = root / "movie.zho.ass"
            sidecar.write_text("[Script Info]\n", encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            config = BatchConfig(output_dir=out_dir, skip_existing=True, target_lang="zho")

            remaining, skipped = batch.filter_existing_outputs([media], config)

        self.assertEqual(remaining, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("external subtitle", skipped[0][1])

    def test_filter_skips_external_sidecar_with_lang_alias(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "movie.mp4"
            media.write_text("x", encoding="utf-8")
            sidecar = root / "movie.chi.srt"
            sidecar.write_text("sub", encoding="utf-8")
            config = BatchConfig(skip_existing=True, target_lang="zho")

            remaining, skipped = batch.filter_existing_outputs([media], config)

        self.assertEqual(remaining, [])
        self.assertIn("external subtitle", skipped[0][1])

    def test_filter_skips_embedded_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "movie.mp4"
            media.write_text("x", encoding="utf-8")
            config = BatchConfig(skip_existing=True, target_lang="zho")

            with patch("batch.probe_subtitle_languages", return_value=["chi"]):
                remaining, skipped = batch.filter_existing_outputs([media], config)

        self.assertEqual(remaining, [])
        self.assertIn("embedded subtitle", skipped[0][1])

    def test_filter_does_not_skip_unrelated_embedded_lang(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "movie.mp4"
            media.write_text("x", encoding="utf-8")
            config = BatchConfig(skip_existing=True, target_lang="zho")

            with patch("batch.probe_subtitle_languages", return_value=["eng"]):
                remaining, skipped = batch.filter_existing_outputs([media], config)

        self.assertEqual(remaining, [media])
        self.assertEqual(skipped, [])


class ServerCliTests(unittest.TestCase):
    def test_mlx_whisper_selects_default_translator_for_chinese_target(self) -> None:
        args = server.argparse.Namespace(
            backend="mlx-whisper",
            model=None,
            translator_model=None,
            no_translate=False,
            source_lang="eng",
            target_lang="zho",
        )

        self.assertEqual(server.selected_model(args), server.DEFAULT_MLX_WHISPER_MODEL)
        self.assertEqual(
            server.selected_translator_model(args),
            server.DEFAULT_MLX_TRANSLATOR_MODEL,
        )

    def test_mlx_whisper_does_not_need_translator_for_source_only(self) -> None:
        args = server.argparse.Namespace(
            backend="mlx-whisper",
            model=None,
            translator_model=None,
            no_translate=True,
            source_lang="eng",
            target_lang="zho",
        )

        self.assertIsNone(server.selected_translator_model(args))

    def test_prefer_mlx_gpu_sets_default_device(self) -> None:
        class FakeMlx:
            gpu = "gpu0"

            def __init__(self) -> None:
                self.selected = "cpu0"

            def set_default_device(self, device: str) -> None:
                self.selected = device

            def default_device(self) -> str:
                return self.selected

        fake_mlx = FakeMlx()

        self.assertEqual(prefer_mlx_gpu(fake_mlx), "gpu0")

    def test_dry_run_prints_mapping_without_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "clip.wav"
            media.write_text("x", encoding="utf-8")
            output = StringIO()

            with patch.object(batch, "MLXWhisperEngine") as mock_engine:
                with patch.object(
                    sys,
                    "argv",
                    ["server.py", str(media), "--dry-run", "--output-dir", str(root / "out")],
                ):
                    with redirect_stdout(output):
                        self.assertEqual(server.main(), 0)

        mock_engine.assert_not_called()
        self.assertIn("inputs=1", output.getvalue())
        self.assertIn("process\t", output.getvalue())
        self.assertIn("clip.wav", output.getvalue())
        self.assertIn("clip.zho.srt", output.getvalue())

    def test_dry_run_mlx_backend_prints_models_without_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "clip.wav"
            media.write_text("x", encoding="utf-8")
            output = StringIO()

            with patch.object(batch, "MLXWhisperEngine") as mock_engine:
                with patch.object(
                    sys,
                    "argv",
                    [
                        "server.py",
                        str(media),
                        "--backend",
                        "mlx-whisper",
                        "--dry-run",
                        "--output-dir",
                        str(root / "out"),
                    ],
                ):
                    with redirect_stdout(output):
                        self.assertEqual(server.main(), 0)

        mock_engine.assert_not_called()
        self.assertIn("backend=mlx-whisper", output.getvalue())
        self.assertIn("mlx-community/Qwen3-1.7B-4bit", output.getvalue())

    def test_skip_existing_allows_exit_without_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "clip.wav"
            media.write_text("x", encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            (out_dir / "clip.zho.srt").write_text("existing", encoding="utf-8")
            output = StringIO()

            with patch.object(batch, "MLXWhisperEngine") as mock_engine:
                with patch.object(
                    sys,
                    "argv",
                    ["server.py", str(media), "--skip-existing", "--output-dir", str(out_dir)],
                ):
                    with redirect_stdout(output):
                        self.assertEqual(server.main(), 0)

        mock_engine.assert_not_called()
        self.assertIn("skip:", output.getvalue())
        self.assertIn("output exists", output.getvalue())

    def test_check_model_exits_without_inputs(self) -> None:
        with patch.object(
            server,
            "get_model_info",
            return_value=ModelInfo("facebook/hf-seamless-m4t-medium", "abc123", True, 10, 1024),
        ) as mock_info:
            with patch.object(sys, "argv", ["server.py", "--backend", "seamless", "--check-model"]):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(server.main(), 0)

        mock_info.assert_called_once_with("facebook/seamless-m4t-medium")
        self.assertIn("files=10", output.getvalue())
        self.assertIn("size=1.0 KiB", output.getvalue())

    def test_download_model_exits_without_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            with patch.object(server, "resolve_model_dir", return_value=target) as mock_resolve:
                with patch.object(sys, "argv", ["server.py", "--backend", "seamless", "--download-model"]):
                    with redirect_stdout(StringIO()):
                        self.assertEqual(server.main(), 0)

        mock_resolve.assert_called_once_with("facebook/seamless-m4t-medium", local_only=False)

    def test_default_download_model_resolves_mlx_models(self) -> None:
        with patch.object(server, "resolve_mlx_model_path", side_effect=["/whisper", "/translator"]) as mock_resolve:
            with patch.object(sys, "argv", ["server.py", "--download-model"]):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(server.main(), 0)

        self.assertEqual(
            mock_resolve.call_args_list,
            [
                unittest.mock.call("mlx-community/whisper-tiny", local_only=False),
                unittest.mock.call("mlx-community/Qwen3-1.7B-4bit", local_only=False),
            ],
        )
        self.assertIn("/whisper", output.getvalue())
        self.assertIn("/translator", output.getvalue())

    def test_download_model_passes_local_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td)
            with patch.object(server, "resolve_model_dir", return_value=target) as mock_resolve:
                with patch.object(sys, "argv", ["server.py", "--backend", "seamless", "--download-model", "--local-only"]):
                    with redirect_stdout(StringIO()):
                        self.assertEqual(server.main(), 0)

        mock_resolve.assert_called_once_with("facebook/seamless-m4t-medium", local_only=True)

    def test_check_model_reports_runtime_error_without_traceback(self) -> None:
        with patch.object(server, "get_model_info", side_effect=RuntimeError("proxy failed")):
            with patch.object(sys, "argv", ["server.py", "--backend", "seamless", "--check-model"]):
                self.assertEqual(server.main(), 1)

    def test_default_check_model_reports_mlx_models(self) -> None:
        with patch.object(sys, "argv", ["server.py", "--check-model"]):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(server.main(), 0)

        self.assertIn("backend=mlx-whisper", output.getvalue())
        self.assertIn("translator=mlx-community/Qwen3-1.7B-4bit", output.getvalue())

    def test_check_network_exits_without_inputs(self) -> None:
        with patch.object(server, "check_network", return_value="ok status=200 url=https://example.test") as mock_check:
            with patch.object(sys, "argv", ["server.py", "--check-network"]):
                with redirect_stdout(StringIO()):
                    self.assertEqual(server.main(), 0)

        mock_check.assert_called_once_with()

    def test_check_proxy_exits_without_inputs(self) -> None:
        status = ProxyConnectStatus(
            proxy="http://127.0.0.1:8080",
            target="huggingface.co:443",
            ok=True,
            response="HTTP/1.1 200 Connection established",
        )
        with patch.object(server, "check_proxy_connect", return_value=status) as mock_check:
            with patch.object(sys, "argv", ["server.py", "--check-proxy"]):
                with redirect_stdout(StringIO()):
                    self.assertEqual(server.main(), 0)

        mock_check.assert_called_once_with()

    def test_model_status_exits_without_inputs(self) -> None:
        status = LocalSnapshotStatus(
            model_id="facebook/hf-seamless-m4t-medium",
            revision="rev123",
            path=Path("/tmp/model"),
            file_count=4,
            total_size_bytes=1024,
            temp_file_count=1,
            temp_size_bytes=2048,
            has_config=True,
        )
        with patch.object(server, "get_local_model_status", return_value=[status]) as mock_status:
            with patch.object(sys, "argv", ["server.py", "--backend", "seamless", "--model-status"]):
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(server.main(), 0)

        mock_status.assert_called_once_with("facebook/seamless-m4t-medium")
        self.assertIn("partial_files=1", output.getvalue())

    def test_apply_proxy_args_sets_override_proxy(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            server.apply_proxy_args("http://proxy.test:8080", False)
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.test:8080")
            self.assertEqual(os.environ["https_proxy"], "http://proxy.test:8080")

    def test_apply_proxy_args_removes_proxy(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.test:8080",
                "https_proxy": "http://proxy.test:8080",
            },
            clear=True,
        ):
            server.apply_proxy_args(None, True)
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertNotIn("https_proxy", os.environ)


class AudioUtilsSubtitleTests(unittest.TestCase):
    def test_langs_match_same_code(self) -> None:
        from audio_utils import langs_match
        self.assertTrue(langs_match("eng", "eng"))

    def test_langs_match_aliases(self) -> None:
        from audio_utils import langs_match
        self.assertTrue(langs_match("zho", "chi"))
        self.assertTrue(langs_match("zh", "zho"))
        self.assertTrue(langs_match("chi", "zh"))

    def test_langs_match_different(self) -> None:
        from audio_utils import langs_match
        self.assertFalse(langs_match("eng", "zho"))
        self.assertFalse(langs_match("jpn", "kor"))

    def test_normalize_lang_unknown_code_passthrough(self) -> None:
        from audio_utils import normalize_lang
        self.assertEqual(normalize_lang("xyz"), "xyz")

    def test_find_external_subtitles_returns_match(self) -> None:
        from audio_utils import find_external_subtitles
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "clip.mp4"
            video.write_text("x", encoding="utf-8")
            sub = root / "clip.zho.srt"
            sub.write_text("sub", encoding="utf-8")

            result = find_external_subtitles(video, "zho")

        self.assertEqual(result, sub)

    def test_find_external_subtitles_returns_none_for_wrong_lang(self) -> None:
        from audio_utils import find_external_subtitles
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "clip.mp4"
            video.write_text("x", encoding="utf-8")
            sub = root / "clip.eng.srt"
            sub.write_text("sub", encoding="utf-8")

            result = find_external_subtitles(video, "zho")

        self.assertIsNone(result)

    def test_probe_subtitle_languages_parses_json(self) -> None:
        from audio_utils import probe_subtitle_languages
        ffprobe_output = json.dumps({
            "streams": [
                {"index": 2, "codec_type": "subtitle", "tags": {"language": "chi"}},
                {"index": 3, "codec_type": "subtitle", "tags": {"language": "eng"}},
            ]
        })
        with patch("audio_utils.shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("audio_utils.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=ffprobe_output, stderr=""
                )
                result = probe_subtitle_languages(Path("/fake/video.mkv"))

        self.assertEqual(result, ["chi", "eng"])

    def test_probe_subtitle_languages_returns_empty_on_failure(self) -> None:
        from audio_utils import probe_subtitle_languages
        with patch("audio_utils.shutil.which", return_value="/usr/bin/ffprobe"):
            with patch("audio_utils.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="error"
                )
                result = probe_subtitle_languages(Path("/fake/video.mkv"))

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
