from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import hf_utils


class HuggingFaceMirrorTests(unittest.TestCase):
    def test_default_endpoint_is_official_huggingface(self) -> None:
        self.assertEqual(hf_utils.HF_ENDPOINT, "https://huggingface.co")

    def test_resolve_model_dir_accepts_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(hf_utils.resolve_model_dir(td), Path(td).resolve())

    def test_resolve_model_dir_downloads_selected_transformers_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {hf_utils.MODEL_CACHE_ENV: td}, clear=True):
                with patch.object(
                    hf_utils,
                    "_remote_model_info",
                    return_value={
                        "revision": "rev123",
                        "files": [
                            hf_utils.RemoteFile("config.json", 10),
                            hf_utils.RemoteFile("pytorch_model.bin", 20),
                            hf_utils.RemoteFile("m4t_v2_multitask_unity2.pt", 30),
                        ],
                    },
                ):
                    with patch.object(hf_utils, "_download_file") as mock_download:
                        resolved = hf_utils.resolve_model_dir("facebook/seamless-m4t-medium")

            self.assertEqual(resolved, Path(td, "facebook--hf-seamless-m4t-medium", "rev123").resolve())
            self.assertEqual([call.args[2] for call in mock_download.call_args_list], [
                "config.json",
                "pytorch_model.bin",
            ])

    def test_resolve_model_dir_uses_complete_local_snapshot_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            snapshot = Path(td) / "facebook--hf-seamless-m4t-medium" / "rev123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "pytorch_model.bin").write_bytes(b"weights")

            with patch.dict(os.environ, {hf_utils.MODEL_CACHE_ENV: td}, clear=True):
                with patch.object(hf_utils, "download_model_snapshot") as mock_download:
                    resolved = hf_utils.resolve_model_dir("facebook/seamless-m4t-medium")

        self.assertEqual(resolved, snapshot.resolve())
        mock_download.assert_not_called()

    def test_resolve_model_dir_ignores_partial_local_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            snapshot = Path(td) / "facebook--hf-seamless-m4t-medium" / "rev123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "pytorch_model.bin.tmp").write_bytes(b"partial")

            target = Path(td) / "downloaded"
            with patch.dict(os.environ, {hf_utils.MODEL_CACHE_ENV: td}, clear=True):
                with patch.object(hf_utils, "download_model_snapshot", return_value=target) as mock_download:
                    resolved = hf_utils.resolve_model_dir("facebook/seamless-m4t-medium")

        self.assertEqual(resolved, target)
        mock_download.assert_called_once_with("facebook/hf-seamless-m4t-medium")

    def test_resolve_model_dir_local_only_rejects_missing_complete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            snapshot = Path(td) / "facebook--hf-seamless-m4t-medium" / "rev123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "pytorch_model.bin.tmp").write_bytes(b"partial")

            with patch.dict(os.environ, {hf_utils.MODEL_CACHE_ENV: td}, clear=True):
                with patch.object(hf_utils, "download_model_snapshot") as mock_download:
                    with self.assertRaises(RuntimeError):
                        hf_utils.resolve_model_dir("facebook/seamless-m4t-medium", local_only=True)

        mock_download.assert_not_called()

    def test_get_model_info_reports_selected_download_size(self) -> None:
        fake_info = SimpleNamespace(
            modelId="facebook/hf-seamless-m4t-medium",
            sha="rev123",
            siblings=[
                SimpleNamespace(rfilename="config.json", size=10),
                SimpleNamespace(rfilename="pytorch_model.bin", size=20),
                SimpleNamespace(rfilename="seamlessM4T_v2_large.pt", size=30),
            ],
        )
        fake_api = SimpleNamespace(model_info=lambda *args, **kwargs: fake_info)
        with patch("huggingface_hub.HfApi", return_value=fake_api):
            info = hf_utils.get_model_info("facebook/seamless-m4t-medium")

        self.assertEqual(info.file_count, 2)
        self.assertEqual(info.total_size_bytes, 30)
        self.assertTrue(info.has_config)

    def test_resolve_model_id_aliases_medium_to_transformers_layout(self) -> None:
        self.assertEqual(
            hf_utils.resolve_model_id("facebook/seamless-m4t-medium"),
            "facebook/hf-seamless-m4t-medium",
        )

    def test_format_bytes(self) -> None:
        self.assertEqual(hf_utils.format_bytes(0), "0 B")
        self.assertEqual(hf_utils.format_bytes(1024), "1.0 KiB")
        self.assertEqual(hf_utils.format_bytes(1024 * 1024 * 3), "3.0 MiB")

    def test_load_env_file_does_not_override_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "HF_TOKEN=file-token\nEXISTING=file-value\n# ignored\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"EXISTING": "process-value"}, clear=True):
                hf_utils.load_env_file(env_path)
                self.assertEqual(os.environ["HF_TOKEN"], "file-token")
                self.assertEqual(os.environ["EXISTING"], "process-value")

    def test_load_env_file_mirrors_proxy_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text("HTTPS_PROXY=http://file-proxy:1\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                hf_utils.load_env_file(env_path)
                self.assertEqual(os.environ["HTTPS_PROXY"], "http://file-proxy:1")
                self.assertEqual(os.environ["https_proxy"], "http://file-proxy:1")

    def test_load_env_file_preserves_proxy_override_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / ".env"
            env_path.write_text(
                "HTTPS_PROXY=http://file-proxy:1\nhttps_proxy=http://file-proxy:1\nHF_TOKEN=file-token\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    hf_utils.PROXY_OVERRIDE_ENV: "override",
                    "HTTPS_PROXY": "http://cli-proxy:2",
                    "https_proxy": "http://cli-proxy:2",
                },
                clear=True,
            ):
                hf_utils.load_env_file(env_path)
                self.assertEqual(os.environ["HTTPS_PROXY"], "http://cli-proxy:2")
                self.assertEqual(os.environ["https_proxy"], "http://cli-proxy:2")
                self.assertEqual(os.environ["HF_TOKEN"], "file-token")

    def test_check_network_uses_urlopen(self) -> None:
        @contextmanager
        def fake_response(request, timeout):
            self.assertEqual(timeout, 15)
            self.assertEqual(request.full_url, "https://example.test")
            yield SimpleNamespace(status=200)

        with patch.object(hf_utils, "urlopen", side_effect=fake_response):
            self.assertIn("status=200", hf_utils.check_network("https://example.test"))

    def test_check_proxy_connect_reports_proxy_response(self) -> None:
        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def sendall(self, data):
                self.request = data

            def recv(self, size):
                return b"HTTP/1.1 200 Connection established\r\n\r\n"

        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:8080"}, clear=True):
            with patch.object(hf_utils.socket, "create_connection", return_value=FakeSocket()):
                status = hf_utils.check_proxy_connect()

        self.assertTrue(status.ok)
        self.assertEqual(status.response, "HTTP/1.1 200 Connection established")

    def test_get_local_model_status_reports_partial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot = root / "facebook--hf-seamless-m4t-medium" / "rev123"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"{}")
            (snapshot / "pytorch_model.bin.tmp").write_bytes(b"partial")

            with patch.dict(os.environ, {hf_utils.MODEL_CACHE_ENV: td}, clear=True):
                statuses = hf_utils.get_local_model_status("facebook/seamless-m4t-medium")

        self.assertEqual(len(statuses), 1)
        self.assertTrue(statuses[0].has_config)
        self.assertEqual(statuses[0].file_count, 1)
        self.assertEqual(statuses[0].temp_file_count, 1)
        self.assertEqual(statuses[0].temp_size_bytes, len(b"partial"))


if __name__ == "__main__":
    unittest.main()
