"""Lightweight web frontend for Subtitle Anywhere configuration.

Serves a single-page app (web/index.html) plus a small JSON API for
reading and writing the user's saved folders. Uses only the Python
standard library so the runtime stays dependency-free.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from run_manager import get_manager

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
CONFIG_PATH = Path(
    os.environ.get(
        "SUBTITLE_ANYWHERE_CONFIG",
        str(Path.home() / ".config" / "subtitle_anywhere" / "config.json"),
    )
)

DEFAULT_CONFIG = {
    "source_folder": "",
    "dest_folder": "",
    "source_lang": "auto",
    "target_lang": "zho",
    "overwrite": False,
}

ALLOWED_KEYS = set(DEFAULT_CONFIG.keys())


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    for key, value in data.items():
        if key in ALLOWED_KEYS:
            merged[key] = value
    return merged


def save_config(payload: dict) -> dict:
    sanitized = dict(DEFAULT_CONFIG)
    for key, value in payload.items():
        if key not in ALLOWED_KEYS:
            continue
        if key in {"source_folder", "dest_folder", "source_lang", "target_lang"}:
            sanitized[key] = str(value or "").strip()
        elif key == "overwrite":
            sanitized[key] = bool(value)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(sanitized, indent=2, sort_keys=True), encoding="utf-8")
    return sanitized


def pick_folder(prompt: str, default: str = "") -> str | None:
    """Open a native macOS folder picker via osascript.

    Returns the chosen POSIX path, or None if the user cancelled.
    """
    default_clause = ""
    if default:
        default_path = Path(default).expanduser()
        if default_path.is_dir():
            default_clause = f' default location POSIX file (item 2 of argv)'
    script = (
        'on run argv\n'
        'try\n'
        f'  set theFolder to choose folder with prompt (item 1 of argv){default_clause}\n'
        '  return POSIX path of theFolder\n'
        'on error number -128\n'
        '  return ""\n'
        'end try\n'
        'end run'
    )
    cmd = ["osascript", "-e", script, prompt]
    if default_clause:
        cmd.append(str(Path(default).expanduser()))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("folder picker failed: %s", exc)
        return None
    path = result.stdout.strip()
    if not path:
        return None
    return path.rstrip("/") or "/"


def pick_file(prompt: str, default: str = "") -> str | None:
    """Open a native macOS file picker via osascript.

    Returns the chosen POSIX path, or None if the user cancelled.
    """
    default_clause = ""
    if default:
        default_path = Path(default).expanduser()
        parent = default_path.parent if default_path.is_file() else default_path
        if parent.is_dir():
            default_clause = f' default location POSIX file (item 2 of argv)'
    script = (
        'on run argv\n'
        'try\n'
        f'  set theFile to choose file with prompt (item 1 of argv){default_clause}\n'
        '  return POSIX path of theFile\n'
        'on error number -128\n'
        '  return ""\n'
        'end try\n'
        'end run'
    )
    cmd = ["osascript", "-e", script, prompt]
    if default_clause:
        parent = Path(default).expanduser()
        if parent.is_file():
            parent = parent.parent
        cmd.append(str(parent))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning("file picker failed: %s", exc)
        return None
    path = result.stdout.strip()
    if not path:
        return None
    return path


def folder_info(path_str: str) -> dict:
    info = {"path": path_str, "exists": False, "is_dir": False, "video_count": 0}
    if not path_str:
        return info
    path = Path(path_str).expanduser()
    info["exists"] = path.exists()
    info["is_dir"] = path.is_dir()
    if info["is_dir"]:
        exts = (".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm")
        info["video_count"] = sum(1 for p in path.rglob("*") if p.suffix.lower() in exts)
    return info


class WebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), format % args)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length") or "0")
        if not length:
            return None
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/config":
            cfg = load_config()
            src = cfg.get("source_folder", "")
            dst = cfg.get("dest_folder", "") or src
            self._json(
                HTTPStatus.OK,
                {
                    "config": cfg,
                    "source_info": folder_info(src),
                    "dest_info": folder_info(dst),
                    "config_path": str(CONFIG_PATH),
                },
            )
            return
        if self.path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True})
            return
        if self.path == "/api/run":
            self._json(HTTPStatus.OK, get_manager().status())
            return
        if self.path == "/api/run/preview":
            self._json(HTTPStatus.OK, get_manager().preview_status())
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/pick-folder":
            params = parse_qs(parsed.query)
            prompt = (params.get("prompt") or ["Choose a folder"])[0]
            default = (params.get("default") or [""])[0]
            picked = pick_folder(prompt, default)
            if picked is None:
                self._json(HTTPStatus.OK, {"cancelled": True})
            else:
                self._json(HTTPStatus.OK, {"path": picked, "info": folder_info(picked)})
            return
        if parsed.path == "/api/pick-file":
            params = parse_qs(parsed.query)
            prompt = (params.get("prompt") or ["Choose a video file"])[0]
            default = (params.get("default") or [""])[0]
            picked = pick_file(prompt, default)
            if picked is None:
                self._json(HTTPStatus.OK, {"cancelled": True})
            else:
                self._json(HTTPStatus.OK, {"path": picked})
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/run/cancel":
            self._json(HTTPStatus.OK, get_manager().cancel())
            return
        if self.path == "/api/run/preview":
            payload = self._read_json_body()
            cfg = save_config(payload) if isinstance(payload, dict) else load_config()
            self._json(HTTPStatus.OK, get_manager().preview(cfg))
            return
        if self.path == "/api/run":
            payload = self._read_json_body()
            cfg = save_config(payload) if isinstance(payload, dict) else load_config()
            self._json(HTTPStatus.OK, get_manager().start(cfg))
            return
        if self.path != "/api/config":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        payload = self._read_json_body()
        if payload is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
            return
        saved = save_config(payload)
        src = saved.get("source_folder", "")
        dst = saved.get("dest_folder", "") or src
        self._json(
            HTTPStatus.OK,
            {
                "config": saved,
                "source_info": folder_info(src),
                "dest_info": folder_info(dst),
                "config_path": str(CONFIG_PATH),
            },
        )


def serve(host: str, port: int) -> None:
    if not WEB_DIR.is_dir():
        raise RuntimeError(f"web directory missing: {WEB_DIR}")
    server = HTTPServer((host, port), WebHandler)
    logger.info("Subtitle Anywhere web UI on http://%s:%d (config %s)", host, port, CONFIG_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
