"""Single-run background controller for the web UI.

Spawns ``server/server.py`` as a subprocess, parses its log output to track
file-level progress, and exposes a thread-safe status dict for polling.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / "server" / "server.py"
LOG_BUFFER_LINES = 400
LOG_TAIL_LINES = 80

PROCESSING_RE = re.compile(r"\[(\d+)/(\d+)\] Processing (.+)")
COMPLETED_RE = re.compile(r"^(.+\.srt) \((\d+) segments\)\s*$")
NO_INPUTS_RE = re.compile(r"No supported input media files found")
DURATION_RE = re.compile(r"Audio duration ([\d.]+)s")
CHUNK_RE = re.compile(r"Processing ([\d.]+)s-([\d.]+)s")


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._state: str = "idle"
        self._error: str = ""
        self._exit_code: int | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._current_index: int = 0
        self._total: int = 0
        self._current_file: str = ""
        self._first_file_started_at: float | None = None
        self._completed: int = 0
        self._file_duration_s: float = 0.0
        self._chunk_start_s: float = 0.0
        self._chunk_end_s: float = 0.0
        self._chunk_count: int = 0
        self._log: deque[str] = deque(maxlen=LOG_BUFFER_LINES)
        self._cmd: list[str] = []

    def start(self, config: dict) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._status_locked()
            cmd_or_error = self._build_command(config)
            if isinstance(cmd_or_error, str):
                self._reset_locked()
                self._state = "failed"
                self._error = cmd_or_error
                self._finished_at = time.monotonic()
                return self._status_locked()
            self._reset_locked()
            self._cmd = cmd_or_error
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            try:
                self._proc = subprocess.Popen(
                    cmd_or_error,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    env=env,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._state = "failed"
                self._error = str(exc)
                self._finished_at = time.monotonic()
                return self._status_locked()
            self._state = "running"
            self._started_at = time.monotonic()
            self._log.append("$ " + " ".join(cmd_or_error))
            self._reader = threading.Thread(target=self._read_loop, daemon=True)
            self._reader.start()
            return self._status_locked()

    def cancel(self) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except OSError as exc:
                    logger.warning("terminate failed: %s", exc)
                if self._state == "running":
                    self._state = "cancelling"
            return self._status_locked()

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _reset_locked(self) -> None:
        self._proc = None
        self._reader = None
        self._state = "idle"
        self._error = ""
        self._exit_code = None
        self._started_at = None
        self._finished_at = None
        self._current_index = 0
        self._total = 0
        self._current_file = ""
        self._first_file_started_at = None
        self._completed = 0
        self._file_duration_s = 0.0
        self._chunk_start_s = 0.0
        self._chunk_end_s = 0.0
        self._chunk_count = 0
        self._log.clear()
        self._cmd = []

    def _build_command(self, config: dict):
        source = (config.get("source_folder") or "").strip()
        if not source:
            return "source folder is empty — set it in the form above first"
        source_path = Path(source).expanduser()
        if not source_path.is_dir():
            return f"source folder does not exist: {source_path}"
        cmd = [
            sys.executable,
            str(SERVER_SCRIPT),
            str(source_path),
            "--source-lang",
            (config.get("source_lang") or "auto").strip(),
            "--target-lang",
            (config.get("target_lang") or "zho").strip(),
            "--log-level",
            "INFO",
        ]
        dest = (config.get("dest_folder") or "").strip()
        if dest and dest != source:
            cmd.extend(["--output-dir", str(Path(dest).expanduser())])
        if config.get("overwrite"):
            cmd.append("--overwrite")
        else:
            cmd.append("--skip-existing")
        return cmd

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                with self._lock:
                    self._log.append(line)
                    self._consume_line_locked(line)
        except Exception as exc:
            logger.warning("run output reader error: %s", exc)
        rc = proc.wait()
        with self._lock:
            if self._proc is not proc:
                return
            self._finished_at = time.monotonic()
            self._exit_code = rc
            if self._state == "cancelling":
                self._state = "cancelled"
            elif rc == 0:
                self._state = "done"
                if self._total > 0:
                    self._completed = self._total
            else:
                self._state = "failed"
                if not self._error:
                    self._error = f"process exited with code {rc}"

    def _consume_line_locked(self, line: str) -> None:
        m = PROCESSING_RE.search(line)
        if m:
            self._current_index = int(m.group(1))
            self._total = int(m.group(2))
            self._current_file = Path(m.group(3)).name
            self._file_duration_s = 0.0
            self._chunk_start_s = 0.0
            self._chunk_end_s = 0.0
            self._chunk_count = 0
            if self._first_file_started_at is None:
                self._first_file_started_at = time.monotonic()
            return
        m = DURATION_RE.search(line)
        if m:
            self._file_duration_s = float(m.group(1))
            return
        m = CHUNK_RE.search(line)
        if m:
            self._chunk_start_s = float(m.group(1))
            self._chunk_end_s = float(m.group(2))
            self._chunk_count += 1
            return
        m = COMPLETED_RE.match(line)
        if m and self._current_index > 0:
            self._completed = max(self._completed, self._current_index)
            if self._file_duration_s > 0:
                self._chunk_start_s = self._file_duration_s
                self._chunk_end_s = self._file_duration_s
            return
        if NO_INPUTS_RE.search(line):
            self._error = "no supported media files in the source folder"

    def _status_locked(self) -> dict:
        now = time.monotonic()
        if self._started_at is None:
            elapsed = 0.0
        elif self._finished_at is not None:
            elapsed = self._finished_at - self._started_at
        else:
            elapsed = now - self._started_at

        eta: float | None = None
        if self._state == "running" and self._total > 0:
            if self._completed > 0 and self._first_file_started_at is not None:
                work_elapsed = now - self._first_file_started_at
                avg = work_elapsed / self._completed
                eta = max(0.0, avg * (self._total - self._completed))

        return {
            "state": self._state,
            "current_index": self._current_index,
            "total": self._total,
            "completed": self._completed,
            "current_file": self._current_file,
            "file_duration_s": round(self._file_duration_s, 1),
            "chunk_start_s": round(self._chunk_start_s, 1),
            "chunk_end_s": round(self._chunk_end_s, 1),
            "chunk_count": self._chunk_count,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "exit_code": self._exit_code,
            "error": self._error,
            "command": self._cmd,
            "log_tail": list(self._log)[-LOG_TAIL_LINES:],
        }


    def preview(self, config: dict) -> dict:
        """Run --dry-run and return structured results."""
        cmd_or_error = self._build_command(config)
        if isinstance(cmd_or_error, str):
            return {"error": cmd_or_error, "to_process": [], "skipped": []}
        cmd = cmd_or_error + ["--dry-run"]
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=env,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"error": str(exc), "to_process": [], "skipped": []}
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            return {"error": stderr or f"dry-run exited with code {result.returncode}", "to_process": [], "skipped": []}
        return _parse_dry_run(result.stdout)


def _parse_dry_run(output: str) -> dict:
    to_process: list[dict] = []
    skipped: list[dict] = []
    for line in output.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == "skip":
            skipped.append({"path": parts[1], "name": Path(parts[1]).name, "reason": parts[2]})
        elif len(parts) >= 3 and parts[0] == "process":
            to_process.append({"path": parts[1], "name": Path(parts[1]).name, "output": parts[2]})
    return {"to_process": to_process, "skipped": skipped, "error": ""}


_manager = RunManager()


def get_manager() -> RunManager:
    return _manager
