"""Single-run background controller for the web UI.

Runs the subtitle pipeline in-process so that the MLX engine (and its
loaded model weights) stay warm across consecutive runs.  The dry-run
preview still shells out to ``server/cli.py --dry-run`` because it
never needs the models.
"""

from __future__ import annotations

import logging
import os
import select
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from batch import (
    collect_inputs,
    create_engine,
    filter_existing_outputs,
    find_output_collisions,
    validate_batch_config,
)
from config import BatchConfig
from pipeline import SubtitlePipeline

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / "server" / "cli.py"
LOG_BUFFER_LINES = 400
LOG_TAIL_LINES = 80


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
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
        self._engine: object | None = None
        self._engine_key: tuple | None = None
        self._preview_proc: subprocess.Popen | None = None
        self._preview_state: str = "idle"
        self._preview_error: str = ""
        self._preview_progress: str = ""
        self._preview_lines: list[str] = []
        self._preview_result: dict | None = None

    def _log_line(self, msg: str) -> None:
        self._log.append(msg)

    def _get_or_create_engine(self, config: BatchConfig):
        key = (config.backend, config.model_id, config.translator_model_id, config.local_only, config.device)
        if self._engine is not None and self._engine_key == key:
            return self._engine
        self._engine = create_engine(config)
        self._engine_key = key
        return self._engine

    def start(self, config: dict) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._status_locked()
            self._reset_locked()

            source_file = (config.get("source_file") or "").strip()
            source_folder = (config.get("source_folder") or "").strip()

            if source_file:
                source_path = Path(source_file).expanduser()
                if not source_path.is_file():
                    self._state = "failed"
                    self._error = f"source file does not exist: {source_path}"
                    self._finished_at = time.monotonic()
                    return self._status_locked()
                input_files = [source_path.resolve()]
            elif source_folder:
                source_path = Path(source_folder).expanduser()
                if not source_path.is_dir():
                    self._state = "failed"
                    self._error = f"source folder does not exist: {source_path}"
                    self._finished_at = time.monotonic()
                    return self._status_locked()
                input_files = None
            else:
                self._state = "failed"
                self._error = "no source selected — pick a file or folder first"
                self._finished_at = time.monotonic()
                return self._status_locked()

            batch_config = self._build_batch_config(config, source_path)
            self._state = "running"
            self._started_at = time.monotonic()
            self._cancel_event.clear()
            self._log_line(f"Starting in-process run: source={source_path}")
            self._thread = threading.Thread(
                target=self._run_pipeline,
                args=(batch_config, source_path, input_files),
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def cancel(self) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._cancel_event.set()
                if self._state == "running":
                    self._state = "cancelling"
            return self._status_locked()

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _reset_locked(self) -> None:
        self._thread = None
        self._cancel_event = threading.Event()
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

    def _build_batch_config(self, config: dict, source_path: Path) -> BatchConfig:
        dest = (config.get("dest_folder") or "").strip()
        output_dir = Path(dest).expanduser() if dest and dest != str(source_path) else None
        overwrite = bool(config.get("overwrite"))
        return BatchConfig(
            source_lang=(config.get("source_lang") or "auto").strip(),
            target_lang=(config.get("target_lang") or "zho").strip(),
            output_dir=output_dir,
            overwrite=overwrite,
            skip_existing=not overwrite,
            stage_input=bool(config.get("stage_input")),
        )

    def _run_pipeline(self, config: BatchConfig, source_path: Path, input_files: list[Path] | None = None) -> None:
        try:
            if input_files is not None:
                inputs = input_files
            else:
                with self._lock:
                    self._log_line("Collecting input files...")
                inputs = collect_inputs([str(source_path)])
            if not inputs:
                with self._lock:
                    self._state = "failed"
                    self._error = "no supported media files in the source folder"
                    self._finished_at = time.monotonic()
                return

            Path(config.work_dir).mkdir(parents=True, exist_ok=True)
            if config.output_dir:
                config.output_dir.mkdir(parents=True, exist_ok=True)

            validation_errors = validate_batch_config(config)
            if validation_errors:
                with self._lock:
                    self._state = "failed"
                    self._error = "; ".join(validation_errors)
                    self._finished_at = time.monotonic()
                return

            inputs, skipped_existing = filter_existing_outputs(inputs, config)
            collisions = find_output_collisions(inputs, config)
            if collisions:
                with self._lock:
                    self._state = "failed"
                    self._error = "output path collisions detected"
                    self._finished_at = time.monotonic()
                return

            for input_path, reason in skipped_existing:
                with self._lock:
                    self._log_line(f"skip: {input_path.name} ({reason})")

            if not inputs:
                with self._lock:
                    self._state = "done"
                    self._finished_at = time.monotonic()
                    self._log_line("No inputs to process after skipping existing outputs")
                return

            with self._lock:
                self._total = len(inputs)
                self._log_line(f"Loading engine (model weights stay warm across runs)...")

            engine = self._get_or_create_engine(config)
            pipeline = SubtitlePipeline(engine, config)

            with self._lock:
                self._log_line(f"Engine ready, processing {len(inputs)} file(s)")

            failed = 0
            for index, input_path in enumerate(inputs, start=1):
                if self._cancel_event.is_set():
                    with self._lock:
                        self._state = "cancelled"
                        self._finished_at = time.monotonic()
                        self._log_line("Cancelled by user")
                    return

                with self._lock:
                    self._current_index = index
                    self._current_file = input_path.name
                    self._file_duration_s = 0.0
                    self._chunk_start_s = 0.0
                    self._chunk_end_s = 0.0
                    self._chunk_count = 0
                    if self._first_file_started_at is None:
                        self._first_file_started_at = time.monotonic()
                    self._log_line(f"[{index}/{len(inputs)}] Processing {input_path}")

                try:
                    result = pipeline.process_file(input_path)
                    with self._lock:
                        self._completed = index
                        self._log_line(f"{result.output_path} ({result.segment_count} segments)")
                except Exception:
                    failed += 1
                    logger.exception("Failed: %s", input_path)
                    with self._lock:
                        self._log_line(f"FAILED: {input_path}")

            with self._lock:
                self._finished_at = time.monotonic()
                if failed:
                    self._state = "failed"
                    self._error = f"{failed} of {len(inputs)} files failed"
                    self._exit_code = 1
                else:
                    self._state = "done"
                    self._completed = len(inputs)
                    self._exit_code = 0

        except Exception as exc:
            logger.exception("Pipeline run failed")
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
                self._finished_at = time.monotonic()
                self._exit_code = 1

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
        """Start an async dry-run preview. Returns immediately."""
        with self._lock:
            if self._preview_proc is not None and self._preview_proc.poll() is None:
                return self._preview_status_locked()
            cmd_or_error = self._build_preview_command(config)
            if isinstance(cmd_or_error, str):
                self._preview_reset_locked()
                self._preview_state = "error"
                self._preview_error = cmd_or_error
                return self._preview_status_locked()
            self._preview_reset_locked()
            cmd = cmd_or_error + ["--dry-run"]
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            try:
                self._preview_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO_ROOT),
                    env=env,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._preview_state = "error"
                self._preview_error = str(exc)
                return self._preview_status_locked()
            self._preview_state = "scanning"
            self._preview_lines = []
            reader = threading.Thread(target=self._preview_read_loop, daemon=True)
            reader.start()
            return self._preview_status_locked()

    def preview_status(self) -> dict:
        with self._lock:
            return self._preview_status_locked()

    def preview_cancel(self) -> dict:
        with self._lock:
            if self._preview_proc is not None and self._preview_proc.poll() is None:
                self._preview_proc.kill()
                self._preview_state = "error"
                self._preview_error = "cancelled"
            return self._preview_status_locked()

    def _build_preview_command(self, config: dict):
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
        if config.get("stage_input"):
            cmd.append("--stage-input")
        if config.get("overwrite"):
            cmd.append("--overwrite")
        else:
            cmd.append("--skip-existing")
        return cmd

    def _preview_reset_locked(self) -> None:
        self._preview_proc = None
        self._preview_state = "idle"
        self._preview_error = ""
        self._preview_progress = ""
        self._preview_lines = []
        self._preview_result = None

    def _preview_read_loop(self) -> None:
        proc = self._preview_proc
        if proc is None or proc.stdout is None:
            return
        lines: list[str] = []
        try:
            while True:
                ready, _, _ = select.select([proc.stdout], [], [], 30)
                if not ready:
                    proc.kill()
                    with self._lock:
                        if self._preview_proc is proc:
                            self._preview_state = "error"
                            self._preview_error = "scan timed out (no progress for 30s)"
                    return
                raw = proc.stdout.readline()
                if not raw:
                    break
                line = raw.rstrip("\n")
                lines.append(line)
                parts = line.split("\t")
                if parts[0] == "progress":
                    with self._lock:
                        if self._preview_proc is proc:
                            self._preview_progress = "\t".join(parts[1:])
        except Exception as exc:
            proc.kill()
            with self._lock:
                if self._preview_proc is proc:
                    self._preview_state = "error"
                    self._preview_error = str(exc)
            return
        rc = proc.wait()
        with self._lock:
            if self._preview_proc is not proc:
                return
            if rc != 0:
                output = "\n".join(lines).strip()
                self._preview_state = "error"
                self._preview_error = output or f"dry-run exited with code {rc}"
            else:
                self._preview_state = "done"
                self._preview_result = _parse_dry_run("\n".join(lines))

    def _preview_status_locked(self) -> dict:
        result = self._preview_result or {"to_process": [], "skipped": []}
        return {
            "state": self._preview_state,
            "progress": self._preview_progress,
            "error": self._preview_error,
            "to_process": result.get("to_process", []),
            "skipped": result.get("skipped", []),
        }


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
