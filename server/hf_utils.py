"""Hugging Face model download helpers."""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

HF_ENDPOINT = "https://huggingface.co"
HF_TOKEN_ENV = "HF_TOKEN"
MODEL_CACHE_ENV = "SUBTITLE_ANYWHERE_MODEL_CACHE"
PROXY_OVERRIDE_ENV = "SUBTITLE_ANYWHERE_PROXY_OVERRIDE"
DEFAULT_CACHE_DIR = Path("~/.cache/subtitle_anywhere/huggingface").expanduser()
PROXY_ENV_KEYS = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
}
PROXY_ENV_PAIRS = {
    "HTTP_PROXY": "http_proxy",
    "HTTPS_PROXY": "https_proxy",
    "ALL_PROXY": "all_proxy",
    "http_proxy": "HTTP_PROXY",
    "https_proxy": "HTTPS_PROXY",
    "all_proxy": "ALL_PROXY",
}
MODEL_ALIASES = {
    # User-facing upstream repo. The sibling hf-* repo is the Transformers layout.
    "facebook/seamless-m4t-medium": "facebook/hf-seamless-m4t-medium",
}
TRANSFORMERS_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "spm_char_lang38_tc.model",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model*.bin",
)


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    revision: str
    has_config: bool
    file_count: int = 0
    total_size_bytes: int = 0


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


@dataclass(frozen=True)
class LocalSnapshotStatus:
    model_id: str
    revision: str
    path: Path
    file_count: int
    total_size_bytes: int
    temp_file_count: int
    temp_size_bytes: int
    has_config: bool


@dataclass(frozen=True)
class ProxyConnectStatus:
    proxy: str
    target: str
    ok: bool
    response: str


def resolve_model_dir(model_id_or_path: str, local_only: bool = False) -> Path:
    """Return a local model directory, downloading from Hugging Face if needed."""
    load_env_file()
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return path.resolve()
    model_id = resolve_model_id(model_id_or_path)
    local_snapshot = find_complete_local_snapshot(model_id)
    if local_snapshot:
        logger.info("Using cached model snapshot: %s", local_snapshot)
        return local_snapshot
    if local_only:
        raise RuntimeError(
            f"No complete local snapshot found for {model_id}. "
            "Run without --local-only or use --download-model first."
        )
    return download_model_snapshot(model_id)


def get_model_info(model_id_or_path: str) -> ModelInfo:
    """Fetch lightweight model metadata through the configured HF endpoint."""
    load_env_file()
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        files = [item for item in path.rglob("*") if item.is_file()]
        return ModelInfo(
            model_id=str(path.resolve()),
            revision="local",
            has_config=(path / "config.json").exists(),
            file_count=len(files),
            total_size_bytes=sum(item.stat().st_size for item in files),
        )

    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to check models. "
            "Install dependencies with `pip install -r server/requirements.txt`."
        ) from exc

    endpoint = os.environ.get("HF_ENDPOINT", HF_ENDPOINT)
    resolved_model_id = resolve_model_id(model_id_or_path)
    try:
        info = HfApi(endpoint=endpoint, token=os.environ.get(HF_TOKEN_ENV)).model_info(
            resolved_model_id,
            files_metadata=True,
        )
    except Exception as exc:
        raise RuntimeError(_network_error_message("check model metadata", exc)) from exc
    files = _select_transformers_files(
        [RemoteFile(sibling.rfilename, sibling.size or 0) for sibling in info.siblings]
    )
    return ModelInfo(
        model_id=info.modelId,
        revision=info.sha or "",
        has_config=any(file.path == "config.json" for file in files),
        file_count=len(files),
        total_size_bytes=sum(file.size for file in files),
    )


def download_model_snapshot(model_id: str) -> Path:
    """Download the Transformers snapshot files from Hugging Face."""
    load_env_file()
    model_id = resolve_model_id(model_id)
    model_info = _remote_model_info(model_id)
    files = _select_transformers_files(model_info["files"])
    if not files:
        raise RuntimeError(f"No downloadable Transformers files found for {model_id}")

    revision = model_info["revision"]
    target_dir = _snapshot_dir(model_id, revision)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %d files for %s@%s via %s (%s)",
        len(files),
        model_id,
        revision[:12] if revision else "unknown",
        os.environ.get("HF_ENDPOINT", HF_ENDPOINT),
        format_bytes(sum(file.size for file in files)),
    )
    for index, remote_file in enumerate(files, start=1):
        destination = target_dir / remote_file.path
        if destination.exists() and (remote_file.size == 0 or destination.stat().st_size == remote_file.size):
            logger.info("[%d/%d] cached %s", index, len(files), remote_file.path)
            continue
        logger.info("[%d/%d] downloading %s (%s)", index, len(files), remote_file.path, format_bytes(remote_file.size))
        _download_file(model_id, revision, remote_file.path, destination, remote_file.size)

    return target_dir.resolve()


def check_network(url: str = "https://huggingface.co/api/models/facebook/hf-seamless-m4t-medium") -> str:
    """Check whether the configured endpoint/proxy can reach Hugging Face."""
    load_env_file()
    request = Request(url, method="HEAD", headers={"User-Agent": "subtitle-anywhere/1.0"})
    token = os.environ.get(HF_TOKEN_ENV)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(request, timeout=15) as response:
            return f"ok status={response.status} url={url}"
    except (OSError, URLError) as exc:
        raise RuntimeError(_network_error_message("connect to Hugging Face", exc, endpoint=url)) from exc


def check_proxy_connect(target_host: str = "huggingface.co", target_port: int = 443) -> ProxyConnectStatus:
    """Check whether configured proxy accepts HTTP CONNECT to the target."""
    load_env_file()
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY")
    if not proxy:
        return ProxyConnectStatus(proxy="not set", target=f"{target_host}:{target_port}", ok=False, response="proxy not set")

    parsed = urlparse(proxy)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.port:
        return ProxyConnectStatus(proxy=proxy, target=f"{target_host}:{target_port}", ok=False, response="unsupported proxy URL")

    request = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        "User-Agent: subtitle-anywhere/1.0\r\n"
        "Proxy-Connection: Keep-Alive\r\n"
        "\r\n"
    ).encode("ascii")
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as sock:
            sock.sendall(request)
            raw = sock.recv(4096)
    except OSError as exc:
        return ProxyConnectStatus(proxy=proxy, target=f"{target_host}:{target_port}", ok=False, response=str(exc))

    response_line = raw.splitlines()[0].decode("iso-8859-1", errors="replace") if raw else "empty response"
    return ProxyConnectStatus(
        proxy=proxy,
        target=f"{target_host}:{target_port}",
        ok=response_line.startswith("HTTP/") and " 200 " in response_line,
        response=response_line,
    )


def get_local_model_status(model_id_or_path: str) -> list[LocalSnapshotStatus]:
    """Return local snapshot/cache status without touching the network."""
    load_env_file()
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return [_snapshot_status(str(path.resolve()), "local", path.resolve())]

    model_id = resolve_model_id(model_id_or_path)
    safe_model_id = model_id.replace("/", "--")
    cache_root = Path(os.environ.get(MODEL_CACHE_ENV, os.fspath(DEFAULT_CACHE_DIR))).expanduser()
    model_root = cache_root / safe_model_id
    if not model_root.exists():
        return []

    snapshots = [item for item in model_root.iterdir() if item.is_dir()]
    snapshots.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [_snapshot_status(model_id, snapshot.name, snapshot) for snapshot in snapshots]


def find_complete_local_snapshot(model_id_or_path: str) -> Path | None:
    """Return the newest local snapshot that looks loadable without network."""
    for status in get_local_model_status(model_id_or_path):
        if _is_complete_snapshot(status):
            return status.path.resolve()
    return None


def _remote_model_info(model_id: str) -> dict:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    load_env_file()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to check models. "
            "Install dependencies with `pip install -r server/requirements.txt`."
        ) from exc

    endpoint = os.environ.get("HF_ENDPOINT", HF_ENDPOINT)
    model_id = resolve_model_id(model_id)
    try:
        info = HfApi(endpoint=endpoint, token=os.environ.get(HF_TOKEN_ENV)).model_info(
            model_id,
            files_metadata=True,
        )
    except Exception as exc:
        raise RuntimeError(_network_error_message("check model metadata", exc)) from exc
    return {
        "revision": info.sha or "main",
        "files": [RemoteFile(sibling.rfilename, sibling.size or 0) for sibling in info.siblings],
    }


def _select_transformers_files(files: list[RemoteFile]) -> list[RemoteFile]:
    selected = [
        file for file in files
        if any(fnmatch(file.path, pattern) for pattern in TRANSFORMERS_ALLOW_PATTERNS)
    ]
    return sorted(selected, key=lambda item: item.path)


def _snapshot_dir(model_id: str, revision: str) -> Path:
    cache_root = Path(os.environ.get(MODEL_CACHE_ENV, os.fspath(DEFAULT_CACHE_DIR))).expanduser()
    safe_model_id = model_id.replace("/", "--")
    return cache_root / safe_model_id / revision


def _snapshot_status(model_id: str, revision: str, path: Path) -> LocalSnapshotStatus:
    files = [item for item in path.rglob("*") if item.is_file()]
    temp_files = [item for item in files if item.name.endswith(".tmp")]
    complete_files = [item for item in files if not item.name.endswith(".tmp")]
    return LocalSnapshotStatus(
        model_id=model_id,
        revision=revision,
        path=path,
        file_count=len(complete_files),
        total_size_bytes=sum(item.stat().st_size for item in complete_files),
        temp_file_count=len(temp_files),
        temp_size_bytes=sum(item.stat().st_size for item in temp_files),
        has_config=(path / "config.json").exists(),
    )


def _is_complete_snapshot(status: LocalSnapshotStatus) -> bool:
    if not status.has_config or status.temp_file_count:
        return False
    return any(
        item.is_file() and (
            fnmatch(item.name, "pytorch_model*.bin")
            or fnmatch(item.name, "model-*.safetensors")
            or item.name == "model.safetensors"
        )
        for item in status.path.rglob("*")
    )


def resolve_model_id(model_id_or_path: str) -> str:
    return MODEL_ALIASES.get(model_id_or_path, model_id_or_path)


def _download_file(model_id: str, revision: str, filename: str, destination: Path, expected_size: int = 0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    quoted_model = "/".join(quote(part) for part in model_id.split("/"))
    quoted_filename = "/".join(quote(part) for part in filename.split("/"))
    endpoint = os.environ.get("HF_ENDPOINT", HF_ENDPOINT).rstrip("/")
    url = f"{endpoint}/{quoted_model}/resolve/{quote(revision)}/{quoted_filename}"
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    resume_at = temp_path.stat().st_size if temp_path.exists() else 0
    headers = {"User-Agent": "subtitle-anywhere/1.0"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    token = os.environ.get(HF_TOKEN_ENV)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            mode = "ab" if resume_at and response.status == 206 else "wb"
            downloaded = resume_at if mode == "ab" else 0
            started_at = time.monotonic()
            last_report_at = started_at
            with temp_path.open(mode) as fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_report_at >= 10:
                        logger.info(
                            "%s %s/%s at %s/s",
                            filename,
                            format_bytes(downloaded),
                            format_bytes(expected_size) if expected_size else "unknown",
                            format_bytes(int(downloaded / max(0.001, now - started_at))),
                        )
                        last_report_at = now
    except (OSError, URLError) as exc:
        raise RuntimeError(_network_error_message(f"download {filename}", exc)) from exc
    temp_path.replace(destination)


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env.

    Repo-local .env is treated as authoritative for model download settings so
    inherited shell defaults do not accidentally route downloads elsewhere.
    """
    env_path = path or _default_env_path()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if os.environ.get(PROXY_OVERRIDE_ENV) and key in PROXY_ENV_KEYS:
            continue
        if key:
            os.environ[key] = value
            paired_key = PROXY_ENV_PAIRS.get(key)
            if paired_key:
                os.environ[paired_key] = value


def _default_env_path() -> Path:
    return Path(__file__).resolve().parents[1] / ".env"


def _network_error_message(action: str, exc: BaseException, endpoint: str | None = None) -> str:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("ALL_PROXY") or "not set"
    endpoint_text = endpoint or os.environ.get("HF_ENDPOINT", HF_ENDPOINT)
    return (
        f"Failed to {action}: {exc}. "
        f"endpoint={endpoint_text} HTTPS_PROXY={proxy}. "
        "Check that the proxy accepts HTTPS CONNECT requests, then retry; partial downloads are resumable."
    )


def format_bytes(size: int) -> str:
    """Format byte counts for CLI output."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"
