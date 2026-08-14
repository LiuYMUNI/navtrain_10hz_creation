#!/usr/bin/env python3
"""Selectively pack exact NAVSIM 10 Hz camera JPEGs from nuPlan TAR objects.

This is the low-transfer companion to ``download_navsim_nuplan10hz_cameras.py``.
It reads one validated 512-byte TAR header at a time, jumps over non-target
payloads, and fetches bytes only for the immutable DB-planned target inventory.
Target JPEGs are concatenated into one pack per upstream archive and indexed
in SQLite.  Existing OpenScene/NAVSIM 2 Hz JPEGs may be reused by exact path
and source-declared size, avoiding duplicate storage.

The operation is resumable and crash-safe:

* source objects are pinned by ETag and listed size;
* each archive has an independent partial pack;
* pack bytes are fsynced before SQLite publishes their offsets;
* uncommitted trailing bytes are truncated on resume;
* every stored JPEG has a SHA-256 and JPEG marker check.

It never downloads native LiDAR, maps, mini, test, or unreferenced camera
payloads.  The resulting packs preserve the original JPEG payload bytes; they
can be read directly by offset or materialized into loose files later.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import time
from typing import Any, Mapping, Sequence

import requests
from urllib3.exceptions import HTTPError as Urllib3HTTPError

from download_navsim_nuplan10hz import atomic_write_json
from download_navsim_nuplan10hz_cameras import (
    ArchiveSpec,
    CameraDownloadError,
    DEFAULT_LISTING_URL,
    DEFAULT_STATE,
    archive_groups_from_initialized_state,
    discover_archives,
    exclusive_state_lock,
    jpeg_sha256,
    open_state,
    select_archives,
    utc_now,
)
from nuplan_range_tar import TAR_BLOCK_SIZE, RangeTarError, TarMember, _parse_header


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_ROOT = REPO_ROOT / "dataset/raw/nuplan-v1.1-navtrain-10hz/camera_packs"
DEFAULT_PACK_STATE = REPO_ROOT / "dataset/cache/navsim_nuplan10hz/camera_pack_state.sqlite"
DEFAULT_REPORT = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_pack_report.json"
)
DEFAULT_ARCHIVE_BASE_URL = "https://d1qinkmu0ju04f.cloudfront.net"
PACK_INDEX_FORMAT = "navsim_nuplan10hz_camera_pack_state_v1"
REPORT_FORMAT = "navsim_nuplan10hz_camera_pack_report_v1"
CONTENT_RANGE_PATTERN = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
RETRYABLE_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
MAX_JPEG_BYTES = 64 * 1024 * 1024


class CameraPackError(RuntimeError):
    """A source, pack, state, or integrity failure."""


def fsync_directory(path: Path) -> None:
    """Make a completed atomic rename durable before publishing SQLite state."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def normalized_etag(value: str) -> str:
    normalized = str(value).strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1]
    if not normalized:
        raise CameraPackError("Source ETag is empty")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-state-sqlite",
        type=Path,
        default=DEFAULT_STATE,
        help="Initialized immutable target ledger made by the camera downloader.",
    )
    parser.add_argument("--pack-state-sqlite", type=Path, default=DEFAULT_PACK_STATE)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument(
        "--existing-camera-root",
        type=Path,
        help=(
            "Optional existing OpenScene camera root containing <log>/CAM_*/<image>.jpg. "
            "Matching regular files with the source-declared size are indexed instead "
            "of duplicated into packs."
        ),
    )
    parser.add_argument("--listing-url", default=DEFAULT_LISTING_URL)
    parser.add_argument(
        "--archive-base-url",
        default=DEFAULT_ARCHIVE_BASE_URL,
        help=(
            "Range-capable base URL. The immutable public archive key is appended to "
            "this URL; archive size and ETag are still checked against the official listing."
        ),
    )
    parser.add_argument(
        "--archive-key",
        action="append",
        default=[],
        help="Process only this exact inventory-referenced public archive key. Repeatable.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--max-headers-per-archive",
        type=int,
        default=0,
        help="Stop each selected archive after this many new TAR headers (0 completes it).",
    )
    parser.add_argument("--checkpoint-headers", type=int, default=1000)
    parser.add_argument("--checkpoint-targets", type=int, default=100)
    parser.add_argument("--connect-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--max-http-attempts", type=int, default=8)
    parser.add_argument("--listing-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


class PersistentRangeClient:
    """Strict range reader with one reusable HTTP connection per archive worker."""

    def __init__(
        self,
        *,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_attempts: int,
    ) -> None:
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise ValueError("HTTP timeouts must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._timeout = (float(connect_timeout_seconds), float(read_timeout_seconds))
        self._max_attempts = int(max_attempts)
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=0,
            pool_block=True,
        )
        self._session.mount("https://", adapter)
        self._session.headers.update(
            {
                "Accept-Encoding": "identity",
                "User-Agent": "navsim-nuplan10hz-camera-pack/1",
            }
        )

    def close(self) -> None:
        self._session.close()

    def fetch(
        self,
        url: str,
        start: int,
        end: int,
        *,
        etag: str | None,
    ) -> tuple[bytes, int, str]:
        if start < 0 or end < start:
            raise ValueError("invalid byte range")
        expected_length = end - start + 1
        headers = {"Range": f"bytes={start}-{end}"}
        if etag is not None:
            headers["If-Match"] = etag

        last_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self._timeout,
                    allow_redirects=False,
                    stream=True,
                )
                try:
                    if (
                        response.status_code in RETRYABLE_HTTP_STATUSES
                        and attempt < self._max_attempts
                    ):
                        retry_after = response.headers.get("Retry-After")
                        delay = min(30.0, float(retry_after)) if retry_after else min(
                            10.0, 0.5 * (2 ** (attempt - 1))
                        )
                        time.sleep(delay)
                        continue
                    if response.status_code != 206:
                        raise CameraPackError(
                            f"Range server returned HTTP {response.status_code}, not 206, "
                            f"for {url} bytes {start}-{end}"
                        )
                    content_range = response.headers.get("Content-Range")
                    match = (
                        CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
                        if content_range is not None
                        else None
                    )
                    if match is None:
                        raise CameraPackError(
                            f"Range response has malformed Content-Range {content_range!r}"
                        )
                    actual_start, actual_end, total_size = (
                        int(value) for value in match.groups()
                    )
                    if (actual_start, actual_end) != (start, end) or total_size <= end:
                        raise CameraPackError(
                            f"Range response does not match bytes {start}-{end}: "
                            f"{content_range!r}"
                        )
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != expected_length:
                        raise CameraPackError(
                            f"Range response Content-Length {content_length!r} does not "
                            f"match {expected_length}"
                        )
                    payload = response.raw.read(expected_length + 1, decode_content=False)
                    if len(payload) != expected_length:
                        raise CameraPackError(
                            f"Range payload has {len(payload)} bytes, expected {expected_length}"
                        )
                    response_etag = response.headers.get("ETag")
                    if not response_etag:
                        raise CameraPackError(f"Range response lacks ETag for {url}")
                    if etag is not None and response_etag != etag:
                        raise CameraPackError(
                            f"Source ETag changed: expected {etag!r}, got {response_etag!r}"
                        )
                    return payload, total_size, response_etag
                finally:
                    response.close()
            except (
                requests.RequestException,
                Urllib3HTTPError,
                TimeoutError,
                CameraPackError,
                ValueError,
            ) as exc:
                last_error = exc
                if isinstance(exc, CameraPackError) and "ETag changed" in str(exc):
                    raise
                if attempt >= self._max_attempts:
                    break
                time.sleep(min(10.0, 0.5 * (2 ** (attempt - 1))))
                self._session.close()
                self._session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=1,
                    pool_maxsize=1,
                    max_retries=0,
                    pool_block=True,
                )
                self._session.mount("https://", adapter)
                self._session.headers.update(
                    {
                        "Accept-Encoding": "identity",
                        "User-Agent": "navsim-nuplan10hz-camera-pack/1",
                    }
                )
        assert last_error is not None
        raise CameraPackError(
            f"Range request failed after {self._max_attempts} attempts for "
            f"{url} bytes {start}-{end}: {last_error}"
        ) from last_error


def open_target_state_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise CameraPackError(f"Missing initialized target state: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=120.0)
    connection.execute("PRAGMA query_only = ON")
    required = {"metadata", "target"}
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if not required.issubset(tables):
        connection.close()
        raise CameraPackError(f"Target state lacks required tables: {path}")
    return connection


def open_pack_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120.0)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 120000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS archive_progress (
            archive_key TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            listed_size INTEGER NOT NULL,
            remote_etag TEXT NOT NULL,
            next_header_offset INTEGER NOT NULL,
            headers_scanned INTEGER NOT NULL,
            target_total INTEGER NOT NULL,
            target_done INTEGER NOT NULL,
            packed_target_count INTEGER NOT NULL,
            existing_target_count INTEGER NOT NULL,
            pack_relative_path TEXT NOT NULL,
            pack_bytes INTEGER NOT NULL,
            status TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS member_storage (
            archive_key TEXT NOT NULL,
            tar_member_name TEXT NOT NULL,
            destination_relative_path TEXT NOT NULL,
            storage_kind TEXT NOT NULL,
            storage_relative_path TEXT NOT NULL,
            pack_offset INTEGER,
            payload_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            source_header_offset INTEGER NOT NULL,
            source_data_offset INTEGER NOT NULL,
            PRIMARY KEY (archive_key, tar_member_name)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS member_storage_by_destination
            ON member_storage (destination_relative_path);
        CREATE INDEX IF NOT EXISTS member_storage_by_pack
            ON member_storage (storage_kind, storage_relative_path, pack_offset);
        """
    )
    prior_format = connection.execute(
        "SELECT value FROM metadata WHERE key = 'format'"
    ).fetchone()
    if prior_format is not None and str(prior_format[0]) != PACK_INDEX_FORMAT:
        connection.close()
        raise CameraPackError(
            f"Pack state has unexpected format {prior_format[0]!r}: {path}"
        )
    connection.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('format', ?)",
        (PACK_INDEX_FORMAT,),
    )
    connection.commit()
    return connection


def target_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    rows = {
        str(key): str(value)
        for key, value in connection.execute(
            "SELECT key, value FROM metadata WHERE key IN "
            "('inventory_sha256', 'inventory_row_count', 'navsim_scene_filter_sha256')"
        )
    }
    for key in ("inventory_sha256", "inventory_row_count", "navsim_scene_filter_sha256"):
        if key not in rows:
            raise CameraPackError(f"Target state lacks metadata key {key!r}")
    return rows


def initialize_pack_metadata(
    connection: sqlite3.Connection,
    source_metadata: Mapping[str, str],
) -> None:
    for key, value in source_metadata.items():
        prior = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if prior is not None and str(prior[0]) != value:
            raise CameraPackError(
                f"Pack state belongs to different target metadata: {key}"
            )
        connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
    connection.commit()


def replace_archive_url(spec: ArchiveSpec, base_url: str) -> ArchiveSpec:
    base = str(base_url).strip().rstrip("/")
    if not base.startswith("https://"):
        raise CameraPackError("--archive-base-url must be an https:// URL")
    return replace(spec, url=f"{base}/{spec.key}")


def pack_relative_path(spec: ArchiveSpec) -> str:
    split_directory = f"{spec.source_split}_set"
    filename = PurePosixPath(spec.key).name
    if not filename.endswith(".zip"):
        raise CameraPackError(f"Unexpected camera archive filename: {filename}")
    return str(PurePosixPath(split_directory) / f"{filename[:-4]}.pack")


def safe_existing_path(existing_root: Path, destination_relative_path: str) -> Path:
    relative = PurePosixPath(destination_relative_path)
    if (
        relative.is_absolute()
        or len(relative.parts) < 4
        or relative.parts[0] != "sensor_blobs"
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CameraPackError(
            f"Unsafe target destination path: {destination_relative_path!r}"
        )
    return existing_root.joinpath(*relative.parts[1:])


def jpeg_bytes_sha256(payload: bytes, *, member: TarMember) -> str:
    if (
        member.size <= 0
        or member.size > MAX_JPEG_BYTES
        or len(payload) != member.size
        or not payload.startswith(b"\xff\xd8")
        or not payload.endswith(b"\xff\xd9")
    ):
        raise CameraPackError(
            f"JPEG payload integrity check failed for {member.name!r}"
        )
    return hashlib.sha256(payload).hexdigest()


def load_archive_targets(
    target_state_path: Path,
    archive_key: str,
) -> dict[str, str]:
    connection = open_target_state_readonly(target_state_path)
    try:
        rows = connection.execute(
            "SELECT tar_member_name, destination_relative_path FROM target "
            "WHERE archive_key = ?",
            (archive_key,),
        ).fetchall()
    finally:
        connection.close()
    targets = {str(name): str(destination) for name, destination in rows}
    if len(targets) != len(rows) or not targets:
        raise CameraPackError(
            f"Archive has no unique immutable targets in target state: {archive_key}"
        )
    return targets


def prepare_archive_progress(
    connection: sqlite3.Connection,
    *,
    spec: ArchiveSpec,
    remote_size: int,
    remote_etag: str,
    target_total: int,
    relative_pack: str,
) -> tuple[int, int, int, int, int, int, str]:
    if remote_size != spec.listed_size:
        raise CameraPackError(
            f"Listed/source size mismatch for {spec.key}: "
            f"{spec.listed_size} != {remote_size}"
        )
    if (
        spec.listed_etag is not None
        and normalized_etag(remote_etag) != normalized_etag(spec.listed_etag)
    ):
        raise CameraPackError(
            f"Listed/source ETag mismatch for {spec.key}: "
            f"{spec.listed_etag!r} != {remote_etag!r}"
        )
    row = connection.execute(
        "SELECT source_url, listed_size, remote_etag, next_header_offset, "
        "headers_scanned, target_total, target_done, packed_target_count, "
        "existing_target_count, pack_relative_path, pack_bytes, status "
        "FROM archive_progress WHERE archive_key = ?",
        (spec.key,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO archive_progress "
            "(archive_key, source_url, listed_size, remote_etag, next_header_offset, "
            "headers_scanned, target_total, target_done, packed_target_count, "
            "existing_target_count, pack_relative_path, pack_bytes, status) "
            "VALUES (?, ?, ?, ?, 0, 0, ?, 0, 0, 0, ?, 0, 'running')",
            (
                spec.key,
                spec.url,
                spec.listed_size,
                remote_etag,
                target_total,
                relative_pack,
            ),
        )
        connection.commit()
        return 0, 0, 0, 0, 0, 0, "running"
    expected = (
        spec.url,
        spec.listed_size,
        remote_etag,
        target_total,
        relative_pack,
    )
    observed = (str(row[0]), int(row[1]), str(row[2]), int(row[5]), str(row[9]))
    if observed != expected:
        raise CameraPackError(
            f"Pack progress source/target contract changed for {spec.key}"
        )
    return (
        int(row[3]),
        int(row[4]),
        int(row[6]),
        int(row[7]),
        int(row[8]),
        int(row[10]),
        str(row[11]),
    )


def checkpoint_pack(
    connection: sqlite3.Connection,
    *,
    spec: ArchiveSpec,
    pack_handle: Any,
    next_header_offset: int,
    headers_scanned: int,
    target_done: int,
    packed_target_count: int,
    existing_target_count: int,
    status: str,
    updates: Sequence[tuple[Any, ...]],
) -> None:
    pack_handle.flush()
    os.fsync(pack_handle.fileno())
    pack_bytes = int(pack_handle.tell())
    connection.execute("BEGIN IMMEDIATE")
    try:
        if updates:
            connection.executemany(
                "INSERT INTO member_storage "
                "(archive_key, tar_member_name, destination_relative_path, "
                "storage_kind, storage_relative_path, pack_offset, payload_size, "
                "sha256, source_header_offset, source_data_offset) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                updates,
            )
        connection.execute(
            "UPDATE archive_progress SET next_header_offset = ?, headers_scanned = ?, "
            "target_done = ?, packed_target_count = ?, existing_target_count = ?, "
            "pack_bytes = ?, status = ? WHERE archive_key = ?",
            (
                next_header_offset,
                headers_scanned,
                target_done,
                packed_target_count,
                existing_target_count,
                pack_bytes,
                status,
                spec.key,
            ),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def open_partial_pack(
    pack_root: Path,
    relative_pack: str,
    *,
    committed_bytes: int,
    status: str,
) -> tuple[Any, Path, Path]:
    final_path = pack_root.joinpath(*PurePosixPath(relative_pack).parts)
    partial_path = Path(f"{final_path}.partial")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if status == "complete":
        if not final_path.is_file() or final_path.stat().st_size != committed_bytes:
            raise CameraPackError(
                f"Completed pack is missing or has wrong size: {final_path}"
            )
        return final_path.open("rb"), partial_path, final_path
    if (final_path.exists() or final_path.is_symlink()) and not (
        partial_path.exists() or partial_path.is_symlink()
    ):
        metadata = final_path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != committed_bytes:
            raise CameraPackError(
                f"Recoverable final pack has wrong type or size: {final_path}"
            )
        os.replace(final_path, partial_path)
    elif final_path.exists() or final_path.is_symlink():
        raise CameraPackError(
            f"Incomplete archive has both final and partial packs: {final_path}"
        )
    if partial_path.exists() or partial_path.is_symlink():
        metadata = partial_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise CameraPackError(f"Partial pack is not a regular file: {partial_path}")
        if metadata.st_size < committed_bytes:
            raise CameraPackError(
                f"Partial pack is shorter than committed state: {partial_path}"
            )
        handle = partial_path.open("r+b")
        if metadata.st_size > committed_bytes:
            handle.truncate(committed_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        if committed_bytes:
            raise CameraPackError(
                f"Committed pack state has no partial pack file: {partial_path}"
            )
        handle = partial_path.open("w+b")
    handle.seek(committed_bytes)
    return handle, partial_path, final_path


def finalize_pack(
    connection: sqlite3.Connection,
    *,
    spec: ArchiveSpec,
    pack_handle: Any,
    partial_path: Path,
    final_path: Path,
    next_header_offset: int,
    headers_scanned: int,
    target_done: int,
    packed_target_count: int,
    existing_target_count: int,
    updates: Sequence[tuple[Any, ...]],
) -> int:
    """Publish all bytes first, then mark the archive complete in SQLite."""

    checkpoint_pack(
        connection,
        spec=spec,
        pack_handle=pack_handle,
        next_header_offset=next_header_offset,
        headers_scanned=headers_scanned,
        target_done=target_done,
        packed_target_count=packed_target_count,
        existing_target_count=existing_target_count,
        status="running",
        updates=updates,
    )
    committed_bytes = int(pack_handle.tell())
    pack_handle.close()
    os.replace(partial_path, final_path)
    fsync_directory(final_path.parent)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "UPDATE archive_progress SET status = 'complete' WHERE archive_key = ?",
            (spec.key,),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return committed_bytes


def pack_archive(
    *,
    target_state_path: Path,
    pack_state_path: Path,
    pack_root: Path,
    existing_camera_root: Path | None,
    spec: ArchiveSpec,
    max_headers: int,
    checkpoint_headers: int,
    checkpoint_targets: int,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    max_http_attempts: int,
) -> dict[str, Any]:
    targets = load_archive_targets(target_state_path, spec.key)
    client = PersistentRangeClient(
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        max_attempts=max_http_attempts,
    )
    connection = open_pack_state(pack_state_path)
    pack_handle: Any | None = None
    try:
        first_header, remote_size, remote_etag = client.fetch(
            spec.url, 0, TAR_BLOCK_SIZE - 1, etag=None
        )
        relative_pack = pack_relative_path(spec)
        (
            next_header_offset,
            headers_before,
            target_done,
            packed_target_count,
            existing_target_count,
            committed_bytes,
            prior_status,
        ) = prepare_archive_progress(
            connection,
            spec=spec,
            remote_size=remote_size,
            remote_etag=remote_etag,
            target_total=len(targets),
            relative_pack=relative_pack,
        )
        if prior_status == "complete":
            handle, _partial_path, final_path = open_partial_pack(
                pack_root,
                relative_pack,
                committed_bytes=committed_bytes,
                status=prior_status,
            )
            handle.close()
            return {
                "key": spec.key,
                "status": "complete",
                "headers_scanned": headers_before,
                "target_done": target_done,
                "packed_target_count": packed_target_count,
                "existing_target_count": existing_target_count,
                "pack_bytes": committed_bytes,
                "pack_path": str(final_path),
            }

        completed = {
            str(row[0])
            for row in connection.execute(
                "SELECT tar_member_name FROM member_storage WHERE archive_key = ?",
                (spec.key,),
            )
        }
        if len(completed) != target_done or not completed.issubset(targets):
            raise CameraPackError(
                f"Pack member/progress count mismatch for {spec.key}"
            )
        pending = set(targets).difference(completed)
        pack_handle, partial_path, final_path = open_partial_pack(
            pack_root,
            relative_pack,
            committed_bytes=committed_bytes,
            status=prior_status,
        )

        scanned_now = 0
        targets_since_checkpoint = 0
        updates: list[tuple[Any, ...]] = []
        offset = next_header_offset
        if offset == 0:
            header = first_header
        else:
            header, observed_size, observed_etag = client.fetch(
                spec.url,
                offset,
                offset + TAR_BLOCK_SIZE - 1,
                etag=remote_etag,
            )
            if observed_size != remote_size or observed_etag != remote_etag:
                raise CameraPackError(f"Source identity changed for {spec.key}")

        while True:
            if max_headers and scanned_now >= max_headers:
                checkpoint_pack(
                    connection,
                    spec=spec,
                    pack_handle=pack_handle,
                    next_header_offset=offset,
                    headers_scanned=headers_before + scanned_now,
                    target_done=target_done,
                    packed_target_count=packed_target_count,
                    existing_target_count=existing_target_count,
                    status="partial",
                    updates=updates,
                )
                return {
                    "key": spec.key,
                    "status": "partial",
                    "headers_scanned": headers_before + scanned_now,
                    "target_done": target_done,
                    "packed_target_count": packed_target_count,
                    "existing_target_count": existing_target_count,
                    "pack_bytes": int(pack_handle.tell()),
                    "pack_path": str(partial_path),
                }

            if header == b"\0" * TAR_BLOCK_SIZE:
                second, observed_size, observed_etag = client.fetch(
                    spec.url,
                    offset + TAR_BLOCK_SIZE,
                    offset + 2 * TAR_BLOCK_SIZE - 1,
                    etag=remote_etag,
                )
                if (
                    second != b"\0" * TAR_BLOCK_SIZE
                    or observed_size != remote_size
                    or observed_etag != remote_etag
                ):
                    raise CameraPackError(
                        f"Invalid TAR termination for {spec.key} at {offset}"
                    )
                if pending:
                    missing = ", ".join(sorted(pending)[:3])
                    raise CameraPackError(
                        f"TAR ended before {len(pending)} target(s) were found "
                        f"in {spec.key}: {missing}"
                    )
                committed_size = finalize_pack(
                    connection,
                    spec=spec,
                    pack_handle=pack_handle,
                    partial_path=partial_path,
                    final_path=final_path,
                    next_header_offset=offset,
                    headers_scanned=headers_before + scanned_now,
                    target_done=target_done,
                    packed_target_count=packed_target_count,
                    existing_target_count=existing_target_count,
                    updates=updates,
                )
                pack_handle = None
                return {
                    "key": spec.key,
                    "status": "complete",
                    "headers_scanned": headers_before + scanned_now,
                    "target_done": target_done,
                    "packed_target_count": packed_target_count,
                    "existing_target_count": existing_target_count,
                    "pack_bytes": committed_size,
                    "pack_path": str(final_path),
                }

            member = _parse_header(header, offset=offset, archive_size=remote_size)
            scanned_now += 1
            destination = targets.get(member.name)
            if destination is not None and member.name in pending:
                if (
                    not member.is_regular_file
                    or member.size <= 0
                    or member.size > MAX_JPEG_BYTES
                    or not member.name.lower().endswith(".jpg")
                ):
                    raise CameraPackError(
                        f"Target is not a valid regular JPEG member: {member.name!r}"
                    )

                existing_path: Path | None = None
                if existing_camera_root is not None:
                    candidate = safe_existing_path(existing_camera_root, destination)
                    if candidate.exists() or candidate.is_symlink():
                        metadata = candidate.lstat()
                        if stat.S_ISREG(metadata.st_mode) and metadata.st_size == member.size:
                            existing_path = candidate

                if existing_path is not None:
                    size, digest = jpeg_sha256(
                        existing_path,
                        expected_size=member.size,
                    )
                    storage_kind = "existing"
                    try:
                        storage_relative_path = str(
                            existing_path.relative_to(existing_camera_root)
                        )
                    except ValueError as exc:
                        raise CameraPackError(
                            f"Existing JPEG escaped configured root: {existing_path}"
                        ) from exc
                    pack_offset: int | None = None
                    existing_target_count += 1
                else:
                    payload, observed_size, observed_etag = client.fetch(
                        spec.url,
                        member.data_offset,
                        member.data_offset + member.size - 1,
                        etag=remote_etag,
                    )
                    if observed_size != remote_size or observed_etag != remote_etag:
                        raise CameraPackError(f"Source identity changed for {spec.key}")
                    size = len(payload)
                    digest = jpeg_bytes_sha256(payload, member=member)
                    pack_offset = int(pack_handle.tell())
                    pack_handle.write(payload)
                    storage_kind = "pack"
                    storage_relative_path = relative_pack
                    packed_target_count += 1

                updates.append(
                    (
                        spec.key,
                        member.name,
                        destination,
                        storage_kind,
                        storage_relative_path,
                        pack_offset,
                        size,
                        digest,
                        member.header_offset,
                        member.data_offset,
                    )
                )
                pending.remove(member.name)
                target_done += 1
                targets_since_checkpoint += 1

            next_offset = member.next_header_offset
            should_checkpoint = (
                scanned_now % checkpoint_headers == 0
                or targets_since_checkpoint >= checkpoint_targets
                or not pending
            )
            if should_checkpoint:
                checkpoint_pack(
                    connection,
                    spec=spec,
                    pack_handle=pack_handle,
                    next_header_offset=next_offset,
                    headers_scanned=headers_before + scanned_now,
                    target_done=target_done,
                    packed_target_count=packed_target_count,
                    existing_target_count=existing_target_count,
                    status="running",
                    updates=updates,
                )
                updates.clear()
                targets_since_checkpoint = 0

            if not pending:
                committed_size = finalize_pack(
                    connection,
                    spec=spec,
                    pack_handle=pack_handle,
                    partial_path=partial_path,
                    final_path=final_path,
                    next_header_offset=next_offset,
                    headers_scanned=headers_before + scanned_now,
                    target_done=target_done,
                    packed_target_count=packed_target_count,
                    existing_target_count=existing_target_count,
                    updates=updates,
                )
                pack_handle = None
                return {
                    "key": spec.key,
                    "status": "complete",
                    "headers_scanned": headers_before + scanned_now,
                    "target_done": target_done,
                    "packed_target_count": packed_target_count,
                    "existing_target_count": existing_target_count,
                    "pack_bytes": committed_size,
                    "pack_path": str(final_path),
                }

            offset = next_offset
            header, observed_size, observed_etag = client.fetch(
                spec.url,
                offset,
                offset + TAR_BLOCK_SIZE - 1,
                etag=remote_etag,
            )
            if observed_size != remote_size or observed_etag != remote_etag:
                raise CameraPackError(f"Source identity changed for {spec.key}")
    finally:
        if pack_handle is not None:
            pack_handle.close()
        connection.close()
        client.close()


def pack_state_summary(connection: sqlite3.Connection) -> dict[str, int]:
    archive_rows = {
        str(status): int(count)
        for status, count in connection.execute(
            "SELECT status, COUNT(*) FROM archive_progress GROUP BY status"
        )
    }
    row = connection.execute(
        "SELECT COALESCE(SUM(target_total), 0), COALESCE(SUM(target_done), 0), "
        "COALESCE(SUM(packed_target_count), 0), "
        "COALESCE(SUM(existing_target_count), 0), "
        "COALESCE(SUM(pack_bytes), 0), COALESCE(SUM(headers_scanned), 0) "
        "FROM archive_progress"
    ).fetchone()
    return {
        "archive_total": sum(archive_rows.values()),
        "archives_complete": archive_rows.get("complete", 0),
        "archives_partial": archive_rows.get("partial", 0),
        "archives_running": archive_rows.get("running", 0),
        "target_total": int(row[0]),
        "target_done": int(row[1]),
        "target_pending": int(row[0]) - int(row[1]),
        "packed_target_count": int(row[2]),
        "existing_target_count": int(row[3]),
        "pack_bytes": int(row[4]),
        "headers_scanned": int(row[5]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1:
        raise CameraPackError("--workers must be positive")
    if args.max_headers_per_archive < 0:
        raise CameraPackError("--max-headers-per-archive must be nonnegative")
    if args.checkpoint_headers < 1 or args.checkpoint_targets < 1:
        raise CameraPackError("checkpoint intervals must be positive")
    if args.max_http_attempts < 1:
        raise CameraPackError("--max-http-attempts must be positive")
    if args.pack_state_sqlite.suffix != ".sqlite":
        raise CameraPackError("--pack-state-sqlite must have a .sqlite suffix")
    if args.report_json.suffix != ".json":
        raise CameraPackError("--report-json must have a .json suffix")
    if args.existing_camera_root is not None and not args.existing_camera_root.is_dir():
        raise CameraPackError(
            f"Existing camera root is not a directory: {args.existing_camera_root}"
        )
    if args.target_state_sqlite.resolve() == args.pack_state_sqlite.resolve():
        raise CameraPackError(
            "--target-state-sqlite and --pack-state-sqlite must be different files"
        )

    target_connection = open_target_state_readonly(args.target_state_sqlite)
    try:
        metadata = target_metadata(target_connection)
        groups = archive_groups_from_initialized_state(target_connection)
    finally:
        target_connection.close()
    archives = discover_archives(
        listing_url=args.listing_url,
        timeout_seconds=float(args.listing_timeout_seconds),
        groups=groups,
    )
    archives = tuple(
        replace_archive_url(spec, args.archive_base_url) for spec in archives
    )
    archives = select_archives(archives, args.archive_key)

    with exclusive_state_lock(args.pack_state_sqlite):
        connection = open_pack_state(args.pack_state_sqlite)
        try:
            initialize_pack_metadata(connection, metadata)
            if args.dry_run:
                return {
                    "format": REPORT_FORMAT,
                    "generated_at_utc": utc_now(),
                    "status": "dry_run",
                    "archive_count": len(archives),
                    "inventory_target_count": int(metadata["inventory_row_count"]),
                    "inventory_sha256": metadata["inventory_sha256"],
                    "archives": [
                        {
                            "key": spec.key,
                            "url": spec.url,
                            "size_bytes": spec.listed_size,
                        }
                        for spec in archives
                    ],
                    "state": pack_state_summary(connection),
                }
        finally:
            connection.close()

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(
            max_workers=min(int(args.workers), len(archives))
        ) as executor:
            futures = {
                executor.submit(
                    pack_archive,
                    target_state_path=args.target_state_sqlite,
                    pack_state_path=args.pack_state_sqlite,
                    pack_root=args.pack_root,
                    existing_camera_root=args.existing_camera_root,
                    spec=spec,
                    max_headers=int(args.max_headers_per_archive),
                    checkpoint_headers=int(args.checkpoint_headers),
                    checkpoint_targets=int(args.checkpoint_targets),
                    connect_timeout_seconds=float(args.connect_timeout_seconds),
                    read_timeout_seconds=float(args.read_timeout_seconds),
                    max_http_attempts=int(args.max_http_attempts),
                ): spec
                for spec in archives
            }
            try:
                for future in as_completed(futures):
                    results.append(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

        connection = open_pack_state(args.pack_state_sqlite)
        try:
            summary = pack_state_summary(connection)
        finally:
            connection.close()
        selected_complete = all(result["status"] == "complete" for result in results)
        return {
            "format": REPORT_FORMAT,
            "generated_at_utc": utc_now(),
            "status": "pass" if selected_complete else "partial",
            "archive_count": len(archives),
            "inventory_target_count": int(metadata["inventory_row_count"]),
            "inventory_sha256": metadata["inventory_sha256"],
            "archive_results": sorted(results, key=lambda item: str(item["key"])),
            "state": summary,
            "contract": {
                "camera_archives_are_tar_streams": True,
                "source_etag_and_size_are_pinned": True,
                "reads_only_512_byte_headers_for_non_targets": True,
                "fetches_payloads_only_for_missing_inventory_targets": True,
                "reuses_existing_jpegs_only_after_path_size_and_jpeg_validation": True,
                "pack_payloads_preserve_exact_source_jpeg_bytes": True,
                "excluded_data": ["test", "mini", "maps", "native_lidar"],
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (
        CameraDownloadError,
        CameraPackError,
        RangeTarError,
        OSError,
        ValueError,
        sqlite3.Error,
        requests.RequestException,
    ) as exc:
        report = {
            "format": REPORT_FORMAT,
            "generated_at_utc": utc_now(),
            "status": "fail",
            "error": str(exc),
        }
    if not args.dry_run:
        atomic_write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "partial", "dry_run"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
