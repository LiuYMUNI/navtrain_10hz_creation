#!/usr/bin/env python3
"""Index or stage the DB-planned native nuPlan 10 Hz JPEG inventory from public TARs.

The public nuPlan camera objects use a ``.zip`` suffix but are uncompressed
TAR streams.  They have no central directory, so this program advances
through validated TAR headers, checkpointing each archive offset in SQLite.
The ``stage`` operation writes image payload bytes solely for members in the
verified ``plan_navsim_nuplan10hz.py`` inventory.  The ``offset-index``
operation reads TAR headers only, records byte offsets for those same
members, and publishes a compact immutable target-only TAR-offset index.

This is intentionally a derived NAVSIM 10 Hz camera protocol.  It does not
download nuPlan test, mini, maps, or native LiDAR data, and it does not alter
the released 2 Hz NAVSIM root.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from download_navsim_nuplan10hz import (
    archive_url_from_listing_url,
    atomic_write_json,
    list_public_s3_objects,
)
from nuplan_range_tar import RangeTarArchive, RangeTarError, TarMember


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_NUPLAN_ROOT = REPO_ROOT / "dataset/raw/nuplan-v1.1-navtrain-10hz"
DEFAULT_INVENTORY = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_inventory.jsonl"
)
DEFAULT_TRAIN_GROUPS = (
    REPO_ROOT / "reference/upstream/public_set_train_sensor.txt"
)
DEFAULT_VAL_GROUPS = (
    REPO_ROOT / "reference/upstream/public_set_val_sensor.txt"
)
DEFAULT_LISTING_URL = (
    "https://motional-nuplan.s3.ap-northeast-1.amazonaws.com/"
    "?list-type=2&prefix=public%2Fnuplan-v1.1%2Fsensor_blobs%2F"
)
DEFAULT_STATE = REPO_ROOT / "dataset/cache/navsim_nuplan10hz/camera_download_state.sqlite"
DEFAULT_REPORT = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_download_report.json"
)
DEFAULT_TAR_OFFSET_INDEX = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_tar_offsets.sqlite"
)
DEFAULT_TAR_SCAN_CHUNK_BYTES = 8 * 1024 * 1024

INVENTORY_FORMAT = "navsim_nuplan10hz_camera_inventory_v1"
REPORT_FORMAT = "navsim_nuplan10hz_camera_download_report_v1"
TAR_OFFSET_INDEX_FORMAT = "navsim_nuplan10hz_tar_offset_index_v1"
PUBLIC_PREFIX = "public/nuplan-v1.1/sensor_blobs"


class CameraDownloadError(RuntimeError):
    """A source, state, staging, or integrity failure."""


@dataclass(frozen=True)
class ArchiveSpec:
    key: str
    url: str
    source_split: str
    group: int
    listed_size: int
    listed_etag: str | None


@contextmanager
def exclusive_state_lock(state_path: Path) -> Iterable[None]:
    """Prevent overlapping processes from mutating one resumable state ledger."""

    lock_path = Path(f"{state_path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CameraDownloadError(
                f"Another camera downloader already owns the state lock: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-jsonl", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--nuplan-root", type=Path, default=DEFAULT_NUPLAN_ROOT)
    parser.add_argument("--train-group-manifest", type=Path, default=DEFAULT_TRAIN_GROUPS)
    parser.add_argument("--val-group-manifest", type=Path, default=DEFAULT_VAL_GROUPS)
    parser.add_argument("--listing-url", default=DEFAULT_LISTING_URL)
    parser.add_argument("--state-sqlite", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--operation",
        choices=("initialize", "stage", "offset-index"),
        default="stage",
        help=(
            "initialize: create the immutable target ledger without scanning TARs; "
            "stage: write exact loose JPEGs; offset-index: record only exact TAR "
            "member byte offsets and publish a target-only offset index."
        ),
    )
    parser.add_argument(
        "--tar-offset-index-sqlite",
        type=Path,
        default=DEFAULT_TAR_OFFSET_INDEX,
        help=(
            "Immutable target-only TAR-offset index published after a complete "
            "--operation offset-index scan."
        ),
    )
    parser.add_argument(
        "--archive-key",
        action="append",
        default=[],
        help=(
            "Stage or inspect this exact public camera archive key only; may be repeated. "
            "By default all archives referenced by the immutable inventory are selected."
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--tar-scan-chunk-bytes",
        type=int,
        default=DEFAULT_TAR_SCAN_CHUNK_BYTES,
        help=(
            "Bytes fetched per sequential TAR scan range (default: 8388608 / 8 MiB). "
            "Must be a positive multiple of 512."
        ),
    )
    parser.add_argument(
        "--max-headers-per-archive",
        type=int,
        default=0,
        help="Checkpoint after this many TAR headers per archive (0 means run to completion).",
    )
    parser.add_argument("--listing-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def parse_group_manifest(path: Path, *, split: str) -> dict[str, int]:
    if not path.is_file():
        raise CameraDownloadError(f"Missing official {split} sensor-group manifest: {path}")
    mapping: dict[str, int] = {}
    group: int | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("File group:"):
            value = line.split(":", 1)[1].strip()
            try:
                group = int(value)
            except ValueError as exc:
                raise CameraDownloadError(f"Malformed group marker in {path}: {line!r}") from exc
            if group < 0:
                raise CameraDownloadError(f"Negative group marker in {path}: {line!r}")
            continue
        if group is None:
            raise CameraDownloadError(f"Log appears before a group marker in {path}: {line!r}")
        if not line.replace(".", "").replace("_", "").replace("-", "").isalnum():
            raise CameraDownloadError(f"Unsafe log name in {path}: {line!r}")
        if line in mapping:
            raise CameraDownloadError(f"Duplicate log name in {path}: {line}")
        mapping[line] = group
    if not mapping:
        raise CameraDownloadError(f"No log groups found in {path}")
    return mapping


def safe_inventory_path(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if record.get("format") != INVENTORY_FORMAT:
        raise CameraDownloadError(f"Unexpected inventory record format: {record.get('format')!r}")
    log_name = str(record.get("navsim_log_name") or "")
    channel = str(record.get("camera_channel") or "")
    filename = str(record.get("native_image_filename_jpg") or "")
    destination = str(record.get("native_blob_relative_path") or "")
    filename_path = PurePosixPath(filename)
    expected = PurePosixPath(log_name) / channel / filename_path.name
    destination_path = PurePosixPath(destination)
    expected_destination = PurePosixPath("sensor_blobs") / log_name / channel / filename_path.name
    if (
        not log_name
        or not channel.startswith("CAM_")
        or filename_path != expected
        or filename_path.suffix.lower() != ".jpg"
        or filename_path.is_absolute()
        or any(part in {"", ".", ".."} for part in filename_path.parts)
        or destination_path != expected_destination
    ):
        raise CameraDownloadError(
            f"Inventory record has an unsafe or noncanonical native path: {log_name!r}, {channel!r}, {filename!r}"
        )
    return log_name, channel, filename, str(destination_path)


def open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120.0)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS target (
            archive_key TEXT NOT NULL,
            tar_member_name TEXT NOT NULL,
            destination_relative_path TEXT NOT NULL,
            log_name TEXT NOT NULL,
            camera_channel TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            tar_member_size INTEGER,
            sha256 TEXT,
            tar_header_offset INTEGER,
            tar_data_offset INTEGER,
            PRIMARY KEY (archive_key, tar_member_name)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS target_pending_by_archive
            ON target (archive_key, status);
        CREATE INDEX IF NOT EXISTS target_status_summary
            ON target (status, tar_member_size);
        CREATE TABLE IF NOT EXISTS archive_progress (
            archive_key TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            listed_size INTEGER NOT NULL,
            remote_etag TEXT NOT NULL,
            next_header_offset INTEGER NOT NULL,
            headers_scanned INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tar_offset_progress (
            archive_key TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            listed_size INTEGER NOT NULL,
            remote_etag TEXT NOT NULL,
            next_header_offset INTEGER NOT NULL,
            headers_scanned INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        """
    )
    target_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(target)").fetchall()
    }
    if "tar_header_offset" not in target_columns:
        connection.execute("ALTER TABLE target ADD COLUMN tar_header_offset INTEGER")
    if "tar_data_offset" not in target_columns:
        connection.execute("ALTER TABLE target ADD COLUMN tar_data_offset INTEGER")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS target_offset_pending_by_archive "
        "ON target (archive_key, tar_data_offset)"
    )
    connection.commit()
    return connection


def get_metadata(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row[0])


def archive_groups_from_initialized_state(connection: sqlite3.Connection) -> set[tuple[str, int]]:
    """Recover the exact upstream archive groups from a completed target ledger."""

    prefix = tuple(PurePosixPath(PUBLIC_PREFIX).parts)
    groups: set[tuple[str, int]] = set()
    rows = connection.execute("SELECT DISTINCT archive_key FROM target")
    for (raw_key,) in rows:
        key = str(raw_key)
        parts = tuple(PurePosixPath(key).parts)
        remainder = parts[len(prefix) :]
        if parts[: len(prefix)] != prefix or len(remainder) != 2:
            raise CameraDownloadError(f"Stored camera archive key is malformed: {key!r}")
        split_directory, filename = remainder
        if not split_directory.endswith("_set"):
            raise CameraDownloadError(f"Stored camera archive split is malformed: {key!r}")
        source_split = split_directory[: -len("_set")]
        if source_split not in {"train", "val"}:
            raise CameraDownloadError(f"Stored camera archive has an unsupported split: {key!r}")
        filename_prefix = f"nuplan-v1.1_{source_split}_camera_"
        if not filename.startswith(filename_prefix) or not filename.endswith(".zip"):
            raise CameraDownloadError(f"Stored camera archive filename is malformed: {key!r}")
        group_text = filename[len(filename_prefix) : -len(".zip")]
        try:
            group = int(group_text)
        except ValueError as exc:
            raise CameraDownloadError(f"Stored camera archive group is malformed: {key!r}") from exc
        if group < 0 or str(group) != group_text:
            raise CameraDownloadError(f"Stored camera archive group is malformed: {key!r}")
        groups.add((source_split, group))
    if not groups:
        raise CameraDownloadError("Initialized camera download state has no archive groups.")
    return groups


def initialize_targets(
    connection: sqlite3.Connection,
    *,
    inventory: Path,
    train_groups: Mapping[str, int],
    val_groups: Mapping[str, int],
) -> tuple[int, set[tuple[str, int]], str]:
    if not inventory.is_file():
        raise CameraDownloadError(f"Missing verified camera inventory: {inventory}")
    inventory_sha256 = sha256_file(inventory)
    prior_sha256 = get_metadata(connection, "inventory_sha256")
    if prior_sha256 is not None and prior_sha256 != inventory_sha256:
        raise CameraDownloadError(
            "Camera download state belongs to a different inventory; use a new --state-sqlite path."
        )
    if prior_sha256 is not None:
        prior_count = get_metadata(connection, "inventory_row_count")
        if prior_count is None:
            raise CameraDownloadError("Initialized camera download state lacks its inventory row count.")
        try:
            expected_count = int(prior_count)
        except ValueError as exc:
            raise CameraDownloadError("Initialized camera download state has an invalid inventory row count.") from exc
        if expected_count < 1:
            raise CameraDownloadError("Initialized camera download state has an invalid inventory row count.")
        target_count = int(connection.execute("SELECT COUNT(*) FROM target").fetchone()[0])
        if target_count != expected_count:
            raise CameraDownloadError(
                "Initialized camera download state target count does not match its immutable inventory."
            )
        filter_sha256 = get_metadata(connection, "navsim_scene_filter_sha256")
        if filter_sha256 is None or len(filter_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in filter_sha256
        ):
            raise CameraDownloadError(
                "Initialized camera download state lacks a valid NAVSIM SceneFilter SHA-256."
            )
        return expected_count, archive_groups_from_initialized_state(connection), inventory_sha256

    groups: set[tuple[str, int]] = set()
    filter_sha256: str | None = None
    total = 0
    with inventory.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise CameraDownloadError(f"Blank inventory record at {inventory}:{line_number}")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise CameraDownloadError(f"Malformed inventory JSON at {inventory}:{line_number}") from exc
            if not isinstance(record, Mapping):
                raise CameraDownloadError(f"Non-mapping inventory record at {inventory}:{line_number}")
            log_name, channel, filename, destination = safe_inventory_path(record)
            record_filter_sha256 = str(record.get("navsim_scene_filter_sha256") or "")
            if len(record_filter_sha256) != 64 or any(
                character not in "0123456789abcdef" for character in record_filter_sha256
            ):
                raise CameraDownloadError(
                    f"Inventory record lacks a valid NAVSIM SceneFilter SHA-256 at {inventory}:{line_number}"
                )
            if filter_sha256 is None:
                filter_sha256 = record_filter_sha256
            elif filter_sha256 != record_filter_sha256:
                raise CameraDownloadError(f"Inventory mixes NAVSIM SceneFilter hashes at {inventory}:{line_number}")
            matches = [("train", train_groups.get(log_name)), ("val", val_groups.get(log_name))]
            matches = [(split, group) for split, group in matches if group is not None]
            if len(matches) != 1:
                raise CameraDownloadError(
                    f"Could not resolve exactly one native split/group for inventory log {log_name}"
                )
            source_split, group = matches[0]
            root_name = f"nuplan-v1.1_{source_split}_camera_{group}"
            archive_key = f"{PUBLIC_PREFIX}/{source_split}_set/{root_name}.zip"
            member_name = f"{root_name}/{filename}"
            existing = connection.execute(
                "SELECT destination_relative_path, log_name, camera_channel FROM target "
                "WHERE archive_key = ? AND tar_member_name = ?",
                (archive_key, member_name),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO target (archive_key, tar_member_name, destination_relative_path, log_name, camera_channel) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (archive_key, member_name, destination, log_name, channel),
                )
            elif tuple(existing) != (destination, log_name, channel):
                raise CameraDownloadError(f"Conflicting duplicate inventory image at {inventory}:{line_number}")
            groups.add((source_split, int(group)))
            total += 1
            if total % 10_000 == 0:
                connection.commit()
    existing_count = int(connection.execute("SELECT COUNT(*) FROM target").fetchone()[0])
    if existing_count != total:
        raise CameraDownloadError(
            f"Inventory contains duplicate camera records: {total} rows, {existing_count} unique TAR members"
        )
    connection.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('inventory_sha256', ?)", (inventory_sha256,))
    connection.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('inventory_row_count', ?)", (str(total),))
    assert filter_sha256 is not None
    prior_filter_sha256 = get_metadata(connection, "navsim_scene_filter_sha256")
    if prior_filter_sha256 is not None and prior_filter_sha256 != filter_sha256:
        raise CameraDownloadError("Camera download state belongs to a different NAVSIM SceneFilter hash.")
    connection.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('navsim_scene_filter_sha256', ?)",
        (filter_sha256,),
    )
    connection.commit()
    return total, groups, inventory_sha256


def discover_archives(
    *,
    listing_url: str,
    timeout_seconds: float,
    groups: Iterable[tuple[str, int]],
) -> tuple[ArchiveSpec, ...]:
    records = list_public_s3_objects(listing_url, timeout_seconds)
    objects = {key: (size, etag) for key, size, etag in records}
    specs: list[ArchiveSpec] = []
    for source_split, group in sorted(groups):
        root_name = f"nuplan-v1.1_{source_split}_camera_{group}"
        key = f"{PUBLIC_PREFIX}/{source_split}_set/{root_name}.zip"
        if key not in objects:
            raise CameraDownloadError(f"Public S3 listing is missing required camera archive: {key}")
        size, etag = objects[key]
        if size <= 0:
            raise CameraDownloadError(f"Public S3 camera archive has invalid size: {key}")
        specs.append(
            ArchiveSpec(
                key=key,
                url=archive_url_from_listing_url(listing_url, key),
                source_split=source_split,
                group=group,
                listed_size=size,
                listed_etag=etag,
            )
        )
    return tuple(specs)


def select_archives(
    archives: Sequence[ArchiveSpec],
    requested_keys: Sequence[str],
) -> tuple[ArchiveSpec, ...]:
    """Restrict a run to explicitly named archives without changing its ledger."""

    if not requested_keys:
        return tuple(archives)
    requested = {str(key).strip() for key in requested_keys if str(key).strip()}
    if not requested:
        raise CameraDownloadError("--archive-key values must be nonempty")
    available = {archive.key: archive for archive in archives}
    unknown = sorted(requested - set(available))
    if unknown:
        raise CameraDownloadError(
            "--archive-key is not referenced by the immutable inventory: " + ", ".join(unknown[:3])
        )
    return tuple(available[key] for key in sorted(requested))


def jpeg_sha256(path: Path, *, expected_size: int | None = None) -> tuple[int, str]:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CameraDownloadError(f"Expected a regular JPEG file, found unsafe path: {path}")
    if expected_size is not None and metadata.st_size != expected_size:
        raise CameraDownloadError(f"JPEG size mismatch at {path}: {metadata.st_size} != {expected_size}")
    if metadata.st_size < 4:
        raise CameraDownloadError(f"JPEG is too short: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(2)
        digest.update(head)
        previous = head
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            previous = (previous + block)[-2:]
    if head != b"\xff\xd8" or previous != b"\xff\xd9":
        raise CameraDownloadError(f"JPEG marker check failed: {path}")
    return metadata.st_size, digest.hexdigest()


def stage_member(
    archive: RangeTarArchive,
    member: TarMember,
    destination_root: Path,
    destination_relative: str,
) -> tuple[int, str]:
    destination = destination_root / PurePosixPath(destination_relative)
    if destination.exists() or destination.is_symlink():
        return jpeg_sha256(destination, expected_size=member.size)
    archive.extract_member(member, destination_root, relative_path=destination_relative)
    return jpeg_sha256(destination, expected_size=member.size)


def initialize_archive_progress(connection: sqlite3.Connection, spec: ArchiveSpec, archive: RangeTarArchive) -> tuple[int, int]:
    etag = archive.metadata.etag
    if not etag:
        raise CameraDownloadError(f"Range response lacks an ETag for {spec.key}")
    row = connection.execute(
        "SELECT url, listed_size, remote_etag, next_header_offset, headers_scanned FROM archive_progress WHERE archive_key = ?",
        (spec.key,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO archive_progress (archive_key, url, listed_size, remote_etag, next_header_offset, headers_scanned, status) "
            "VALUES (?, ?, ?, ?, 0, 0, 'running')",
            (spec.key, spec.url, spec.listed_size, etag),
        )
        connection.commit()
        return 0, 0
    if tuple(row[:3]) != (spec.url, spec.listed_size, etag):
        raise CameraDownloadError(f"Archive source changed since the checkpoint: {spec.key}")
    return int(row[3]), int(row[4])


def checkpoint_archive(
    connection: sqlite3.Connection,
    spec: ArchiveSpec,
    *,
    next_offset: int,
    headers_scanned: int,
    status: str,
    staged_updates: Sequence[tuple[int, str, str]] = (),
) -> None:
    if staged_updates:
        connection.executemany(
            "UPDATE target SET status = 'staged', tar_member_size = ?, sha256 = ? "
            "WHERE archive_key = ? AND tar_member_name = ?",
            [
                (size, digest, spec.key, member_name)
                for size, digest, member_name in staged_updates
            ],
        )
    connection.execute(
        "UPDATE archive_progress SET next_header_offset = ?, headers_scanned = ?, status = ? WHERE archive_key = ?",
        (next_offset, headers_scanned, status, spec.key),
    )
    connection.commit()


def scan_archive(
    *,
    state_path: Path,
    spec: ArchiveSpec,
    nuplan_root: Path,
    max_headers: int,
    tar_scan_chunk_bytes: int,
) -> dict[str, Any]:
    connection = sqlite3.connect(state_path, timeout=120.0)
    connection.execute("PRAGMA busy_timeout = 120000")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        archive = RangeTarArchive.open(spec.url, scan_chunk_bytes=tar_scan_chunk_bytes)
        next_offset, scanned_before = initialize_archive_progress(connection, spec, archive)
        connection.execute("UPDATE archive_progress SET status = 'running' WHERE archive_key = ?", (spec.key,))
        connection.commit()
        pending_rows = connection.execute(
            "SELECT tar_member_name, destination_relative_path FROM target WHERE archive_key = ? AND status != 'staged'",
            (spec.key,),
        ).fetchall()
        pending = {str(name): str(destination) for name, destination in pending_rows}
        if not pending:
            checkpoint_archive(
                connection, spec, next_offset=next_offset, headers_scanned=scanned_before, status="complete"
            )
            return {"key": spec.key, "status": "complete", "headers_scanned": scanned_before, "staged": 0}
        scanned_now = 0
        staged = 0
        last_next_offset = next_offset
        staged_updates: list[tuple[int, str, str]] = []
        for member in archive.iter_members(start_header_offset=next_offset):
            if max_headers and scanned_now >= max_headers:
                checkpoint_archive(
                    connection,
                    spec,
                    next_offset=last_next_offset,
                    headers_scanned=scanned_before + scanned_now,
                    status="partial",
                    staged_updates=staged_updates,
                )
                return {
                    "key": spec.key,
                    "status": "partial",
                    "headers_scanned": scanned_before + scanned_now,
                    "staged": staged,
                }
            scanned_now += 1
            last_next_offset = member.next_header_offset
            destination = pending.get(member.name)
            if destination is not None:
                if not member.is_regular_file or member.size <= 0 or not member.name.lower().endswith(".jpg"):
                    raise CameraDownloadError(f"Expected a regular JPEG TAR member, found {member.name!r}")
                size, digest = stage_member(archive, member, nuplan_root, destination)
                staged_updates.append((size, digest, member.name))
                pending.pop(member.name)
                staged += 1
            if scanned_now % 1000 == 0 or not pending:
                checkpoint_archive(
                    connection,
                    spec,
                    next_offset=last_next_offset,
                    headers_scanned=scanned_before + scanned_now,
                    status="running",
                    staged_updates=staged_updates,
                )
                staged_updates.clear()
            if not pending:
                checkpoint_archive(
                    connection,
                    spec,
                    next_offset=last_next_offset,
                    headers_scanned=scanned_before + scanned_now,
                    status="complete",
                )
                return {
                    "key": spec.key,
                    "status": "complete",
                    "headers_scanned": scanned_before + scanned_now,
                    "staged": staged,
                }
        missing = ", ".join(sorted(pending)[:3])
        raise CameraDownloadError(f"TAR ended before {len(pending)} planned image(s) were found in {spec.key}: {missing}")
    finally:
        connection.close()


def initialize_tar_offset_progress(
    connection: sqlite3.Connection,
    spec: ArchiveSpec,
    archive: RangeTarArchive,
) -> tuple[int, int]:
    """Open or verify a resumable header-only scan for one immutable TAR."""

    etag = archive.metadata.etag
    if not etag:
        raise CameraDownloadError(f"Range response lacks an ETag for {spec.key}")
    row = connection.execute(
        "SELECT url, listed_size, remote_etag, next_header_offset, headers_scanned "
        "FROM tar_offset_progress WHERE archive_key = ?",
        (spec.key,),
    ).fetchone()
    if row is None:
        connection.execute(
            "INSERT INTO tar_offset_progress "
            "(archive_key, url, listed_size, remote_etag, next_header_offset, headers_scanned, status) "
            "VALUES (?, ?, ?, ?, 0, 0, 'running')",
            (spec.key, spec.url, spec.listed_size, etag),
        )
        connection.commit()
        return 0, 0
    if tuple(row[:3]) != (spec.url, spec.listed_size, etag):
        raise CameraDownloadError(f"Archive source changed since the offset checkpoint: {spec.key}")
    return int(row[3]), int(row[4])


def checkpoint_tar_offsets(
    connection: sqlite3.Connection,
    spec: ArchiveSpec,
    *,
    next_offset: int,
    headers_scanned: int,
    status: str,
    offset_updates: Sequence[tuple[int, int, int, str]] = (),
) -> None:
    """Commit short, atomic batches of target offsets and scan progress."""

    if offset_updates:
        connection.executemany(
            "UPDATE target SET tar_header_offset = ?, tar_data_offset = ?, tar_member_size = ? "
            "WHERE archive_key = ? AND tar_member_name = ?",
            [
                (header_offset, data_offset, member_size, spec.key, member_name)
                for header_offset, data_offset, member_size, member_name in offset_updates
            ],
        )
    connection.execute(
        "UPDATE tar_offset_progress "
        "SET next_header_offset = ?, headers_scanned = ?, status = ? WHERE archive_key = ?",
        (next_offset, headers_scanned, status, spec.key),
    )
    connection.commit()


def scan_archive_tar_offsets(
    *,
    state_path: Path,
    spec: ArchiveSpec,
    max_headers: int,
    tar_scan_chunk_bytes: int,
) -> dict[str, Any]:
    """Record exact byte positions for this archive's selected JPEG members only."""

    connection = sqlite3.connect(state_path, timeout=120.0)
    connection.execute("PRAGMA busy_timeout = 120000")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        archive = RangeTarArchive.open(spec.url, scan_chunk_bytes=tar_scan_chunk_bytes)
        next_offset, scanned_before = initialize_tar_offset_progress(connection, spec, archive)
        connection.execute("UPDATE tar_offset_progress SET status = 'running' WHERE archive_key = ?", (spec.key,))
        connection.commit()
        pending_rows = connection.execute(
            "SELECT tar_member_name, tar_member_size FROM target "
            "WHERE archive_key = ? AND tar_data_offset IS NULL",
            (spec.key,),
        ).fetchall()
        pending_sizes = {str(name): (None if size is None else int(size)) for name, size in pending_rows}
        if not pending_sizes:
            checkpoint_tar_offsets(
                connection,
                spec,
                next_offset=next_offset,
                headers_scanned=scanned_before,
                status="complete",
            )
            return {"key": spec.key, "status": "complete", "headers_scanned": scanned_before, "indexed": 0}

        scanned_now = 0
        indexed = 0
        last_next_offset = next_offset
        offset_updates: list[tuple[int, int, int, str]] = []
        for member in archive.iter_members(start_header_offset=next_offset):
            if max_headers and scanned_now >= max_headers:
                checkpoint_tar_offsets(
                    connection,
                    spec,
                    next_offset=last_next_offset,
                    headers_scanned=scanned_before + scanned_now,
                    status="partial",
                    offset_updates=offset_updates,
                )
                return {
                    "key": spec.key,
                    "status": "partial",
                    "headers_scanned": scanned_before + scanned_now,
                    "indexed": indexed,
                }
            scanned_now += 1
            last_next_offset = member.next_header_offset
            planned_size = pending_sizes.get(member.name)
            if member.name in pending_sizes:
                if not member.is_regular_file or member.size <= 0 or not member.name.lower().endswith(".jpg"):
                    raise CameraDownloadError(f"Expected a regular JPEG TAR member, found {member.name!r}")
                if planned_size is not None and planned_size != member.size:
                    raise CameraDownloadError(
                        f"TAR member size changed for indexed JPEG {member.name!r}: "
                        f"{member.size} != {planned_size}"
                    )
                offset_updates.append((member.header_offset, member.data_offset, member.size, member.name))
                pending_sizes.pop(member.name)
                indexed += 1
            if scanned_now % 1000 == 0 or not pending_sizes:
                checkpoint_tar_offsets(
                    connection,
                    spec,
                    next_offset=last_next_offset,
                    headers_scanned=scanned_before + scanned_now,
                    status="running",
                    offset_updates=offset_updates,
                )
                offset_updates.clear()
            if not pending_sizes:
                checkpoint_tar_offsets(
                    connection,
                    spec,
                    next_offset=last_next_offset,
                    headers_scanned=scanned_before + scanned_now,
                    status="complete",
                )
                return {
                    "key": spec.key,
                    "status": "complete",
                    "headers_scanned": scanned_before + scanned_now,
                    "indexed": indexed,
                }
        missing = ", ".join(sorted(pending_sizes)[:3])
        raise CameraDownloadError(
            f"TAR ended before {len(pending_sizes)} planned image(s) were offset-indexed in {spec.key}: {missing}"
        )
    finally:
        connection.close()


def tar_offset_summary(connection: sqlite3.Connection) -> dict[str, int]:
    """Summarize the independent header-only offset-index pass."""

    target_total = int(connection.execute("SELECT COUNT(*) FROM target").fetchone()[0])
    target_indexed = int(
        connection.execute(
            "SELECT COUNT(*) FROM target "
            "WHERE tar_header_offset IS NOT NULL AND tar_data_offset IS NOT NULL AND tar_member_size IS NOT NULL"
        ).fetchone()[0]
    )
    malformed_offsets = int(
        connection.execute(
            "SELECT COUNT(*) FROM target WHERE "
            "(tar_header_offset IS NULL) != (tar_data_offset IS NULL) "
            "OR (tar_data_offset IS NOT NULL AND tar_member_size IS NULL)"
        ).fetchone()[0]
    )
    archive_rows = dict(
        connection.execute("SELECT status, COUNT(*) FROM tar_offset_progress GROUP BY status").fetchall()
    )
    headers = int(
        connection.execute("SELECT COALESCE(SUM(headers_scanned), 0) FROM tar_offset_progress").fetchone()[0]
    )
    expected_archives = int(connection.execute("SELECT COUNT(DISTINCT archive_key) FROM target").fetchone()[0])
    return {
        "target_total": target_total,
        "target_indexed": target_indexed,
        "target_pending": target_total - target_indexed,
        "target_malformed_offsets": malformed_offsets,
        "archive_total": expected_archives,
        "archives_complete": int(archive_rows.get("complete", 0)),
        "archives_partial": int(archive_rows.get("partial", 0)),
        "archives_running": int(archive_rows.get("running", 0)),
        "tar_headers_scanned": headers,
    }


def tar_offset_index_complete(summary: Mapping[str, int]) -> bool:
    """Whether every immutable target has a source-pinned TAR byte range."""

    return (
        int(summary["target_total"]) > 0
        and int(summary["target_indexed"]) == int(summary["target_total"])
        and int(summary["target_malformed_offsets"]) == 0
        and int(summary["archives_complete"]) == int(summary["archive_total"])
        and int(summary["archives_partial"]) == 0
        and int(summary["archives_running"]) == 0
    )


def publish_tar_offset_index(
    state: sqlite3.Connection,
    destination: Path,
    *,
    inventory_sha256: str,
) -> str:
    """Atomically publish the complete compact target-only TAR-offset index."""

    summary = tar_offset_summary(state)
    if not tar_offset_index_complete(summary):
        raise CameraDownloadError("Refusing to publish an incomplete TAR-offset index.")
    filter_sha256 = get_metadata(state, "navsim_scene_filter_sha256")
    if filter_sha256 is None:
        raise CameraDownloadError("Download state lacks the immutable NAVSIM SceneFilter hash.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    published = False
    output: sqlite3.Connection | None = None
    try:
        output = sqlite3.connect(temporary)
        output.execute("PRAGMA journal_mode = DELETE")
        output.execute("PRAGMA synchronous = FULL")
        output.execute("PRAGMA foreign_keys = ON")
        output.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE source_archive (
                archive_key TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                listed_size INTEGER NOT NULL,
                remote_etag TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE member_offset (
                native_blob_relative_path TEXT PRIMARY KEY,
                archive_key TEXT NOT NULL REFERENCES source_archive(archive_key),
                tar_member_name TEXT NOT NULL,
                tar_header_offset INTEGER NOT NULL,
                tar_data_offset INTEGER NOT NULL,
                tar_member_size INTEGER NOT NULL,
                navsim_log_name TEXT NOT NULL,
                camera_channel TEXT NOT NULL,
                UNIQUE (archive_key, tar_member_name)
            ) WITHOUT ROWID;
            CREATE INDEX member_offset_by_archive
                ON member_offset (archive_key, tar_data_offset);
            """
        )
        output.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("format", TAR_OFFSET_INDEX_FORMAT),
                ("created_at_utc", utc_now()),
                ("inventory_sha256", inventory_sha256),
                ("navsim_scene_filter_sha256", filter_sha256),
                ("target_member_count", str(summary["target_total"])),
                ("source_archive_count", str(summary["archive_total"])),
                ("source_format", "uncompressed_tar_streams_with_pinned_etags"),
                ("payload_policy", "offsets_only_no_native_jpeg_bytes_read"),
            ],
        )
        archive_rows = state.execute(
            """
            SELECT archive_key, url, listed_size, remote_etag
            FROM tar_offset_progress
            WHERE status = 'complete'
            ORDER BY archive_key
            """
        )
        output.executemany(
            "INSERT INTO source_archive (archive_key, url, listed_size, remote_etag) VALUES (?, ?, ?, ?)",
            archive_rows,
        )
        member_rows = state.execute(
            """
            SELECT destination_relative_path, archive_key, tar_member_name,
                   tar_header_offset, tar_data_offset, tar_member_size,
                   log_name, camera_channel
            FROM target
            WHERE tar_header_offset IS NOT NULL
              AND tar_data_offset IS NOT NULL
              AND tar_member_size IS NOT NULL
            ORDER BY archive_key, tar_data_offset
            """
        )
        batch: list[tuple[Any, ...]] = []
        for row in member_rows:
            batch.append(tuple(row))
            if len(batch) >= 10_000:
                output.executemany(
                    """
                    INSERT INTO member_offset (
                        native_blob_relative_path, archive_key, tar_member_name,
                        tar_header_offset, tar_data_offset, tar_member_size,
                        navsim_log_name, camera_channel
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                batch.clear()
        if batch:
            output.executemany(
                """
                INSERT INTO member_offset (
                    native_blob_relative_path, archive_key, tar_member_name,
                    tar_header_offset, tar_data_offset, tar_member_size,
                    navsim_log_name, camera_channel
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
        count = int(output.execute("SELECT COUNT(*) FROM member_offset").fetchone()[0])
        if count != int(summary["target_total"]):
            raise CameraDownloadError(
                f"Published TAR-offset member count mismatch: {count} != {summary['target_total']}"
            )
        foreign_key_errors = output.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise CameraDownloadError(f"Published TAR-offset index foreign-key failure: {foreign_key_errors[:1]}")
        if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise CameraDownloadError("Published TAR-offset index integrity check failed.")
        output.execute("ANALYZE")
        output.commit()
        output.close()
        output = None
        os.replace(temporary, destination)
        published = True
        return sha256_file(destination)
    finally:
        if output is not None:
            output.close()
        if not published:
            temporary.unlink(missing_ok=True)


def state_summary(connection: sqlite3.Connection) -> dict[str, int]:
    rows = dict(connection.execute("SELECT status, COUNT(*) FROM target GROUP BY status").fetchall())
    staged_bytes = int(
        connection.execute("SELECT COALESCE(SUM(tar_member_size), 0) FROM target WHERE status = 'staged'").fetchone()[0]
    )
    archive_rows = dict(connection.execute("SELECT status, COUNT(*) FROM archive_progress GROUP BY status").fetchall())
    headers = int(connection.execute("SELECT COALESCE(SUM(headers_scanned), 0) FROM archive_progress").fetchone()[0])
    return {
        "target_total": int(connection.execute("SELECT COUNT(*) FROM target").fetchone()[0]),
        "target_staged": int(rows.get("staged", 0)),
        "target_pending": int(rows.get("pending", 0)),
        "staged_bytes": staged_bytes,
        "archives_complete": int(archive_rows.get("complete", 0)),
        "archives_partial": int(archive_rows.get("partial", 0)),
        "archives_running": int(archive_rows.get("running", 0)),
        "tar_headers_scanned": headers,
    }


def run_tar_offset_index(
    args: argparse.Namespace,
    *,
    archives: Sequence[ArchiveSpec],
    targets: int,
    inventory_sha256: str,
) -> dict[str, Any]:
    """Run the resumable header-only pass and publish only when globally complete."""

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(int(args.workers), len(archives))) as executor:
        futures = {
            executor.submit(
                scan_archive_tar_offsets,
                state_path=args.state_sqlite,
                spec=spec,
                max_headers=int(args.max_headers_per_archive),
                tar_scan_chunk_bytes=int(args.tar_scan_chunk_bytes),
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

    connection = open_state(args.state_sqlite)
    try:
        summary = tar_offset_summary(connection)
        complete = tar_offset_index_complete(summary)
        index_sha256 = (
            publish_tar_offset_index(
                connection,
                args.tar_offset_index_sqlite,
                inventory_sha256=inventory_sha256,
            )
            if complete
            else None
        )
    finally:
        connection.close()
    return {
        "format": REPORT_FORMAT,
        "generated_at_utc": utc_now(),
        "status": "pass" if complete else "partial",
        "operation": "offset-index",
        "inventory_sha256": inventory_sha256,
        "inventory_target_count": targets,
        "archive_count": len(archives),
        "archive_results": sorted(results, key=lambda item: str(item["key"])),
        "offset_state": summary,
        "output": {
            "tar_offset_index_format": TAR_OFFSET_INDEX_FORMAT,
            "tar_offset_index_sqlite": str(args.tar_offset_index_sqlite) if complete else None,
            "tar_offset_index_sha256": index_sha256,
        },
        "contract": {
            "camera_archives_are_tar_streams": True,
            "indexes_only_db_verified_inventory_members": True,
            "tar_scan_chunk_bytes": int(args.tar_scan_chunk_bytes),
            "native_jpeg_bytes_read": False,
            "excluded_data": ["test", "mini", "maps", "native_lidar"],
            "source_etag_is_pinned_per_archive": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.workers > 55:
        raise CameraDownloadError("--workers must be between 1 and 55")
    if args.max_headers_per_archive < 0:
        raise CameraDownloadError("--max-headers-per-archive must be nonnegative")
    if (
        args.tar_scan_chunk_bytes < 512
        or args.tar_scan_chunk_bytes % 512
    ):
        raise CameraDownloadError("--tar-scan-chunk-bytes must be a positive multiple of 512")
    if args.report_json.suffix != ".json":
        raise CameraDownloadError("--report-json must have a .json suffix")
    if args.tar_offset_index_sqlite.suffix != ".sqlite":
        raise CameraDownloadError("--tar-offset-index-sqlite must have a .sqlite suffix")
    with exclusive_state_lock(args.state_sqlite):
        train_groups = parse_group_manifest(args.train_group_manifest, split="train")
        val_groups = parse_group_manifest(args.val_group_manifest, split="val")
        connection = open_state(args.state_sqlite)
        try:
            targets, groups, inventory_sha = initialize_targets(
                connection,
                inventory=args.inventory_jsonl,
                train_groups=train_groups,
                val_groups=val_groups,
            )
            archives = discover_archives(
                listing_url=args.listing_url,
                timeout_seconds=float(args.listing_timeout_seconds),
                groups=groups,
            )
            archives = select_archives(archives, args.archive_key)
            if args.operation == "initialize" and not args.dry_run:
                return {
                    "format": REPORT_FORMAT,
                    "generated_at_utc": utc_now(),
                    "status": "pass",
                    "operation": "initialize",
                    "inventory_sha256": inventory_sha,
                    "inventory_target_count": targets,
                    "archive_count": len(archives),
                    "archives": [
                        {
                            "key": item.key,
                            "size_bytes": item.listed_size,
                            "split": item.source_split,
                            "group": item.group,
                        }
                        for item in archives
                    ],
                    "state": state_summary(connection),
                    "contract": {
                        "native_jpeg_bytes_read": False,
                        "tar_headers_scanned": False,
                        "target_ledger_is_inventory_derived": True,
                    },
                }
            if args.dry_run:
                summary = state_summary(connection)
                offset_summary = tar_offset_summary(connection)
                return {
                    "format": REPORT_FORMAT,
                    "generated_at_utc": utc_now(),
                    "status": "dry_run",
                    "operation": args.operation,
                    "inventory_sha256": inventory_sha,
                    "inventory_target_count": targets,
                    "archive_count": len(archives),
                    "archives": [
                        {"key": item.key, "size_bytes": item.listed_size, "split": item.source_split, "group": item.group}
                        for item in archives
                    ],
                    "state": summary,
                    "offset_state": offset_summary,
                }
        finally:
            connection.close()

        if args.operation == "offset-index":
            return run_tar_offset_index(
                args,
                archives=archives,
                targets=targets,
                inventory_sha256=inventory_sha,
            )

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(int(args.workers), len(archives))) as executor:
            futures = {
                executor.submit(
                    scan_archive,
                    state_path=args.state_sqlite,
                    spec=spec,
                    nuplan_root=args.nuplan_root,
                    max_headers=int(args.max_headers_per_archive),
                    tar_scan_chunk_bytes=int(args.tar_scan_chunk_bytes),
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

        connection = open_state(args.state_sqlite)
        try:
            summary = state_summary(connection)
        finally:
            connection.close()
        complete = summary["target_staged"] == summary["target_total"] and summary["archives_partial"] == 0
        return {
            "format": REPORT_FORMAT,
            "generated_at_utc": utc_now(),
            "status": "pass" if complete else "partial",
            "inventory_sha256": inventory_sha,
            "inventory_target_count": targets,
            "archive_count": len(archives),
            "archive_results": sorted(results, key=lambda item: str(item["key"])),
            "state": summary,
            "contract": {
                "camera_archives_are_tar_streams": True,
                "writes_only_db_verified_inventory_members": True,
                "tar_scan_chunk_bytes": int(args.tar_scan_chunk_bytes),
                "excluded_data": ["test", "mini", "maps", "native_lidar"],
                "all_staged_images_have_tar_header_size_and_sha256": True,
            },
        }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (CameraDownloadError, RangeTarError, OSError, ValueError, sqlite3.Error) as exc:
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
