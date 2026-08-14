#!/usr/bin/env python3
"""Build a metadata-only, native nuPlan 10 Hz index for NAVSIM ``navtrain``.

OpenScene/NAVSIM retains 2 Hz observations.  This program proves every
selected NAVSIM anchor against the matching native nuPlan v1.1 SQLite DB and
publishes a normalized SQLite index for a *derived* camera-history protocol.
It does not read, download, or require any native JPEG bytes.

The index is deliberately range-normalized: an ``anchor_camera`` row records
the inclusive native stream-index interval for one anchor/channel, while
``native_image`` stores each image once.  This preserves exact 10 Hz input
sequences without storing roughly fifteen duplicate image associations for
every anchor/channel.

Official NAVSIM remains a 2 Hz benchmark.  This index alone does not modify
the stock NAVSIM loader or create 10 Hz prediction anchors, labels, or test
data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence, TextIO

import audit_navsim_nuplan10hz_lineage as lineage


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_NUPLAN_ROOT = REPO_ROOT / "dataset/raw/nuplan-v1.1-navtrain-10hz"
DEFAULT_INDEX = (
    REPO_ROOT
    / "dataset/manifests/navsim_nuplan10hz/"
    "navtrain_camera_10hz_index.sqlite"
)
DEFAULT_REPORT = (
    REPO_ROOT
    / "dataset/manifests/navsim_nuplan10hz/"
    "navtrain_camera_10hz_plan_report.json"
)

INDEX_FORMAT = "navsim_nuplan10hz_index_v1"
INVENTORY_FORMAT = "navsim_nuplan10hz_camera_inventory_v1"
REPORT_FORMAT = "navsim_nuplan10hz_camera_plan_report_v2"
NAVSIM_RETAINED_CAMERA_INTERVAL_US = 500_000
NOMINAL_NATIVE_HISTORY_FRAMES = 16
NATIVE_HISTORY_FRAME_POLICY = lineage.NATIVE_HISTORY_FRAME_POLICY
NATIVE_CAMERA_CAPTURE_GAP_POLICY = lineage.NATIVE_CAMERA_CAPTURE_GAP_POLICY


@dataclass(frozen=True)
class AnchorContext:
    """The selected NAVSIM anchor plus its actual 2 Hz history boundary."""

    log_name: str
    row: Mapping[str, Any]
    row_index: int
    history_rows: tuple[Mapping[str, Any], ...]
    navsim_history_start_timestamp_us: int


@dataclass
class PlanState:
    anchors_seen: int = 0
    anchors_verified: int = 0
    logs_scanned: int = 0
    native_databases_verified: int = 0
    retained_camera_records_verified: int = 0
    native_10hz_image_references: int = 0
    channel_reference_counts: Counter[str] = field(default_factory=Counter)
    history_frame_counts: Counter[str] = field(default_factory=Counter)
    navsim_history_cadence_exceptions: int = 0
    native_stream_capture_gap_count: int = 0
    native_stream_capture_gap_delta_us: Counter[int] = field(default_factory=Counter)
    native_stream_capture_gap_logs: set[str] = field(default_factory=set)
    native_stream_capture_gap_streams: set[tuple[str, str]] = field(default_factory=set)
    native_history_capture_gap_count: int = 0
    native_history_capture_gap_delta_us: Counter[int] = field(default_factory=Counter)
    native_history_capture_gap_anchor_tokens: set[str] = field(default_factory=set)
    native_history_capture_gap_anchor_channels: int = 0
    mapping_methods: Counter[str] = field(default_factory=Counter)
    partial: bool = False


@dataclass
class NativeImageWriteCache:
    """Per-source-log write cache for normalized native-image records.

    A four-frame NAVSIM history overlaps heavily with neighboring anchors.
    Keeping IDs and reference counts in memory avoids a SQLite select/update
    round trip for every repeated native image while retaining the same
    normalized on-disk representation.
    """

    image_ids: dict[tuple[str, str], int] = field(default_factory=dict)
    reference_counts: Counter[int] = field(default_factory=Counter)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-root", type=Path, default=lineage.DEFAULT_NAVSIM_ROOT)
    parser.add_argument("--log-path", type=Path, default=None)
    # Retained only so old invocations do not fail.  The canonical planner
    # reads paths from the pickle metadata but intentionally never reads JPEGs.
    parser.add_argument("--sensor-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scene-filter-yaml", type=Path, default=lineage.DEFAULT_FILTER)
    parser.add_argument(
        "--nuplan-root",
        type=Path,
        default=DEFAULT_NUPLAN_ROOT,
        help="Native root containing splits/trainval/<log>.db.",
    )
    parser.add_argument("--nuplan-split", choices=("trainval",), default="trainval")
    parser.add_argument("--channels", default=",".join(lineage.NUPLAN_CAMERA_CHANNELS))
    parser.add_argument("--retained-camera-tolerance-us", type=int, default=50_000)
    parser.add_argument("--camera-cadence-tolerance-us", type=int, default=50_000)
    parser.add_argument(
        "--navsim-camera-cadence-tolerance-us",
        type=int,
        default=50_000,
        help="Allowed deviation from NAVSIM's retained 2 Hz / 500 ms camera cadence.",
    )
    parser.add_argument(
        "--anchor-token",
        action="append",
        default=[],
        help="Plan this official navtrain anchor token only; may be repeated.",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="Plan this many anchors for a deterministic pilot; 0 plans all requested anchors.",
    )
    parser.add_argument(
        "--allow-nonofficial-filter",
        action="store_true",
        help="Permit a fixture/custom SceneFilter; disabled for the production contract.",
    )
    parser.add_argument(
        "--index-sqlite",
        type=Path,
        default=DEFAULT_INDEX,
        help="Canonical normalized metadata index (.sqlite).",
    )
    parser.add_argument(
        "--inventory-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional compatibility export for the later camera extractor. The SQLite index remains "
            "canonical; no JSONL is written unless this is explicitly supplied."
        ),
    )
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and count the plan without publishing an index, inventory, or report.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> None:
    if args.log_path is None:
        args.log_path = args.navsim_root / "navsim_logs" / "trainval"
    if args.max_anchors < 0:
        raise lineage.AuditError("--max-anchors must be nonnegative")
    if (
        args.retained_camera_tolerance_us < 0
        or args.camera_cadence_tolerance_us < 0
        or args.navsim_camera_cadence_tolerance_us < 0
    ):
        raise lineage.AuditError("Timestamp tolerances must be nonnegative")
    if args.index_sqlite.suffix != ".sqlite":
        raise lineage.AuditError("--index-sqlite must have a .sqlite suffix")
    if args.inventory_jsonl is not None and args.inventory_jsonl.suffix != ".jsonl":
        raise lineage.AuditError("--inventory-jsonl must have a .jsonl suffix")
    if args.report_json.suffix != ".json":
        raise lineage.AuditError("--report-json must have a .json suffix")


def safe_blob_relative_path(log_name: str, image: lineage.NativeImage) -> str:
    """Return the canonical staged path used by the native nuPlan devkit."""

    filename = str(image.filename_jpg)
    path = PurePosixPath(filename)
    expected_channel_path = PurePosixPath(image.channel) / path.name
    expected_log_channel_path = PurePosixPath(log_name) / expected_channel_path
    if (
        not filename
        or "\\" in filename
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() != ".jpg"
        or path
        not in {
            PurePosixPath(path.name),
            expected_channel_path,
            expected_log_channel_path,
        }
    ):
        raise lineage.AuditError(
            f"Unsafe native JPEG filename for {log_name}/{image.channel}: {filename!r}"
        )
    if not log_name or "/" in log_name or "\\" in log_name:
        raise lineage.AuditError(f"Unsafe native log name: {log_name!r}")
    return str(PurePosixPath("sensor_blobs") / log_name / image.channel / path.name)


def canonical_archive_relative_path(
    *,
    log_name: str,
    channel: str,
    native_blob_relative_path: str,
) -> str:
    """Derive a public camera-TAR member path from a staged blob path.

    This is intentionally used only for the optional compatibility inventory.
    The canonical SQLite index stores the staged path, which is the path a
    future image reader needs; retaining this second spelling for every image
    would duplicate metadata at navtrain scale.
    """

    path = PurePosixPath(native_blob_relative_path)
    expected_prefix = ("sensor_blobs", log_name, channel)
    if (
        not native_blob_relative_path
        or "\\" in native_blob_relative_path
        or path.is_absolute()
        or len(path.parts) != 4
        or tuple(path.parts[:3]) != expected_prefix
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix.lower() != ".jpg"
    ):
        raise lineage.AuditError(
            f"Unsafe native blob path for compatibility inventory: {native_blob_relative_path!r}"
        )
    return str(PurePosixPath(*path.parts[1:]))


def load_log_rows(log_file: Path) -> list[Mapping[str, Any]]:
    try:
        with log_file.open("rb") as handle:
            rows = pickle.load(handle)
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise lineage.AuditError(
                "NAVSIM metadata pickles contain NumPy arrays, but this interpreter has no numpy. "
                "Run this planner with the NAVSIM environment."
            ) from exc
        raise lineage.AuditError(f"Could not load NAVSIM metadata pickle {log_file}: {exc!r}") from exc
    except Exception as exc:
        raise lineage.AuditError(f"Could not load NAVSIM metadata pickle {log_file}: {exc!r}") from exc
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise lineage.AuditError(f"NAVSIM metadata pickle is not a list of mappings: {log_file}")
    return rows


def iter_anchor_contexts(
    *,
    log_path: Path,
    scene_filter: Mapping[str, Any],
    requested_tokens: set[str],
    max_anchors: int,
    navsim_camera_cadence_tolerance_us: int,
    state: PlanState,
) -> Iterable[AnchorContext]:
    """Resolve official anchor tokens while preserving NAVSIM's 2 Hz history start."""

    if not log_path.is_dir():
        raise lineage.AuditError(f"NAVSIM log directory does not exist: {log_path}")
    history_frames = int(scene_filter["history_frames"])
    future_frames = int(scene_filter["future_frames"])
    remaining = set(requested_tokens)
    emitted = 0
    for log_name in scene_filter["logs"]:
        if max_anchors and emitted >= max_anchors:
            state.partial = True
            break
        log_file = log_path / f"{log_name}.pkl"
        if not log_file.is_file():
            raise lineage.AuditError(f"Missing NAVSIM OpenScene metadata pickle: {log_file}")
        rows = load_log_rows(log_file)
        state.logs_scanned += 1
        for row_index, row in enumerate(rows):
            if max_anchors and emitted >= max_anchors:
                state.partial = True
                break
            token = lineage.as_hex(row.get("token"))
            if token not in remaining:
                continue
            if str(row.get("log_name") or "") != log_name:
                raise lineage.AuditError(
                    f"NAVSIM token {token} is stored under {log_name!r} but row.log_name is "
                    f"{row.get('log_name')!r}"
                )
            if row_index < history_frames - 1:
                raise lineage.AuditError(
                    f"NAVSIM anchor {token} at row {row_index} lacks the required {history_frames} history frames"
                )
            if row_index + future_frames >= len(rows):
                raise lineage.AuditError(
                    f"NAVSIM anchor {token} at row {row_index} lacks the required {future_frames} future frames"
                )
            try:
                anchor_timestamp_us = int(row["timestamp"])
                inherited_start_timestamp_us = int(rows[row_index - history_frames + 1]["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise lineage.AuditError(f"NAVSIM anchor {token} lacks a valid history timestamp") from exc
            if inherited_start_timestamp_us > anchor_timestamp_us:
                raise lineage.AuditError(
                    f"NAVSIM anchor {token} has a history timestamp later than its anchor timestamp"
                )
            history_rows = tuple(rows[row_index - history_frames + 1 : row_index + 1])
            if len(history_rows) != history_frames:
                raise lineage.AuditError(
                    f"NAVSIM anchor {token} does not expose its complete {history_frames}-frame history"
                )
            try:
                history_timestamps_us = tuple(int(history_row["timestamp"]) for history_row in history_rows)
            except (KeyError, TypeError, ValueError) as exc:
                raise lineage.AuditError(f"NAVSIM anchor {token} lacks a valid retained camera timestamp") from exc
            for previous_timestamp_us, current_timestamp_us in zip(
                history_timestamps_us,
                history_timestamps_us[1:],
            ):
                delta_us = current_timestamp_us - previous_timestamp_us
                if delta_us <= 0:
                    raise lineage.AuditError(
                        f"NAVSIM retained camera history for {token} is not strictly increasing: "
                        f"{delta_us} us between retained rows"
                    )
                # Keep the source timestamps exactly.  Most NAVSIM histories
                # are nominally 2 Hz, but dropped/irregular retained rows are
                # part of the released data and must not be silently padded or
                # discarded.  The report records their count for downstream
                # fixed-length feature-builder decisions.
                if abs(delta_us - NAVSIM_RETAINED_CAMERA_INTERVAL_US) > navsim_camera_cadence_tolerance_us:
                    state.navsim_history_cadence_exceptions += 1
            remaining.remove(token)
            emitted += 1
            yield AnchorContext(
                log_name=log_name,
                row=row,
                row_index=row_index,
                history_rows=history_rows,
                navsim_history_start_timestamp_us=int(inherited_start_timestamp_us),
            )
        if not remaining:
            break
    if remaining and not state.partial:
        preview = ", ".join(sorted(remaining)[:8])
        raise lineage.AuditError(
            f"{len(remaining)} requested navtrain token(s) were not found in the allowed NAVSIM log pickles; "
            f"first: {preview}"
        )


def create_index_store(destination: Path) -> tuple[Path, sqlite3.Connection]:
    """Create an unpublished SQLite index on the destination filesystem."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("PRAGMA cache_size = -262144")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE source_log (
                navsim_log_name TEXT PRIMARY KEY,
                navsim_log_token TEXT NOT NULL,
                native_db_relative_path TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE anchor (
                navsim_anchor_token TEXT PRIMARY KEY,
                navsim_log_name TEXT NOT NULL REFERENCES source_log(navsim_log_name),
                navsim_log_token TEXT NOT NULL,
                navsim_scene_token TEXT NOT NULL,
                navsim_anchor_timestamp_us INTEGER NOT NULL,
                navsim_row_index INTEGER NOT NULL,
                navsim_history_start_timestamp_us INTEGER NOT NULL,
                native_lidar_token TEXT NOT NULL,
                native_lidar_timestamp_us INTEGER NOT NULL,
                native_lidar_scene_token TEXT NOT NULL,
                native_lidar_filename TEXT NOT NULL,
                native_lidar_mapping_method TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE native_image (
                image_id INTEGER PRIMARY KEY,
                navsim_log_name TEXT NOT NULL REFERENCES source_log(navsim_log_name),
                camera_channel TEXT NOT NULL,
                stream_index INTEGER NOT NULL,
                native_image_token TEXT NOT NULL,
                native_image_timestamp_us INTEGER NOT NULL,
                native_image_filename_jpg TEXT NOT NULL,
                native_camera_token TEXT NOT NULL,
                native_blob_relative_path TEXT NOT NULL,
                anchor_reference_count INTEGER NOT NULL CHECK(anchor_reference_count > 0),
                UNIQUE(navsim_log_name, camera_channel, stream_index),
                UNIQUE(navsim_log_name, camera_channel, native_image_token),
                UNIQUE(native_blob_relative_path)
            );

            CREATE TABLE anchor_camera (
                navsim_anchor_token TEXT NOT NULL REFERENCES anchor(navsim_anchor_token),
                camera_channel TEXT NOT NULL,
                history_start_retained_image_id INTEGER NOT NULL REFERENCES native_image(image_id),
                history_start_retained_native_image_timestamp_us INTEGER NOT NULL,
                history_end_retained_image_id INTEGER NOT NULL REFERENCES native_image(image_id),
                history_end_retained_native_image_timestamp_us INTEGER NOT NULL,
                history_start_timestamp_us INTEGER NOT NULL,
                history_end_timestamp_us INTEGER NOT NULL,
                first_stream_index INTEGER NOT NULL,
                last_stream_index INTEGER NOT NULL,
                frame_count INTEGER NOT NULL CHECK(frame_count > 0),
                PRIMARY KEY(navsim_anchor_token, camera_channel),
                CHECK(last_stream_index >= first_stream_index),
                CHECK(frame_count = last_stream_index - first_stream_index + 1)
            ) WITHOUT ROWID;

            CREATE INDEX anchor_by_log
                ON anchor(navsim_log_name, navsim_anchor_timestamp_us);
            """
        )
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path, connection


def set_metadata(connection: sqlite3.Connection, values: Mapping[str, Any]) -> None:
    """Store scalar metadata as plain SQLite text.

    The index is a cross-package contract, so values such as ``format`` must
    be directly comparable without every consumer needing JSON decoding.
    Structured metadata does not belong in this small key/value table.
    """

    serialized: list[tuple[str, str]] = []
    for key, value in values.items():
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            text = str(value)
        else:
            raise lineage.AuditError(
                f"Index metadata {key!r} must be a scalar, got {type(value).__name__}"
            )
        serialized.append((str(key), text))
    connection.executemany(
        "INSERT INTO metadata (key, value) VALUES (?, ?)",
        serialized,
    )


def insert_source_log(
    connection: sqlite3.Connection,
    *,
    log_name: str,
    log_token: str,
    native_split: str,
) -> None:
    relative_db = str(PurePosixPath("splits") / native_split / f"{log_name}.db")
    connection.execute(
        "INSERT INTO source_log (navsim_log_name, navsim_log_token, native_db_relative_path) VALUES (?, ?, ?)",
        (log_name, log_token, relative_db),
    )


def insert_anchor(
    connection: sqlite3.Connection,
    *,
    context: AnchorContext,
    native_lidar: lineage.NativeLidar,
) -> str:
    token = lineage.as_hex(context.row.get("token"))
    if not token:
        raise lineage.AuditError(f"NAVSIM row in {context.log_name} has an empty anchor token")
    try:
        connection.execute(
            """
            INSERT INTO anchor (
                navsim_anchor_token, navsim_log_name, navsim_log_token, navsim_scene_token,
                navsim_anchor_timestamp_us, navsim_row_index, navsim_history_start_timestamp_us,
                native_lidar_token, native_lidar_timestamp_us,
                native_lidar_scene_token, native_lidar_filename, native_lidar_mapping_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token,
                context.log_name,
                lineage.as_hex(context.row.get("log_token")),
                lineage.as_hex(context.row.get("scene_token")),
                int(context.row["timestamp"]),
                context.row_index,
                context.navsim_history_start_timestamp_us,
                native_lidar.token,
                native_lidar.timestamp_us,
                native_lidar.scene_token,
                native_lidar.filename,
                native_lidar.mapping_method,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise lineage.AuditError(f"Duplicate or invalid NAVSIM anchor row for {token}") from exc
    return token


def ensure_native_image(
    connection: sqlite3.Connection,
    *,
    log_name: str,
    image: lineage.NativeImage,
    stream_index: int,
    cache: NativeImageWriteCache,
) -> int:
    """Insert one image once, cache its ID, and count this anchor reference."""

    cache_key = (image.channel, image.token)
    cached_image_id = cache.image_ids.get(cache_key)
    if cached_image_id is not None:
        cache.reference_counts[cached_image_id] += 1
        return cached_image_id
    blob_relative_path = safe_blob_relative_path(log_name, image)
    try:
        cursor = connection.execute(
            """
            INSERT INTO native_image (
                navsim_log_name, camera_channel, stream_index, native_image_token,
                native_image_timestamp_us, native_image_filename_jpg, native_camera_token,
                native_blob_relative_path, anchor_reference_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                log_name,
                image.channel,
                int(stream_index),
                image.token,
                int(image.timestamp_us),
                str(image.filename_jpg),
                str(image.camera_token),
                blob_relative_path,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise lineage.AuditError(
            f"Native image identity is inconsistent for {log_name}/{image.channel}/{image.token}"
        ) from exc
    image_id = int(cursor.lastrowid)
    cache.image_ids[cache_key] = image_id
    cache.reference_counts[image_id] = 1
    return image_id


def flush_native_image_reference_counts(
    connection: sqlite3.Connection,
    cache: NativeImageWriteCache,
) -> None:
    """Persist the per-log reference totals after all overlapping anchors are seen."""

    updates = [
        (int(reference_count), int(image_id))
        for image_id, reference_count in cache.reference_counts.items()
        if reference_count != 1
    ]
    if updates:
        connection.executemany(
            "UPDATE native_image SET anchor_reference_count = ? WHERE image_id = ?",
            updates,
        )


def insert_anchor_camera(
    connection: sqlite3.Connection,
    *,
    anchor_token: str,
    channel: str,
    history_start_retained_image_id: int,
    history_start_retained_timestamp_us: int,
    history_end_retained_image_id: int,
    history_end_retained_timestamp_us: int,
    history_start_timestamp_us: int,
    history_end_timestamp_us: int,
    first_stream_index: int,
    last_stream_index: int,
    frame_count: int,
) -> None:
    connection.execute(
        """
        INSERT INTO anchor_camera (
            navsim_anchor_token, camera_channel, history_start_retained_image_id,
            history_start_retained_native_image_timestamp_us, history_end_retained_image_id,
            history_end_retained_native_image_timestamp_us, history_start_timestamp_us,
            history_end_timestamp_us, first_stream_index, last_stream_index, frame_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            anchor_token,
            channel,
            history_start_retained_image_id,
            history_start_retained_timestamp_us,
            history_end_retained_image_id,
            history_end_retained_timestamp_us,
            history_start_timestamp_us,
            history_end_timestamp_us,
            first_stream_index,
            last_stream_index,
            frame_count,
        ),
    )


def write_compatibility_inventory(
    connection: sqlite3.Connection,
    destination: Path,
    *,
    scene_filter_sha256: str,
) -> tuple[int, int, str]:
    """Export the old camera-extractor JSONL only when explicitly requested."""

    temporary, handle = lineage.open_manifest_writer(destination, dry_run=False)
    assert temporary is not None and handle is not None
    digest = hashlib.sha256()
    count = 0
    references = 0
    success = False
    try:
        rows = connection.execute(
            """
            SELECT navsim_log_name, camera_channel, native_image_token,
                   native_image_timestamp_us, native_camera_token, native_blob_relative_path,
                   anchor_reference_count
            FROM native_image
            ORDER BY navsim_log_name, camera_channel, stream_index
            """
        )
        for row in rows:
            archive_relative_path = canonical_archive_relative_path(
                log_name=str(row[0]),
                channel=str(row[1]),
                native_blob_relative_path=str(row[5]),
            )
            payload = {
                "format": INVENTORY_FORMAT,
                "navsim_scene_filter_sha256": scene_filter_sha256,
                "navsim_log_name": str(row[0]),
                "camera_channel": str(row[1]),
                "native_image_filename_jpg": archive_relative_path,
                "native_image_token": str(row[2]),
                "native_image_timestamp_us": int(row[3]),
                "native_camera_token": str(row[4]),
                "native_blob_relative_path": str(row[5]),
                "anchor_reference_count": int(row[6]),
            }
            encoded = json.dumps(payload, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode("utf-8"))
            count += 1
            references += int(row[6])
        success = True
        return count, references, digest.hexdigest()
    finally:
        lineage.publish_manifest(temporary, handle, destination, success=success)


def query_counts(connection: sqlite3.Connection) -> tuple[int, int, int]:
    anchors = int(connection.execute("SELECT COUNT(*) FROM anchor").fetchone()[0])
    images = int(connection.execute("SELECT COUNT(*) FROM native_image").fetchone()[0])
    references = int(
        connection.execute("SELECT COALESCE(SUM(anchor_reference_count), 0) FROM native_image").fetchone()[0]
    )
    return anchors, images, references


def finalize_index(connection: sqlite3.Connection, temporary_path: Path, destination: Path, *, success: bool) -> str | None:
    try:
        if success:
            connection.commit()
            connection.execute("ANALYZE")
            connection.commit()
    finally:
        connection.close()
    if not success:
        temporary_path.unlink(missing_ok=True)
        return None
    # Closing a DELETE-journal database leaves one self-contained file, so an
    # atomic replace never exposes a partially built index.
    os.replace(temporary_path, destination)
    return sha256_file(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def history_frame_counts_report(state: PlanState, channels: Sequence[str]) -> dict[str, dict[int, int]]:
    report = {channel: {} for channel in channels}
    for key, total in state.history_frame_counts.items():
        channel, count_text = key.rsplit(":", 1)
        if channel in report:
            report[channel][int(count_text)] = int(total)
    return {
        channel: dict(sorted(counts.items()))
        for channel, counts in report.items()
    }


def capture_gap_delta_report(counter: Counter[int]) -> dict[int, int]:
    """Return stable JSON-safe native timestamp-gap counts."""

    return {int(delta_us): int(total) for delta_us, total in sorted(counter.items())}


def build_report(
    *,
    args: argparse.Namespace,
    scene_filter: Mapping[str, Any],
    channels: Sequence[str],
    state: PlanState,
    anchor_count: int,
    unique_images: int,
    image_references: int,
    index_sha256: str | None,
    inventory_sha256: str | None,
) -> dict[str, Any]:
    return {
        "format": REPORT_FORMAT,
        "generated_at_utc": lineage.utc_now(),
        "status": "partial_pass" if state.partial else "pass",
        "partial": state.partial,
        "dry_run": bool(args.dry_run),
        "navsim": {
            "log_path": str(args.log_path),
            "scene_filter_yaml": str(args.scene_filter_yaml),
            "scene_filter_sha256": scene_filter["sha256"],
            "selected_anchor_count": len(scene_filter["tokens"]),
            "requested_anchor_count": anchor_count,
            "history_frames": int(scene_filter["history_frames"]),
            "future_frames": int(scene_filter["future_frames"]),
            "retained_camera_hz": 2,
            "camera_cadence_tolerance_us": int(args.navsim_camera_cadence_tolerance_us),
            "retained_history_cadence_exceptions": state.navsim_history_cadence_exceptions,
        },
        "native_nuplan": {
            "root": str(args.nuplan_root),
            "split": args.nuplan_split,
            "camera_hz": 10,
            "lidar_hz": 20,
            "channels": list(channels),
            "nominal_camera_interval_us": lineage.NOMINAL_NATIVE_CAMERA_INTERVAL_US,
            "source_capture_gap_policy": NATIVE_CAMERA_CAPTURE_GAP_POLICY,
        },
        "checks": {
            "anchors_seen": state.anchors_seen,
            "anchors_verified": state.anchors_verified,
            "logs_scanned": state.logs_scanned,
            "native_databases_verified": state.native_databases_verified,
            "retained_camera_records_verified": state.retained_camera_records_verified,
            "native_10hz_image_references": state.native_10hz_image_references,
            "native_10hz_references_by_channel": dict(sorted(state.channel_reference_counts.items())),
            "native_history_frame_counts_by_channel": history_frame_counts_report(state, channels),
            "native_stream_capture_gap_count": state.native_stream_capture_gap_count,
            "native_stream_capture_gap_delta_us_counts": capture_gap_delta_report(
                state.native_stream_capture_gap_delta_us
            ),
            "native_stream_capture_gap_affected_log_count": len(state.native_stream_capture_gap_logs),
            "native_stream_capture_gap_affected_stream_count": len(state.native_stream_capture_gap_streams),
            "native_history_capture_gap_count": state.native_history_capture_gap_count,
            "native_history_capture_gap_delta_us_counts": capture_gap_delta_report(
                state.native_history_capture_gap_delta_us
            ),
            "native_history_capture_gap_affected_anchor_count": len(
                state.native_history_capture_gap_anchor_tokens
            ),
            "native_history_capture_gap_affected_anchor_channel_count": (
                state.native_history_capture_gap_anchor_channels
            ),
            "native_lidar_mapping_methods": dict(sorted(state.mapping_methods.items())),
            "unique_native_10hz_images": unique_images,
            "native_image_reference_count": image_references,
            "native_jpeg_bytes_read": False,
        },
        "output": {
            "index_sqlite": str(args.index_sqlite),
            "index_format": INDEX_FORMAT,
            "index_sha256": index_sha256,
            "inventory_jsonl": None if args.inventory_jsonl is None else str(args.inventory_jsonl),
            "inventory_format": None if args.inventory_jsonl is None else INVENTORY_FORMAT,
            "inventory_sha256": inventory_sha256,
        },
        "contract": {
            "does_not_read_or_modify_raw_sensor_data": True,
            "derived_protocol": "NAVSIM-derived-10Hz-camera-history",
            "official_navsim_anchor_hz": 2,
            "native_nuplan_camera_hz": 10,
            "native_nuplan_lidar_hz": 20,
            "nominal_native_history_frames": NOMINAL_NATIVE_HISTORY_FRAMES,
            "native_history_frame_policy": NATIVE_HISTORY_FRAME_POLICY,
            "native_camera_capture_gap_policy": NATIVE_CAMERA_CAPTURE_GAP_POLICY,
            "native_camera_timestamp_policy": "preserve_exact_source_timestamps",
            "index_relationship_encoding": "native stream index ranges per anchor/channel",
            "camera_history_endpoint": (
                "exact contiguous native camera range between the first and current retained NAVSIM frames; "
                "all four retained observations are required and source timestamp gaps are preserved"
            ),
            "does_not_create_10hz_prediction_anchors_labels_or_evaluation": True,
        },
    }


def plan(args: argparse.Namespace) -> dict[str, Any]:
    resolve_paths(args)
    lineage.reject_heldout_paths(args)
    channels = lineage.parse_channels(args.channels)
    scene_filter = lineage.load_scene_filter(
        args.scene_filter_yaml,
        allow_nonofficial=bool(args.allow_nonofficial_filter),
    )
    official_tokens = set(scene_filter["tokens"])
    requested_tokens = (
        {lineage.as_hex(token) for token in args.anchor_token} if args.anchor_token else official_tokens
    )
    unknown = sorted(requested_tokens - official_tokens)
    if unknown:
        raise lineage.AuditError(f"Requested --anchor-token values are not in navtrain: {unknown[:8]}")
    state = PlanState(
        partial=(bool(args.anchor_token) and len(requested_tokens) != len(official_tokens))
        or (bool(args.max_anchors) and int(args.max_anchors) < len(requested_tokens))
    )
    temporary_index, store = create_index_store(args.index_sqlite)
    native_connection: sqlite3.Connection | None = None
    active_log_name: str | None = None
    timelines: dict[str, lineage.NativeImageTimeline] = {}
    timeline_indices: dict[str, dict[str, int]] = {}
    native_image_cache = NativeImageWriteCache()
    index_success = False
    index_sha256: str | None = None
    inventory_sha256: str | None = None
    anchor_count = 0
    unique_images = 0
    image_references = 0
    try:
        set_metadata(
            store,
            {
                "format": INDEX_FORMAT,
                "schema_version": 1,
                "created_at_utc": lineage.utc_now(),
                "derived_protocol": "NAVSIM-derived-10Hz-camera-history",
                "navsim_scene_filter_sha256": scene_filter["sha256"],
                "navsim_history_frames": int(scene_filter["history_frames"]),
                "navsim_future_frames": int(scene_filter["future_frames"]),
                "native_nuplan_split": args.nuplan_split,
                "native_nuplan_camera_hz": 10,
                "native_nuplan_lidar_hz": 20,
                "retained_camera_tolerance_us": int(args.retained_camera_tolerance_us),
                "camera_cadence_tolerance_us": int(args.camera_cadence_tolerance_us),
                "native_camera_nominal_interval_us": lineage.NOMINAL_NATIVE_CAMERA_INTERVAL_US,
                "native_camera_capture_gap_policy": NATIVE_CAMERA_CAPTURE_GAP_POLICY,
                "native_camera_timestamp_policy": "preserve_exact_source_timestamps",
                "navsim_retained_camera_hz": 2,
                "navsim_camera_cadence_tolerance_us": int(args.navsim_camera_cadence_tolerance_us),
                "navsim_retained_history_cadence_policy": "preserve_source_timestamps",
                "native_jpeg_bytes_read": False,
                "native_jpegs_verified": False,
                "native_history_frame_policy": NATIVE_HISTORY_FRAME_POLICY,
                "native_history_nominal_frame_count": NOMINAL_NATIVE_HISTORY_FRAMES,
                "image_relationship_encoding": "anchor_camera native stream index ranges",
                "camera_history_endpoint": (
                    "exact contiguous native camera range between first/current retained NAVSIM frames"
                ),
            },
        )
        for context in iter_anchor_contexts(
            log_path=args.log_path,
            scene_filter=scene_filter,
            requested_tokens=requested_tokens,
            max_anchors=int(args.max_anchors),
            navsim_camera_cadence_tolerance_us=int(args.navsim_camera_cadence_tolerance_us),
            state=state,
        ):
            if context.log_name != active_log_name:
                if active_log_name is not None:
                    flush_native_image_reference_counts(store, native_image_cache)
                if native_connection is not None:
                    native_connection.close()
                db_path = lineage.native_db_path(args.nuplan_root, args.nuplan_split, context.log_name)
                native_connection = lineage.open_native_db(db_path)
                expected_log_token = lineage.as_hex(context.row.get("log_token"))
                lineage.require_native_log(native_connection, context.log_name, expected_log_token, db_path)
                insert_source_log(
                    store,
                    log_name=context.log_name,
                    log_token=expected_log_token,
                    native_split=args.nuplan_split,
                )
                timelines = {
                    channel: lineage.load_native_image_timeline(native_connection, channel) for channel in channels
                }
                for channel, timeline in timelines.items():
                    stream_gaps = lineage.validate_10hz_sequence(
                        timeline.images,
                        channel=channel,
                        anchor_token=f"source-log:{context.log_name}",
                        history_seconds=0.0,
                        cadence_tolerance_us=int(args.camera_cadence_tolerance_us),
                    )
                    if stream_gaps:
                        state.native_stream_capture_gap_logs.add(context.log_name)
                        state.native_stream_capture_gap_streams.add((context.log_name, channel))
                        state.native_stream_capture_gap_count += len(stream_gaps)
                        state.native_stream_capture_gap_delta_us.update(
                            gap.delta_us for gap in stream_gaps
                        )
                timeline_indices = {
                    channel: {image.token: index for index, image in enumerate(timeline.images)}
                    for channel, timeline in timelines.items()
                }
                native_image_cache = NativeImageWriteCache()
                active_log_name = context.log_name
                state.native_databases_verified += 1
            assert native_connection is not None
            state.anchors_seen += 1
            native_lidar = lineage.resolve_native_lidar(native_connection, context.row, context.log_name)
            state.mapping_methods[native_lidar.mapping_method] += 1
            anchor_token = insert_anchor(store, context=context, native_lidar=native_lidar)
            for channel in channels:
                retained_history = [
                    lineage.retained_camera_image(
                        timelines[channel],
                        history_row,
                        channel,
                        int(args.retained_camera_tolerance_us),
                    )[0]
                    for history_row in context.history_rows
                ]
                history_start_retained = retained_history[0]
                history_end_retained = retained_history[-1]
                history_start_timestamp_us = history_start_retained.timestamp_us
                history_end_timestamp_us = history_end_retained.timestamp_us
                if history_end_timestamp_us < history_start_timestamp_us:
                    raise lineage.AuditError(
                        f"Retained {channel} camera history is reversed for NAVSIM anchor {anchor_token}"
                    )
                sequence = timelines[channel].in_window(
                    history_start_timestamp_us,
                    history_end_timestamp_us,
                )
                if any(retained not in sequence for retained in retained_history):
                    raise lineage.AuditError(
                        f"A retained {channel} image is outside the native history window for NAVSIM anchor "
                        f"{anchor_token}"
                    )
                history_capture_gaps = lineage.validate_10hz_sequence(
                    sequence,
                    channel=channel,
                    anchor_token=anchor_token,
                    history_seconds=(history_end_timestamp_us - history_start_timestamp_us) / 1_000_000.0,
                    cadence_tolerance_us=int(args.camera_cadence_tolerance_us),
                )
                if not sequence:
                    raise lineage.AuditError(f"Native {channel} history is empty for NAVSIM anchor {anchor_token}")
                if history_capture_gaps:
                    state.native_history_capture_gap_count += len(history_capture_gaps)
                    state.native_history_capture_gap_delta_us.update(
                        gap.delta_us for gap in history_capture_gaps
                    )
                    state.native_history_capture_gap_anchor_tokens.add(anchor_token)
                    state.native_history_capture_gap_anchor_channels += 1
                # Preserve the exact native contiguous range.  A nominal
                # four-row / 2 Hz history is usually 16 frames, but source
                # timestamp gaps can legitimately produce 15, 17, 21, ...
                # frames.  Padding or dropping those frames would no longer
                # be a provenance-preserving 10 Hz sidecar.
                state.history_frame_counts[f"{channel}:{len(sequence)}"] += 1
                image_ids: dict[str, int] = {}
                for image in sequence:
                    stream_index = timeline_indices[channel].get(image.token)
                    if stream_index is None:
                        raise lineage.AuditError(
                            f"Native {channel} stream index is missing image {image.token} for {anchor_token}"
                        )
                    image_ids[image.token] = ensure_native_image(
                        store,
                        log_name=context.log_name,
                        image=image,
                        stream_index=stream_index,
                        cache=native_image_cache,
                    )
                    state.native_10hz_image_references += 1
                    state.channel_reference_counts[channel] += 1
                first_stream_index = timeline_indices[channel][sequence[0].token]
                last_stream_index = timeline_indices[channel][sequence[-1].token]
                if last_stream_index - first_stream_index + 1 != len(sequence):
                    raise lineage.AuditError(
                        f"Native {channel} sequence is not a contiguous stream range for NAVSIM anchor {anchor_token}"
                    )
                insert_anchor_camera(
                    store,
                    anchor_token=anchor_token,
                    channel=channel,
                    history_start_retained_image_id=image_ids[history_start_retained.token],
                    history_start_retained_timestamp_us=history_start_retained.timestamp_us,
                    history_end_retained_image_id=image_ids[history_end_retained.token],
                    history_end_retained_timestamp_us=history_end_retained.timestamp_us,
                    history_start_timestamp_us=history_start_timestamp_us,
                    history_end_timestamp_us=history_end_timestamp_us,
                    first_stream_index=first_stream_index,
                    last_stream_index=last_stream_index,
                    frame_count=len(sequence),
                )
                state.retained_camera_records_verified += len(retained_history)
            state.anchors_verified += 1
            if state.anchors_seen % 256 == 0:
                store.commit()
        if active_log_name is not None:
            flush_native_image_reference_counts(store, native_image_cache)
        store.commit()
        anchor_count, unique_images, image_references = query_counts(store)
        if anchor_count != state.anchors_verified:
            raise lineage.AuditError(f"Index anchor mismatch: {anchor_count} != {state.anchors_verified}")
        if image_references != state.native_10hz_image_references:
            raise lineage.AuditError(
                f"Index image-reference mismatch: {image_references} != {state.native_10hz_image_references}"
            )
        if args.inventory_jsonl is not None and not args.dry_run:
            _inventory_count, inventory_references, inventory_sha256 = write_compatibility_inventory(
                store,
                args.inventory_jsonl,
                scene_filter_sha256=scene_filter["sha256"],
            )
            if inventory_references != image_references:
                raise lineage.AuditError(
                    f"Compatibility inventory reference mismatch: {inventory_references} != {image_references}"
                )
        index_success = not args.dry_run
    finally:
        if native_connection is not None:
            native_connection.close()
        index_sha256 = finalize_index(
            store,
            temporary_index,
            args.index_sqlite,
            success=index_success,
        )
    return build_report(
        args=args,
        scene_filter=scene_filter,
        channels=channels,
        state=state,
        anchor_count=anchor_count,
        unique_images=unique_images,
        image_references=image_references,
        index_sha256=index_sha256,
        inventory_sha256=inventory_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = plan(args)
    except lineage.AuditError as exc:
        report = {
            "format": REPORT_FORMAT,
            "generated_at_utc": lineage.utc_now(),
            "status": "fail",
            "error": str(exc),
        }
    except Exception as exc:
        report = {
            "format": REPORT_FORMAT,
            "generated_at_utc": lineage.utc_now(),
            "status": "error",
            "error": f"Unexpected {type(exc).__name__}: {exc}",
        }
    if not args.dry_run:
        lineage.atomic_write_json(args.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "partial_pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
