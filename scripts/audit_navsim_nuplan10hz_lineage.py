#!/usr/bin/env python3
"""Fail-closed auditor for staged JPEGs referenced by NAVSIM 10 Hz metadata.

The released OpenScene/NAVSIM files retain only nominal 2 Hz camera
observations. The canonical metadata planner proves their licensed native
nuPlan v1.1 lineage and writes exact contiguous native camera ranges to
SQLite. Clean four-row histories usually contain 16 frames; source timestamp
gaps are preserved rather than padded or discarded. This
tool is the later JPEG-availability stage: it reads those immutable SQLite
references, checks only the referenced image paths, and can emit an optional
compatibility manifest.  It never downloads, copies, deletes, or edits raw
sensor data.

The normal workflow is deliberately two-stage:

1. Build the canonical metadata index after staging native databases.
2. After the referenced JPEGs are staged, run a small pilot with
   ``--max-anchors`` and then the full audit.  The optional manifest is
   published atomically only if every requested canonical reference exists.

The output is a derived-data manifest, not an official NAVSIM replacement.
Standard NAVSIM evaluation remains a 2 Hz input protocol.
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
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_FILTER = (
    REPO_ROOT / "reference/navtrain.yaml"
)
DEFAULT_NAVSIM_ROOT = REPO_ROOT / "dataset/navsim2"
DEFAULT_INDEX = (
    REPO_ROOT
    / "dataset/manifests/navsim_nuplan10hz/"
    "navtrain_camera_10hz_index.sqlite"
)

OFFICIAL_NAVTRAIN_FILTER_SHA256 = "5f3c406a06c961e75d69d33bc127acd5704a3a440a7186bfa2212493ecf9fc01"
OFFICIAL_NAVTRAIN_LOG_COUNT = 1_192
OFFICIAL_NAVTRAIN_TOKEN_COUNT = 103_288
NUPLAN_CAMERA_CHANNELS = (
    "CAM_F0",
    "CAM_L0",
    "CAM_L1",
    "CAM_L2",
    "CAM_R0",
    "CAM_R1",
    "CAM_R2",
    "CAM_B0",
)
NAVSIM_RETAINED_CAMERA_INTERVAL_US = 500_000
HELDOUT_MARKERS = ("navtest", "navhard", "warmup", "private", "test")
MANIFEST_FORMAT = "navsim_nuplan10hz_camera_manifest_v2"
REPORT_FORMAT = "navsim_nuplan10hz_lineage_report_v2"
NOMINAL_NATIVE_HISTORY_FRAMES = 16
NATIVE_HISTORY_FRAME_POLICY = "native_contiguous_camera_range_preserve_source_capture_gaps"
NATIVE_CAMERA_CAPTURE_GAP_POLICY = "preserve_source_capture_gaps"
NOMINAL_NATIVE_CAMERA_INTERVAL_US = 100_000


class AuditError(RuntimeError):
    """A data-contract failure which should produce a nonzero exit code."""


@dataclass(frozen=True)
class NativeLidar:
    token: str
    timestamp_us: int
    scene_token: str
    filename: str
    mapping_method: str


@dataclass(frozen=True)
class NativeImage:
    token: str
    timestamp_us: int
    filename_jpg: str
    channel: str
    camera_token: str


@dataclass(frozen=True)
class NativeImageTimeline:
    """One native camera stream, indexed once for all anchors in a source log."""

    images: tuple[NativeImage, ...]
    timestamps_us: tuple[int, ...]

    def in_window(self, start_timestamp_us: int, end_timestamp_us: int) -> tuple[NativeImage, ...]:
        start = bisect_left(self.timestamps_us, int(start_timestamp_us))
        end = bisect_right(self.timestamps_us, int(end_timestamp_us))
        return self.images[start:end]


@dataclass(frozen=True)
class NativeCameraCaptureGap:
    """A genuine longer-than-nominal interval in one native camera stream.

    The source database provides the two endpoint images and their exact
    timestamps.  This record is descriptive only: callers must retain both
    endpoints and must not fill the interval with fabricated images.
    """

    previous_token: str
    previous_timestamp_us: int
    current_token: str
    current_timestamp_us: int
    delta_us: int

    @property
    def omitted_nominal_frame_count(self) -> int:
        """Conservative nominal-frame estimate for reporting only."""

        return max(0, int(round(self.delta_us / NOMINAL_NATIVE_CAMERA_INTERVAL_US)) - 1)


@dataclass(frozen=True)
class NavsimAnchorContext:
    """Raw NAVSIM rows retained only for the optional OpenScene-file check."""

    log_name: str
    row: Mapping[str, Any]
    row_index: int
    history_rows: tuple[Mapping[str, Any], ...]


@dataclass
class AuditState:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    anchors_seen: int = 0
    anchors_verified: int = 0
    logs_scanned: int = 0
    native_databases_verified: int = 0
    retained_camera_records_verified: int = 0
    native_10hz_images_verified: int = 0
    mapping_methods: Counter[str] = field(default_factory=Counter)
    channel_frame_counts: Counter[str] = field(default_factory=Counter)
    history_frame_counts: Counter[str] = field(default_factory=Counter)
    navsim_history_cadence_exceptions: int = 0
    partial: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--navsim-root",
        type=Path,
        default=DEFAULT_NAVSIM_ROOT,
        help="Root containing NAVSIM navsim_logs and sensor_blobs directories.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="NAVSIM trainval pickle directory; defaults to <navsim-root>/navsim_logs/trainval.",
    )
    parser.add_argument(
        "--sensor-path",
        type=Path,
        default=None,
        help="NAVSIM trainval sensor directory; used only with --require-openscene-files.",
    )
    parser.add_argument(
        "--scene-filter-yaml",
        type=Path,
        default=DEFAULT_FILTER,
        help="Official NAVSIM navtrain SceneFilter YAML.",
    )
    parser.add_argument(
        "--nuplan-root",
        type=Path,
        required=True,
        help=(
            "Licensed nuPlan v1.1 root containing the sensor_blobs paths referenced by "
            "the canonical index."
        ),
    )
    parser.add_argument("--nuplan-split", choices=("trainval",), default="trainval")
    parser.add_argument(
        "--index-sqlite",
        type=Path,
        default=DEFAULT_INDEX,
        help="Canonical navsim_nuplan10hz_index_v1 SQLite metadata index.",
    )
    parser.add_argument(
        "--channels",
        default=",".join(NUPLAN_CAMERA_CHANNELS),
        help="Comma-separated native camera channels to verify (default: all eight).",
    )
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=1.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--retained-camera-tolerance-us",
        type=int,
        default=50_000,
        help="Maximum native image-to-LiDAR timestamp offset allowed for a retained 2 Hz image.",
    )
    parser.add_argument(
        "--camera-cadence-tolerance-us",
        type=int,
        default=50_000,
        help="Maximum deviation from 100000 us allowed between adjacent native camera frames.",
    )
    parser.add_argument(
        "--navsim-camera-cadence-tolerance-us",
        type=int,
        default=50_000,
        help="Deviation from NAVSIM's nominal 500000 us retained-camera cadence recorded as an exception.",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="Audit at most this many anchors for a deterministic pilot; 0 audits all navtrain anchors.",
    )
    parser.add_argument(
        "--anchor-token",
        action="append",
        default=[],
        help="Audit this official navtrain anchor token only; may be repeated.",
    )
    parser.add_argument(
        "--require-openscene-files",
        action="store_true",
        help="Also require every retained 2 Hz OpenScene image file to be staged locally.",
    )
    parser.add_argument(
        "--allow-nonofficial-filter",
        action="store_true",
        help="Permit a fixture/custom SceneFilter. This disables the pinned official navtrain hash/count guard.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional output JSONL of verified native 10 Hz image records. Published atomically after a clean audit.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional atomic JSON report path. Without it, the report is printed to stdout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write a report or manifest; still performs all requested read-only checks.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def as_hex(value: Any) -> str:
    """Normalize an opaque token returned by SQLite or a NAVSIM pickle."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.hex().lower()
    return str(value).lower()


def basename(value: Any) -> str:
    return Path(str(value)).name


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not channels:
        raise AuditError("At least one --channels value is required")
    unknown = sorted(set(channels) - set(NUPLAN_CAMERA_CHANNELS))
    if unknown:
        raise AuditError(f"Unsupported nuPlan camera channel(s): {unknown}")
    if len(channels) != len(set(channels)):
        raise AuditError("--channels contains duplicates")
    return channels


def resolve_paths(args: argparse.Namespace) -> None:
    if args.log_path is None:
        args.log_path = args.navsim_root / "navsim_logs" / "trainval"
    if args.sensor_path is None:
        args.sensor_path = args.navsim_root / "sensor_blobs" / "trainval"
    if abs(float(args.history_seconds) - 1.5) > 1e-9:
        raise AuditError(
            "The canonical navtrain JPEG audit always uses the four retained 2 Hz frames "
            "and their exact native 10 Hz index range; --history-seconds is unsupported."
        )
    if args.max_anchors < 0:
        raise AuditError("--max-anchors must be nonnegative")
    if (
        args.retained_camera_tolerance_us < 0
        or args.camera_cadence_tolerance_us < 0
        or args.navsim_camera_cadence_tolerance_us < 0
    ):
        raise AuditError("Timestamp tolerances must be nonnegative")
    if args.manifest_jsonl is not None and args.manifest_jsonl.suffix != ".jsonl":
        raise AuditError("--manifest-jsonl must have a .jsonl suffix")
    if args.report_json is not None and args.report_json.suffix != ".json":
        raise AuditError("--report-json must have a .json suffix")
    if args.index_sqlite.suffix != ".sqlite":
        raise AuditError("--index-sqlite must have a .sqlite suffix")


def open_canonical_index(index_path: Path) -> Any:
    """Open the metadata contract used by later 10 Hz-aware feature builders."""

    package_root = REPO_ROOT
    package_root_text = str(package_root)
    if package_root_text not in sys.path:
        sys.path.insert(0, package_root_text)
    try:
        from navtrain10hz.loader import NuPlan10HzCameraHistoryIndex, NuPlan10HzIndexError
    except ImportError as exc:
        raise AuditError(
            f"Could not import the navtrain10hz index reader from {package_root}: {exc}"
        ) from exc
    try:
        return NuPlan10HzCameraHistoryIndex(index_path)
    except NuPlan10HzIndexError as exc:
        raise AuditError(f"Could not open canonical 10 Hz metadata index {index_path}: {exc}") from exc


def require_canonical_index_contract(
    index: Any,
    *,
    scene_filter: Mapping[str, Any],
    native_split: str,
) -> None:
    """Reject an index from a different NAVSIM filter or native split."""

    metadata = index.metadata
    if metadata.get("navsim_scene_filter_sha256") != scene_filter["sha256"]:
        raise AuditError(
            "Canonical 10 Hz index SceneFilter hash does not match the requested NAVSIM filter: "
            f"index={metadata.get('navsim_scene_filter_sha256')!r}, "
            f"requested={scene_filter['sha256']!r}"
        )
    if metadata.get("native_nuplan_split") != native_split:
        raise AuditError(
            "Canonical 10 Hz index native split does not match this audit: "
            f"index={metadata.get('native_nuplan_split')!r}, requested={native_split!r}"
        )
    if metadata.get("native_history_frame_policy") != NATIVE_HISTORY_FRAME_POLICY:
        raise AuditError(
            "Canonical 10 Hz index does not preserve exact contiguous native camera ranges: "
            f"policy={metadata.get('native_history_frame_policy')!r}"
        )
    if metadata.get("native_camera_capture_gap_policy") != NATIVE_CAMERA_CAPTURE_GAP_POLICY:
        raise AuditError(
            "Canonical 10 Hz index does not declare source capture-gap preservation: "
            f"policy={metadata.get('native_camera_capture_gap_policy')!r}"
        )


def load_scene_filter(path: Path, *, allow_nonofficial: bool) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"SceneFilter YAML does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AuditError(f"Could not parse SceneFilter YAML {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AuditError(f"SceneFilter YAML is not a mapping: {path}")
    required = {
        "_target_",
        "_convert_",
        "num_history_frames",
        "num_future_frames",
        "frame_interval",
        "has_route",
        "max_scenes",
        "log_names",
        "tokens",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise AuditError(f"SceneFilter YAML is missing keys {missing}: {path}")
    logs = tuple(str(value) for value in payload["log_names"])
    tokens = tuple(as_hex(value) for value in payload["tokens"])
    if not logs or len(logs) != len(set(logs)):
        raise AuditError("SceneFilter log_names are empty or duplicated")
    if not tokens or len(tokens) != len(set(tokens)):
        raise AuditError("SceneFilter tokens are empty or duplicated")
    if payload["_target_"] != "navsim.common.dataclasses.SceneFilter":
        raise AuditError(f"Unexpected SceneFilter target: {payload['_target_']!r}")
    if int(payload["num_history_frames"]) != 4 or int(payload["num_future_frames"]) != 10:
        raise AuditError("This tool requires the official 4-history / 10-future navtrain filter")
    if int(payload["frame_interval"]) != 1 or payload["has_route"] is not True or payload["max_scenes"] is not None:
        raise AuditError("This tool requires the official navtrain sampling semantics")

    filter_sha256 = sha256_file(path)
    if not allow_nonofficial:
        mismatches: list[str] = []
        if filter_sha256 != OFFICIAL_NAVTRAIN_FILTER_SHA256:
            mismatches.append(f"sha256={filter_sha256}")
        if len(logs) != OFFICIAL_NAVTRAIN_LOG_COUNT:
            mismatches.append(f"log_count={len(logs)}")
        if len(tokens) != OFFICIAL_NAVTRAIN_TOKEN_COUNT:
            mismatches.append(f"token_count={len(tokens)}")
        if mismatches:
            raise AuditError(
                "SceneFilter is not the pinned official navtrain contract ("
                + ", ".join(mismatches)
                + "). Pass --allow-nonofficial-filter only for a fixture or intentional custom protocol."
            )
    return {
        "logs": logs,
        "tokens": tokens,
        "sha256": filter_sha256,
        "history_frames": int(payload["num_history_frames"]),
        "future_frames": int(payload["num_future_frames"]),
    }


def reject_heldout_paths(args: argparse.Namespace) -> None:
    # A 10 Hz derivative can only be built from navtrain/trainval. The parser
    # does not offer an override because a held-out test derivative would make
    # the provenance report unsuitable for the intended finetuning protocol.
    values = (str(args.log_path), str(args.sensor_path), str(args.scene_filter_yaml))
    matched = sorted(
        marker
        for marker in HELDOUT_MARKERS
        if any(marker in value.lower() for value in values)
    )
    # The official path has "train_test_split" in it; only reject an explicit
    # test split directory/name, not that configuration-parent spelling.
    matched = [marker for marker in matched if marker != "test" or "/test/" in "/".join(values).lower()]
    if matched:
        raise AuditError(f"Refusing held-out NAVSIM input marker(s): {matched}")


def native_db_path(nuplan_root: Path, split: str, log_name: str) -> Path:
    return nuplan_root / "splits" / split / f"{log_name}.db"


def open_native_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise AuditError(f"Missing native nuPlan database: {path}")
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise AuditError(f"Could not open native nuPlan database {path}: {exc}") from exc
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.Error as exc:
        connection.close()
        raise AuditError(f"Could not place native database in query-only mode {path}: {exc}") from exc
    return connection


def require_native_log(
    connection: sqlite3.Connection,
    log_name: str,
    expected_log_token: str,
    db_path: Path,
) -> None:
    try:
        rows = connection.execute("SELECT logfile, lower(hex(token)) AS token_hex FROM log").fetchall()
    except sqlite3.Error as exc:
        raise AuditError(f"Native database does not expose expected log table: {db_path}: {exc}") from exc
    if len(rows) != 1:
        raise AuditError(f"Expected exactly one log row in {db_path}, found {len(rows)}")
    native_name = str(rows[0]["logfile"])
    if native_name != log_name:
        raise AuditError(
            f"Native database identity mismatch: expected log.logfile={log_name!r}, got {native_name!r} in {db_path}"
        )
    if not expected_log_token:
        raise AuditError(f"NAVSIM metadata lacks log_token for {log_name}")
    native_log_token = as_hex(rows[0]["token_hex"])
    if native_log_token != expected_log_token:
        raise AuditError(
            f"Native database log-token mismatch for {log_name}: "
            f"native={native_log_token}, openscene={expected_log_token}"
        )


def fetch_lidar_by_token(connection: sqlite3.Connection, token: str) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT lower(hex(token)) AS token_hex, timestamp, lower(hex(scene_token)) AS scene_token_hex, filename
        FROM lidar_pc
        WHERE lower(hex(token)) = ?
        """,
        (token.lower(),),
    ).fetchall()


def fetch_lidar_by_timestamp(connection: sqlite3.Connection, timestamp_us: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT lower(hex(token)) AS token_hex, timestamp, lower(hex(scene_token)) AS scene_token_hex, filename
        FROM lidar_pc
        WHERE timestamp = ?
        """,
        (int(timestamp_us),),
    ).fetchall()


def resolve_native_lidar(connection: sqlite3.Connection, row: Mapping[str, Any], log_name: str) -> NativeLidar:
    token = as_hex(row.get("token"))
    timestamp_us = int(row.get("timestamp"))
    scene_token = as_hex(row.get("scene_token"))
    lidar_path = str(row.get("lidar_path") or "")
    if not token or not scene_token or not lidar_path:
        raise AuditError(f"NAVSIM row in {log_name} lacks token, scene_token, or lidar_path")

    candidates = fetch_lidar_by_token(connection, token)
    mapping_method = "native_lidar_token"
    if not candidates:
        expected_name = basename(lidar_path)
        candidates = [
            candidate
            for candidate in fetch_lidar_by_timestamp(connection, timestamp_us)
            if basename(candidate["filename"]) == expected_name
        ]
        mapping_method = "native_lidar_timestamp_and_filename"
    if len(candidates) != 1:
        raise AuditError(
            f"Could not resolve exactly one native LiDAR record for NAVSIM token {token} in {log_name}; "
            f"found {len(candidates)} candidate(s)"
        )
    candidate = candidates[0]
    native = NativeLidar(
        token=as_hex(candidate["token_hex"]),
        timestamp_us=int(candidate["timestamp"]),
        scene_token=as_hex(candidate["scene_token_hex"]),
        filename=str(candidate["filename"]),
        mapping_method=mapping_method,
    )
    mismatches: list[str] = []
    if native.timestamp_us != timestamp_us:
        mismatches.append(f"timestamp native={native.timestamp_us} openscene={timestamp_us}")
    if native.scene_token != scene_token:
        mismatches.append(f"scene_token native={native.scene_token} openscene={scene_token}")
    if basename(native.filename) != basename(lidar_path):
        mismatches.append(f"filename native={native.filename!r} openscene={lidar_path!r}")
    if mapping_method == "native_lidar_token" and native.token != token:
        mismatches.append(f"token native={native.token} openscene={token}")
    if mismatches:
        raise AuditError(f"Native LiDAR provenance mismatch for {log_name}/{token}: " + "; ".join(mismatches))
    return native


def load_native_image_timeline(connection: sqlite3.Connection, channel: str) -> NativeImageTimeline:
    try:
        rows = connection.execute(
            """
            SELECT lower(hex(image.token)) AS token_hex, image.timestamp AS timestamp_us,
                   image.filename_jpg AS filename_jpg, camera.channel AS channel,
                   lower(hex(camera.token)) AS camera_token_hex
            FROM image
            JOIN camera ON image.camera_token = camera.token
            WHERE camera.channel = ?
            ORDER BY image.timestamp ASC, lower(hex(image.token)) ASC
            """,
            (channel,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise AuditError(f"Could not query native image/camera tables for {channel}: {exc}") from exc
    images = tuple(
        NativeImage(
            token=as_hex(item["token_hex"]),
            timestamp_us=int(item["timestamp_us"]),
            filename_jpg=str(item["filename_jpg"]),
            channel=str(item["channel"]),
            camera_token=as_hex(item["camera_token_hex"]),
        )
        for item in rows
    )
    return NativeImageTimeline(
        images=images,
        timestamps_us=tuple(image.timestamp_us for image in images),
    )


def native_blob_path(nuplan_root: Path, log_name: str, image: NativeImage) -> Path:
    log_root = (nuplan_root / "sensor_blobs" / log_name).resolve()
    candidates = (
        log_root / image.filename_jpg,
        log_root / image.channel / basename(image.filename_jpg),
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(log_root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    raise AuditError(
        f"Native camera blob is absent for {log_name}/{image.channel}/{image.filename_jpg}; "
        f"expected below {log_root}"
    )


def verify_openscene_blob(sensor_path: Path, camera_path: str, channel: str) -> None:
    root = sensor_path.resolve()
    candidate = (root / camera_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AuditError(f"OpenScene {channel} path escapes sensor root: {camera_path!r}") from exc
    if not candidate.is_file():
        raise AuditError(f"Retained OpenScene {channel} image is absent: {candidate}")


def retained_camera_image(
    timeline: NativeImageTimeline,
    row: Mapping[str, Any],
    channel: str,
    tolerance_us: int,
) -> tuple[NativeImage, str]:
    cameras = row.get("cams")
    if not isinstance(cameras, Mapping):
        raise AuditError("NAVSIM row lacks a camera mapping")
    camera = cameras.get(channel)
    if not isinstance(camera, Mapping) or not camera.get("data_path"):
        raise AuditError(f"NAVSIM row lacks a retained {channel} data_path")
    expected_path = str(camera["data_path"])
    timestamp_us = int(row["timestamp"])
    candidates = timeline.in_window(timestamp_us - tolerance_us, timestamp_us + tolerance_us)
    matches = [image for image in candidates if basename(image.filename_jpg) == basename(expected_path)]
    if len(matches) != 1:
        raise AuditError(
            f"Could not match exactly one retained {channel} image {expected_path!r} "
            f"near NAVSIM timestamp {timestamp_us}; found {len(matches)}"
        )
    return matches[0], expected_path


def validate_10hz_sequence(
    images: Sequence[NativeImage],
    *,
    channel: str,
    anchor_token: str,
    history_seconds: float,
    cadence_tolerance_us: int,
) -> tuple[NativeCameraCaptureGap, ...]:
    """Validate an exact native range and return its source capture gaps.

    A native nuPlan camera stream is nominally 10 Hz, but an occasional
    longer capture interval is source data rather than permission to drop an
    anchor or synthesize a frame.  The canonical derived protocol preserves
    those intervals exactly.  It remains fail-closed for empty, non-monotonic,
    or implausibly *short* intervals, which would not represent a missing
    nominal capture.

    ``history_seconds`` remains an explicit call-site record of the requested
    source interval.  It is intentionally not used to force a minimum frame
    count: a source capture gap may legitimately reduce that count.
    """

    del history_seconds
    if not images:
        raise AuditError(f"Native {channel} stream is empty for NAVSIM anchor {anchor_token}")
    capture_gaps: list[NativeCameraCaptureGap] = []
    for previous, current in zip(images, images[1:]):
        delta_us = current.timestamp_us - previous.timestamp_us
        if delta_us <= 0:
            raise AuditError(
                f"Native {channel} stream is not strictly increasing for NAVSIM anchor {anchor_token}: "
                f"{delta_us} us between {previous.token} and {current.token}"
            )
        if delta_us < NOMINAL_NATIVE_CAMERA_INTERVAL_US - cadence_tolerance_us:
            raise AuditError(
                f"Native {channel} stream has an implausibly short camera interval for "
                f"NAVSIM anchor {anchor_token}: "
                f"{delta_us} us between {previous.token} and {current.token}"
            )
        if delta_us > NOMINAL_NATIVE_CAMERA_INTERVAL_US + cadence_tolerance_us:
            capture_gaps.append(
                NativeCameraCaptureGap(
                    previous_token=previous.token,
                    previous_timestamp_us=previous.timestamp_us,
                    current_token=current.token,
                    current_timestamp_us=current.timestamp_us,
                    delta_us=delta_us,
                )
            )
    return tuple(capture_gaps)


def iter_navsim_anchors(
    *,
    log_path: Path,
    log_names: Sequence[str],
    requested_tokens: set[str],
    max_anchors: int,
    history_frames: int,
    camera_cadence_tolerance_us: int,
    state: AuditState,
) -> Iterable[NavsimAnchorContext]:
    if not log_path.is_dir():
        raise AuditError(f"NAVSIM log directory does not exist: {log_path}")
    remaining = set(requested_tokens)
    emitted = 0
    for log_name in log_names:
        if max_anchors and emitted >= max_anchors:
            state.partial = True
            break
        log_file = log_path / f"{log_name}.pkl"
        if not log_file.is_file():
            raise AuditError(f"Missing NAVSIM OpenScene metadata pickle: {log_file}")
        try:
            with log_file.open("rb") as handle:
                rows = pickle.load(handle)
        except ModuleNotFoundError as exc:
            if exc.name == "numpy":
                raise AuditError(
                    "NAVSIM metadata pickles contain NumPy arrays, but this interpreter has no numpy. "
                    "Run this auditor with the existing NAVSIM environment, for example "
                    ".venv/bin/python scripts/audit_navsim_nuplan10hz_lineage.py ..."
                ) from exc
            raise AuditError(f"Could not load NAVSIM metadata pickle {log_file}: {exc!r}") from exc
        except Exception as exc:
            raise AuditError(f"Could not load NAVSIM metadata pickle {log_file}: {exc!r}") from exc
        if not isinstance(rows, list):
            raise AuditError(f"NAVSIM metadata pickle is not a row list: {log_file}")
        state.logs_scanned += 1
        for row_index, row in enumerate(rows):
            if max_anchors and emitted >= max_anchors:
                state.partial = True
                break
            if not isinstance(row, Mapping):
                raise AuditError(f"NAVSIM metadata contains a non-mapping row: {log_file}")
            token = as_hex(row.get("token"))
            if token not in remaining:
                continue
            row_log_name = str(row.get("log_name") or "")
            if row_log_name != log_name:
                raise AuditError(
                    f"NAVSIM token {token} is stored under {log_name!r} but row.log_name is {row_log_name!r}"
                )
            if row_index < history_frames - 1:
                raise AuditError(
                    f"NAVSIM anchor {token} at row {row_index} lacks its {history_frames}-frame camera history"
                )
            history_rows = tuple(rows[row_index - history_frames + 1 : row_index + 1])
            if len(history_rows) != history_frames or any(
                not isinstance(history_row, Mapping) for history_row in history_rows
            ):
                raise AuditError(f"NAVSIM anchor {token} has an invalid retained camera history")
            try:
                history_timestamps = tuple(int(history_row["timestamp"]) for history_row in history_rows)
            except (KeyError, TypeError, ValueError) as exc:
                raise AuditError(f"NAVSIM anchor {token} has an invalid retained camera timestamp") from exc
            for previous_timestamp, current_timestamp in zip(history_timestamps, history_timestamps[1:]):
                delta_us = current_timestamp - previous_timestamp
                if delta_us <= 0:
                    raise AuditError(
                        f"NAVSIM retained camera history for {token} is not strictly increasing: "
                        f"{delta_us} us between retained rows"
                    )
                # Preserve irregular source timestamps.  The official filter
                # selects four rows, but released logs can contain dropped
                # retained observations; this audit must not invent or remove
                # native camera frames to force a nominal 2 Hz duration.
                if abs(delta_us - NAVSIM_RETAINED_CAMERA_INTERVAL_US) > camera_cadence_tolerance_us:
                    state.navsim_history_cadence_exceptions += 1
            remaining.remove(token)
            emitted += 1
            yield NavsimAnchorContext(
                log_name=log_name,
                row=row,
                row_index=row_index,
                history_rows=history_rows,
            )
        if not remaining:
            break
    if remaining and not state.partial:
        preview = ", ".join(sorted(remaining)[:8])
        raise AuditError(
            f"{len(remaining)} requested navtrain token(s) were not found in the allowed NAVSIM log pickles; "
            f"first: {preview}"
        )


def open_manifest_writer(path: Path | None, dry_run: bool) -> tuple[Path | None, TextIO | None]:
    if path is None or dry_run:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return Path(temporary_name), os.fdopen(fd, "w", encoding="utf-8")


def publish_manifest(temporary_path: Path | None, handle: TextIO | None, destination: Path | None, *, success: bool) -> None:
    if handle is not None:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    if temporary_path is None:
        return
    if success and destination is not None:
        os.replace(temporary_path, destination)
    else:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def openscene_camera_path(row: Mapping[str, Any], channel: str) -> str:
    cameras = row.get("cams")
    if not isinstance(cameras, Mapping):
        raise AuditError("NAVSIM row lacks a camera mapping")
    camera = cameras.get(channel)
    if not isinstance(camera, Mapping) or not camera.get("data_path"):
        raise AuditError(f"NAVSIM row lacks a retained {channel} data_path")
    return str(camera["data_path"])


def verify_openscene_history(
    context: NavsimAnchorContext,
    *,
    channels: Sequence[str],
    sensor_path: Path,
) -> int:
    """Check all four retained 2 Hz OpenScene camera paths when requested."""

    count = 0
    for history_row in context.history_rows:
        for channel in channels:
            verify_openscene_blob(sensor_path, openscene_camera_path(history_row, channel), channel)
            count += 1
    return count


def canonical_manifest_entry(
    *,
    filter_sha256: str,
    anchor: Any,
    image: Any,
) -> dict[str, Any]:
    """Serialize one exact canonical image reference without opening JPEG bytes."""

    return {
        "format": MANIFEST_FORMAT,
        "navsim_scene_filter_sha256": filter_sha256,
        "navsim_log_name": anchor.navsim_log_name,
        "navsim_log_token": anchor.navsim_log_token,
        "navsim_anchor_token": anchor.navsim_anchor_token,
        "navsim_anchor_timestamp_us": anchor.navsim_anchor_timestamp_us,
        "navsim_scene_token": anchor.navsim_scene_token,
        "native_lidar_token": anchor.native_lidar_token,
        "native_lidar_mapping_method": anchor.native_lidar_mapping_method,
        "camera_channel": image.camera_channel,
        "native_camera_token": image.native_camera_token,
        "native_image_token": image.native_image_token,
        "native_image_timestamp_us": image.native_image_timestamp_us,
        "native_image_offset_us": image.native_image_timestamp_us - anchor.navsim_anchor_timestamp_us,
        "native_image_filename_jpg": image.native_image_filename_jpg,
        "native_blob_relative_path": str(image.native_blob_relative_path),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    resolve_paths(args)
    reject_heldout_paths(args)
    channels = parse_channels(args.channels)
    scene_filter = load_scene_filter(
        args.scene_filter_yaml,
        allow_nonofficial=bool(args.allow_nonofficial_filter),
    )
    official_tokens = set(scene_filter["tokens"])
    requested_tokens = set(as_hex(token) for token in args.anchor_token) if args.anchor_token else official_tokens
    unknown = sorted(requested_tokens - official_tokens)
    if unknown:
        raise AuditError(f"Requested --anchor-token values are not in this navtrain SceneFilter: {unknown[:8]}")
    if args.anchor_token:
        # Explicit token pilots are intentionally partial even if all listed
        # tokens pass: they do not validate the complete official filter.
        partial_requested = len(requested_tokens) != len(official_tokens)
    else:
        partial_requested = False

    state = AuditState(
        partial=partial_requested
        or (bool(args.max_anchors) and int(args.max_anchors) < len(requested_tokens))
    )
    index = open_canonical_index(args.index_sqlite)
    temporary_manifest: Path | None = None
    manifest_handle: TextIO | None = None
    manifest_records = 0
    source_logs: set[str] = set()
    database_paths: set[str] = set()
    success = False
    try:
        require_canonical_index_contract(
            index,
            scene_filter=scene_filter,
            native_split=args.nuplan_split,
        )
        index_tokens = set(index.tokens())
        unexpected_index_tokens = sorted(index_tokens - official_tokens)
        if unexpected_index_tokens:
            raise AuditError(
                "Canonical 10 Hz index contains token(s) outside the requested NAVSIM SceneFilter: "
                + ", ".join(unexpected_index_tokens[:8])
            )
        missing_requested_tokens = sorted(requested_tokens - index_tokens)
        if missing_requested_tokens:
            raise AuditError(
                "Canonical 10 Hz index lacks requested NAVSIM token(s): "
                + ", ".join(missing_requested_tokens[:8])
            )
        if not args.anchor_token and not args.max_anchors and index_tokens != official_tokens:
            missing = sorted(official_tokens - index_tokens)
            raise AuditError(
                "A full navtrain JPEG audit requires a complete canonical 10 Hz index; "
                f"{len(missing)} NAVSIM token(s) are missing"
            )

        ordered_tokens = [token for token in scene_filter["tokens"] if token in requested_tokens]
        if args.max_anchors:
            ordered_tokens = ordered_tokens[: int(args.max_anchors)]
        if not ordered_tokens:
            raise AuditError("No NAVSIM anchors were selected for the canonical JPEG audit")

        raw_contexts: dict[str, NavsimAnchorContext] = {}
        if args.require_openscene_files:
            raw_contexts = {
                as_hex(context.row.get("token")): context
                for context in iter_navsim_anchors(
                    log_path=args.log_path,
                    log_names=scene_filter["logs"],
                    requested_tokens=set(ordered_tokens),
                    max_anchors=0,
                    history_frames=int(scene_filter["history_frames"]),
                    camera_cadence_tolerance_us=int(args.navsim_camera_cadence_tolerance_us),
                    state=state,
                )
            }
            missing_contexts = sorted(set(ordered_tokens) - set(raw_contexts))
            if missing_contexts:
                raise AuditError(
                    "NAVSIM source metadata lacks requested canonical token(s): "
                    + ", ".join(missing_contexts[:8])
                )

        temporary_manifest, manifest_handle = open_manifest_writer(args.manifest_jsonl, bool(args.dry_run))
        for token in ordered_tokens:
            try:
                anchor = index.get_anchor(token, required_channels=channels)
            except RuntimeError as exc:
                raise AuditError(f"Canonical 10 Hz index could not load NAVSIM anchor {token}: {exc}") from exc
            state.anchors_seen += 1
            source_logs.add(anchor.navsim_log_name)
            database_paths.add(str(anchor.native_db_relative_path))
            expected_database_prefix = ("splits", args.nuplan_split)
            if tuple(anchor.native_db_relative_path.parts[:2]) != expected_database_prefix:
                raise AuditError(
                    f"Canonical 10 Hz index has unexpected native database path for {token}: "
                    f"{anchor.native_db_relative_path}"
                )
            state.mapping_methods[anchor.native_lidar_mapping_method] += 1

            if args.require_openscene_files:
                context = raw_contexts[token]
                raw_mismatches: list[str] = []
                if context.log_name != anchor.navsim_log_name:
                    raw_mismatches.append(
                        f"log_name source={context.log_name!r} index={anchor.navsim_log_name!r}"
                    )
                if as_hex(context.row.get("log_token")) != anchor.navsim_log_token:
                    raw_mismatches.append("log_token")
                if as_hex(context.row.get("scene_token")) != anchor.navsim_scene_token:
                    raw_mismatches.append("scene_token")
                if int(context.row.get("timestamp", -1)) != anchor.navsim_anchor_timestamp_us:
                    raw_mismatches.append("anchor_timestamp")
                if context.row_index != anchor.navsim_row_index:
                    raw_mismatches.append("row_index")
                if raw_mismatches:
                    raise AuditError(
                        f"Canonical 10 Hz index provenance mismatches current NAVSIM source for {token}: "
                        + ", ".join(raw_mismatches)
                    )
                state.retained_camera_records_verified += verify_openscene_history(
                    context,
                    channels=channels,
                    sensor_path=args.sensor_path,
                )
            else:
                # The canonical planner verified all four retained frames per channel before publishing the index.
                state.retained_camera_records_verified += int(scene_filter["history_frames"]) * len(channels)

            for channel in channels:
                history = anchor.camera_histories[channel]
                if history.frame_count < 1:
                    raise AuditError(
                        f"Canonical 10 Hz index has an empty native history for {token}/{channel}"
                    )
                state.history_frame_counts[f"{channel}:{history.frame_count}"] += 1
                missing_paths: list[str] = []
                for image in history.references:
                    if not image.blob_path(args.nuplan_root).is_file():
                        missing_paths.append(str(image.native_blob_relative_path))
                if missing_paths:
                    raise AuditError(
                        f"Native JPEGs are unavailable for NAVSIM anchor {token}/{channel}: "
                        f"{len(missing_paths)} missing; first: {', '.join(missing_paths[:3])}"
                    )
                for image in history.references:
                    state.native_10hz_images_verified += 1
                    state.channel_frame_counts[channel] += 1
                    if manifest_handle is not None:
                        json.dump(
                            canonical_manifest_entry(
                                filter_sha256=scene_filter["sha256"],
                                anchor=anchor,
                                image=image,
                            ),
                            manifest_handle,
                            sort_keys=True,
                        )
                        manifest_handle.write("\n")
                        manifest_records += 1
            state.anchors_verified += 1
        success = True
    finally:
        index.close()
        publish_manifest(
            temporary_manifest,
            manifest_handle,
            args.manifest_jsonl,
            success=success,
        )

    return {
        "format": REPORT_FORMAT,
        "generated_at_utc": utc_now(),
        "status": "partial_pass" if state.partial else "pass",
        "partial": state.partial,
        "dry_run": bool(args.dry_run),
        "navsim": {
            "log_path": str(args.log_path),
            "sensor_path": str(args.sensor_path),
            "scene_filter_yaml": str(args.scene_filter_yaml),
            "scene_filter_sha256": scene_filter["sha256"],
            "selected_anchor_count": len(ordered_tokens),
            "retained_history_cadence_exceptions": state.navsim_history_cadence_exceptions,
        },
        "native_nuplan": {
            "root": str(args.nuplan_root),
            "split": args.nuplan_split,
            "camera_hz": 10,
            "lidar_hz": 20,
            "channels": list(channels),
            "nominal_camera_history_frames": NOMINAL_NATIVE_HISTORY_FRAMES,
            "native_history_frame_policy": NATIVE_HISTORY_FRAME_POLICY,
        },
        "checks": {
            "anchors_seen": state.anchors_seen,
            "anchors_verified": state.anchors_verified,
            "logs_scanned": state.logs_scanned,
            "native_databases_referenced": len(database_paths),
            "source_logs_referenced": len(source_logs),
            "retained_camera_records_verified": state.retained_camera_records_verified,
            "native_10hz_images_verified": state.native_10hz_images_verified,
            "mapping_methods": dict(sorted(state.mapping_methods.items())),
            "native_10hz_images_by_channel": dict(sorted(state.channel_frame_counts.items())),
            "native_history_frame_counts_by_channel": {
                channel: {
                    int(key.rsplit(":", 1)[1]): int(total)
                    for key, total in state.history_frame_counts.items()
                    if key.startswith(f"{channel}:")
                }
                for channel in channels
            },
            "manifest_records": manifest_records,
            "require_openscene_files": bool(args.require_openscene_files),
        },
        "contract": {
            "does_not_modify_raw_data": True,
            "manifest_is_derived_protocol": True,
            "canonical_index_sqlite": str(args.index_sqlite),
            "canonical_index_is_camera_bounded": True,
            "nominal_camera_history_frames": NOMINAL_NATIVE_HISTORY_FRAMES,
            "native_history_frame_policy": NATIVE_HISTORY_FRAME_POLICY,
            "navsim_camera_cadence_tolerance_us": int(args.navsim_camera_cadence_tolerance_us),
            "native_jpeg_bytes_read": False,
            "official_navsim_input_hz": 2,
            "native_nuplan_camera_hz": 10,
            "native_nuplan_lidar_hz": 20,
        },
        "errors": state.errors,
        "warnings": state.warnings,
    }


def main() -> int:
    args = parse_args()
    try:
        report = audit(args)
    except AuditError as exc:
        report = {
            "format": REPORT_FORMAT,
            "generated_at_utc": utc_now(),
            "status": "fail",
            "error": str(exc),
            "contract": {"does_not_modify_raw_data": True},
        }
    except Exception as exc:  # Keep unexpected exceptions auditable without hiding them.
        report = {
            "format": REPORT_FORMAT,
            "generated_at_utc": utc_now(),
            "status": "error",
            "error": f"Unexpected {type(exc).__name__}: {exc}",
            "contract": {"does_not_modify_raw_data": True},
        }

    if args.report_json is not None and not args.dry_run:
        atomic_write_json(args.report_json, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"pass", "partial_pass"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
