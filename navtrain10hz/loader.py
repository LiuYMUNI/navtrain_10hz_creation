"""Source-faithful nuPlan camera histories for existing NAVSIM 2 Hz anchors.

The official NAVSIM/OpenScene logs stay untouched and keep their 2 Hz
prediction anchors and labels.  This module reads the repository's canonical
SQLite index, which maps those anchors to native nuPlan camera image
references.  ``NuPlan10HzCameraHistoryIndex`` is metadata-only, while
``Navsim10HzImageLoader`` decodes only the exact staged JPEGs named by that
immutable index.

The index is a derived-data protocol, not an official NAVSIM input format and
not a 10 Hz prediction-anchor dataset.
"""

from __future__ import annotations

import sqlite3
import hashlib
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image


INDEX_FORMAT = "navsim_nuplan10hz_index_v1"
NATIVE_CAMERA_HZ = 10
NAVSIM_LABEL_HZ = 2
# Sixteen is the nominal count for a clean four-row / 1.5 s NAVSIM history.
# Real OpenScene rows can contain timestamp gaps, so the canonical index keeps
# the exact contiguous native range and may expose a different frame count.
NOMINAL_NATIVE_CAMERA_HISTORY_FRAMES = 16
NAVTRAIN_NATIVE_CAMERA_HISTORY_FRAMES = NOMINAL_NATIVE_CAMERA_HISTORY_FRAMES
NATIVE_HISTORY_FRAME_POLICY = "native_contiguous_camera_range_preserve_source_capture_gaps"
NATIVE_CAMERA_CAPTURE_GAP_POLICY = "preserve_source_capture_gaps"
NOMINAL_NATIVE_CAMERA_INTERVAL_US = 100_000


class NuPlan10HzIndexError(RuntimeError):
    """Raised when a derived-camera index violates its on-disk contract."""


class NuPlan10HzImagesUnavailableError(NuPlan10HzIndexError):
    """Raised only when a caller explicitly requests unavailable image files."""


STORAGE_INDEX_FORMAT = "navsim_nuplan10hz_image_storage_v1"


@dataclass(frozen=True)
class NativeImageStorage:
    """Durable location and integrity metadata for one indexed JPEG."""

    destination_relative_path: PurePosixPath
    storage_kind: str
    storage_relative_path: PurePosixPath
    pack_offset: Optional[int]
    payload_size: int
    sha256: str


class NativeImageStorageIndex:
    """Read-only destination-to-byte-location map for the packed dataset."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._connection: Optional[sqlite3.Connection] = None
        if not self.path.is_file():
            raise NuPlan10HzIndexError(f"Image storage index does not exist: {self.path}")
        try:
            self._connection = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro", uri=True
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA query_only = ON")
            tables = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {"metadata", "image_storage"}.issubset(tables):
                raise NuPlan10HzIndexError(
                    "Image storage index is missing metadata/image_storage tables"
                )
            metadata = dict(
                self._connection.execute("SELECT key, value FROM metadata").fetchall()
            )
            if metadata.get("format") != STORAGE_INDEX_FORMAT:
                raise NuPlan10HzIndexError(
                    f"Unsupported image storage format {metadata.get('format')!r}"
                )
            self._metadata = {str(k): str(v) for k, v in metadata.items()}
        except BaseException:
            if self._connection is not None:
                self._connection.close()
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()

    def __enter__(self) -> "NativeImageStorageIndex":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    @property
    def metadata(self) -> Mapping[str, str]:
        return dict(self._metadata)

    def get(self, destination_relative_path: PurePosixPath) -> NativeImageStorage:
        row = self._connection.execute(
            "SELECT destination_relative_path, storage_kind, storage_relative_path, "
            "pack_offset, payload_size, sha256 FROM image_storage "
            "WHERE destination_relative_path = ?",
            (str(destination_relative_path),),
        ).fetchone()
        if row is None:
            raise NuPlan10HzImagesUnavailableError(
                f"No certified storage row for {destination_relative_path}"
            )
        kind = str(row["storage_kind"])
        if kind not in {"existing", "pack", "supplemental_pack"}:
            raise NuPlan10HzIndexError(f"Unsupported storage kind {kind!r}")
        relative = PurePosixPath(str(row["storage_relative_path"]))
        digest = str(row["sha256"])
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise NuPlan10HzIndexError(
                f"Unsafe storage row for {destination_relative_path}"
            )
        offset = None if row["pack_offset"] is None else int(row["pack_offset"])
        if kind == "existing" and offset is not None:
            raise NuPlan10HzIndexError("Existing storage row unexpectedly has pack_offset")
        if kind != "existing" and (offset is None or offset < 0):
            raise NuPlan10HzIndexError("Packed storage row lacks a valid pack_offset")
        return NativeImageStorage(
            destination_relative_path=PurePosixPath(str(row["destination_relative_path"])),
            storage_kind=kind,
            storage_relative_path=relative,
            pack_offset=offset,
            payload_size=int(row["payload_size"]),
            sha256=digest,
        )


@dataclass(frozen=True)
class NativeCameraReference:
    """One native nuPlan camera image referenced by a derived NAVSIM anchor."""

    image_id: int
    camera_channel: str
    stream_index: int
    native_image_token: str
    native_image_timestamp_us: int
    native_image_filename_jpg: str
    native_camera_token: str
    native_blob_relative_path: PurePosixPath

    def blob_path(self, nuplan_root: Path) -> Path:
        """Resolve the planned path below a native nuPlan root without reading it."""

        root = Path(nuplan_root).resolve()
        candidate = (root / self.native_blob_relative_path).resolve()
        try:
            candidate.relative_to(root / "sensor_blobs")
        except ValueError as exc:
            raise NuPlan10HzIndexError(
                f"Derived camera path escapes native sensor_blobs: {self.native_blob_relative_path!s}"
            ) from exc
        return candidate


@dataclass(frozen=True)
class NativeCameraHistory:
    """Native 10 Hz image references for one camera at one 2 Hz NAVSIM anchor."""

    camera_channel: str
    history_start_retained_image_id: int
    history_start_retained_native_image_timestamp_us: int
    history_end_retained_image_id: int
    history_end_retained_native_image_timestamp_us: int
    history_start_timestamp_us: int
    history_end_timestamp_us: int
    camera_cadence_tolerance_us: int
    references: tuple[NativeCameraReference, ...]

    @property
    def frame_count(self) -> int:
        """Number of native image references in this history window."""

        return len(self.references)

    @property
    def native_image_timestamp_us(self) -> tuple[int, ...]:
        """Exact native camera timestamps in chronological stream order."""

        return tuple(reference.native_image_timestamp_us for reference in self.references)

    @property
    def timestamp_deltas_us(self) -> tuple[int, ...]:
        """Adjacent exact source timestamp intervals; never resampled."""

        timestamps = self.native_image_timestamp_us
        return tuple(current - previous for previous, current in zip(timestamps, timestamps[1:]))

    @property
    def source_capture_gap_deltas_us(self) -> tuple[int, ...]:
        """Longer-than-nominal source intervals retained in this history."""

        return tuple(
            delta_us
            for delta_us in self.timestamp_deltas_us
            if delta_us > NOMINAL_NATIVE_CAMERA_INTERVAL_US + self.camera_cadence_tolerance_us
        )

    @property
    def has_source_capture_gaps(self) -> bool:
        """Whether this exact range crosses at least one native capture gap."""

        return bool(self.source_capture_gap_deltas_us)

    @property
    def history_start_retained_reference(self) -> NativeCameraReference:
        """Return the native image proven to match NAVSIM's first history frame."""

        for reference in self.references:
            if reference.image_id == self.history_start_retained_image_id:
                return reference
        raise NuPlan10HzIndexError(
            f"{self.camera_channel} history does not include its first retained image id "
            f"{self.history_start_retained_image_id}"
        )

    @property
    def history_end_retained_reference(self) -> NativeCameraReference:
        """Return the native image proven to match NAVSIM's current history frame."""

        for reference in self.references:
            if reference.image_id == self.history_end_retained_image_id:
                return reference
        raise NuPlan10HzIndexError(
            f"{self.camera_channel} history does not include its current retained image id "
            f"{self.history_end_retained_image_id}"
        )

    @property
    def retained_image_id(self) -> int:
        """Compatibility alias for the current (last) retained NAVSIM frame."""

        return self.history_end_retained_image_id

    @property
    def retained_native_image_timestamp_us(self) -> int:
        """Compatibility alias for the current (last) retained NAVSIM frame."""

        return self.history_end_retained_native_image_timestamp_us

    @property
    def retained_reference(self) -> NativeCameraReference:
        """Compatibility alias for the current (last) retained NAVSIM frame."""

        return self.history_end_retained_reference


@dataclass(frozen=True)
class LoadedNativeCameraHistory:
    """Decoded exact JPEG sequence for one camera, with no temporal padding."""

    history: NativeCameraHistory
    images: tuple[npt.NDArray[np.uint8], ...]
    native_image_timestamp_us: tuple[int, ...]
    valid_mask: tuple[bool, ...]

    @property
    def camera_channel(self) -> str:
        """Camera channel of the decoded source sequence."""

        return self.history.camera_channel

    @property
    def frame_count(self) -> int:
        """Actual unpadded source-frame count."""

        return len(self.images)

    def padded(self, frame_count: Optional[int] = None) -> "PaddedNativeCameraHistory":
        """Return explicit zero padding plus a false validity mask for batching.

        The padded slots have timestamp ``-1`` and ``False`` validity.  They
        are not images from the source and must be ignored by consumers.
        """

        target_frame_count = self.frame_count if frame_count is None else int(frame_count)
        if target_frame_count < self.frame_count:
            raise NuPlan10HzIndexError(
                f"Cannot pad {self.camera_channel} from {self.frame_count} down to {target_frame_count} frames"
            )
        if not self.images:
            raise NuPlan10HzIndexError(f"Decoded {self.camera_channel} history is empty")
        image_shape = self.images[0].shape
        image_dtype = self.images[0].dtype
        if any(image.shape != image_shape or image.dtype != image_dtype for image in self.images):
            raise NuPlan10HzIndexError(
                f"Decoded {self.camera_channel} source images have inconsistent array shapes or dtypes"
            )
        images = np.zeros((target_frame_count, *image_shape), dtype=image_dtype)
        images[: self.frame_count] = np.stack(self.images, axis=0)
        timestamps = np.full(target_frame_count, -1, dtype=np.int64)
        timestamps[: self.frame_count] = np.asarray(self.native_image_timestamp_us, dtype=np.int64)
        valid_mask = np.zeros(target_frame_count, dtype=np.bool_)
        valid_mask[: self.frame_count] = True
        return PaddedNativeCameraHistory(
            camera_channel=self.camera_channel,
            images=images,
            native_image_timestamp_us=timestamps,
            valid_mask=valid_mask,
        )


@dataclass(frozen=True)
class PaddedNativeCameraHistory:
    """Optional batch-ready view with explicit invalid padded positions."""

    camera_channel: str
    images: npt.NDArray[np.uint8]
    native_image_timestamp_us: npt.NDArray[np.int64]
    valid_mask: npt.NDArray[np.bool_]


@dataclass(frozen=True)
class DerivedNavsimAnchor:
    """A 2 Hz NAVSIM label anchor paired with native 10 Hz camera histories."""

    navsim_anchor_token: str
    navsim_log_name: str
    navsim_log_token: str
    navsim_scene_token: str
    navsim_anchor_timestamp_us: int
    navsim_row_index: int
    navsim_history_start_timestamp_us: int
    native_lidar_token: str
    native_lidar_timestamp_us: int
    native_lidar_scene_token: str
    native_lidar_filename: str
    native_lidar_mapping_method: str
    native_db_relative_path: PurePosixPath
    camera_histories: Mapping[str, NativeCameraHistory]

    @property
    def camera_hz(self) -> int:
        """The cadence of native image references, not the label cadence."""

        return NATIVE_CAMERA_HZ

    @property
    def label_hz(self) -> int:
        """The retained NAVSIM prediction/label cadence."""

        return NAVSIM_LABEL_HZ


@dataclass(frozen=True)
class LoadedNavsim10HzAnchor:
    """An existing NAVSIM label anchor plus decoded native camera histories."""

    anchor: DerivedNavsimAnchor
    camera_histories: Mapping[str, LoadedNativeCameraHistory]

    def padded_camera_histories(
        self,
        frame_count: Optional[int] = None,
    ) -> Mapping[str, PaddedNativeCameraHistory]:
        """Pad each camera only for batching, preserving an explicit mask."""

        target_frame_count = (
            max(history.frame_count for history in self.camera_histories.values())
            if frame_count is None
            else int(frame_count)
        )
        return {
            channel: history.padded(target_frame_count)
            for channel, history in self.camera_histories.items()
        }


@dataclass(frozen=True)
class NativeJpegAvailability:
    """Non-failing filesystem availability report for one derived anchor."""

    checked: bool
    total_references: int
    present_references: int
    missing_relative_paths: tuple[PurePosixPath, ...]

    @property
    def complete(self) -> bool:
        """Whether every reference exists as a regular file at the checked root."""

        return self.checked and not self.missing_relative_paths


class NuPlan10HzCameraHistoryIndex:
    """Read-only loader for ``navsim_nuplan10hz_index_v1`` SQLite indexes.

    The index stores image streams once per native source log and describes
    each anchor/channel history as an inclusive stream-index range.  Loading
    an anchor therefore reads metadata only and does not depend on JPEGs being
    available locally.
    """

    _REQUIRED_COLUMNS = {
        "metadata": {"key", "value"},
        "source_log": {"navsim_log_name", "navsim_log_token", "native_db_relative_path"},
        "anchor": {
            "navsim_anchor_token",
            "navsim_log_name",
            "navsim_log_token",
            "navsim_scene_token",
            "navsim_anchor_timestamp_us",
            "navsim_row_index",
            "native_lidar_token",
            "native_lidar_timestamp_us",
            "native_lidar_scene_token",
            "native_lidar_filename",
            "native_lidar_mapping_method",
            "navsim_history_start_timestamp_us",
        },
        "native_image": {
            "image_id",
            "navsim_log_name",
            "camera_channel",
            "stream_index",
            "native_image_token",
            "native_image_timestamp_us",
            "native_image_filename_jpg",
            "native_camera_token",
            "native_blob_relative_path",
        },
        "anchor_camera": {
            "navsim_anchor_token",
            "camera_channel",
            "history_start_retained_image_id",
            "history_start_retained_native_image_timestamp_us",
            "history_end_retained_image_id",
            "history_end_retained_native_image_timestamp_us",
            "history_start_timestamp_us",
            "history_end_timestamp_us",
            "first_stream_index",
            "last_stream_index",
            "frame_count",
        },
    }

    def __init__(
        self,
        index_path: Path,
        *,
        expected_history_frames: Optional[int] = None,
    ):
        self._index_path = Path(index_path)
        if expected_history_frames is not None and expected_history_frames < 1:
            raise NuPlan10HzIndexError("expected_history_frames must be positive or None")
        self._expected_history_frames = expected_history_frames
        if not self._index_path.is_file():
            raise NuPlan10HzIndexError(f"Derived 10 Hz index does not exist: {self._index_path}")
        try:
            self._connection = sqlite3.connect(f"{self._index_path.resolve().as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise NuPlan10HzIndexError(f"Could not open derived 10 Hz index {self._index_path}: {exc}") from exc
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA query_only = ON")
            self._metadata = self._load_and_validate_schema()
        except BaseException:
            self._connection.close()
            raise

    def __enter__(self) -> "NuPlan10HzCameraHistoryIndex":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the read-only SQLite connection."""

        self._connection.close()

    @property
    def index_path(self) -> Path:
        """Path to the immutable metadata index."""

        return self._index_path

    @property
    def metadata(self) -> Mapping[str, str]:
        """Immutable-on-convention metadata emitted by the index builder."""

        return dict(self._metadata)

    @property
    def native_jpegs_verified(self) -> bool:
        """Whether the builder declared all referenced JPEGs verified.

        Metadata-only indexes normally return ``False``.  This is descriptive;
        callers that need a current filesystem view should use
        :meth:`jpeg_availability`.
        """

        value = self._metadata.get("native_jpegs_verified", "false").strip().lower()
        return value in {"1", "true", "yes"}

    @property
    def expected_history_frames(self) -> Optional[int]:
        """Optional fixed native frame count enforced by this reader.

        The canonical index preserves the exact contiguous native range for
        each anchor.  ``None`` (the default) therefore accepts variable counts
        caused by timestamp gaps in the retained NAVSIM rows.  A caller may
        pass ``16`` when auditing a separately defined fixed-length protocol.
        """

        return self._expected_history_frames

    def tokens(self) -> tuple[str, ...]:
        """Return all 2 Hz NAVSIM anchor tokens represented by the index."""

        rows = self._connection.execute("SELECT navsim_anchor_token FROM anchor ORDER BY navsim_anchor_token").fetchall()
        return tuple(str(row["navsim_anchor_token"]) for row in rows)

    def has_anchor(self, token: str) -> bool:
        """Return whether an existing NAVSIM anchor has derived camera metadata."""

        normalized = self._normalize_token(token)
        row = self._connection.execute(
            "SELECT 1 FROM anchor WHERE navsim_anchor_token = ?", (normalized,)
        ).fetchone()
        return row is not None

    def get_anchor(
        self,
        token: str,
        *,
        required_channels: Optional[Sequence[str]] = None,
        nuplan_root: Optional[Path] = None,
        require_images: bool = False,
    ) -> DerivedNavsimAnchor:
        """Load native camera references for one existing 2 Hz NAVSIM anchor.

        This method only reads index rows unless ``require_images`` is set.
        In metadata-first mode it succeeds before sensor blobs are staged,
        provided the metadata index is complete and internally consistent.
        ``require_images=True`` performs path existence checks only; callers
        should use it immediately before a feature builder opens pixels.
        """

        normalized = self._normalize_token(token)
        anchor = self._connection.execute(
            """
            SELECT anchor.navsim_anchor_token, anchor.navsim_log_name, anchor.navsim_log_token,
                   anchor.navsim_scene_token, anchor.navsim_anchor_timestamp_us, anchor.navsim_row_index,
                   anchor.navsim_history_start_timestamp_us, anchor.native_lidar_token,
                   anchor.native_lidar_timestamp_us, anchor.native_lidar_scene_token,
                   anchor.native_lidar_filename, anchor.native_lidar_mapping_method,
                   source_log.native_db_relative_path
            FROM anchor
            JOIN source_log ON source_log.navsim_log_name = anchor.navsim_log_name
            WHERE anchor.navsim_anchor_token = ?
            """,
            (normalized,),
        ).fetchone()
        if anchor is None:
            raise NuPlan10HzIndexError(f"No derived 10 Hz metadata for NAVSIM anchor token {token!r}")

        log_name = str(anchor["navsim_log_name"])
        channel_rows = self._connection.execute(
            """
            SELECT camera_channel,
                   history_start_retained_image_id,
                   history_start_retained_native_image_timestamp_us,
                   history_end_retained_image_id,
                   history_end_retained_native_image_timestamp_us,
                   history_start_timestamp_us, history_end_timestamp_us,
                   first_stream_index, last_stream_index, frame_count
            FROM anchor_camera
            WHERE navsim_anchor_token = ?
            ORDER BY camera_channel
            """,
            (normalized,),
        ).fetchall()
        if not channel_rows:
            raise NuPlan10HzIndexError(f"Anchor {anchor['navsim_anchor_token']} has no camera histories")

        histories: dict[str, NativeCameraHistory] = {}
        for channel_row in channel_rows:
            history = self._load_camera_history(
                navsim_anchor_token=str(anchor["navsim_anchor_token"]),
                navsim_log_name=log_name,
                navsim_history_start_timestamp_us=int(anchor["navsim_history_start_timestamp_us"]),
                navsim_anchor_timestamp_us=int(anchor["navsim_anchor_timestamp_us"]),
                channel_row=channel_row,
            )
            if history.camera_channel in histories:
                raise NuPlan10HzIndexError(
                    f"Anchor {anchor['navsim_anchor_token']} has duplicate {history.camera_channel} histories"
                )
            histories[history.camera_channel] = history

        requested = tuple(required_channels or ())
        missing_channels = sorted(set(requested) - set(histories))
        if missing_channels:
            raise NuPlan10HzIndexError(
                f"Anchor {anchor['navsim_anchor_token']} lacks required camera channel(s): {', '.join(missing_channels)}"
            )

        native_db_relative_path = self._safe_relative_path(
            str(anchor["native_db_relative_path"]),
            field="native_db_relative_path",
            expected_prefix=("splits",),
        )
        derived_anchor = DerivedNavsimAnchor(
            navsim_anchor_token=str(anchor["navsim_anchor_token"]),
            navsim_log_name=log_name,
            navsim_log_token=str(anchor["navsim_log_token"]),
            navsim_scene_token=str(anchor["navsim_scene_token"]),
            navsim_anchor_timestamp_us=int(anchor["navsim_anchor_timestamp_us"]),
            navsim_row_index=int(anchor["navsim_row_index"]),
            navsim_history_start_timestamp_us=int(anchor["navsim_history_start_timestamp_us"]),
            native_lidar_token=str(anchor["native_lidar_token"]),
            native_lidar_timestamp_us=int(anchor["native_lidar_timestamp_us"]),
            native_lidar_scene_token=str(anchor["native_lidar_scene_token"]),
            native_lidar_filename=str(anchor["native_lidar_filename"]),
            native_lidar_mapping_method=str(anchor["native_lidar_mapping_method"]),
            native_db_relative_path=native_db_relative_path,
            camera_histories=histories,
        )
        if require_images:
            if nuplan_root is None:
                raise NuPlan10HzImagesUnavailableError(
                    "require_images=True needs a native nuPlan root for JPEG availability checks"
                )
            availability = self.jpeg_availability(derived_anchor, nuplan_root)
            if not availability.complete:
                preview = ", ".join(str(path) for path in availability.missing_relative_paths[:3])
                raise NuPlan10HzImagesUnavailableError(
                    f"Native JPEGs are unavailable for NAVSIM anchor {derived_anchor.navsim_anchor_token}: "
                    f"{len(availability.missing_relative_paths)} missing; first: {preview}"
                )
        return derived_anchor

    def jpeg_availability(self, anchor: DerivedNavsimAnchor, nuplan_root: Path) -> NativeJpegAvailability:
        """Report staged JPEG availability without failing or opening image contents."""

        references = [
            reference
            for history in anchor.camera_histories.values()
            for reference in history.references
        ]
        missing = tuple(
            sorted(
                {
                    reference.native_blob_relative_path
                    for reference in references
                    if not reference.blob_path(nuplan_root).is_file()
                },
                key=str,
            )
        )
        return NativeJpegAvailability(
            checked=True,
            total_references=len(references),
            present_references=sum(1 for reference in references if reference.native_blob_relative_path not in missing),
            missing_relative_paths=missing,
        )

    def _load_and_validate_schema(self) -> dict[str, str]:
        for table, required_columns in self._REQUIRED_COLUMNS.items():
            try:
                rows = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            except sqlite3.Error as exc:
                raise NuPlan10HzIndexError(f"Could not inspect derived-index table {table}: {exc}") from exc
            available = {str(row["name"]) for row in rows}
            missing = sorted(required_columns - available)
            if missing:
                raise NuPlan10HzIndexError(
                    f"Derived 10 Hz index table {table!r} is missing required column(s): {', '.join(missing)}"
                )
        try:
            rows = self._connection.execute("SELECT key, value FROM metadata").fetchall()
        except sqlite3.Error as exc:
            raise NuPlan10HzIndexError(f"Could not load derived-index metadata: {exc}") from exc
        metadata = {str(row["key"]): str(row["value"]) for row in rows}
        if metadata.get("format") != INDEX_FORMAT:
            raise NuPlan10HzIndexError(
                f"Unsupported derived 10 Hz index format {metadata.get('format')!r}; expected {INDEX_FORMAT!r}"
            )
        return metadata

    def _load_camera_history(
        self,
        *,
        navsim_anchor_token: str,
        navsim_log_name: str,
        navsim_history_start_timestamp_us: int,
        navsim_anchor_timestamp_us: int,
        channel_row: sqlite3.Row,
    ) -> NativeCameraHistory:
        channel = str(channel_row["camera_channel"])
        first_stream_index = int(channel_row["first_stream_index"])
        last_stream_index = int(channel_row["last_stream_index"])
        expected_frame_count = int(channel_row["frame_count"])
        history_start_timestamp_us = int(channel_row["history_start_timestamp_us"])
        history_end_timestamp_us = int(channel_row["history_end_timestamp_us"])
        history_start_retained_image_id = int(channel_row["history_start_retained_image_id"])
        history_start_retained_timestamp_us = int(
            channel_row["history_start_retained_native_image_timestamp_us"]
        )
        history_end_retained_image_id = int(channel_row["history_end_retained_image_id"])
        history_end_retained_timestamp_us = int(
            channel_row["history_end_retained_native_image_timestamp_us"]
        )
        if not channel.startswith("CAM_"):
            raise NuPlan10HzIndexError(f"Invalid camera channel {channel!r} for anchor {navsim_anchor_token}")
        if first_stream_index < 0 or last_stream_index < first_stream_index:
            raise NuPlan10HzIndexError(
                f"Invalid stream range {first_stream_index}..{last_stream_index} for {navsim_anchor_token}/{channel}"
            )
        if expected_frame_count != last_stream_index - first_stream_index + 1:
            raise NuPlan10HzIndexError(
                f"Range/frame-count mismatch for {navsim_anchor_token}/{channel}: "
                f"{first_stream_index}..{last_stream_index} versus {expected_frame_count} frame(s)"
            )
        if self._expected_history_frames is not None and expected_frame_count != self._expected_history_frames:
            raise NuPlan10HzIndexError(
                f"Native camera history for {navsim_anchor_token}/{channel} has {expected_frame_count} frame(s), "
                f"but this derived protocol requires {self._expected_history_frames}"
            )
        if history_start_timestamp_us > history_end_timestamp_us:
            raise NuPlan10HzIndexError(
                f"History start exceeds end for {navsim_anchor_token}/{channel}: "
                f"{history_start_timestamp_us} > {history_end_timestamp_us}"
            )
        retained_tolerance_us = self._metadata_int("retained_camera_tolerance_us", default=50_000)
        if abs(history_start_retained_timestamp_us - navsim_history_start_timestamp_us) > retained_tolerance_us:
            raise NuPlan10HzIndexError(
                f"First retained camera timestamp is not near the NAVSIM history start for "
                f"{navsim_anchor_token}/{channel}"
            )
        if abs(history_end_retained_timestamp_us - navsim_anchor_timestamp_us) > retained_tolerance_us:
            raise NuPlan10HzIndexError(
                f"Current retained camera timestamp is not near the NAVSIM anchor for "
                f"{navsim_anchor_token}/{channel}"
            )
        if history_start_timestamp_us != history_start_retained_timestamp_us:
            raise NuPlan10HzIndexError(
                f"Native history start is not the first retained camera timestamp for "
                f"{navsim_anchor_token}/{channel}"
            )
        if history_end_timestamp_us != history_end_retained_timestamp_us:
            raise NuPlan10HzIndexError(
                f"Native history end is not the current retained camera timestamp for "
                f"{navsim_anchor_token}/{channel}"
            )

        rows = self._connection.execute(
            """
            SELECT image_id, camera_channel, stream_index, native_image_token,
                   native_image_timestamp_us, native_image_filename_jpg,
                   native_camera_token, native_blob_relative_path
            FROM native_image
            WHERE navsim_log_name = ? AND camera_channel = ?
              AND stream_index BETWEEN ? AND ?
            ORDER BY stream_index ASC
            """,
            (navsim_log_name, channel, first_stream_index, last_stream_index),
        ).fetchall()
        if len(rows) != expected_frame_count:
            raise NuPlan10HzIndexError(
                f"Image range for {navsim_anchor_token}/{channel} yielded {len(rows)} row(s), "
                f"expected {expected_frame_count}"
            )

        references: list[NativeCameraReference] = []
        previous_timestamp_us: Optional[int] = None
        cadence_tolerance_us = self._metadata_int("camera_cadence_tolerance_us", default=50_000)
        capture_gap_policy = self._native_camera_capture_gap_policy()
        for expected_stream_index, row in enumerate(rows, start=first_stream_index):
            if int(row["stream_index"]) != expected_stream_index:
                raise NuPlan10HzIndexError(
                    f"Image stream index is not contiguous for {navsim_anchor_token}/{channel}"
                )
            timestamp_us = int(row["native_image_timestamp_us"])
            if timestamp_us < history_start_timestamp_us or timestamp_us > history_end_timestamp_us:
                raise NuPlan10HzIndexError(
                    f"Native image timestamp {timestamp_us} is outside the declared history window for "
                    f"{navsim_anchor_token}/{channel}"
                )
            if previous_timestamp_us is not None and timestamp_us <= previous_timestamp_us:
                raise NuPlan10HzIndexError(
                    f"Native image timestamps are not strictly increasing for {navsim_anchor_token}/{channel}"
                )
            if previous_timestamp_us is not None:
                delta_us = timestamp_us - previous_timestamp_us
                if delta_us < NOMINAL_NATIVE_CAMERA_INTERVAL_US - cadence_tolerance_us:
                    raise NuPlan10HzIndexError(
                        f"Native image cadence is implausibly short for {navsim_anchor_token}/{channel}: "
                        f"{delta_us} us"
                    )
                if (
                    delta_us > NOMINAL_NATIVE_CAMERA_INTERVAL_US + cadence_tolerance_us
                    and capture_gap_policy != NATIVE_CAMERA_CAPTURE_GAP_POLICY
                ):
                    raise NuPlan10HzIndexError(
                        f"Native image cadence has an undeclared source gap for "
                        f"{navsim_anchor_token}/{channel}: {delta_us} us"
                    )
            previous_timestamp_us = timestamp_us
            references.append(
                NativeCameraReference(
                    image_id=int(row["image_id"]),
                    camera_channel=channel,
                    stream_index=int(row["stream_index"]),
                    native_image_token=str(row["native_image_token"]),
                    native_image_timestamp_us=timestamp_us,
                    native_image_filename_jpg=str(row["native_image_filename_jpg"]),
                    native_camera_token=str(row["native_camera_token"]),
                    native_blob_relative_path=self._safe_blob_relative_path(
                        str(row["native_blob_relative_path"]),
                        log_name=navsim_log_name,
                        channel=channel,
                    ),
                )
            )

        history = NativeCameraHistory(
            camera_channel=channel,
            history_start_retained_image_id=history_start_retained_image_id,
            history_start_retained_native_image_timestamp_us=history_start_retained_timestamp_us,
            history_end_retained_image_id=history_end_retained_image_id,
            history_end_retained_native_image_timestamp_us=history_end_retained_timestamp_us,
            history_start_timestamp_us=history_start_timestamp_us,
            history_end_timestamp_us=history_end_timestamp_us,
            camera_cadence_tolerance_us=cadence_tolerance_us,
            references=tuple(references),
        )
        start_retained = history.history_start_retained_reference
        if start_retained.native_image_timestamp_us != history.history_start_retained_native_image_timestamp_us:
            raise NuPlan10HzIndexError(
                f"First retained timestamp/image mismatch for {navsim_anchor_token}/{channel}"
            )
        end_retained = history.history_end_retained_reference
        if end_retained.native_image_timestamp_us != history.history_end_retained_native_image_timestamp_us:
            raise NuPlan10HzIndexError(
                f"Current retained timestamp/image mismatch for {navsim_anchor_token}/{channel}"
            )
        if history.references[0] != start_retained:
            raise NuPlan10HzIndexError(
                f"First retained camera image is not the native history endpoint for "
                f"{navsim_anchor_token}/{channel}"
            )
        if history.references[-1] != end_retained:
            raise NuPlan10HzIndexError(
                f"Current retained camera image is not the native history endpoint for "
                f"{navsim_anchor_token}/{channel}"
            )
        return history

    def _metadata_int(self, key: str, *, default: int) -> int:
        value = self._metadata.get(key)
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise NuPlan10HzIndexError(
                f"Derived 10 Hz metadata {key!r} must be an integer, got {value!r}"
            ) from exc
        if parsed < 0:
            raise NuPlan10HzIndexError(
                f"Derived 10 Hz metadata {key!r} must be nonnegative, got {parsed}"
            )
        return parsed

    def _native_camera_capture_gap_policy(self) -> str:
        """Return the declared source-gap policy, failing closed for unknown values."""

        policy = self._metadata.get("native_camera_capture_gap_policy")
        if policy is None:
            # Older v1 indexes did not state their gap policy.  Preserve their
            # stricter reader behavior rather than silently accepting a gap.
            return "strict_nominal_10hz"
        if policy != NATIVE_CAMERA_CAPTURE_GAP_POLICY:
            raise NuPlan10HzIndexError(
                "Unsupported native camera capture-gap policy in derived 10 Hz index: "
                f"{policy!r}"
            )
        return policy

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = str(token).strip().lower()
        if not normalized:
            raise NuPlan10HzIndexError("NAVSIM anchor token must be a non-empty string")
        return normalized

    @staticmethod
    def _safe_relative_path(value: str, *, field: str, expected_prefix: tuple[str, ...]) -> PurePosixPath:
        path = PurePosixPath(value.replace("\\", "/"))
        if (
            not value
            or path.is_absolute()
            or len(path.parts) < len(expected_prefix)
            or tuple(path.parts[: len(expected_prefix)]) != expected_prefix
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise NuPlan10HzIndexError(f"Unsafe {field} in derived 10 Hz index: {value!r}")
        return path

    @classmethod
    def _safe_blob_relative_path(cls, value: str, *, log_name: str, channel: str) -> PurePosixPath:
        path = cls._safe_relative_path(
            value,
            field="native_blob_relative_path",
            expected_prefix=("sensor_blobs", log_name, channel),
        )
        if len(path.parts) != 4 or path.suffix.lower() != ".jpg":
            raise NuPlan10HzIndexError(
                f"Native blob path is not canonical for {log_name}/{channel}: {value!r}"
            )
        return path


class Navsim10HzImageLoader:
    """Decode exact staged native JPEGs for the canonical NAVSIM overlay.

    This class deliberately does not alter ``SceneLoader`` or manufacture
    NAVSIM labels at 10 Hz.  It pairs the existing 2 Hz anchor metadata with
    its source-faithful native camera history.  Histories can have different
    valid lengths, so callers should consume ``valid_mask`` when batching.
    """

    def __init__(
        self,
        index: NuPlan10HzCameraHistoryIndex,
        nuplan_root: Path,
        *,
        color_mode: str = "RGB",
        storage_index: Optional[NativeImageStorageIndex] = None,
        pack_root: Optional[Path] = None,
        supplemental_pack_root: Optional[Path] = None,
        existing_camera_root: Optional[Path] = None,
        verify_sha256: bool = True,
        _owns_index: bool = False,
        _owns_storage_index: bool = False,
    ):
        if not color_mode:
            raise NuPlan10HzIndexError("color_mode must be a non-empty PIL image mode")
        self._index = index
        self._nuplan_root = Path(nuplan_root)
        self._color_mode = color_mode
        self._storage_index = storage_index
        self._pack_root = None if pack_root is None else Path(pack_root).resolve()
        self._supplemental_pack_root = (
            None if supplemental_pack_root is None else Path(supplemental_pack_root).resolve()
        )
        self._existing_camera_root = (
            None if existing_camera_root is None else Path(existing_camera_root).resolve()
        )
        self._verify_sha256 = bool(verify_sha256)
        self._owns_index = _owns_index
        self._owns_storage_index = _owns_storage_index

    @classmethod
    def from_index_path(
        cls,
        index_path: Path,
        nuplan_root: Path,
        *,
        expected_history_frames: Optional[int] = None,
        color_mode: str = "RGB",
        storage_index_path: Optional[Path] = None,
        pack_root: Optional[Path] = None,
        supplemental_pack_root: Optional[Path] = None,
        existing_camera_root: Optional[Path] = None,
        verify_sha256: bool = True,
    ) -> "Navsim10HzImageLoader":
        """Open an index owned by this loader; call ``close`` when finished."""

        storage_index = (
            NativeImageStorageIndex(storage_index_path)
            if storage_index_path is not None
            else None
        )
        return cls(
            NuPlan10HzCameraHistoryIndex(
                index_path,
                expected_history_frames=expected_history_frames,
            ),
            nuplan_root,
            color_mode=color_mode,
            storage_index=storage_index,
            pack_root=pack_root,
            supplemental_pack_root=supplemental_pack_root,
            existing_camera_root=existing_camera_root,
            verify_sha256=verify_sha256,
            _owns_index=True,
            _owns_storage_index=storage_index is not None,
        )

    def __enter__(self) -> "Navsim10HzImageLoader":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close an index opened by ``from_index_path``."""

        if self._owns_index:
            self._index.close()
            self._owns_index = False
        if self._owns_storage_index and self._storage_index is not None:
            self._storage_index.close()
            self._owns_storage_index = False

    def load_anchor(
        self,
        token: str,
        *,
        required_channels: Optional[Sequence[str]] = None,
    ) -> LoadedNavsim10HzAnchor:
        """Decode one existing NAVSIM anchor's exact indexed JPEG sequences."""

        anchor = self._index.get_anchor(
            token,
            required_channels=required_channels,
            nuplan_root=self._nuplan_root,
            require_images=self._storage_index is None,
        )
        histories: dict[str, LoadedNativeCameraHistory] = {}
        for channel, history in anchor.camera_histories.items():
            images = tuple(self._decode_jpeg(reference) for reference in history.references)
            timestamps = history.native_image_timestamp_us
            if len(images) != history.frame_count or len(timestamps) != history.frame_count:
                raise NuPlan10HzIndexError(
                    f"Decoded native camera length mismatch for {anchor.navsim_anchor_token}/{channel}"
                )
            histories[channel] = LoadedNativeCameraHistory(
                history=history,
                images=images,
                native_image_timestamp_us=timestamps,
                valid_mask=tuple(True for _ in images),
            )
        return LoadedNavsim10HzAnchor(anchor=anchor, camera_histories=histories)

    def _decode_jpeg(self, reference: NativeCameraReference) -> npt.NDArray[np.uint8]:
        """Decode a single indexed regular JPEG without following an unsafe path."""

        payload: bytes
        storage = None
        if self._storage_index is not None:
            storage = self._storage_index.get(reference.native_blob_relative_path)
        if storage is not None and storage.storage_kind != "existing":
            pack_root = (
                self._supplemental_pack_root
                if storage.storage_kind == "supplemental_pack"
                else self._pack_root
            )
            if pack_root is None or storage.pack_offset is None:
                raise NuPlan10HzImagesUnavailableError("Packed JPEG root is not configured")
            pack_path = pack_root.joinpath(*storage.storage_relative_path.parts)
            try:
                pack_path.resolve(strict=True).relative_to(pack_root)
                if pack_path.resolve() != pack_path.absolute() or not pack_path.is_file():
                    raise ValueError
                with pack_path.open("rb") as handle:
                    handle.seek(storage.pack_offset)
                    payload = handle.read(storage.payload_size)
            except (OSError, ValueError) as exc:
                raise NuPlan10HzImagesUnavailableError(
                    f"Packed JPEG is unavailable: {storage.storage_relative_path}"
                ) from exc
            if len(payload) != storage.payload_size:
                raise NuPlan10HzImagesUnavailableError("Packed JPEG slice is truncated")
            expected_sha = storage.sha256
        else:
            if storage is not None and self._existing_camera_root is not None:
                root = self._existing_camera_root
                image_path = root.joinpath(*storage.storage_relative_path.parts)
                try:
                    resolved = image_path.resolve(strict=True)
                    resolved.relative_to(root)
                    if resolved != image_path.absolute() or not resolved.is_file():
                        raise ValueError
                    image_path = resolved
                except (OSError, ValueError) as exc:
                    raise NuPlan10HzImagesUnavailableError(
                        f"Certified official NAVSIM JPEG is unavailable: "
                        f"{storage.storage_relative_path}"
                    ) from exc
            else:
                image_path = reference.blob_path(self._nuplan_root)
            if not image_path.is_file():
                raise NuPlan10HzImagesUnavailableError(
                    f"Indexed native JPEG is unavailable: {reference.native_blob_relative_path}"
                )
            try:
                payload = image_path.read_bytes()
            except OSError as exc:
                raise NuPlan10HzImagesUnavailableError(
                    f"Could not read indexed native JPEG {reference.native_blob_relative_path}: {exc}"
                ) from exc
            expected_sha = None if storage is None else storage.sha256
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise NuPlan10HzImagesUnavailableError("Indexed payload is not a complete JPEG")
        if self._verify_sha256 and expected_sha is not None:
            observed_sha = hashlib.sha256(payload).hexdigest()
            if observed_sha != expected_sha:
                raise NuPlan10HzImagesUnavailableError("Indexed JPEG SHA-256 mismatch")
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                pixels = np.asarray(image.convert(self._color_mode), dtype=np.uint8).copy()
        except (OSError, ValueError) as exc:
            raise NuPlan10HzImagesUnavailableError(
                f"Could not decode indexed native JPEG {reference.native_blob_relative_path}: {exc}"
            ) from exc
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise NuPlan10HzIndexError(
                f"Decoded {reference.native_blob_relative_path} is not an RGB HxWx3 image: {pixels.shape!r}"
            )
        return pixels


class Navsim10HzMetadataAdapter:
    """Restrict a native-camera index to the existing tokens of a NAVSIM loader.

    ``SceneLoader`` remains the owner of scene labels, future trajectory, and
    official 2 Hz sensor records.  This overlay only supplies the additional
    native camera references required by a future 10 Hz-aware feature builder.
    It never calls ``SceneLoader.get_agent_input_from_token`` and cannot make
    the stock loader consume native JPEGs by itself.
    """

    def __init__(self, index: NuPlan10HzCameraHistoryIndex, navsim_tokens: Iterable[str]):
        self._index = index
        self._navsim_tokens = {str(token).strip().lower() for token in navsim_tokens if str(token).strip()}

    @classmethod
    def from_scene_loader(cls, index: NuPlan10HzCameraHistoryIndex, scene_loader: Any) -> "Navsim10HzMetadataAdapter":
        """Build an overlay from an existing stock NAVSIM ``SceneLoader``."""

        return cls(index, scene_loader.tokens)

    @property
    def tokens(self) -> tuple[str, ...]:
        """2 Hz NAVSIM tokens with a corresponding native-camera history."""

        return tuple(token for token in self._index.tokens() if token in self._navsim_tokens)

    def get_anchor(
        self,
        token: str,
        *,
        required_channels: Optional[Sequence[str]] = None,
        nuplan_root: Optional[Path] = None,
        require_images: bool = False,
    ) -> DerivedNavsimAnchor:
        """Return metadata only after proving the token belongs to stock NAVSIM."""

        normalized = NuPlan10HzCameraHistoryIndex._normalize_token(token)
        if normalized not in self._navsim_tokens:
            raise NuPlan10HzIndexError(
                f"Token {token!r} is not present in the supplied NAVSIM source-loader token set"
            )
        return self._index.get_anchor(
            normalized,
            required_channels=required_channels,
            nuplan_root=nuplan_root,
            require_images=require_images,
        )
