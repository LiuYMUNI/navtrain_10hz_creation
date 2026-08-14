#!/usr/bin/env python3
"""Render reproducible random NAVSIM/navtrain native-10 Hz camera panels."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import random
import sqlite3
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = (
    REPO_ROOT
    / "dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_index.sqlite"
)
DEFAULT_PACK_ROOT = (
    REPO_ROOT / "dataset/raw/nuplan-v1.1-navtrain-10hz/camera_packs"
)
DEFAULT_EXISTING_ROOT = REPO_ROOT / "dataset/navsim2/sensor_blobs/trainval"
DEFAULT_OUTPUT = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/visual_audit"
)


class VisualAuditError(RuntimeError):
    """Raised when the sampled mapping or stored JPEG violates its contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--source-state", type=Path, action="append", default=[])
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--existing-root", type=Path, default=DEFAULT_EXISTING_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--camera-channel", default="CAM_F0")
    return parser.parse_args()


def readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise VisualAuditError(f"Missing SQLite file: {path}")
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def safe_regular_path(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise VisualAuditError(f"Unsafe relative storage path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*posix.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise VisualAuditError(f"Storage path escapes or is missing: {relative}") from exc
    if resolved != candidate.absolute() or not resolved.is_file():
        raise VisualAuditError(f"Storage path is not a regular non-symlink file: {relative}")
    return resolved


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


class StorageResolver:
    def __init__(
        self,
        states: list[Path],
        *,
        pack_root: Path,
        existing_root: Path,
    ) -> None:
        self.connections = [readonly(path) for path in states]
        self.pack_root = pack_root
        self.existing_root = existing_root

    def close(self) -> None:
        for connection in self.connections:
            connection.close()

    def get(self, destination: str) -> sqlite3.Row:
        rows = []
        for connection in self.connections:
            row = connection.execute(
                "SELECT destination_relative_path, storage_kind, storage_relative_path, "
                "pack_offset, payload_size, sha256, archive_key, tar_member_name "
                "FROM member_storage WHERE destination_relative_path = ?",
                (destination,),
            ).fetchone()
            if row is not None:
                rows.append(row)
        if len(rows) != 1:
            raise VisualAuditError(
                f"Expected exactly one storage row for {destination}, found {len(rows)}"
            )
        return rows[0]

    def read_jpeg(self, row: sqlite3.Row) -> bytes:
        kind = str(row["storage_kind"])
        size = int(row["payload_size"])
        if size <= 0:
            raise VisualAuditError("Stored JPEG has a non-positive payload size")
        if kind == "existing":
            path = safe_regular_path(self.existing_root, str(row["storage_relative_path"]))
            payload = path.read_bytes()
        elif kind == "pack":
            path = safe_regular_path(self.pack_root, str(row["storage_relative_path"]))
            offset = row["pack_offset"]
            if offset is None or int(offset) < 0:
                raise VisualAuditError("Packed JPEG has an invalid offset")
            descriptor = os.open(path, os.O_RDONLY)
            try:
                payload = os.pread(descriptor, size, int(offset))
            finally:
                os.close(descriptor)
        else:
            raise VisualAuditError(f"Unsupported visual-audit storage kind: {kind!r}")
        if len(payload) != size:
            raise VisualAuditError("Stored JPEG size does not match its storage row")
        if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise VisualAuditError("Stored payload is not a complete JPEG")
        observed = hashlib.sha256(payload).hexdigest()
        if observed != str(row["sha256"]):
            raise VisualAuditError("Stored JPEG SHA-256 does not match its storage row")
        return payload


def sample_anchors(
    connection: sqlite3.Connection,
    *,
    samples: int,
    seed: int,
    camera_channel: str,
) -> list[sqlite3.Row]:
    if samples < 1:
        raise VisualAuditError("--samples must be positive")
    generator = random.Random(seed)
    selected: list[sqlite3.Row] = []
    selected_logs: set[str] = set()
    attempts = 0
    while len(selected) < samples and attempts < samples * 200:
        attempts += 1
        probe = f"{generator.getrandbits(64):016x}"
        anchor_row = connection.execute(
            "SELECT navsim_anchor_token, navsim_log_name, navsim_scene_token, "
            "navsim_anchor_timestamp_us, navsim_row_index FROM anchor "
            "WHERE navsim_anchor_token >= ? ORDER BY navsim_anchor_token LIMIT 1",
            (probe,),
        ).fetchone()
        if anchor_row is None:
            continue
        camera_row = connection.execute(
            "SELECT first_stream_index, last_stream_index, frame_count, "
            "history_start_retained_native_image_timestamp_us, "
            "history_end_retained_native_image_timestamp_us "
            "FROM anchor_camera WHERE navsim_anchor_token = ? AND camera_channel = ?",
            (anchor_row["navsim_anchor_token"], camera_channel),
        ).fetchone()
        if camera_row is None:
            continue
        row = dict(anchor_row)
        row.update(dict(camera_row))
        if row is None or str(row["navsim_log_name"]) in selected_logs:
            continue
        if int(row["frame_count"]) < 2:
            continue
        selected.append(row)
        selected_logs.add(str(row["navsim_log_name"]))
    if len(selected) != samples:
        raise VisualAuditError(
            f"Could only select {len(selected)} distinct logs for {samples} panels"
        )
    return selected


def render_panel(
    *,
    index: sqlite3.Connection,
    resolver: StorageResolver,
    anchor: sqlite3.Row,
    camera_channel: str,
    panel_number: int,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    references = index.execute(
        "SELECT image_id, stream_index, native_image_timestamp_us, "
        "native_blob_relative_path FROM native_image "
        "WHERE navsim_log_name = ? AND camera_channel = ? "
        "AND stream_index BETWEEN ? AND ? ORDER BY stream_index",
        (
            anchor["navsim_log_name"],
            camera_channel,
            anchor["first_stream_index"],
            anchor["last_stream_index"],
        ),
    ).fetchall()
    if len(references) != int(anchor["frame_count"]):
        raise VisualAuditError("Index frame_count does not match its native image range")

    cell_width, image_height, label_height = 320, 180, 70
    columns = 4
    rows = (len(references) + columns - 1) // columns
    header_height = 150
    canvas = Image.new(
        "RGB", (columns * cell_width, header_height + rows * (image_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23, bold=True)
    body_font = load_font(16)
    small_font = load_font(14)
    log_name = str(anchor["navsim_log_name"])
    token = str(anchor["navsim_anchor_token"])
    anchor_timestamp = int(anchor["navsim_anchor_timestamp_us"])
    draw.text((14, 10), f"NAVSIM navtrain native-10 Hz visual audit #{panel_number}", fill="black", font=title_font)
    draw.text((14, 43), f"log: {log_name}", fill="black", font=body_font)
    draw.text(
        (14, 67),
        f"anchor: {token}   camera: {camera_channel}   anchor timestamp: {anchor_timestamp} us",
        fill="black",
        font=body_font,
    )
    draw.text(
        (14, 94),
        "GREEN = official NAVSIM 2 Hz frame    BLUE = added native nuPlan frame",
        fill="black",
        font=body_font,
    )
    draw.text(
        (14, 120),
        "Exact source timestamps are shown; chronological frames are not interpolated."
        f"  Reproducible seed: {seed}",
        fill="black",
        font=small_font,
    )

    frame_records: list[dict[str, Any]] = []
    first_timestamp = int(references[0]["native_image_timestamp_us"])
    previous_timestamp: int | None = None
    existing_count = 0
    for position, reference in enumerate(references):
        destination = str(reference["native_blob_relative_path"])
        storage = resolver.get(destination)
        payload = resolver.read_jpeg(storage)
        with Image.open(BytesIO(payload)) as source:
            source.load()
            frame = ImageOps.fit(source.convert("RGB"), (cell_width - 8, image_height - 8))
        kind = str(storage["storage_kind"])
        is_navsim = kind == "existing"
        existing_count += int(is_navsim)
        border = (25, 155, 75) if is_navsim else (35, 105, 210)
        row, column = divmod(position, columns)
        x = column * cell_width
        y = header_height + row * (image_height + label_height)
        draw.rectangle((x, y, x + cell_width - 1, y + image_height - 1), fill=border)
        canvas.paste(frame, (x + 4, y + 4))
        timestamp = int(reference["native_image_timestamp_us"])
        relative_s = (timestamp - anchor_timestamp) / 1_000_000.0
        delta_us = None if previous_timestamp is None else timestamp - previous_timestamp
        source_label = "NAVSIM 2 Hz" if is_navsim else "nuPlan added"
        draw.text(
            (x + 7, y + image_height + 3),
            f"#{position:02d}  t={relative_s:+.3f}s  {source_label}",
            fill=border,
            font=small_font,
        )
        draw.text(
            (x + 7, y + image_height + 25),
            f"timestamp={timestamp} us",
            fill="black",
            font=small_font,
        )
        delta_text = "start" if delta_us is None else f"delta={delta_us / 1000.0:.3f} ms"
        draw.text((x + 7, y + image_height + 47), delta_text, fill="black", font=small_font)
        frame_records.append(
            {
                "position": position,
                "image_id": int(reference["image_id"]),
                "stream_index": int(reference["stream_index"]),
                "native_image_timestamp_us": timestamp,
                "relative_to_anchor_us": timestamp - anchor_timestamp,
                "delta_from_previous_us": delta_us,
                "destination_relative_path": destination,
                "storage_kind": kind,
                "sha256": str(storage["sha256"]),
            }
        )
        previous_timestamp = timestamp

    if frame_records[0]["native_image_timestamp_us"] != int(
        anchor["history_start_retained_native_image_timestamp_us"]
    ):
        raise VisualAuditError("First visualized frame does not match retained NAVSIM history start")
    if frame_records[-1]["native_image_timestamp_us"] != int(
        anchor["history_end_retained_native_image_timestamp_us"]
    ):
        raise VisualAuditError("Last visualized frame does not match retained NAVSIM history end")
    if frame_records[0]["storage_kind"] != "existing" or frame_records[-1]["storage_kind"] != "existing":
        raise VisualAuditError("Retained NAVSIM history endpoints are not official loose frames")

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return {
        "panel": output.name,
        "navsim_anchor_token": token,
        "navsim_log_name": log_name,
        "navsim_scene_token": str(anchor["navsim_scene_token"]),
        "navsim_row_index": int(anchor["navsim_row_index"]),
        "navsim_anchor_timestamp_us": anchor_timestamp,
        "camera_channel": camera_channel,
        "frame_count": len(frame_records),
        "official_navsim_frame_count": existing_count,
        "added_native_nuplan_frame_count": len(frame_records) - existing_count,
        "first_to_last_duration_us": frame_records[-1]["native_image_timestamp_us"] - first_timestamp,
        "all_jpeg_sha256_rows_verified": True,
        "frames": frame_records,
    }


def make_overview(output_dir: Path, panel_paths: list[Path]) -> Path:
    thumb_width = 700
    thumbs = []
    for path in panel_paths:
        with Image.open(path) as panel:
            ratio = thumb_width / panel.width
            thumbs.append(panel.convert("RGB").resize((thumb_width, int(panel.height * ratio))))
    columns = 2
    rows = (len(thumbs) + columns - 1) // columns
    cell_height = max(image.height for image in thumbs)
    overview = Image.new("RGB", (columns * thumb_width, rows * cell_height), "white")
    for position, thumb in enumerate(thumbs):
        row, column = divmod(position, columns)
        overview.paste(thumb, (column * thumb_width, row * cell_height))
    path = output_dir / "overview.png"
    overview.save(path, format="PNG", optimize=True)
    return path


def main() -> int:
    args = parse_args()
    state_paths = args.source_state
    if not state_paths:
        raise VisualAuditError(
            "At least one --source-state pack-state SQLite file is required"
        )
    index = readonly(args.index)
    resolver = StorageResolver(
        state_paths,
        pack_root=args.pack_root,
        existing_root=args.existing_root,
    )
    try:
        anchors = sample_anchors(
            index,
            samples=args.samples,
            seed=args.seed,
            camera_channel=args.camera_channel,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        panel_paths = []
        for number, anchor in enumerate(anchors, start=1):
            panel_path = args.output_dir / f"panel_{number:02d}_{anchor['navsim_anchor_token']}.png"
            records.append(
                render_panel(
                    index=index,
                    resolver=resolver,
                    anchor=anchor,
                    camera_channel=args.camera_channel,
                    panel_number=number,
                    seed=args.seed,
                    output=panel_path,
                )
            )
            panel_paths.append(panel_path)
        overview = make_overview(args.output_dir, panel_paths)
        manifest = {
            "format": "navsim_nuplan10hz_visual_audit_v1",
            "purpose": (
                "Human-readable audit that official NAVSIM 2 Hz frames and added native "
                "nuPlan frames form one timestamp-ordered camera sequence."
            ),
            "seed": args.seed,
            "sample_count": len(records),
            "camera_channel": args.camera_channel,
            "selection_policy": "deterministic random anchors from distinct navtrain logs",
            "overview": overview.name,
            "source_states": [str(path.resolve()) for path in state_paths],
            "panels": records,
        }
        manifest_path = args.output_dir / "manifest.json"
        temporary = manifest_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        print(json.dumps({
            "status": "pass",
            "manifest": str(manifest_path),
            "overview": str(overview),
            "panels": len(records),
        }, indent=2))
        return 0
    finally:
        resolver.close()
        index.close()


if __name__ == "__main__":
    raise SystemExit(main())
