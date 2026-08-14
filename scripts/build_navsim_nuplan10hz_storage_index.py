#!/usr/bin/env python3
"""Build the immutable destination-to-byte storage index for NAVSIM 10 Hz."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
from typing import Iterable


FORMAT = "navsim_nuplan10hz_image_storage_v1"
SOURCE_FORMAT = "navsim_nuplan10hz_camera_pack_state_v1"
VERIFICATION_FORMAT = "navsim_nuplan10hz_existing_verification_state_v1"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-state", type=Path, required=True)
    p.add_argument("--source-state", type=Path, action="append", required=True)
    p.add_argument("--verification-state", type=Path)
    p.add_argument(
        "--existing-policy",
        choices=("require-native-verification", "trust-official-navsim"),
        default="require-native-verification",
    )
    p.add_argument("--visual-audit-manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pack-root", type=Path, required=True)
    p.add_argument("--supplemental-pack-root", type=Path, required=True)
    p.add_argument("--inventory-json", type=Path, required=True)
    return p.parse_args()


def ro(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA query_only = ON")
    return c


def metadata(c: sqlite3.Connection) -> dict[str, str]:
    return {str(r[0]): str(r[1]) for r in c.execute("select key,value from metadata")}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_pack_size(root: Path, relative: str, cache: dict[Path, int]) -> int:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise RuntimeError(f"unsafe pack path: {relative!r}")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*posix.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"missing or unsafe pack: {candidate}") from exc
    if resolved != candidate.absolute() or not resolved.is_file():
        raise RuntimeError(f"pack must be a regular non-symlink file: {candidate}")
    if resolved not in cache:
        cache[resolved] = resolved.stat().st_size
    return cache[resolved]


def main() -> int:
    a = args()
    target = ro(a.target_state)
    sources = [ro(p) for p in a.source_state]
    trust_official_navsim = a.existing_policy == "trust-official-navsim"
    if trust_official_navsim:
        if a.visual_audit_manifest is None:
            raise RuntimeError(
                "trust-official-navsim requires --visual-audit-manifest"
            )
        if a.verification_state is not None:
            raise RuntimeError(
                "--verification-state must be omitted with trust-official-navsim"
            )
        verify = None
    else:
        if a.verification_state is None:
            raise RuntimeError(
                "--verification-state is required by require-native-verification"
            )
        if a.visual_audit_manifest is not None:
            raise RuntimeError(
                "--visual-audit-manifest is only valid with trust-official-navsim"
            )
        verify = ro(a.verification_state)
    try:
        target_meta = metadata(target)
        if target_meta.get("inventory_row_count") is None:
            raise RuntimeError("target state lacks immutable inventory metadata")
        source_meta = [metadata(c) for c in sources]
        if any(m.get("format") != SOURCE_FORMAT for m in source_meta):
            raise RuntimeError("unexpected source state format")
        if any(m != source_meta[0] for m in source_meta[1:]):
            raise RuntimeError("source states have different immutable metadata")
        audit_sha = None
        if trust_official_navsim:
            assert a.visual_audit_manifest is not None
            audit = json.loads(a.visual_audit_manifest.read_text())
            panels = audit.get("panels")
            if (
                audit.get("format") != "navsim_nuplan10hz_visual_audit_v1"
                or not isinstance(panels, list)
                or len(panels) != int(audit.get("sample_count", -1))
                or len(panels) < 1
            ):
                raise RuntimeError("visual audit manifest is incomplete or unsupported")
            for panel in panels:
                if (
                    not panel.get("all_jpeg_sha256_rows_verified")
                    or int(panel.get("official_navsim_frame_count", 0)) < 2
                    or int(panel.get("added_native_nuplan_frame_count", 0)) < 1
                ):
                    raise RuntimeError("visual audit panel lacks mixed, SHA-checked coverage")
                panel_path = a.visual_audit_manifest.parent / str(panel.get("panel", ""))
                if not panel_path.is_file():
                    raise RuntimeError(f"visual audit panel is missing: {panel_path}")
            audit_sha = sha256_file(a.visual_audit_manifest)
        else:
            assert verify is not None
            if metadata(verify).get("format") != VERIFICATION_FORMAT:
                raise RuntimeError("unexpected verification state format")
            vm = metadata(verify)
            for key in ("inventory_sha256", "inventory_row_count", "navsim_scene_filter_sha256"):
                if vm.get(key) != source_meta[0].get(key):
                    raise RuntimeError(f"verification metadata mismatch: {key}")
            verification_totals = verify.execute(
                "select count(*), sum(status = 'complete'), "
                "coalesce(sum(source_existing_count),0), coalesce(sum(checked_count),0) "
                "from archive_verification"
            ).fetchone()
        source_existing_total = sum(
            int(c.execute(
                "select coalesce(sum(existing_target_count),0) from archive_progress "
                "where status = 'complete'"
            ).fetchone()[0])
            for c in sources
        )
        if not trust_official_navsim:
            if (
                int(verification_totals[0]) != 55
                or int(verification_totals[1] or 0) != 55
                or int(verification_totals[2]) != source_existing_total
                or int(verification_totals[3]) != source_existing_total
            ):
                raise RuntimeError("native verification is not complete for all reused JPEGs")

        archive_progress = []
        for connection in sources:
            archive_progress.extend(
                connection.execute(
                    "select archive_key,target_total,target_done,status from archive_progress"
                ).fetchall()
            )
        archive_keys_seen = [str(row[0]) for row in archive_progress]
        if (
            len(archive_progress) != 55
            or len(set(archive_keys_seen)) != 55
            or any(str(row[3]) != "complete" for row in archive_progress)
            or any(int(row[1]) != int(row[2]) for row in archive_progress)
        ):
            raise RuntimeError("source pack states do not contain 55 unique complete archives")

        target_count = int(source_meta[0]["inventory_row_count"])
        observed_target_count = target.execute("select count(*) from target").fetchone()[0]
        if int(observed_target_count) != target_count:
            raise RuntimeError("target row count does not match immutable metadata")

        output = a.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".partial")
        temp.unlink(missing_ok=True)
        out = sqlite3.connect(temp)
        try:
            out.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
                CREATE TABLE image_storage (
                    destination_relative_path TEXT PRIMARY KEY,
                    archive_key TEXT NOT NULL,
                    tar_member_name TEXT NOT NULL,
                    storage_kind TEXT NOT NULL,
                    storage_relative_path TEXT NOT NULL,
                    pack_offset INTEGER,
                    payload_size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    UNIQUE (archive_key, tar_member_name)
                ) WITHOUT ROWID;
            """)
            inventory_sha = sha256_file(a.inventory_json)
            output_metadata = {
                "format": FORMAT,
                "inventory_sha256": source_meta[0]["inventory_sha256"],
                "inventory_row_count": source_meta[0]["inventory_row_count"],
                "navsim_scene_filter_sha256": source_meta[0]["navsim_scene_filter_sha256"],
                "inventory_file_sha256": inventory_sha,
                "existing_source_policy": a.existing_policy,
                "native_byte_verification_complete": (
                    "false" if trust_official_navsim else "true"
                ),
                "consumer_sha256_check_required": "true",
            }
            if trust_official_navsim:
                output_metadata["visual_audit_manifest_sha256"] = str(audit_sha)
            else:
                assert a.verification_state is not None
                output_metadata["verification_state_sha256"] = sha256_file(
                    a.verification_state
                )
            for k, v in output_metadata.items():
                out.execute("insert into metadata values (?,?)", (k, v))
            archive_keys = [str(r[0]) for r in target.execute("select distinct archive_key from target order by archive_key")]
            inserted_count = 0
            pack_size_cache: dict[Path, int] = {}
            for archive_key in archive_keys:
                source_rows = []
                for c in sources:
                    source_rows.extend(c.execute(
                        "select archive_key,tar_member_name,destination_relative_path,storage_kind,"
                        "storage_relative_path,pack_offset,payload_size,sha256 from member_storage "
                        "where archive_key = ? order by tar_member_name", (archive_key,)
                    ).fetchall())
                source_by_member = {str(r[1]): r for r in source_rows}
                if len(source_by_member) != len(source_rows):
                    raise RuntimeError(f"duplicate source member for archive: {archive_key}")
                verify_by_member = {}
                if verify is not None:
                    verify_by_member = {
                        str(r[1]): r for r in verify.execute(
                            "select archive_key,tar_member_name,destination_relative_path,effective_storage_kind,"
                            "effective_storage_relative_path,effective_pack_offset,payload_size,native_sha256,"
                            "verification_status from member_verification where archive_key = ?",
                            (archive_key,),
                        )
                    }
                for row in target.execute(
                    "select archive_key,tar_member_name,destination_relative_path from target "
                    "where archive_key = ? order by tar_member_name", (archive_key,)
                ):
                    member_name = str(row[1]); key = (archive_key, member_name)
                    src = source_by_member.get(member_name)
                    if src is None:
                        raise RuntimeError(f"target member missing from source states: {key}")
                    kind = str(src[3]); path = str(src[4]); offset = src[5]
                    size = int(src[6]); digest = str(src[7])
                    if kind not in {"existing", "pack", "supplemental_pack"}:
                        raise RuntimeError(f"unsupported storage kind for {key}: {kind!r}")
                    if kind == "existing" and not trust_official_navsim:
                        vr = verify_by_member.get(member_name)
                        if vr is None or str(vr[8]) not in {"certified_equal", "backfilled"}:
                            raise RuntimeError(f"existing member lacks completed native verification: {key}")
                        kind, path, offset, size, digest = str(vr[3]), str(vr[4]), vr[5], int(vr[6]), str(vr[7])
                    destination = str(row[2])
                    if str(src[2]) != destination:
                        raise RuntimeError(
                            f"source/target destination mismatch for {key}: "
                            f"{src[2]!r} != {destination!r}"
                        )
                    if size <= 0 or len(digest) != 64 or any(
                        character not in "0123456789abcdef" for character in digest.lower()
                    ):
                        raise RuntimeError(f"invalid stored JPEG metadata: {key}")
                    if kind != "existing":
                        pack_root = (
                            a.supplemental_pack_root
                            if kind == "supplemental_pack"
                            else a.pack_root
                        )
                        if offset is None or int(offset) < 0:
                            raise RuntimeError(f"invalid pack offset: {key}")
                        pack_size = checked_pack_size(pack_root, path, pack_size_cache)
                        if int(offset) + size > pack_size:
                            raise RuntimeError(f"packed JPEG exceeds pack bounds: {key}")
                    out.execute("insert into image_storage values (?,?,?,?,?,?,?,?)",
                                (destination, key[0], key[1], kind, path, offset, size, digest))
                    inserted_count += 1
                out.commit()
            if inserted_count != target_count:
                raise RuntimeError("storage index coverage mismatch")
            out.commit()
            out.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            out.commit()
        finally:
            out.close()
        temp.replace(output)
        print(json.dumps({"status":"pass","output":str(output),"rows":target_count}, indent=2))
        return 0
    finally:
        target.close()
        for c in sources: c.close()
        if verify is not None:
            verify.close()


if __name__ == "__main__":
    raise SystemExit(main())
