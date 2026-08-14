#!/usr/bin/env python3
"""Safely pull completed NAVSIM 10 Hz packs from a relay and free relay space.

The relay packer publishes immutable ``*.pack`` files only after their source
archive is complete.  This tool runs on HPC and:

1. inventories only complete, regular pack files represented by relay SQLite;
2. triggers at a configurable byte threshold, or flushes when packing is idle;
3. rsyncs packs with resumable append verification;
4. snapshots live SQLite state with SQLite's backup API and copies reports;
5. compares full SHA-256 hashes and validates the copied SQLite snapshots;
6. deletes only the exact, unchanged relay pack files that passed every check.

State databases, reports, manifests, partial files, and unrelated relay data
are never deleted.  A local flock prevents overlapping offload processes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_PACK_ROOT = (
    REPO_ROOT / "dataset/raw/nuplan-v1.1-navtrain-10hz/camera_packs"
)
DEFAULT_LOCAL_STATE_ROOT = (
    REPO_ROOT / "dataset/cache/navsim_nuplan10hz/relay_states"
)
DEFAULT_LOCAL_REPORT_ROOT = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/relay_reports"
)
DEFAULT_LOCAL_MANIFEST_ROOT = (
    REPO_ROOT / "dataset/manifests/navsim_nuplan10hz/offload_manifests"
)
DEFAULT_TARGET_STATE = (
    REPO_ROOT / "dataset/cache/navsim_nuplan10hz/camera_download_state.sqlite"
)
DEFAULT_LOCK = REPO_ROOT / "dataset/cache/navsim_nuplan10hz/offload.lock"
PACK_PATTERN = re.compile(
    r"^(train_set/nuplan-v1\.1_train_camera_\d+|"
    r"val_set/nuplan-v1\.1_val_camera_\d+)\.pack$"
)
MANIFEST_FORMAT = "navsim_nuplan10hz_relay_offload_v1"
TARGET_METADATA_KEYS = (
    "inventory_sha256",
    "inventory_row_count",
    "navsim_scene_filter_sha256",
)


class OffloadError(RuntimeError):
    """A remote inventory, transfer, integrity, or deletion failure."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="SSH destination, for example user@relay")
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--local-pack-root", type=Path, required=True)
    parser.add_argument("--local-state-root", type=Path, required=True)
    parser.add_argument("--local-report-root", type=Path, required=True)
    parser.add_argument(
        "--local-manifest-root", type=Path, required=True
    )
    parser.add_argument(
        "--target-state-sqlite",
        type=Path,
        required=True,
        help="Canonical immutable target ledger used to validate every copied pack index.",
    )
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--threshold-gb",
        type=float,
        default=200.0,
        help="Trigger at this many decimal GB of completed relay packs.",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--hash-workers", type=int, default=4)
    parser.add_argument(
        "--relay-min-free-gb",
        type=float,
        default=500.0,
        help=(
            "Offload any finalized packs when relay free space falls below this "
            "decimal-GB floor, independently of packer process liveness."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def utc_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_pack_relative_path(value: str) -> str:
    path = PurePosixPath(str(value))
    normalized = str(path)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or PACK_PATTERN.fullmatch(normalized) is None
    ):
        raise OffloadError(f"Unsafe or unexpected pack path: {value!r}")
    return normalized


def ssh_command(args: argparse.Namespace) -> list[str]:
    if not args.ssh_key.is_file():
        raise OffloadError(f"Missing dedicated SSH key: {args.ssh_key}")
    return [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
    ]


def rsync_rsh(args: argparse.Namespace) -> str:
    return " ".join(shlex.quote(item) for item in ssh_command(args))


def run_remote_python(args: argparse.Namespace, source: str) -> str:
    command = [*ssh_command(args), args.remote, "python3", "-"]
    result = subprocess.run(
        command,
        input=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise OffloadError(
            f"Remote Python failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def remote_inventory(args: argparse.Namespace) -> dict[str, Any]:
    root_literal = repr(str(args.remote_root))
    source = f"""
import glob, json, os, sqlite3, stat
from pathlib import Path, PurePosixPath
root = Path({root_literal})
pack_root = root / 'packs'
resolved_pack_root = pack_root.resolve()
complete = {{}}
for state_name in sorted(glob.glob(str(root / 'state' / 'camera_pack_state*.sqlite'))):
    connection = sqlite3.connect(state_name, timeout=30)
    try:
        for archive_key, relative_path, pack_bytes, status in connection.execute(
            'SELECT archive_key, pack_relative_path, pack_bytes, status FROM archive_progress'
        ):
            if status != 'complete':
                continue
            prior = complete.get(relative_path)
            value = {{'archive_key': archive_key, 'size': int(pack_bytes), 'state': Path(state_name).name}}
            if prior is not None and prior != value:
                raise RuntimeError('conflicting complete pack state: ' + relative_path)
            complete[relative_path] = value
    finally:
        connection.close()
packs = []
for path in sorted(pack_root.glob('*/*.pack')):
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError('pack is not a regular file: ' + str(path))
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_pack_root)
    except ValueError:
        raise RuntimeError('pack escapes configured root: ' + str(path))
    if resolved_path != path.absolute():
        raise RuntimeError('pack path contains a symlink: ' + str(path))
    relative = str(PurePosixPath(*path.relative_to(pack_root).parts))
    row = complete.get(relative)
    if row is None or row['size'] != metadata.st_size:
        raise RuntimeError('pack lacks matching complete state: ' + relative)
    packs.append({{
        'relative_path': relative,
        'size': metadata.st_size,
        'mtime_ns': metadata.st_mtime_ns,
        'archive_key': row['archive_key'],
        'state': row['state'],
    }})
packer_running = False
for proc in Path('/proc').glob('[0-9]*'):
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\\0', b' ').decode('utf-8', 'replace')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if 'pack_navsim_nuplan10hz_cameras.py' in command:
        packer_running = True
        break
print(json.dumps({{
    'packs': packs,
    'total_bytes': sum(item['size'] for item in packs),
    'packer_running': packer_running,
    'packing_complete': (root / 'state' / 'navtrain10hz_packing_complete.json').is_file(),
    'free_bytes': os.statvfs(root).f_bavail * os.statvfs(root).f_frsize,
}}))
"""
    output = run_remote_python(args, source)
    try:
        inventory = json.loads(output)
    except json.JSONDecodeError as exc:
        raise OffloadError(f"Malformed remote inventory: {output!r}") from exc
    for item in inventory.get("packs", []):
        item["relative_path"] = validate_pack_relative_path(item["relative_path"])
    return inventory


def rsync_selected_packs(
    args: argparse.Namespace,
    relative_paths: Sequence[str],
) -> None:
    args.local_pack_root.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-a",
        "--partial",
        "--append-verify",
        "--fsync",
        "--protect-args",
        "--info=progress2,stats2",
        "--files-from=-",
        "-e",
        rsync_rsh(args),
        f"{args.remote}:{args.remote_root}/packs/",
        f"{args.local_pack_root}/",
    ]
    result = subprocess.run(
        command,
        input="".join(f"{path}\n" for path in relative_paths),
        text=True,
        check=False,
    )
    if result.returncode:
        raise OffloadError(f"Pack rsync failed with exit code {result.returncode}")


def snapshot_remote_states(args: argparse.Namespace, run_id: str) -> str:
    root_literal = repr(str(args.remote_root))
    run_literal = repr(run_id)
    source = f"""
import glob, json, os, sqlite3
from pathlib import Path
root = Path({root_literal})
run_id = {run_literal}
destination = root / 'offload_snapshots' / run_id
if destination.exists():
    raise RuntimeError('snapshot already exists: ' + str(destination))
destination.mkdir(parents=True)
copied = []
for source_name in sorted(glob.glob(str(root / 'state' / 'camera_pack_state*.sqlite'))):
    source_path = Path(source_name)
    temporary = destination / ('.' + source_path.name + '.tmp')
    final = destination / source_path.name
    source_db = sqlite3.connect(f'file:{{source_path}}?mode=ro', uri=True, timeout=30)
    target_db = sqlite3.connect(temporary)
    try:
        source_db.backup(target_db)
        if target_db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('snapshot integrity failure: ' + source_path.name)
    finally:
        target_db.close()
        source_db.close()
    os.replace(temporary, final)
    copied.append(final.name)
print(json.dumps({{'relative_path': str(Path('offload_snapshots') / run_id), 'states': copied}}))
"""
    output = run_remote_python(args, source)
    return str(json.loads(output)["relative_path"])


def rsync_metadata(args: argparse.Namespace, run_id: str, remote_snapshot: str) -> None:
    local_snapshot = args.local_state_root / run_id
    local_snapshot.mkdir(parents=True, exist_ok=False)
    snapshot_command = [
        "rsync",
        "-a",
        "--fsync",
        "--protect-args",
        "-e",
        rsync_rsh(args),
        f"{args.remote}:{args.remote_root}/{remote_snapshot}/",
        f"{local_snapshot}/",
    ]
    reports_command = [
        "rsync",
        "-a",
        "--fsync",
        "--protect-args",
        "-e",
        rsync_rsh(args),
        f"{args.remote}:{args.remote_root}/reports/",
        f"{args.local_report_root}/",
    ]
    args.local_report_root.mkdir(parents=True, exist_ok=True)
    for command in (snapshot_command, reports_command):
        result = subprocess.run(command, check=False)
        if result.returncode:
            raise OffloadError(f"Metadata rsync failed with exit code {result.returncode}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def remote_hashes(
    args: argparse.Namespace,
    packs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root_literal = repr(str(args.remote_root))
    packs_literal = repr(
        [
            {
                "relative_path": item["relative_path"],
                "size": int(item["size"]),
                "mtime_ns": int(item["mtime_ns"]),
            }
            for item in packs
        ]
    )
    source = f"""
import hashlib, json, stat
from pathlib import Path
root = Path({root_literal}) / 'packs'
packs = {packs_literal}
result = []
for item in packs:
    path = root.joinpath(*Path(item['relative_path']).parts)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != item['size'] or metadata.st_mtime_ns != item['mtime_ns']:
        raise RuntimeError('pack changed before hash: ' + item['relative_path'])
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    result.append({{**item, 'sha256': digest.hexdigest()}})
print(json.dumps(result))
"""
    return list(json.loads(run_remote_python(args, source)))


def local_hashes(
    root: Path,
    packs: Sequence[Mapping[str, Any]],
    workers: int,
) -> dict[str, str]:
    def one(item: Mapping[str, Any]) -> tuple[str, str]:
        relative = validate_pack_relative_path(str(item["relative_path"]))
        path = root.joinpath(*PurePosixPath(relative).parts)
        resolved_root = root.resolve()
        try:
            path.resolve(strict=True).relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as exc:
            raise OffloadError(f"Local pack escapes its root: {relative}") from exc
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise OffloadError(f"Local pack is not a regular file: {relative}")
        if metadata.st_size != int(item["size"]):
            raise OffloadError(f"Local pack size mismatch: {relative}")
        return relative, sha256_file(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(one, packs))


def normalized_etag(value: str) -> str:
    normalized = str(value).strip()
    if normalized.startswith("W/"):
        normalized = normalized[2:].strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1]
    if not normalized:
        raise OffloadError("Source ETag is empty")
    return normalized


def open_target_state_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise OffloadError(f"Missing canonical target state: {path}")
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro", uri=True, timeout=120.0
    )
    connection.execute("PRAGMA query_only = ON")
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if not {"metadata", "target", "archive_progress"}.issubset(tables):
        connection.close()
        raise OffloadError(f"Canonical target state has an unexpected schema: {path}")
    return connection


def target_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    metadata = {
        str(key): str(value)
        for key, value in connection.execute(
            "SELECT key, value FROM metadata WHERE key IN (?, ?, ?)",
            TARGET_METADATA_KEYS,
        )
    }
    if set(metadata) != set(TARGET_METADATA_KEYS):
        raise OffloadError("Canonical target state lacks immutable inventory metadata")
    if int(metadata["inventory_row_count"]) != int(
        connection.execute("SELECT COUNT(*) FROM target").fetchone()[0]
    ):
        raise OffloadError("Canonical target count does not match its inventory metadata")
    return metadata


def validate_archive_storage(
    *,
    state: sqlite3.Connection,
    target: sqlite3.Connection,
    archive_key: str,
    relative_pack: str,
    pack_bytes: int,
    target_total: int,
    packed_target_count: int,
    existing_target_count: int,
) -> None:
    target_row = target.execute(
        "SELECT listed_size, remote_etag FROM archive_progress WHERE archive_key = ?",
        (archive_key,),
    ).fetchone()
    state_row = state.execute(
        "SELECT listed_size, remote_etag FROM archive_progress WHERE archive_key = ?",
        (archive_key,),
    ).fetchone()
    if target_row is None or state_row is None:
        raise OffloadError(f"Archive metadata is missing for {archive_key}")
    if int(target_row[0]) != int(state_row[0]) or normalized_etag(
        str(target_row[1])
    ) != normalized_etag(str(state_row[1])):
        raise OffloadError(f"Archive size/ETag contract mismatch for {archive_key}")

    target_rows = target.execute(
        "SELECT tar_member_name, destination_relative_path FROM target "
        "WHERE archive_key = ? ORDER BY tar_member_name",
        (archive_key,),
    )
    storage_rows = state.execute(
        "SELECT tar_member_name, destination_relative_path, storage_kind, "
        "storage_relative_path, pack_offset, payload_size, sha256, "
        "source_header_offset, source_data_offset FROM member_storage "
        "WHERE archive_key = ? ORDER BY tar_member_name",
        (archive_key,),
    )
    observed_total = 0
    observed_packed = 0
    observed_existing = 0
    for expected, stored in zip_longest(target_rows, storage_rows):
        if expected is None or stored is None or tuple(expected) != tuple(stored[:2]):
            raise OffloadError(f"Pack index does not exactly cover target ledger: {archive_key}")
        kind = str(stored[2])
        storage_relative = str(stored[3])
        pack_offset = stored[4]
        payload_size = int(stored[5])
        digest = str(stored[6])
        header_offset = int(stored[7])
        data_offset = int(stored[8])
        if (
            payload_size <= 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or header_offset < 0
            or data_offset != header_offset + 512
        ):
            raise OffloadError(f"Invalid indexed JPEG metadata for {archive_key}")
        if kind == "pack":
            if storage_relative != relative_pack or pack_offset is None:
                raise OffloadError(f"Invalid packed JPEG location for {archive_key}")
            observed_packed += 1
        elif kind == "existing":
            path = PurePosixPath(storage_relative)
            if (
                pack_offset is not None
                or path.is_absolute()
                or len(path.parts) != 3
                or any(part in {"", ".", ".."} for part in path.parts)
                or path.suffix.lower() != ".jpg"
            ):
                raise OffloadError(f"Invalid existing JPEG location for {archive_key}")
            observed_existing += 1
        else:
            raise OffloadError(f"Unexpected storage kind {kind!r} for {archive_key}")
        observed_total += 1

    if (
        observed_total != target_total
        or observed_packed != packed_target_count
        or observed_existing != existing_target_count
    ):
        raise OffloadError(f"Pack member counts do not match progress state: {archive_key}")

    expected_offset = 0
    for offset, payload_size in state.execute(
        "SELECT pack_offset, payload_size FROM member_storage "
        "WHERE archive_key = ? AND storage_kind = 'pack' ORDER BY pack_offset",
        (archive_key,),
    ):
        if int(offset) != expected_offset:
            raise OffloadError(f"Pack has a gap or overlap at byte {expected_offset}: {relative_pack}")
        expected_offset += int(payload_size)
    if expected_offset != pack_bytes:
        raise OffloadError(f"Pack byte coverage does not match file size: {relative_pack}")


def validate_snapshot_states(
    args: argparse.Namespace,
    run_id: str,
    packs: Sequence[Mapping[str, Any]],
) -> None:
    expected = {
        validate_pack_relative_path(str(item["relative_path"])): int(item["size"])
        for item in packs
    }
    observed: dict[str, int] = {}
    state_paths = sorted((args.local_state_root / run_id).glob("camera_pack_state*.sqlite"))
    if not state_paths:
        raise OffloadError("No SQLite state snapshots were transferred")
    target = open_target_state_readonly(args.target_state_sqlite)
    try:
        immutable_metadata = target_metadata(target)
        for state_path in state_paths:
            connection = sqlite3.connect(state_path)
            try:
                if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise OffloadError(f"SQLite snapshot integrity failed: {state_path}")
                state_metadata = {
                    str(key): str(value)
                    for key, value in connection.execute(
                        "SELECT key, value FROM metadata WHERE key IN (?, ?, ?)",
                        TARGET_METADATA_KEYS,
                    )
                }
                if state_metadata != immutable_metadata:
                    raise OffloadError(f"SQLite snapshot targets a different inventory: {state_path}")
                rows = connection.execute(
                    "SELECT archive_key, pack_relative_path, pack_bytes, status, "
                    "target_total, target_done, packed_target_count, existing_target_count "
                    "FROM archive_progress"
                )
                for (
                    archive_key,
                    relative,
                    pack_bytes,
                    status,
                    target_total,
                    target_done,
                    packed_target_count,
                    existing_target_count,
                ) in rows:
                    relative = str(relative)
                    if relative not in expected:
                        continue
                    size = int(pack_bytes)
                    if (
                        status != "complete"
                        or size != expected[relative]
                        or int(target_total) <= 0
                        or int(target_done) != int(target_total)
                        or int(packed_target_count) + int(existing_target_count)
                        != int(target_done)
                    ):
                        raise OffloadError(f"Incomplete or inconsistent snapshot state: {relative}")
                    if relative in observed:
                        raise OffloadError(f"Pack appears in multiple snapshot states: {relative}")
                    validate_archive_storage(
                        state=connection,
                        target=target,
                        archive_key=str(archive_key),
                        relative_pack=relative,
                        pack_bytes=size,
                        target_total=int(target_total),
                        packed_target_count=int(packed_target_count),
                        existing_target_count=int(existing_target_count),
                    )
                    observed[relative] = size
            finally:
                connection.close()
    finally:
        target.close()
    if observed != expected:
        missing = sorted(set(expected).difference(observed))[:3]
        raise OffloadError(f"SQLite snapshots do not cover copied packs: {missing}")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def delete_remote_verified(
    args: argparse.Namespace,
    verified: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root_literal = repr(str(args.remote_root))
    verified_literal = repr(
        [
            {
                "relative_path": item["relative_path"],
                "size": int(item["size"]),
                "mtime_ns": int(item["mtime_ns"]),
            }
            for item in verified
        ]
    )
    source = f"""
import json, re, stat
from pathlib import Path, PurePosixPath
root = Path({root_literal}) / 'packs'
verified = {verified_literal}
pattern = re.compile(r'^(train_set/nuplan-v1\\.1_train_camera_\\d+|val_set/nuplan-v1\\.1_val_camera_\\d+)\\.pack$')
resolved_root = root.resolve()
validated = []
for item in verified:
    relative = str(PurePosixPath(item['relative_path']))
    if pattern.fullmatch(relative) is None or '..' in PurePosixPath(relative).parts:
        raise RuntimeError('unsafe deletion path: ' + relative)
    path = root.joinpath(*PurePosixPath(relative).parts)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != item['size'] or metadata.st_mtime_ns != item['mtime_ns']:
        raise RuntimeError('pack changed before deletion: ' + relative)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        raise RuntimeError('pack escaped deletion root: ' + relative)
    if resolved_path != path.absolute():
        raise RuntimeError('pack deletion path contains a symlink: ' + relative)
    validated.append((relative, path, metadata.st_size))
for _relative, path, _size in validated:
    path.unlink()
free = root.stat().st_dev
vfs = __import__('os').statvfs(root)
print(json.dumps({{
    'deleted': [relative for relative, _path, _size in validated],
    'deleted_bytes': sum(size for _relative, _path, size in validated),
    'free_bytes_after': vfs.f_bavail * vfs.f_frsize,
}}))
"""
    return dict(json.loads(run_remote_python(args, source)))


def offload_trigger_reason(
    *,
    force: bool,
    total_bytes: int,
    threshold: int,
    packing_complete: bool,
    relay_free: int,
    relay_min_free: int,
) -> str | None:
    """Return the independent condition authorizing a verified offload."""

    if force:
        return "force"
    if total_bytes >= threshold:
        return "size_threshold"
    if packing_complete:
        return "packing_complete"
    if relay_free < relay_min_free:
        return "relay_space_pressure"
    return None


def offload_once(args: argparse.Namespace) -> bool:
    inventory = remote_inventory(args)
    packs = list(inventory["packs"])
    total_bytes = int(inventory["total_bytes"])
    threshold = int(args.threshold_gb * 1_000_000_000)
    relay_min_free = int(args.relay_min_free_gb * 1_000_000_000)
    relay_free = int(inventory["free_bytes"])
    print(
        f"relay packs={len(packs)} bytes={total_bytes} "
        f"packer_running={inventory['packer_running']} "
        f"packing_complete={inventory['packing_complete']} "
        f"relay_free_bytes={relay_free}",
        flush=True,
    )
    if not packs:
        return False
    trigger_reason = offload_trigger_reason(
        force=bool(args.force),
        total_bytes=total_bytes,
        threshold=threshold,
        packing_complete=bool(inventory["packing_complete"]),
        relay_free=relay_free,
        relay_min_free=relay_min_free,
    )
    if trigger_reason is None:
        print(
            f"below threshold {threshold}, relay free space is above floor "
            f"{relay_min_free}, and packing is not marked complete; waiting",
            flush=True,
        )
        return False
    print(f"offload_trigger={trigger_reason}", flush=True)
    if args.dry_run:
        print("dry-run: offload would trigger", flush=True)
        return False

    relative_paths = [item["relative_path"] for item in packs]
    print("phase=rsync_packs start", flush=True)
    rsync_selected_packs(args, relative_paths)
    print("phase=rsync_packs complete", flush=True)
    run_id = utc_run_id()
    print(f"phase=snapshot_metadata start run_id={run_id}", flush=True)
    remote_snapshot = snapshot_remote_states(args, run_id)
    rsync_metadata(args, run_id, remote_snapshot)
    print("phase=snapshot_metadata complete", flush=True)
    print("phase=sha256_remote start", flush=True)
    remote_verified = remote_hashes(args, packs)
    print("phase=sha256_remote complete", flush=True)
    print("phase=sha256_local start", flush=True)
    local_verified = local_hashes(
        args.local_pack_root,
        remote_verified,
        workers=args.hash_workers,
    )
    for item in remote_verified:
        relative = item["relative_path"]
        if local_verified.get(relative) != item["sha256"]:
            raise OffloadError(f"SHA-256 mismatch after rsync: {relative}")
    print("phase=sha256_local complete", flush=True)
    validate_snapshot_states(args, run_id, remote_verified)
    print("phase=sqlite_validation complete", flush=True)

    manifest = {
        "format": MANIFEST_FORMAT,
        "generated_at_utc": run_id,
        "remote": args.remote,
        "remote_root": str(args.remote_root),
        "local_pack_root": str(args.local_pack_root),
        "pack_count": len(remote_verified),
        "pack_bytes": sum(int(item["size"]) for item in remote_verified),
        "trigger_reason": trigger_reason,
        "relay_free_bytes_before": relay_free,
        "packs": remote_verified,
        "state_snapshot": str(args.local_state_root / run_id),
        "verification": {
            "all_local_sizes_match": True,
            "all_sha256_match": True,
            "sqlite_integrity_check": "ok",
            "sqlite_complete_pack_coverage": True,
        },
    }
    manifest_path = args.local_manifest_root / f"offload_{run_id}.json"
    atomic_write_json(manifest_path, manifest)
    print(f"phase=remote_delete start manifest={manifest_path}", flush=True)
    deletion = delete_remote_verified(args, remote_verified)
    print(
        f"verified and deleted remote packs={len(deletion['deleted'])} "
        f"bytes={deletion['deleted_bytes']} manifest={manifest_path}",
        flush=True,
    )
    return True


def run(args: argparse.Namespace) -> int:
    if args.threshold_gb <= 0:
        raise OffloadError("--threshold-gb must be positive")
    if args.poll_seconds < 10:
        raise OffloadError("--poll-seconds must be at least 10")
    if args.hash_workers < 1:
        raise OffloadError("--hash-workers must be positive")
    if args.relay_min_free_gb <= 0:
        raise OffloadError("--relay-min-free-gb must be positive")
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_file.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OffloadError(f"Another offloader owns {args.lock_file}") from exc
        while True:
            offload_once(args)
            if not args.watch:
                return 0
            time.sleep(args.poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (OffloadError, OSError, sqlite3.Error, subprocess.SubprocessError) as exc:
        print(f"offload failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
