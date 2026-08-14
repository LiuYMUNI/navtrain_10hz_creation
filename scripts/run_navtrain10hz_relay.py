#!/usr/bin/env python3
"""Run generic resumable camera-pack batches on a network relay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLETE_FORMAT = "navtrain10hz_relay_complete_v1"


class RelayWorkflowError(RuntimeError):
    """Raised when relay configuration or immutable batch state is invalid."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise RelayWorkflowError("Relay configuration root must be a mapping")
    return value


def path_value(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise RelayWorkflowError(f"Relay configuration requires {key!r}")
    return Path(value).expanduser().resolve()


def read_targets(path: Path) -> tuple[dict[str, str], list[str]]:
    if not path.is_file():
        raise RelayWorkflowError(f"Target ledger is missing: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RelayWorkflowError("Target-ledger integrity check failed")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        required = {"inventory_sha256", "inventory_row_count", "navsim_scene_filter_sha256"}
        if not required.issubset(metadata):
            raise RelayWorkflowError("Target ledger lacks immutable metadata")
        keys = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT archive_key FROM target ORDER BY archive_key"
            )
        ]
        count = int(connection.execute("SELECT COUNT(*) FROM target").fetchone()[0])
        if count != int(metadata["inventory_row_count"]):
            raise RelayWorkflowError("Target-ledger row count does not match metadata")
        if len(keys) != 55:
            raise RelayWorkflowError(f"Expected 55 camera archives, found {len(keys)}")
        return ({key: str(metadata[key]) for key in required}, keys)
    finally:
        connection.close()


def batches(keys: list[str], size: int) -> list[list[str]]:
    if size < 1 or size > 55:
        raise RelayWorkflowError("archives_per_batch must be between 1 and 55")
    return [keys[start : start + size] for start in range(0, len(keys), size)]


def state_status(path: Path, expected: Sequence[str]) -> dict[str, str]:
    if not path.is_file():
        return {key: "missing" for key in expected}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = {
            str(key): str(status)
            for key, status in connection.execute(
                "SELECT archive_key,status FROM archive_progress"
            )
        }
    finally:
        connection.close()
    unexpected = set(rows) - set(expected)
    if unexpected:
        raise RelayWorkflowError(
            f"Batch state contains unexpected archives: {sorted(unexpected)}"
        )
    return {key: rows.get(key, "missing") for key in expected}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def wait_for_space(root: Path, minimum_gb: float, seconds: int, dry_run: bool) -> None:
    minimum = int(minimum_gb * 1_000_000_000)
    while shutil.disk_usage(root).free < minimum:
        free = shutil.disk_usage(root).free
        if dry_run:
            raise RelayWorkflowError(
                f"Relay has {free / 1e9:.1f} GB free; needs {minimum_gb:.1f} GB"
            )
        print(
            json.dumps({
                "event": "waiting_for_space",
                "free_gb": round(free / 1e9, 3),
                "required_gb": minimum_gb,
            }),
            flush=True,
        )
        time.sleep(seconds)


def run_batch(
    *,
    config: dict[str, Any],
    target: Path,
    existing: Path,
    work: Path,
    number: int,
    keys: list[str],
    dry_run: bool,
) -> Path:
    state = work / "state" / f"camera_pack_state_{number:03d}.sqlite"
    report = work / "reports" / f"camera_pack_report_{number:03d}.json"
    log = work / "logs" / f"camera_pack_{number:03d}.log"
    status = state_status(state, keys)
    incomplete = [key for key in keys if status[key] != "complete"]
    if not incomplete:
        return state
    has_committed_state = any(value != "missing" for value in status.values())
    minimum = float(
        config.get("resume_min_free_gb", 100.0)
        if has_committed_state
        else config.get("min_free_gb", 500.0)
    )
    wait_for_space(work, minimum, int(config.get("poll_seconds", 300)), dry_run)
    arguments = [
        sys.executable,
        str(REPO_ROOT / "scripts/pack_navsim_nuplan10hz_cameras.py"),
        "--target-state-sqlite", str(target),
        "--pack-state-sqlite", str(state),
        "--pack-root", str(work / "packs"),
        "--existing-camera-root", str(existing),
        "--workers", str(min(int(config.get("workers", 16)), len(incomplete))),
        "--checkpoint-targets", str(int(config.get("checkpoint_targets", 500))),
        "--report-json", str(report),
    ]
    if config.get("archive_base_url"):
        arguments.extend(("--archive-base-url", str(config["archive_base_url"])))
    for key in incomplete:
        arguments.extend(("--archive-key", key))
    print("+", " ".join(arguments), flush=True)
    if dry_run:
        return state
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        completed = subprocess.run(
            arguments,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RelayWorkflowError(
            f"Batch {number:03d} failed; resume with the same command after inspecting {log}"
        )
    final = state_status(state, keys)
    if any(value != "complete" for value in final.values()):
        raise RelayWorkflowError(f"Batch {number:03d} exited without complete state")
    return state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        work = path_value(config, "work_root")
        target = path_value(config, "target_state")
        existing = path_value(config, "existing_camera_root")
        if not existing.is_dir():
            raise RelayWorkflowError(f"Existing NAVSIM camera root is missing: {existing}")
        for name in ("state", "reports", "logs", "packs"):
            (work / name).mkdir(parents=True, exist_ok=True)
        metadata, archive_keys = read_targets(target)
        groups = batches(archive_keys, int(config.get("archives_per_batch", 8)))
        state_paths = []
        for number, keys in enumerate(groups):
            state_paths.append(run_batch(
                config=config,
                target=target,
                existing=existing,
                work=work,
                number=number,
                keys=keys,
                dry_run=args.dry_run,
            ))
        if args.dry_run:
            print(json.dumps({"status": "dry_run", "batches": len(groups), "archives": 55}, indent=2))
            return 0
        marker = {
            "format": COMPLETE_FORMAT,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "archive_count": 55,
            "batch_count": len(groups),
            "target_metadata": metadata,
            "target_state_sha256": sha256_file(target),
            "state_files": [str(path.relative_to(work)) for path in state_paths],
        }
        atomic_json(work / "state/navtrain10hz_packing_complete.json", marker)
        print(json.dumps({"status": "pass", **marker}, indent=2))
        return 0
    except (RelayWorkflowError, OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
