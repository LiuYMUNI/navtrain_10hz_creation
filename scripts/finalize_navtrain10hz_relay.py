#!/usr/bin/env python3
"""Finalize verified relay packs into the same storage contract as local mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RelayFinalizeError(RuntimeError):
    """Raised when no complete verified relay snapshot can be finalized."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-state", type=Path, required=True)
    parser.add_argument("--history-index", type=Path, required=True)
    parser.add_argument("--inventory-jsonl", type=Path, required=True)
    parser.add_argument("--relay-state-root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--existing-camera-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args(argv)


def complete_snapshot(root: Path) -> list[Path]:
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()), reverse=True
    )
    for directory in candidates:
        states = sorted(directory.glob("camera_pack_state*.sqlite"))
        if not states:
            continue
        keys: set[str] = set()
        valid = True
        for state in states:
            connection = sqlite3.connect(f"file:{state.resolve()}?mode=ro", uri=True)
            try:
                rows = connection.execute(
                    "SELECT archive_key,status,target_total,target_done FROM archive_progress"
                ).fetchall()
            except sqlite3.Error:
                valid = False
                rows = []
            finally:
                connection.close()
            for key, status, total, done in rows:
                if str(key) in keys or str(status) != "complete" or int(total) != int(done):
                    valid = False
                keys.add(str(key))
        if valid and len(keys) == 55:
            return states
    raise RelayFinalizeError("No relay-state snapshot covers 55 unique complete archives")


def run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    completed = subprocess.run(arguments, cwd=REPO_ROOT, check=False)
    if completed.returncode:
        raise RelayFinalizeError(f"Command failed with exit code {completed.returncode}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        states = complete_snapshot(args.relay_state_root)
        output = args.output_root.resolve()
        visual = output / "visual_audit"
        storage = output / "navtrain_camera_10hz_storage.sqlite"
        backfills = output / "camera_existing_backfills"
        visual.mkdir(parents=True, exist_ok=True)
        backfills.mkdir(parents=True, exist_ok=True)
        visual_command = [
            sys.executable,
            str(REPO_ROOT / "scripts/render_navsim_nuplan10hz_visual_audit.py"),
            "--index", str(args.history_index),
            "--pack-root", str(args.pack_root),
            "--existing-root", str(args.existing_camera_root),
            "--output-dir", str(visual),
            "--samples", str(args.samples),
            "--seed", str(args.seed),
        ]
        for state in states:
            visual_command.extend(("--source-state", str(state)))
        run(visual_command)
        build_command = [
            sys.executable,
            str(REPO_ROOT / "scripts/build_navsim_nuplan10hz_storage_index.py"),
            "--target-state", str(args.target_state),
            "--existing-policy", "trust-official-navsim",
            "--visual-audit-manifest", str(visual / "manifest.json"),
            "--output", str(storage),
            "--pack-root", str(args.pack_root),
            "--supplemental-pack-root", str(backfills),
            "--inventory-json", str(args.inventory_jsonl),
        ]
        for state in states:
            build_command.extend(("--source-state", str(state)))
        run(build_command)
        print(json.dumps({
            "status": "pass",
            "source_snapshot": str(states[0].parent),
            "source_states": [str(path) for path in states],
            "storage_index": str(storage),
            "visual_audit": str(visual / "overview.png"),
        }, indent=2))
        return 0
    except (RelayFinalizeError, OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
