#!/usr/bin/env python3
"""Deprecated compatibility entry point for the NAVSIM native-camera planner.

Use ``plan_navsim_nuplan10hz.py`` directly.  The canonical output is its
normalized SQLite index.  This wrapper exists only for callers that still
need an extractor-compatible JSONL inventory; it delegates all provenance and
timing decisions to the canonical planner and also writes its sibling SQLite
index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import plan_navsim_nuplan10hz as canonical


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navsim-root", type=Path, default=canonical.lineage.DEFAULT_NAVSIM_ROOT)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--scene-filter-yaml", type=Path, default=canonical.lineage.DEFAULT_FILTER)
    parser.add_argument("--nuplan-root", type=Path, required=True)
    parser.add_argument("--nuplan-split", choices=("trainval",), default="trainval")
    parser.add_argument("--channels", default=",".join(canonical.lineage.NUPLAN_CAMERA_CHANNELS))
    parser.add_argument("--retained-camera-tolerance-us", type=int, default=50_000)
    parser.add_argument("--camera-cadence-tolerance-us", type=int, default=50_000)
    parser.add_argument("--navsim-camera-cadence-tolerance-us", type=int, default=50_000)
    parser.add_argument("--anchor-token", action="append", default=[])
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--allow-nonofficial-filter", action="store_true")
    parser.add_argument(
        "--history-seconds",
        type=float,
        default=None,
        help="Removed: canonical navtrain timing uses matched retained camera endpoints.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true", help="The only supported mode; retained for compatibility.")
    mode.add_argument(
        "--require-native-jpegs",
        action="store_true",
        help="Removed: JPEG checks/extraction are intentionally a later stage.",
    )
    parser.add_argument("--plan-jsonl", type=Path, required=True)
    parser.add_argument(
        "--index-sqlite",
        type=Path,
        default=None,
        help="Canonical index destination; defaults beside --plan-jsonl.",
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.history_seconds is not None:
        print(
            "error: --history-seconds is no longer valid; canonical navtrain windows are bounded by "
            "the matched first/current retained native camera images",
            file=sys.stderr,
        )
        return 2
    if args.require_native_jpegs:
        print(
            "error: --require-native-jpegs is not supported by the metadata-first planner; "
            "extract JPEGs only after the canonical index is verified",
            file=sys.stderr,
        )
        return 2
    if args.plan_jsonl.suffix != ".jsonl":
        print("error: --plan-jsonl must have a .jsonl suffix", file=sys.stderr)
        return 2
    index = args.index_sqlite or args.plan_jsonl.with_suffix(".sqlite")
    report = args.report_json or args.plan_jsonl.with_name(f"{args.plan_jsonl.stem}_report.json")
    forwarded = [
        "--navsim-root",
        str(args.navsim_root),
        "--scene-filter-yaml",
        str(args.scene_filter_yaml),
        "--nuplan-root",
        str(args.nuplan_root),
        "--nuplan-split",
        args.nuplan_split,
        "--channels",
        args.channels,
        "--retained-camera-tolerance-us",
        str(args.retained_camera_tolerance_us),
        "--camera-cadence-tolerance-us",
        str(args.camera_cadence_tolerance_us),
        "--navsim-camera-cadence-tolerance-us",
        str(args.navsim_camera_cadence_tolerance_us),
        "--max-anchors",
        str(args.max_anchors),
        "--index-sqlite",
        str(index),
        "--inventory-jsonl",
        str(args.plan_jsonl),
        "--report-json",
        str(report),
    ]
    if args.log_path is not None:
        forwarded.extend(("--log-path", str(args.log_path)))
    if args.allow_nonofficial_filter:
        forwarded.append("--allow-nonofficial-filter")
    for token in args.anchor_token:
        forwarded.extend(("--anchor-token", str(token)))
    if args.dry_run:
        forwarded.append("--dry-run")
    print(
        "DEPRECATED: plan_navsim_nuplan10hz_images.py delegates to the canonical SQLite index planner.",
        file=sys.stderr,
    )
    return canonical.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
