#!/usr/bin/env python3
"""Run the fast single-machine navtrain native-10 Hz recreation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPECTED_FILTER_SHA256 = (
    "5f3c406a06c961e75d69d33bc127acd5704a3a440a7186bfa2212493ecf9fc01"
)


class LocalWorkflowError(RuntimeError):
    """Raised when configuration, prerequisites, or a workflow stage fails."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "all", "plan", "pack", "finalize", "status"))
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LocalWorkflowError(f"Configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise LocalWorkflowError("Configuration root must be a mapping")
    return value


def required_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LocalWorkflowError(f"Configuration requires a non-empty {key!r}")
    return Path(value).expanduser().resolve()


def paths(config: dict[str, Any]) -> dict[str, Path]:
    work = required_path(config, "work_root")
    navsim = required_path(config, "navsim_root")
    scene_filter = Path(
        config.get("scene_filter", REPO_ROOT / "reference/navtrain.yaml")
    ).expanduser().resolve()
    existing = Path(
        config.get("existing_camera_root", navsim / "sensor_blobs/trainval")
    ).expanduser().resolve()
    return {
        "work": work,
        "navsim": navsim,
        "scene_filter": scene_filter,
        "existing": existing,
        "native": work / "native_nuplan",
        "state": work / "state",
        "manifests": work / "manifests",
        "packs": work / "camera_packs",
        "backfills": work / "camera_existing_backfills",
        "visual": work / "visual_audit",
        "index": work / "manifests/navtrain_camera_10hz_index.sqlite",
        "inventory": work / "manifests/navtrain_camera_10hz_inventory.jsonl",
        "plan_report": work / "manifests/navtrain_camera_10hz_plan_report.json",
        "target_state": work / "state/camera_target_state.sqlite",
        "target_report": work / "manifests/camera_target_initialization.json",
        "pack_state": work / "state/camera_pack_state.sqlite",
        "pack_report": work / "manifests/camera_pack_report.json",
        "storage": work / "manifests/navtrain_camera_10hz_storage.sqlite",
    }


def command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts" / script), *(str(value) for value in arguments)]


def run_checked(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    completed = subprocess.run(arguments, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise LocalWorkflowError(
            f"Stage failed with exit code {completed.returncode}: {arguments[1]}"
        )


def doctor(config: dict[str, Any], p: dict[str, Path]) -> None:
    import hashlib

    errors = []
    if not p["navsim"].is_dir():
        errors.append(f"NAVSIM root is missing: {p['navsim']}")
    for name in ("navsim_logs/trainval", "sensor_blobs/trainval"):
        candidate = p["navsim"] / name
        if not candidate.is_dir():
            errors.append(f"Required NAVSIM directory is missing: {candidate}")
    if not p["scene_filter"].is_file():
        errors.append(f"Scene filter is missing: {p['scene_filter']}")
    else:
        observed = hashlib.sha256(p["scene_filter"].read_bytes()).hexdigest()
        if observed != EXPECTED_FILTER_SHA256:
            errors.append(
                f"navtrain.yaml SHA-256 mismatch: {observed} != {EXPECTED_FILTER_SHA256}"
            )
    for module in ("requests", "yaml", "PIL", "numpy"):
        try:
            __import__(module)
        except ImportError:
            errors.append(f"Python dependency is unavailable: {module}")
    free = shutil.disk_usage(p["work"].parent if p["work"].parent.exists() else Path("/"))
    minimum = float(config.get("minimum_free_gb", 1000.0)) * 1_000_000_000
    if free.free < minimum:
        errors.append(
            f"Free space is below configured minimum: {free.free / 1e9:.1f} GB < {minimum / 1e9:.1f} GB"
        )
    report = {
        "status": "fail" if errors else "pass",
        "workflow": "single_machine_fast_selective_range_packing",
        "navsim_root": str(p["navsim"]),
        "work_root": str(p["work"]),
        "free_gb": round(free.free / 1e9, 3),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise LocalWorkflowError("Doctor checks failed")


def prepare_directories(p: dict[str, Path]) -> None:
    for key in ("state", "manifests", "packs", "backfills", "visual", "native"):
        p[key].mkdir(parents=True, exist_ok=True)


def plan(config: dict[str, Any], p: dict[str, Path]) -> None:
    prepare_directories(p)
    if bool(config.get("download_native_databases", True)):
        run_checked(command(
            "download_navsim_nuplan10hz.py",
            "--scene-filter-yaml", p["scene_filter"],
            "--nuplan-root", p["native"],
            "--report-json", p["manifests"] / "native_db_download_report.json",
            "--workers", int(config.get("metadata_workers", 8)),
        ))
    run_checked(command(
        "plan_navsim_nuplan10hz.py",
        "--navsim-root", p["navsim"],
        "--scene-filter-yaml", p["scene_filter"],
        "--nuplan-root", p["native"],
        "--index-sqlite", p["index"],
        "--inventory-jsonl", p["inventory"],
        "--report-json", p["plan_report"],
    ))
    run_checked(command(
        "download_navsim_nuplan10hz_cameras.py",
        "--operation", "initialize",
        "--inventory-jsonl", p["inventory"],
        "--state-sqlite", p["target_state"],
        "--report-json", p["target_report"],
        "--workers", 1,
    ))


def pack(config: dict[str, Any], p: dict[str, Path]) -> None:
    prepare_directories(p)
    if not p["target_state"].is_file():
        raise LocalWorkflowError("Target state is missing; run the plan stage first")
    workers = int(config.get("pack_workers", 32))
    arguments = command(
        "pack_navsim_nuplan10hz_cameras.py",
        "--target-state-sqlite", p["target_state"],
        "--pack-state-sqlite", p["pack_state"],
        "--pack-root", p["packs"],
        "--existing-camera-root", p["existing"],
        "--workers", workers,
        "--checkpoint-targets", int(config.get("checkpoint_targets", 500)),
        "--report-json", p["pack_report"],
    )
    archive_base_url = config.get("archive_base_url")
    if archive_base_url:
        arguments.extend(("--archive-base-url", str(archive_base_url)))
    run_checked(arguments)


def pack_summary(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"archives": 0, "complete": 0, "targets": 0, "done": 0}
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(status='complete'),0), "
            "COALESCE(SUM(target_total),0), COALESCE(SUM(target_done),0) "
            "FROM archive_progress"
        ).fetchone()
        return {"archives": int(row[0]), "complete": int(row[1]), "targets": int(row[2]), "done": int(row[3])}
    finally:
        connection.close()


def finalize(config: dict[str, Any], p: dict[str, Path]) -> None:
    summary = pack_summary(p["pack_state"])
    if summary["archives"] != 55 or summary["complete"] != 55 or summary["targets"] != summary["done"]:
        raise LocalWorkflowError(f"Packing is incomplete: {summary}")
    samples = int(config.get("visual_audit_samples", 6))
    seed = int(config.get("visual_audit_seed", 20260814))
    run_checked(command(
        "render_navsim_nuplan10hz_visual_audit.py",
        "--index", p["index"],
        "--source-state", p["pack_state"],
        "--pack-root", p["packs"],
        "--existing-root", p["existing"],
        "--output-dir", p["visual"],
        "--samples", samples,
        "--seed", seed,
        "--camera-channel", str(config.get("visual_audit_camera", "CAM_F0")),
    ))
    run_checked(command(
        "build_navsim_nuplan10hz_storage_index.py",
        "--target-state", p["target_state"],
        "--source-state", p["pack_state"],
        "--existing-policy", "trust-official-navsim",
        "--visual-audit-manifest", p["visual"] / "manifest.json",
        "--output", p["storage"],
        "--pack-root", p["packs"],
        "--supplemental-pack-root", p["backfills"],
        "--inventory-json", p["inventory"],
    ))
    smoke_test(p)


def smoke_test(p: dict[str, Path]) -> None:
    from navtrain10hz.loader import Navsim10HzImageLoader

    manifest = json.loads((p["visual"] / "manifest.json").read_text())
    token = str(manifest["panels"][0]["navsim_anchor_token"])
    with Navsim10HzImageLoader.from_index_path(
        p["index"],
        p["navsim"],
        storage_index_path=p["storage"],
        pack_root=p["packs"],
        supplemental_pack_root=p["backfills"],
        existing_camera_root=p["existing"],
    ) as loader:
        loaded = loader.load_anchor(token)
        if len(loaded.camera_histories) != 8:
            raise LocalWorkflowError("Smoke test did not decode all eight cameras")
    print(json.dumps({"status": "pass", "smoke_test_anchor": token}, indent=2))


def status(p: dict[str, Path]) -> None:
    result: dict[str, Any] = {"work_root": str(p["work"]), "packing": pack_summary(p["pack_state"])}
    for name in ("index", "inventory", "target_state", "pack_state", "storage"):
        result[name] = {"path": str(p[name]), "exists": p[name].is_file(), "bytes": p[name].stat().st_size if p[name].is_file() else 0}
    print(json.dumps(result, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        p = paths(config)
        if args.command == "doctor": doctor(config, p)
        elif args.command == "plan": plan(config, p)
        elif args.command == "pack": pack(config, p)
        elif args.command == "finalize": finalize(config, p)
        elif args.command == "status": status(p)
        elif args.command == "all":
            doctor(config, p)
            plan(config, p)
            pack(config, p)
            finalize(config, p)
        return 0
    except (LocalWorkflowError, OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
