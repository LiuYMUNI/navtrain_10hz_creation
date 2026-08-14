"""Safety tests for relay-to-HPC NAVSIM 10 Hz pack offloading."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "offload_navsim_nuplan10hz_relay.py"
SPEC = importlib.util.spec_from_file_location(
    "offload_navsim_nuplan10hz_relay", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RelayOffloadSafetyTest(unittest.TestCase):
    def test_pack_path_allowlist_rejects_escape_and_unexpected_files(self) -> None:
        self.assertEqual(
            MODULE.validate_pack_relative_path(
                "train_set/nuplan-v1.1_train_camera_42.pack"
            ),
            "train_set/nuplan-v1.1_train_camera_42.pack",
        )
        self.assertEqual(
            MODULE.validate_pack_relative_path(
                "val_set/nuplan-v1.1_val_camera_11.pack"
            ),
            "val_set/nuplan-v1.1_val_camera_11.pack",
        )
        for unsafe in (
            "../train_set/nuplan-v1.1_train_camera_0.pack",
            "/train_set/nuplan-v1.1_train_camera_0.pack",
            "train_set/nuplan-v1.1_train_camera_0.pack.partial",
            "train_set/unrelated.pack",
            "state/camera_pack_state.sqlite",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(MODULE.OffloadError):
                MODULE.validate_pack_relative_path(unsafe)

    def test_snapshot_validation_requires_complete_size_matching_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target_state = root / "target.sqlite"
            target = sqlite3.connect(target_state)
            try:
                target.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                target.execute(
                    "CREATE TABLE target (archive_key TEXT, tar_member_name TEXT, "
                    "destination_relative_path TEXT, PRIMARY KEY (archive_key, tar_member_name))"
                )
                target.execute(
                    "CREATE TABLE archive_progress (archive_key TEXT PRIMARY KEY, "
                    "listed_size INTEGER, remote_etag TEXT)"
                )
                target.executemany(
                    "INSERT INTO metadata VALUES (?, ?)",
                    [
                        ("inventory_sha256", "a" * 64),
                        ("inventory_row_count", "1"),
                        ("navsim_scene_filter_sha256", "b" * 64),
                    ],
                )
                target.execute(
                    "INSERT INTO target VALUES (?, ?, ?)",
                    ("archive-0", "root/log/CAM_F0/one.jpg", "sensor_blobs/log/CAM_F0/one.jpg"),
                )
                target.execute(
                    "INSERT INTO archive_progress VALUES (?, ?, ?)",
                    ("archive-0", 1000, '"etag"'),
                )
                target.commit()
            finally:
                target.close()

            state_root = root / "states"
            run_id = "fixture"
            snapshot = state_root / run_id
            snapshot.mkdir(parents=True)
            state = snapshot / "camera_pack_state.sqlite"
            connection = sqlite3.connect(state)
            try:
                connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
                connection.execute(
                    "CREATE TABLE archive_progress ("
                    "archive_key TEXT, pack_relative_path TEXT, pack_bytes INTEGER, "
                    "status TEXT, target_total INTEGER, target_done INTEGER, "
                    "packed_target_count INTEGER, existing_target_count INTEGER, "
                    "listed_size INTEGER, remote_etag TEXT)"
                )
                connection.execute(
                    "CREATE TABLE member_storage (archive_key TEXT, tar_member_name TEXT, "
                    "destination_relative_path TEXT, storage_kind TEXT, "
                    "storage_relative_path TEXT, pack_offset INTEGER, payload_size INTEGER, "
                    "sha256 TEXT, source_header_offset INTEGER, source_data_offset INTEGER)"
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES (?, ?)",
                    [
                        ("inventory_sha256", "a" * 64),
                        ("inventory_row_count", "1"),
                        ("navsim_scene_filter_sha256", "b" * 64),
                    ],
                )
                connection.execute(
                    "INSERT INTO archive_progress VALUES (?, ?, ?, 'complete', 1, 1, 1, 0, ?, ?)",
                    (
                        "archive-0",
                        "train_set/nuplan-v1.1_train_camera_0.pack",
                        123,
                        1000,
                        '"etag"',
                    ),
                )
                connection.execute(
                    "INSERT INTO member_storage VALUES (?, ?, ?, 'pack', ?, 0, 123, ?, 0, 512)",
                    (
                        "archive-0",
                        "root/log/CAM_F0/one.jpg",
                        "sensor_blobs/log/CAM_F0/one.jpg",
                        "train_set/nuplan-v1.1_train_camera_0.pack",
                        "c" * 64,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            args = type(
                "Args",
                (),
                {"local_state_root": state_root, "target_state_sqlite": target_state},
            )()
            packs = [
                {
                    "relative_path": "train_set/nuplan-v1.1_train_camera_0.pack",
                    "size": 123,
                }
            ]
            MODULE.validate_snapshot_states(args, run_id, packs)
            packs[0]["size"] = 124
            with self.assertRaises(MODULE.OffloadError):
                MODULE.validate_snapshot_states(args, run_id, packs)

    def test_low_relay_space_triggers_below_size_threshold(self) -> None:
        self.assertEqual(
            MODULE.offload_trigger_reason(
                force=False,
                total_bytes=86_000_000_000,
                threshold=200_000_000_000,
                packing_complete=False,
                relay_free=476_000_000_000,
                relay_min_free=500_000_000_000,
            ),
            "relay_space_pressure",
        )
        self.assertIsNone(
            MODULE.offload_trigger_reason(
                force=False,
                total_bytes=86_000_000_000,
                threshold=200_000_000_000,
                packing_complete=False,
                relay_free=501_000_000_000,
                relay_min_free=500_000_000_000,
            )
        )


if __name__ == "__main__":
    unittest.main()
