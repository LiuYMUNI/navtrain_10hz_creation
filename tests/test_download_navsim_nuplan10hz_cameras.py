"""Tests for the persistent native-camera download ledger."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "download_navsim_nuplan10hz_cameras.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("download_navsim_nuplan10hz_cameras", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CameraDownloadStateTest(unittest.TestCase):
    def test_archive_selection_is_exact_and_fails_on_unknown_keys(self) -> None:
        archives = (
            MODULE.ArchiveSpec(
                key="public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_camera_7.zip",
                url="https://example.invalid/train-7.zip",
                source_split="train",
                group=7,
                listed_size=1,
                listed_etag="a",
            ),
            MODULE.ArchiveSpec(
                key="public/nuplan-v1.1/sensor_blobs/val_set/nuplan-v1.1_val_camera_3.zip",
                url="https://example.invalid/val-3.zip",
                source_split="val",
                group=3,
                listed_size=1,
                listed_etag="b",
            ),
        )
        selected = MODULE.select_archives(archives, [archives[1].key])
        self.assertEqual(selected, (archives[1],))
        with self.assertRaisesRegex(MODULE.CameraDownloadError, "not referenced"):
            MODULE.select_archives(archives, ["public/nuplan-v1.1/sensor_blobs/train_set/unknown.zip"])

    def test_initialized_ledger_reuses_immutable_inventory_without_reparsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inventory = root / "inventory.jsonl"
            train_groups = root / "train.txt"
            val_groups = root / "val.txt"
            state = root / "state.sqlite"
            train_groups.write_text("File group: 7\nlog_train\n", encoding="utf-8")
            val_groups.write_text("File group: 3\nlog_val\n", encoding="utf-8")
            scene_filter_sha256 = hashlib.sha256(b"fixture-filter").hexdigest()
            inventory.write_text(
                "".join(
                    json.dumps(
                        {
                            "format": MODULE.INVENTORY_FORMAT,
                            "navsim_scene_filter_sha256": scene_filter_sha256,
                            "navsim_log_name": log_name,
                            "camera_channel": "CAM_F0",
                            "native_image_filename_jpg": f"{log_name}/CAM_F0/{image_name}",
                            "native_blob_relative_path": f"sensor_blobs/{log_name}/CAM_F0/{image_name}",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                    for log_name, image_name in (("log_train", "one.jpg"), ("log_val", "two.jpg"))
                ),
                encoding="utf-8",
            )
            train = MODULE.parse_group_manifest(train_groups, split="train")
            val = MODULE.parse_group_manifest(val_groups, split="val")
            connection = MODULE.open_state(state)
            try:
                count, groups, inventory_sha256 = MODULE.initialize_targets(
                    connection,
                    inventory=inventory,
                    train_groups=train,
                    val_groups=val,
                )
                self.assertEqual(count, 2)
                self.assertEqual(groups, {("train", 7), ("val", 3)})
                self.assertEqual(
                    inventory_sha256,
                    hashlib.sha256(inventory.read_bytes()).hexdigest(),
                )
            finally:
                connection.close()

            connection = MODULE.open_state(state)
            try:
                with mock.patch.object(MODULE.json, "loads", side_effect=AssertionError("must not reparse inventory")):
                    count, groups, _inventory_sha256 = MODULE.initialize_targets(
                        connection,
                        inventory=inventory,
                        train_groups=train,
                        val_groups=val,
                    )
                self.assertEqual(count, 2)
                self.assertEqual(groups, {("train", 7), ("val", 3)})
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM target").fetchone()[0],
                    2,
                )
            finally:
                connection.close()

    def test_checkpoint_commits_staged_targets_with_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = root / "state.sqlite"
            archive_key = "public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_camera_7.zip"
            member_name = "nuplan-v1.1_train_camera_7/log_train/CAM_F0/one.jpg"
            spec = MODULE.ArchiveSpec(
                key=archive_key,
                url="https://example.invalid/archive.zip",
                source_split="train",
                group=7,
                listed_size=123,
                listed_etag="fixture-etag",
            )
            connection = MODULE.open_state(state)
            try:
                connection.execute(
                    "INSERT INTO target (archive_key, tar_member_name, destination_relative_path, log_name, camera_channel) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (archive_key, member_name, "sensor_blobs/log_train/CAM_F0/one.jpg", "log_train", "CAM_F0"),
                )
                connection.execute(
                    "INSERT INTO archive_progress "
                    "(archive_key, url, listed_size, remote_etag, next_header_offset, headers_scanned, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (archive_key, spec.url, spec.listed_size, spec.listed_etag, 0, 0, "running"),
                )
                connection.commit()
                MODULE.checkpoint_archive(
                    connection,
                    spec,
                    next_offset=4096,
                    headers_scanned=12,
                    status="partial",
                    staged_updates=[(42, "a" * 64, member_name)],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT status, tar_member_size, sha256 FROM target "
                        "WHERE archive_key = ? AND tar_member_name = ?",
                        (archive_key, member_name),
                    ).fetchone(),
                    ("staged", 42, "a" * 64),
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT next_header_offset, headers_scanned, status FROM archive_progress "
                        "WHERE archive_key = ?",
                        (archive_key,),
                    ).fetchone(),
                    (4096, 12, "partial"),
                )
            finally:
                connection.close()

    def test_tar_offset_checkpoint_and_publish_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            state = root / "state.sqlite"
            destination = root / "navtrain_offsets.sqlite"
            archive_key = "public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_camera_7.zip"
            member_name = "nuplan-v1.1_train_camera_7/log_train/CAM_F0/one.jpg"
            spec = MODULE.ArchiveSpec(
                key=archive_key,
                url="https://example.invalid/archive.zip",
                source_split="train",
                group=7,
                listed_size=123,
                listed_etag="fixture-etag",
            )
            inventory_sha256 = hashlib.sha256(b"fixture-inventory").hexdigest()
            filter_sha256 = hashlib.sha256(b"fixture-filter").hexdigest()
            connection = MODULE.open_state(state)
            try:
                connection.executemany(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    [
                        ("inventory_sha256", inventory_sha256),
                        ("navsim_scene_filter_sha256", filter_sha256),
                    ],
                )
                connection.execute(
                    "INSERT INTO target (archive_key, tar_member_name, destination_relative_path, log_name, camera_channel) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (archive_key, member_name, "sensor_blobs/log_train/CAM_F0/one.jpg", "log_train", "CAM_F0"),
                )
                connection.execute(
                    "INSERT INTO tar_offset_progress "
                    "(archive_key, url, listed_size, remote_etag, next_header_offset, headers_scanned, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (archive_key, spec.url, spec.listed_size, spec.listed_etag, 0, 0, "running"),
                )
                connection.commit()
                MODULE.checkpoint_tar_offsets(
                    connection,
                    spec,
                    next_offset=4096,
                    headers_scanned=12,
                    status="complete",
                    offset_updates=[(1024, 1536, 42, member_name)],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT tar_header_offset, tar_data_offset, tar_member_size FROM target "
                        "WHERE archive_key = ? AND tar_member_name = ?",
                        (archive_key, member_name),
                    ).fetchone(),
                    (1024, 1536, 42),
                )
                summary = MODULE.tar_offset_summary(connection)
                self.assertTrue(MODULE.tar_offset_index_complete(summary))
                digest = MODULE.publish_tar_offset_index(
                    connection,
                    destination,
                    inventory_sha256=inventory_sha256,
                )
            finally:
                connection.close()

            self.assertTrue(destination.is_file())
            self.assertEqual(digest, hashlib.sha256(destination.read_bytes()).hexdigest())
            output = sqlite3.connect(destination)
            try:
                self.assertEqual(
                    output.execute("SELECT value FROM metadata WHERE key = 'format'").fetchone()[0],
                    MODULE.TAR_OFFSET_INDEX_FORMAT,
                )
                self.assertEqual(
                    output.execute(
                        "SELECT archive_key, tar_header_offset, tar_data_offset, tar_member_size "
                        "FROM member_offset"
                    ).fetchone(),
                    (archive_key, 1024, 1536, 42),
                )
                self.assertEqual(output.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            finally:
                output.close()


if __name__ == "__main__":
    unittest.main()
