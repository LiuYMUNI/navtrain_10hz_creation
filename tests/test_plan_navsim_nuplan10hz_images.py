from __future__ import annotations

import json
import pickle
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "scripts" / "plan_navsim_nuplan10hz_images.py"
LOG_NAME = "2021.01.01.00.00.00_veh-01_00000_00099"
LOG_TOKEN = "00000000000000bb"
SCENE_TOKEN = "00000000000000aa"
ANCHOR_TOKENS = ("0000000000000001", "0000000000000002")
CHANNELS = ("CAM_F0", "CAM_R0")


def blob(token: str) -> bytes:
    return bytes.fromhex(token)


class NavsimNuPlan10HzImagePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.navsim_root = self.root / "navsim2"
        self.native_root = self.root / "nuplan-v1.1"
        self.filter_yaml = self.root / "navtrain_fixture.yaml"
        self.plan_path = self.root / "plans" / "native_10hz_images.jsonl"
        self.index_path = self.plan_path.with_suffix(".sqlite")
        self._build_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_fixture(self) -> None:
        self.filter_yaml.write_text(
            f"""_target_: navsim.common.dataclasses.SceneFilter
_convert_: all
num_history_frames: 4
num_future_frames: 10
frame_interval: 1
has_route: true
max_scenes: null
log_names:
  - '{LOG_NAME}'
tokens:
  - '{ANCHOR_TOKENS[0]}'
  - '{ANCHOR_TOKENS[1]}'
""",
            encoding="utf-8",
        )

        native_db = self.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
        native_db.parent.mkdir(parents=True)
        connection = sqlite3.connect(native_db)
        try:
            connection.executescript(
                """
                CREATE TABLE log (token BLOB, logfile TEXT);
                CREATE TABLE lidar_pc (token BLOB, timestamp INTEGER, scene_token BLOB, filename TEXT);
                CREATE TABLE camera (token BLOB, channel TEXT);
                CREATE TABLE image (token BLOB, timestamp INTEGER, filename_jpg TEXT, camera_token BLOB);
                """
            )
            connection.execute("INSERT INTO log VALUES (?, ?)", (blob(LOG_TOKEN), LOG_NAME))
            for channel_index, channel in enumerate(CHANNELS, start=1):
                camera_token = blob(f"{channel_index:016x}")
                connection.execute("INSERT INTO camera VALUES (?, ?)", (camera_token, channel))
                for image_index in range(21):
                    timestamp_us = 1_000_000 + image_index * 100_000
                    image_name = self._image_name(channel, image_index)
                    connection.execute(
                        "INSERT INTO image VALUES (?, ?, ?, ?)",
                        (
                            blob(f"{channel_index * 1000 + image_index + 1:016x}"),
                            timestamp_us,
                            f"{LOG_NAME}/{channel}/{image_name}",
                            camera_token,
                        ),
                    )

            for row_index, token in ((3, ANCHOR_TOKENS[0]), (4, ANCHOR_TOKENS[1])):
                connection.execute(
                    "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
                    (
                        blob(token),
                        1_000_000 + row_index * 500_000,
                        blob(SCENE_TOKEN),
                        f"{LOG_NAME}/MergedPointCloud/{token}.pcd",
                    ),
                )
            connection.commit()
        finally:
            connection.close()

        rows: list[dict[str, object]] = []
        for row_index in range(15):
            token = (
                ANCHOR_TOKENS[row_index - 3]
                if row_index in (3, 4)
                else f"{100 + row_index:016x}"
            )
            row: dict[str, object] = {
                "token": token,
                "timestamp": 1_000_000 + row_index * 500_000,
                "log_name": LOG_NAME,
                "log_token": LOG_TOKEN,
                "scene_token": SCENE_TOKEN,
                "lidar_path": f"{LOG_NAME}/MergedPointCloud/{token}.pcd",
                "cams": {},
            }
            if row_index <= 4:
                image_index = (int(row["timestamp"]) - 1_000_000) // 100_000
                row["cams"] = {
                    channel: {
                        "data_path": f"{LOG_NAME}/{channel}/{self._image_name(channel, image_index)}"
                    }
                    for channel in CHANNELS
                }
            rows.append(row)
        log_path = self.navsim_root / "navsim_logs" / "trainval" / f"{LOG_NAME}.pkl"
        log_path.parent.mkdir(parents=True)
        with log_path.open("wb") as handle:
            pickle.dump(rows, handle)

    @staticmethod
    def _image_name(channel: str, image_index: int) -> str:
        return f"{channel.lower()}_{image_index:03d}.jpg"

    def run_planner(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PLANNER),
                "--navsim-root",
                str(self.navsim_root),
                "--scene-filter-yaml",
                str(self.filter_yaml),
                "--nuplan-root",
                str(self.native_root),
                "--channels",
                ",".join(CHANNELS),
                "--allow-nonofficial-filter",
                "--plan-jsonl",
                str(self.plan_path),
                *extra,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_metadata_only_emits_deduplicated_archive_paths_without_jpegs(self) -> None:
        result = self.run_planner("--metadata-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["output"]["index_format"], "navsim_nuplan10hz_index_v1")
        self.assertEqual(report["checks"]["anchors_verified"], 2)
        self.assertEqual(report["checks"]["retained_camera_records_verified"], 16)
        self.assertEqual(report["checks"]["native_10hz_image_references"], 64)
        self.assertEqual(report["checks"]["unique_native_10hz_images"], 42)
        self.assertTrue(self.index_path.is_file())

        records = [json.loads(line) for line in self.plan_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 42)
        blob_paths = [record["native_blob_relative_path"] for record in records]
        self.assertEqual(len(blob_paths), len(set(blob_paths)))
        self.assertTrue(all(path.startswith(f"sensor_blobs/{LOG_NAME}/") for path in blob_paths))
        self.assertEqual(sum(record["anchor_reference_count"] for record in records), 64)
        self.assertEqual(max(record["anchor_reference_count"] for record in records), 2)
        self.assertTrue(all(not (self.native_root / path).exists() for path in blob_paths))

    def test_require_native_jpegs_fails_without_blobs_and_keeps_plan_unpublished(self) -> None:
        result = self.run_planner("--require-native-jpegs")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not supported by the metadata-first planner", result.stderr)
        self.assertFalse(self.plan_path.exists())
        self.assertFalse(self.index_path.exists())

    def test_metadata_only_fails_closed_on_native_log_identity_mismatch(self) -> None:
        native_db = self.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
        connection = sqlite3.connect(native_db)
        try:
            connection.execute("UPDATE log SET token = ?", (blob("00000000000000cc"),))
            connection.commit()
        finally:
            connection.close()

        result = self.run_planner("--metadata-only")
        self.assertEqual(result.returncode, 2)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "fail")
        self.assertIn("log-token mismatch", report["error"])
        self.assertFalse(self.plan_path.exists())


if __name__ == "__main__":
    unittest.main()
