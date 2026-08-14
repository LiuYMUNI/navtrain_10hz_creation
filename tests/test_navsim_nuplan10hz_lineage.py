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
AUDITOR = REPO_ROOT / "scripts" / "audit_navsim_nuplan10hz_lineage.py"
PLANNER = REPO_ROOT / "scripts" / "plan_navsim_nuplan10hz.py"
LOG_NAME = "2021.01.01.00.00.00_veh-01_00000_00099"
ANCHOR_TOKEN = "0000000000000001"
SCENE_TOKEN = "00000000000000aa"
LOG_TOKEN = "00000000000000bb"
CHANNELS = ("CAM_F0", "CAM_R0")


def blob(token: str) -> bytes:
    return bytes.fromhex(token)


class NavsimNuPlan10HzLineageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.navsim_root = self.root / "navsim2"
        self.native_root = self.root / "nuplan-v1.1"
        self.filter_yaml = self.root / "navtrain_fixture.yaml"
        self.manifest = self.root / "derived" / "camera_manifest.jsonl"
        self.index_path = self.root / "derived" / "navtrain_camera_10hz_index.sqlite"
        self.planner_report = self.root / "derived" / "planner_report.json"
        self._image_paths: dict[tuple[str, int], Path] = {}
        self._build_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_fixture(self) -> None:
        self.filter_yaml.write_text(
            """_target_: navsim.common.dataclasses.SceneFilter
_convert_: all
num_history_frames: 4
num_future_frames: 10
frame_interval: 1
has_route: true
max_scenes: null
log_names:
  - '2021.01.01.00.00.00_veh-01_00000_00099'
tokens:
  - '0000000000000001'
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
            connection.execute(
                "INSERT INTO lidar_pc VALUES (?, ?, ?, ?)",
                (
                    blob(ANCHOR_TOKEN),
                    2_500_000,
                    blob(SCENE_TOKEN),
                    f"MergedPointCloud/{ANCHOR_TOKEN}.pcd",
                ),
            )
            for channel_index, channel in enumerate(CHANNELS, start=1):
                camera_token = blob(f"{channel_index:016x}")
                connection.execute("INSERT INTO camera VALUES (?, ?)", (camera_token, channel))
                for frame_index in range(16):
                    image_token = f"{1000 + channel_index * 100 + frame_index:016x}"
                    filename = f"{channel}/{image_token}.jpg"
                    timestamp = 1_000_000 + frame_index * 100_000
                    connection.execute(
                        "INSERT INTO image VALUES (?, ?, ?, ?)",
                        (blob(image_token), timestamp, filename, camera_token),
                    )
                    image_path = self.native_root / "sensor_blobs" / LOG_NAME / filename
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(b"native-jpeg")
                    self._image_paths[(channel, frame_index)] = image_path
            connection.commit()
        finally:
            connection.close()

        rows: list[dict[str, object]] = []
        for index in range(14):
            rows.append(
                {
                    "token": f"{index + 100:016x}",
                    "timestamp": 1_000_000 + index * 500_000,
                    "log_name": LOG_NAME,
                    "log_token": LOG_TOKEN,
                    "scene_token": SCENE_TOKEN,
                    "lidar_path": f"{LOG_NAME}/MergedPointCloud/{index + 100:016x}.pcd",
                    "cams": {},
                }
            )
        anchor = rows[3]
        anchor["token"] = ANCHOR_TOKEN
        anchor["timestamp"] = 2_500_000
        anchor["lidar_path"] = f"{LOG_NAME}/MergedPointCloud/{ANCHOR_TOKEN}.pcd"
        log_path = self.navsim_root / "navsim_logs" / "trainval" / f"{LOG_NAME}.pkl"
        log_path.parent.mkdir(parents=True)
        with log_path.open("wb") as handle:
            pickle.dump(rows, handle)

    def _prepare_camera_phased_history(self) -> None:
        """Make the retained native camera records occur 20 ms after LiDAR.

        A LiDAR-bounded 1.5-second query would omit each current image and
        return fifteen frames. The canonical planner instead uses the matched
        first/current camera endpoints and must publish sixteen frames.
        """

        database = self.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
        connection = sqlite3.connect(database)
        try:
            connection.execute("UPDATE image SET timestamp = timestamp + 20000")
            connection.commit()
        finally:
            connection.close()

        log_path = self.navsim_root / "navsim_logs" / "trainval" / f"{LOG_NAME}.pkl"
        with log_path.open("rb") as handle:
            rows = pickle.load(handle)
        for row_index in range(4):
            image_index = row_index * 5
            rows[row_index]["cams"] = {
                channel: {
                    "data_path": f"{LOG_NAME}/{channel}/{self._image_paths[(channel, image_index)].name}"
                }
                for channel in CHANNELS
            }
        with log_path.open("wb") as handle:
            pickle.dump(rows, handle)

        for channel in CHANNELS:
            for image_index in (0, 5, 10, 15):
                staged = (
                    self.navsim_root
                    / "sensor_blobs"
                    / "trainval"
                    / LOG_NAME
                    / channel
                    / self._image_paths[(channel, image_index)].name
                )
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(b"openscene-jpeg")

    def _build_canonical_index(self) -> None:
        result = subprocess.run(
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
                "--anchor-token",
                ANCHOR_TOKEN,
                "--allow-nonofficial-filter",
                "--index-sqlite",
                str(self.index_path),
                "--report-json",
                str(self.planner_report),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.planner_report.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checks"]["retained_camera_records_verified"], 8)
        self.assertEqual(report["checks"]["native_10hz_image_references"], 32)
        self.assertTrue(self.index_path.is_file())

    def run_auditor(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(AUDITOR),
                "--navsim-root",
                str(self.navsim_root),
                "--scene-filter-yaml",
                str(self.filter_yaml),
                "--nuplan-root",
                str(self.native_root),
                "--index-sqlite",
                str(self.index_path),
                "--channels",
                ",".join(CHANNELS),
                "--anchor-token",
                ANCHOR_TOKEN,
                "--allow-nonofficial-filter",
                "--require-openscene-files",
                *extra,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_index_backed_audit_emits_nominal_16_native_images_per_channel(self) -> None:
        self._prepare_camera_phased_history()
        self._build_canonical_index()
        result = self.run_auditor("--manifest-jsonl", str(self.manifest))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checks"]["anchors_verified"], 1)
        self.assertEqual(report["checks"]["native_databases_referenced"], 1)
        self.assertEqual(report["checks"]["source_logs_referenced"], 1)
        self.assertEqual(report["checks"]["retained_camera_records_verified"], 8)
        self.assertEqual(report["checks"]["native_10hz_images_verified"], 32)
        self.assertTrue(self.manifest.is_file())
        records = [json.loads(line) for line in self.manifest.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 32)
        self.assertEqual({record["camera_channel"] for record in records}, set(CHANNELS))
        self.assertTrue(all(record["navsim_anchor_token"] == ANCHOR_TOKEN for record in records))
        self.assertTrue(all(record["navsim_log_token"] == LOG_TOKEN for record in records))
        self.assertTrue(all(record["native_camera_token"] for record in records))
        self.assertTrue(all(record["native_blob_relative_path"].startswith("sensor_blobs/") for record in records))
        self.assertEqual(report["contract"]["canonical_index_sqlite"], str(self.index_path))
        self.assertEqual(report["contract"]["nominal_camera_history_frames"], 16)
        self.assertEqual(
            report["contract"]["native_history_frame_policy"],
            "native_contiguous_camera_range_preserve_source_capture_gaps",
        )
        self.assertEqual(report["checks"]["native_history_frame_counts_by_channel"]["CAM_F0"], {"16": 1})
        for channel in CHANNELS:
            channel_records = [record for record in records if record["camera_channel"] == channel]
            self.assertEqual(len(channel_records), 16)
            self.assertEqual(
                [record["native_image_timestamp_us"] for record in channel_records],
                [1_020_000 + frame_index * 100_000 for frame_index in range(16)],
            )

    def test_missing_final_indexed_native_jpeg_fails_and_does_not_publish_manifest(self) -> None:
        self._prepare_camera_phased_history()
        self._build_canonical_index()
        final_image = self._image_paths[("CAM_R0", 15)]
        final_image.unlink()
        result = self.run_auditor("--manifest-jsonl", str(self.manifest))
        self.assertEqual(result.returncode, 2)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "fail")
        self.assertIn("Native JPEGs are unavailable", report["error"])
        self.assertIn(final_image.name, report["error"])
        self.assertFalse(self.manifest.exists())

    def test_optional_openscene_check_covers_all_four_retained_frames(self) -> None:
        self._prepare_camera_phased_history()
        self._build_canonical_index()
        earlier_retained_image = self._image_paths[("CAM_F0", 5)]
        staged = (
            self.navsim_root
            / "sensor_blobs"
            / "trainval"
            / LOG_NAME
            / "CAM_F0"
            / earlier_retained_image.name
        )
        staged.unlink()
        result = self.run_auditor("--manifest-jsonl", str(self.manifest))
        self.assertEqual(result.returncode, 2)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "fail")
        self.assertIn("Retained OpenScene CAM_F0 image is absent", report["error"])
        self.assertFalse(self.manifest.exists())

    def test_auditor_uses_immutable_index_after_native_db_metadata_changes(self) -> None:
        self._prepare_camera_phased_history()
        self._build_canonical_index()
        database = self.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
        connection = sqlite3.connect(database)
        try:
            connection.execute("UPDATE log SET token = ?", (blob("00000000000000cc"),))
            connection.commit()
        finally:
            connection.close()
        result = self.run_auditor("--manifest-jsonl", str(self.manifest))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checks"]["native_10hz_images_verified"], 32)
        self.assertTrue(self.manifest.is_file())


if __name__ == "__main__":
    unittest.main()
