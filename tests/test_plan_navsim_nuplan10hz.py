from __future__ import annotations

import json
import pickle
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "scripts/plan_navsim_nuplan10hz.py"


class NavsimNuPlan10HzPlanTest(unittest.TestCase):
    def _prepare_camera_phased_history(self, fixture: object) -> None:
        """Make all four retained 2 Hz rows point at a +20 ms camera stream.

        The current image is then after the LiDAR anchor.  A planner ending
        the window at the LiDAR timestamp emits fifteen frames, while the
        correct matched-camera endpoint emits the inclusive sixteen frames.
        """

        from tests.test_navsim_nuplan10hz_lineage import CHANNELS, LOG_NAME

        native_db = fixture.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
        connection = sqlite3.connect(native_db)
        try:
            connection.execute("UPDATE image SET timestamp = timestamp + 20000")
            connection.commit()
        finally:
            connection.close()

        log_path = fixture.navsim_root / "navsim_logs" / "trainval" / f"{LOG_NAME}.pkl"
        with log_path.open("rb") as handle:
            rows = pickle.load(handle)
        for row_index in range(4):
            image_index = row_index * 5
            rows[row_index]["cams"] = {
                channel: {
                    "data_path": f"{LOG_NAME}/{channel}/{fixture._image_paths[(channel, image_index)].name}"
                }
                for channel in CHANNELS
            }
        with log_path.open("wb") as handle:
            pickle.dump(rows, handle)

        # The canonical index is metadata-only.  It must not require either
        # native or OpenScene JPEG bytes during construction.
        for image_path in fixture._image_paths.values():
            image_path.unlink()

    def test_planner_publishes_a_normalized_camera_timing_index(self) -> None:
        from tests.test_navsim_nuplan10hz_lineage import CHANNELS, LOG_NAME, NavsimNuPlan10HzLineageTest

        fixture = NavsimNuPlan10HzLineageTest(
            "test_index_backed_audit_emits_nominal_16_native_images_per_channel"
        )
        fixture.setUp()
        try:
            self._prepare_camera_phased_history(fixture)
            index = fixture.root / "navtrain_camera_10hz.sqlite"
            inventory = fixture.root / "inventory.jsonl"
            report_path = fixture.root / "plan_report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PLANNER),
                    "--navsim-root",
                    str(fixture.navsim_root),
                    "--scene-filter-yaml",
                    str(fixture.filter_yaml),
                    "--nuplan-root",
                    str(fixture.native_root),
                    "--channels",
                    ",".join(CHANNELS),
                    "--allow-nonofficial-filter",
                    "--index-sqlite",
                    str(index),
                    "--inventory-jsonl",
                    str(inventory),
                    "--report-json",
                    str(report_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["checks"]["anchors_verified"], 1)
            self.assertEqual(report["checks"]["retained_camera_records_verified"], 8)
            self.assertEqual(report["checks"]["native_10hz_image_references"], 32)
            self.assertTrue(report["checks"]["native_jpeg_bytes_read"] is False)

            records = [json.loads(line) for line in inventory.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 32)
            self.assertTrue(all(record["anchor_reference_count"] == 1 for record in records))
            self.assertTrue(
                all(record["native_blob_relative_path"].startswith("sensor_blobs/") for record in records)
            )
            self.assertTrue(
                all(
                    record["native_image_filename_jpg"]
                    == str(PurePosixPath(*PurePosixPath(record["native_blob_relative_path"]).parts[1:]))
                    for record in records
                )
            )

            connection = sqlite3.connect(index)
            try:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM anchor").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM native_image").fetchone()[0], 32)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM anchor_camera").fetchone()[0], 2)
                rows = connection.execute(
                    """
                    SELECT camera_channel, frame_count, history_end_timestamp_us,
                           history_end_retained_native_image_timestamp_us
                    FROM anchor_camera
                    ORDER BY camera_channel
                    """
                ).fetchall()
                self.assertEqual([row[1] for row in rows], [16, 16])
                self.assertEqual([row[2] for row in rows], [2_520_000, 2_520_000])
                self.assertEqual([row[2] for row in rows], [row[3] for row in rows])
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                native_image_columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(native_image)")
                }
                self.assertNotIn("archive_relative_path", native_image_columns)
                native_image_indexes = {
                    str(row[1]) for row in connection.execute("PRAGMA index_list(native_image)")
                }
                self.assertNotIn("native_image_stream_lookup", native_image_indexes)
                for channel in CHANNELS:
                    count = connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM anchor_camera AS anchor_camera
                        JOIN anchor AS anchor
                          ON anchor.navsim_anchor_token = anchor_camera.navsim_anchor_token
                        JOIN native_image AS image
                          ON image.navsim_log_name = anchor.navsim_log_name
                         AND image.camera_channel = anchor_camera.camera_channel
                         AND image.stream_index BETWEEN anchor_camera.first_stream_index
                                                    AND anchor_camera.last_stream_index
                        WHERE anchor_camera.camera_channel = ?
                        """,
                        (channel,),
                    ).fetchone()[0]
                    self.assertEqual(count, 16)
            finally:
                connection.close()

            # Read the actual planner output through the NAVSIM-side adapter.
            # This proves the SQLite schema is useful before any native JPEG
            # payloads are staged.
            sys.path.insert(0, str(REPO_ROOT))
            try:
                from navtrain10hz.loader import NuPlan10HzCameraHistoryIndex

                with NuPlan10HzCameraHistoryIndex(index) as metadata_index:
                    anchor = metadata_index.get_anchor("0000000000000001", required_channels=CHANNELS)
                    self.assertEqual(anchor.camera_hz, 10)
                    self.assertEqual(anchor.label_hz, 2)
                    for channel in CHANNELS:
                        history = anchor.camera_histories[channel]
                        self.assertEqual(history.frame_count, 16)
                        self.assertEqual(history.history_start_retained_reference, history.references[0])
                        self.assertEqual(history.history_end_retained_reference, history.references[-1])
                    availability = metadata_index.jpeg_availability(anchor, fixture.native_root)
                    self.assertFalse(availability.complete)
                    self.assertEqual(availability.present_references, 0)
            finally:
                sys.path.pop(0)
        finally:
            fixture.tearDown()

    def test_planner_preserves_an_irregular_retained_history_range(self) -> None:
        from tests.test_navsim_nuplan10hz_lineage import CHANNELS, LOG_NAME, NavsimNuPlan10HzLineageTest

        fixture = NavsimNuPlan10HzLineageTest(
            "test_index_backed_audit_emits_nominal_16_native_images_per_channel"
        )
        fixture.setUp()
        try:
            self._prepare_camera_phased_history(fixture)
            log_path = fixture.navsim_root / "navsim_logs" / "trainval" / f"{LOG_NAME}.pkl"
            with log_path.open("rb") as handle:
                rows = pickle.load(handle)
            rows[3]["timestamp"] = 2_400_000
            for channel in CHANNELS:
                rows[3]["cams"][channel]["data_path"] = (
                    f"{LOG_NAME}/{channel}/{fixture._image_paths[(channel, 14)].name}"
                )
            with log_path.open("wb") as handle:
                pickle.dump(rows, handle)

            native_db = fixture.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
            connection = sqlite3.connect(native_db)
            try:
                connection.execute("UPDATE lidar_pc SET timestamp = ?", (2_400_000,))
                connection.commit()
            finally:
                connection.close()

            index = fixture.root / "invalid_cadence.sqlite"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PLANNER),
                    "--navsim-root",
                    str(fixture.navsim_root),
                    "--scene-filter-yaml",
                    str(fixture.filter_yaml),
                    "--nuplan-root",
                    str(fixture.native_root),
                    "--channels",
                    ",".join(CHANNELS),
                    "--allow-nonofficial-filter",
                    "--index-sqlite",
                    str(index),
                    "--report-json",
                    str(fixture.root / "invalid_cadence_report.json"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((fixture.root / "invalid_cadence_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["navsim"]["retained_history_cadence_exceptions"], 1)
            self.assertEqual(report["checks"]["native_10hz_image_references"], 30)
            self.assertEqual(report["checks"]["native_history_frame_counts_by_channel"], {
                "CAM_F0": {"15": 1},
                "CAM_R0": {"15": 1},
            })
            self.assertTrue(index.exists())
        finally:
            fixture.tearDown()

    def test_planner_preserves_a_native_source_capture_gap_and_reports_it(self) -> None:
        """A dropped native capture must not discard the NAVSIM anchor or be filled."""

        from tests.test_navsim_nuplan10hz_lineage import CHANNELS, LOG_NAME, NavsimNuPlan10HzLineageTest

        fixture = NavsimNuPlan10HzLineageTest(
            "test_index_backed_audit_emits_nominal_16_native_images_per_channel"
        )
        fixture.setUp()
        try:
            native_db = fixture.native_root / "splits" / "trainval" / f"{LOG_NAME}.db"
            connection = sqlite3.connect(native_db)
            try:
                # Keep source identities intact but create a real 200 ms
                # interval between native stream entries 7 and 8.
                connection.execute("UPDATE image SET timestamp = timestamp + 100000 WHERE timestamp >= 1800000")
                connection.commit()
            finally:
                connection.close()

            log_path = fixture.navsim_root / "navsim_logs" / "trainval" / f"{LOG_NAME}.pkl"
            with log_path.open("rb") as handle:
                rows = pickle.load(handle)
            for row_index, image_index in enumerate((0, 5, 9, 14)):
                rows[row_index]["cams"] = {
                    channel: {
                        "data_path": f"{LOG_NAME}/{channel}/{fixture._image_paths[(channel, image_index)].name}"
                    }
                    for channel in CHANNELS
                }
            with log_path.open("wb") as handle:
                pickle.dump(rows, handle)

            index = fixture.root / "source_capture_gap.sqlite"
            report_path = fixture.root / "source_capture_gap_report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PLANNER),
                    "--navsim-root",
                    str(fixture.navsim_root),
                    "--scene-filter-yaml",
                    str(fixture.filter_yaml),
                    "--nuplan-root",
                    str(fixture.native_root),
                    "--channels",
                    ",".join(CHANNELS),
                    "--allow-nonofficial-filter",
                    "--index-sqlite",
                    str(index),
                    "--report-json",
                    str(report_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["checks"]["anchors_verified"], 1)
            self.assertEqual(report["checks"]["native_10hz_image_references"], 30)
            self.assertEqual(report["checks"]["native_stream_capture_gap_count"], 2)
            self.assertEqual(report["checks"]["native_stream_capture_gap_delta_us_counts"], {"200000": 2})
            self.assertEqual(report["checks"]["native_history_capture_gap_count"], 2)
            self.assertEqual(report["checks"]["native_history_capture_gap_affected_anchor_count"], 1)
            self.assertEqual(report["checks"]["native_history_capture_gap_affected_anchor_channel_count"], 2)

            sys.path.insert(0, str(REPO_ROOT))
            try:
                from navtrain10hz.loader import NuPlan10HzCameraHistoryIndex

                with NuPlan10HzCameraHistoryIndex(index) as metadata_index:
                    anchor = metadata_index.get_anchor("0000000000000001", required_channels=CHANNELS)
                    for channel in CHANNELS:
                        history = anchor.camera_histories[channel]
                        self.assertEqual(history.frame_count, 15)
                        self.assertTrue(history.has_source_capture_gaps)
                        self.assertEqual(history.source_capture_gap_deltas_us, (200000,))
                        self.assertEqual(history.native_image_timestamp_us[7:9], (1_700_000, 1_900_000))
            finally:
                sys.path.pop(0)
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
