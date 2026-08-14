from __future__ import annotations

import sqlite3
import hashlib
from io import BytesIO
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from navtrain10hz.loader import (  # noqa: E402
    NAVSIM_LABEL_HZ,
    NATIVE_CAMERA_HZ,
    NAVTRAIN_NATIVE_CAMERA_HISTORY_FRAMES,
    Navsim10HzImageLoader,
    NativeImageStorageIndex,
    Navsim10HzMetadataAdapter,
    NuPlan10HzCameraHistoryIndex,
    NuPlan10HzIndexError,
    NuPlan10HzImagesUnavailableError,
)


INDEX_FORMAT = "navsim_nuplan10hz_index_v1"
LOG_NAME = "2021.01.01.00.00.00_veh-01_00000_00099"
ANCHOR_TOKEN = "00000000000000aa"
CHANNELS = ("CAM_F0", "CAM_R0")
FRAME_COUNT = 16


class NavsimNuPlan10HzMetadataAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.index_path = self.root / "navtrain_camera_10hz_index.sqlite"
        self.nuplan_root = self.root / "nuplan-v1.1"
        self._build_index()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_index(self) -> None:
        connection = sqlite3.connect(self.index_path)
        try:
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE source_log (
                    navsim_log_name TEXT PRIMARY KEY,
                    navsim_log_token TEXT NOT NULL,
                    native_db_relative_path TEXT NOT NULL
                );
                CREATE TABLE anchor (
                    navsim_anchor_token TEXT PRIMARY KEY,
                    navsim_log_name TEXT NOT NULL,
                    navsim_log_token TEXT NOT NULL,
                    navsim_scene_token TEXT NOT NULL,
                    navsim_anchor_timestamp_us INTEGER NOT NULL,
                    navsim_row_index INTEGER NOT NULL,
                    native_lidar_token TEXT NOT NULL,
                    native_lidar_timestamp_us INTEGER NOT NULL,
                    native_lidar_scene_token TEXT NOT NULL,
                    native_lidar_filename TEXT NOT NULL,
                    native_lidar_mapping_method TEXT NOT NULL,
                    navsim_history_start_timestamp_us INTEGER NOT NULL
                );
                CREATE TABLE native_image (
                    image_id INTEGER PRIMARY KEY,
                    navsim_log_name TEXT NOT NULL,
                    camera_channel TEXT NOT NULL,
                    stream_index INTEGER NOT NULL,
                    native_image_token TEXT NOT NULL,
                    native_image_timestamp_us INTEGER NOT NULL,
                    native_image_filename_jpg TEXT NOT NULL,
                    native_camera_token TEXT NOT NULL,
                    native_blob_relative_path TEXT NOT NULL,
                    UNIQUE (navsim_log_name, camera_channel, stream_index)
                );
                CREATE TABLE anchor_camera (
                    navsim_anchor_token TEXT NOT NULL,
                    camera_channel TEXT NOT NULL,
                    history_start_retained_image_id INTEGER NOT NULL,
                    history_start_retained_native_image_timestamp_us INTEGER NOT NULL,
                    history_end_retained_image_id INTEGER NOT NULL,
                    history_end_retained_native_image_timestamp_us INTEGER NOT NULL,
                    history_start_timestamp_us INTEGER NOT NULL,
                    history_end_timestamp_us INTEGER NOT NULL,
                    first_stream_index INTEGER NOT NULL,
                    last_stream_index INTEGER NOT NULL,
                    frame_count INTEGER NOT NULL,
                    PRIMARY KEY (navsim_anchor_token, camera_channel)
                );
                """
            )
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [
                    ("format", INDEX_FORMAT),
                    ("native_jpegs_verified", "false"),
                    ("camera_hz", "10"),
                    ("label_hz", "2"),
                    ("retained_camera_tolerance_us", "50000"),
                    ("camera_cadence_tolerance_us", "50000"),
                ],
            )
            connection.execute(
                "INSERT INTO source_log VALUES (?, ?, ?)",
                (LOG_NAME, "00000000000000bb", f"splits/trainval/{LOG_NAME}.db"),
            )
            connection.execute(
                "INSERT INTO anchor VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ANCHOR_TOKEN,
                    LOG_NAME,
                    "00000000000000bb",
                    "00000000000000cc",
                    2_500_000,
                    3,
                    ANCHOR_TOKEN,
                    2_500_000,
                    "00000000000000cc",
                    f"{LOG_NAME}/MergedPointCloud/{ANCHOR_TOKEN}.pcd",
                    "native_lidar_token",
                    1_000_000,
                ),
            )
            image_id = 0
            for channel in CHANNELS:
                first_id = image_id + 1
                for stream_index in range(FRAME_COUNT):
                    image_id += 1
                    image_name = f"{stream_index:016x}.jpg"
                    connection.execute(
                        "INSERT INTO native_image VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            image_id,
                            LOG_NAME,
                            channel,
                            stream_index,
                            f"{image_id:016x}",
                            1_000_000 + stream_index * 100_000,
                            f"{LOG_NAME}/{channel}/{image_name}",
                            f"{100 + image_id:016x}",
                            f"sensor_blobs/{LOG_NAME}/{channel}/{image_name}",
                        ),
                    )
                connection.execute(
                    "INSERT INTO anchor_camera VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ANCHOR_TOKEN,
                        channel,
                        first_id,
                        1_000_000,
                        first_id + FRAME_COUNT - 1,
                        2_500_000,
                        1_000_000,
                        2_500_000,
                        0,
                        FRAME_COUNT - 1,
                        FRAME_COUNT,
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def test_metadata_only_reads_a_nominal_16_frame_history_at_a_2hz_label_anchor(self) -> None:
        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            anchor = index.get_anchor(ANCHOR_TOKEN, required_channels=CHANNELS)

            self.assertFalse(index.native_jpegs_verified)
            self.assertIsNone(index.expected_history_frames)
            self.assertEqual(NAVTRAIN_NATIVE_CAMERA_HISTORY_FRAMES, FRAME_COUNT)
            self.assertEqual(anchor.camera_hz, NATIVE_CAMERA_HZ)
            self.assertEqual(anchor.label_hz, NAVSIM_LABEL_HZ)
            self.assertEqual(anchor.navsim_anchor_token, ANCHOR_TOKEN)
            self.assertEqual(set(anchor.camera_histories), set(CHANNELS))
            for channel in CHANNELS:
                history = anchor.camera_histories[channel]
                self.assertEqual(history.frame_count, FRAME_COUNT)
                self.assertEqual(history.references[0].native_image_timestamp_us, 1_000_000)
                self.assertEqual(history.references[-1].native_image_timestamp_us, 2_500_000)
                self.assertEqual(history.history_start_retained_reference, history.references[0])
                self.assertEqual(history.history_end_retained_reference, history.references[-1])
                self.assertEqual(history.retained_reference, history.references[-1])

            # No native JPEGs were created. Metadata mode still succeeds and
            # reports, rather than raises on, the missing pixel payloads.
            availability = index.jpeg_availability(anchor, self.nuplan_root)
            self.assertFalse(availability.complete)
            self.assertEqual(availability.total_references, len(CHANNELS) * FRAME_COUNT)
            self.assertEqual(availability.present_references, 0)
            self.assertEqual(len(availability.missing_relative_paths), len(CHANNELS) * FRAME_COUNT)

    def test_image_loader_decodes_exact_jpegs_and_exposes_padding_masks(self) -> None:
        """Pixels, timestamps, and masks must follow the immutable index exactly."""

        for channel_index, channel in enumerate(CHANNELS):
            for frame_index in range(FRAME_COUNT):
                image_path = (
                    self.nuplan_root
                    / "sensor_blobs"
                    / LOG_NAME
                    / channel
                    / f"{frame_index:016x}.jpg"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.full(
                    (3, 4, 3),
                    fill_value=channel_index * 32 + frame_index,
                    dtype=np.uint8,
                )
                Image.fromarray(pixels, mode="RGB").save(image_path)

        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            loader = Navsim10HzImageLoader(index, self.nuplan_root)
            loaded = loader.load_anchor(ANCHOR_TOKEN, required_channels=CHANNELS)

        self.assertEqual(loaded.anchor.navsim_anchor_token, ANCHOR_TOKEN)
        for channel_index, channel in enumerate(CHANNELS):
            history = loaded.camera_histories[channel]
            self.assertEqual(history.frame_count, FRAME_COUNT)
            self.assertEqual(history.native_image_timestamp_us[0], 1_000_000)
            self.assertEqual(history.native_image_timestamp_us[-1], 2_500_000)
            self.assertTrue(all(history.valid_mask))
            self.assertEqual(history.images[0].shape, (3, 4, 3))
            self.assertEqual(int(history.images[7][0, 0, 0]), channel_index * 32 + 7)

        padded = loaded.padded_camera_histories(FRAME_COUNT + 2)
        for channel in CHANNELS:
            history = padded[channel]
            self.assertEqual(history.images.shape, (FRAME_COUNT + 2, 3, 4, 3))
            self.assertTrue(np.all(history.valid_mask[:FRAME_COUNT]))
            self.assertTrue(np.all(~history.valid_mask[FRAME_COUNT:]))
            self.assertTrue(np.all(history.native_image_timestamp_us[FRAME_COUNT:] == -1))
            self.assertTrue(np.all(history.images[FRAME_COUNT:] == 0))

    def test_image_loader_reads_verified_pack_slice_and_fails_on_corruption(self) -> None:
        pixels = np.full((3, 4, 3), 77, dtype=np.uint8)
        encoded = BytesIO()
        Image.fromarray(pixels, mode="RGB").save(encoded, format="JPEG")
        payload = encoded.getvalue()
        pack_root = self.root / "packs"
        pack_path = pack_root / "train_set" / "fixture.pack"
        pack_path.parent.mkdir(parents=True)
        prefix = b"not-an-image-prefix"
        pack_path.write_bytes(prefix + payload)

        storage_path = self.root / "storage.sqlite"
        connection = sqlite3.connect(storage_path)
        try:
            connection.executescript(
                """
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE image_storage (
                    destination_relative_path TEXT PRIMARY KEY,
                    archive_key TEXT NOT NULL,
                    tar_member_name TEXT NOT NULL,
                    storage_kind TEXT NOT NULL,
                    storage_relative_path TEXT NOT NULL,
                    pack_offset INTEGER,
                    payload_size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT INTO metadata VALUES ('format', 'navsim_nuplan10hz_image_storage_v1')"
            )
            relative = f"sensor_blobs/{LOG_NAME}/{CHANNELS[0]}/0000000000000000.jpg"
            connection.execute(
                "INSERT INTO image_storage VALUES (?, 'archive', 'member', 'pack', ?, ?, ?, ?)",
                (
                    relative,
                    "train_set/fixture.pack",
                    len(prefix),
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                ),
            )
            connection.commit()
        finally:
            connection.close()

        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            reference = index.get_anchor(ANCHOR_TOKEN).camera_histories[CHANNELS[0]].references[0]
            with NativeImageStorageIndex(storage_path) as storage:
                loader = Navsim10HzImageLoader(
                    index, self.nuplan_root, storage_index=storage, pack_root=pack_root
                )
                decoded = loader._decode_jpeg(reference)
                self.assertEqual(decoded.shape, (3, 4, 3))
                pack_path.write_bytes(prefix + payload[:-1] + b"x")
                with self.assertRaisesRegex(NuPlan10HzImagesUnavailableError, "complete JPEG|SHA-256"):
                    loader._decode_jpeg(reference)

    def test_token_lookup_normalizes_the_request_before_exact_sqlite_equality(self) -> None:
        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            statements: list[str] = []
            index._connection.set_trace_callback(statements.append)
            self.assertTrue(index.has_anchor(f"  {ANCHOR_TOKEN.upper()}  "))
            anchor = index.get_anchor(ANCHOR_TOKEN.upper())
            self.assertEqual(anchor.navsim_anchor_token, ANCHOR_TOKEN)

            token_statements = [
                statement
                for statement in statements
                if "FROM anchor" in statement or "FROM anchor_camera" in statement
            ]
            self.assertTrue(token_statements)
            self.assertFalse(any("lower(" in statement.lower() for statement in token_statements))
            self.assertTrue(
                any("navsim_anchor_token =" in statement for statement in token_statements),
                token_statements,
            )

    def test_require_images_is_the_only_missing_jpeg_failure_path(self) -> None:
        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            index.get_anchor(ANCHOR_TOKEN)  # Metadata-only operation must not need JPEGs.
            with self.assertRaises(NuPlan10HzImagesUnavailableError):
                index.get_anchor(ANCHOR_TOKEN, nuplan_root=self.nuplan_root, require_images=True)

            anchor = index.get_anchor(ANCHOR_TOKEN)
            for history in anchor.camera_histories.values():
                for reference in history.references:
                    image_path = reference.blob_path(self.nuplan_root)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(b"fixture-jpeg-not-decoded")

            loaded = index.get_anchor(ANCHOR_TOKEN, nuplan_root=self.nuplan_root, require_images=True)
            self.assertEqual(loaded.navsim_anchor_token, ANCHOR_TOKEN)
            self.assertTrue(index.jpeg_availability(loaded, self.nuplan_root).complete)

    def test_overlay_does_not_call_or_modify_the_stock_scene_loader(self) -> None:
        class FakeSceneLoader:
            tokens = (ANCHOR_TOKEN,)

            def get_agent_input_from_token(self, token: str):
                raise AssertionError("The metadata adapter must not call stock AgentInput loading")

        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            overlay = Navsim10HzMetadataAdapter.from_scene_loader(index, FakeSceneLoader())
            self.assertEqual(overlay.tokens, (ANCHOR_TOKEN,))
            anchor = overlay.get_anchor(ANCHOR_TOKEN)
            self.assertEqual(anchor.camera_hz, 10)
            self.assertEqual(anchor.label_hz, 2)
            self.assertFalse(hasattr(overlay, "get_agent_input_from_token"))
            with self.assertRaisesRegex(NuPlan10HzIndexError, "not present in the supplied NAVSIM"):
                overlay.get_anchor("00000000000000ff")

    def test_corrupt_range_fails_closed(self) -> None:
        connection = sqlite3.connect(self.index_path)
        try:
            connection.execute(
                "UPDATE anchor_camera SET frame_count = ? WHERE navsim_anchor_token = ? AND camera_channel = ?",
                (FRAME_COUNT - 1, ANCHOR_TOKEN, CHANNELS[0]),
            )
            connection.commit()
        finally:
            connection.close()

        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            with self.assertRaisesRegex(NuPlan10HzIndexError, "Range/frame-count mismatch"):
                index.get_anchor(ANCHOR_TOKEN)

    def test_navtrain_default_accepts_exact_variable_native_ranges(self) -> None:
        connection = sqlite3.connect(self.index_path)
        try:
            connection.execute(
                """
                UPDATE anchor_camera
                SET first_stream_index = ?, frame_count = ?,
                    history_start_retained_image_id = history_start_retained_image_id + 1,
                    history_start_retained_native_image_timestamp_us =
                        history_start_retained_native_image_timestamp_us + 100000,
                    history_start_timestamp_us = history_start_timestamp_us + 100000
                """,
                (1, FRAME_COUNT - 1),
            )
            connection.execute(
                "UPDATE anchor SET navsim_history_start_timestamp_us = ?",
                (1_100_000,),
            )
            connection.commit()
        finally:
            connection.close()

        with NuPlan10HzCameraHistoryIndex(self.index_path) as index:
            anchor = index.get_anchor(ANCHOR_TOKEN)
            self.assertEqual(anchor.camera_histories[CHANNELS[0]].frame_count, FRAME_COUNT - 1)
        with NuPlan10HzCameraHistoryIndex(
            self.index_path,
            expected_history_frames=FRAME_COUNT,
        ) as index:
            with self.assertRaisesRegex(NuPlan10HzIndexError, "requires 16"):
                index.get_anchor(ANCHOR_TOKEN)


if __name__ == "__main__":
    unittest.main()
