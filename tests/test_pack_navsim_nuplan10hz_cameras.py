"""Tests for selective, packed native nuPlan camera staging."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sqlite3
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "pack_navsim_nuplan10hz_cameras.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "pack_navsim_nuplan10hz_cameras", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def build_tar(members: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(
        fileobj=output,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class FakePersistentRangeClient:
    payload = b""
    etag = '"fixture-etag"'

    def __init__(self, **_kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass

    def fetch(
        self,
        _url: str,
        start: int,
        end: int,
        *,
        etag: str | None,
    ) -> tuple[bytes, int, str]:
        if etag is not None and etag != self.etag:
            raise AssertionError("unexpected ETag")
        return self.payload[start : end + 1], len(self.payload), self.etag


class CameraPackTest(unittest.TestCase):
    def test_range_payload_timeout_is_retried(self) -> None:
        payload = b"abcd"

        class Raw:
            def __init__(self, fail: bool) -> None:
                self.fail = fail

            def read(self, _size: int, decode_content: bool = False) -> bytes:
                self.assert_decode_content = decode_content
                if self.fail:
                    raise TimeoutError("fixture timeout")
                return payload

        class Response:
            status_code = 206
            headers = {
                "Content-Range": "bytes 0-3/4",
                "Content-Length": "4",
                "ETag": '"fixture-etag"',
            }

            def __init__(self, fail: bool) -> None:
                self.raw = Raw(fail)

            def close(self) -> None:
                pass

        class Session:
            def __init__(self, fail: bool) -> None:
                self.fail = fail
                self.headers: dict[str, str] = {}

            def mount(self, *_args: object) -> None:
                pass

            def get(self, *_args: object, **_kwargs: object) -> Response:
                return Response(self.fail)

            def close(self) -> None:
                pass

        sessions = iter((Session(True), Session(False)))
        with mock.patch.object(MODULE.requests, "Session", side_effect=lambda: next(sessions)), mock.patch.object(
            MODULE.time, "sleep"
        ):
            client = MODULE.PersistentRangeClient(
                connect_timeout_seconds=1.0,
                read_timeout_seconds=1.0,
                max_attempts=2,
            )
            try:
                observed, size, etag = client.fetch(
                    "https://example.invalid/archive",
                    0,
                    3,
                    etag='"fixture-etag"',
                )
            finally:
                client.close()
        self.assertEqual(observed, payload)
        self.assertEqual(size, len(payload))
        self.assertEqual(etag, '"fixture-etag"')

    def test_partial_resume_reuses_existing_and_packs_only_missing_jpeg(self) -> None:
        archive_key = (
            "public/nuplan-v1.1/sensor_blobs/train_set/"
            "nuplan-v1.1_train_camera_7.zip"
        )
        root_name = "nuplan-v1.1_train_camera_7"
        log_name = "2021.01.01.00.00.00_veh-01_00000_00010"
        existing_payload = b"\xff\xd8existing-source-jpeg\xff\xd9"
        packed_payload = b"\xff\xd8missing-source-jpeg\xff\xd9"
        existing_member = f"{root_name}/{log_name}/CAM_F0/one.jpg"
        packed_member = f"{root_name}/{log_name}/CAM_F0/two.jpg"
        FakePersistentRangeClient.payload = build_tar(
            [
                (existing_member, existing_payload),
                (packed_member, packed_payload),
            ]
        )
        spec = MODULE.ArchiveSpec(
            key=archive_key,
            url="https://example.invalid/archive.zip",
            source_split="train",
            group=7,
            listed_size=len(FakePersistentRangeClient.payload),
            listed_etag=FakePersistentRangeClient.etag,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target_state = root / "target.sqlite"
            pack_state = root / "pack.sqlite"
            pack_root = root / "packs"
            existing_root = root / "existing"
            existing_path = existing_root / log_name / "CAM_F0" / "one.jpg"
            existing_path.parent.mkdir(parents=True)
            existing_path.write_bytes(existing_payload)

            connection = MODULE.open_state(target_state)
            try:
                connection.executemany(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    [
                        ("inventory_sha256", hashlib.sha256(b"inventory").hexdigest()),
                        ("inventory_row_count", "2"),
                        (
                            "navsim_scene_filter_sha256",
                            hashlib.sha256(b"scene-filter").hexdigest(),
                        ),
                    ],
                )
                connection.executemany(
                    "INSERT INTO target "
                    "(archive_key, tar_member_name, destination_relative_path, "
                    "log_name, camera_channel) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            archive_key,
                            existing_member,
                            f"sensor_blobs/{log_name}/CAM_F0/one.jpg",
                            log_name,
                            "CAM_F0",
                        ),
                        (
                            archive_key,
                            packed_member,
                            f"sensor_blobs/{log_name}/CAM_F0/two.jpg",
                            log_name,
                            "CAM_F0",
                        ),
                    ],
                )
                connection.commit()
            finally:
                connection.close()

            kwargs = {
                "target_state_path": target_state,
                "pack_state_path": pack_state,
                "pack_root": pack_root,
                "existing_camera_root": existing_root,
                "spec": spec,
                "checkpoint_headers": 100,
                "checkpoint_targets": 100,
                "connect_timeout_seconds": 1.0,
                "read_timeout_seconds": 1.0,
                "max_http_attempts": 1,
            }
            with mock.patch.object(
                MODULE,
                "PersistentRangeClient",
                FakePersistentRangeClient,
            ):
                partial = MODULE.pack_archive(
                    **kwargs,
                    max_headers=1,
                )
                self.assertEqual(partial["status"], "partial")
                self.assertEqual(partial["target_done"], 1)
                self.assertEqual(partial["existing_target_count"], 1)
                self.assertEqual(partial["packed_target_count"], 0)

                complete = MODULE.pack_archive(
                    **kwargs,
                    max_headers=0,
                )
                self.assertEqual(complete["status"], "complete")
                self.assertEqual(complete["target_done"], 2)
                self.assertEqual(complete["existing_target_count"], 1)
                self.assertEqual(complete["packed_target_count"], 1)

            final_pack = pack_root / "train_set" / "nuplan-v1.1_train_camera_7.pack"
            self.assertEqual(final_pack.read_bytes(), packed_payload)

            connection = sqlite3.connect(pack_state)
            try:
                rows = connection.execute(
                    "SELECT tar_member_name, storage_kind, storage_relative_path, "
                    "pack_offset, payload_size, sha256 FROM member_storage "
                    "ORDER BY tar_member_name"
                ).fetchall()
                self.assertEqual(len(rows), 2)
                by_name = {row[0]: row[1:] for row in rows}
                self.assertEqual(
                    by_name[existing_member],
                    (
                        "existing",
                        f"{log_name}/CAM_F0/one.jpg",
                        None,
                        len(existing_payload),
                        hashlib.sha256(existing_payload).hexdigest(),
                    ),
                )
                self.assertEqual(
                    by_name[packed_member],
                    (
                        "pack",
                        "train_set/nuplan-v1.1_train_camera_7.pack",
                        0,
                        len(packed_payload),
                        hashlib.sha256(packed_payload).hexdigest(),
                    ),
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT status, target_done, packed_target_count, "
                        "existing_target_count, pack_bytes FROM archive_progress"
                    ).fetchone(),
                    ("complete", 2, 1, 1, len(packed_payload)),
                )
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
