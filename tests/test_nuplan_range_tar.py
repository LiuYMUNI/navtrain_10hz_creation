from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from nuplan_range_tar import RangeTarArchive, RangeTarError  # noqa: E402
from test_nuplan_range_zip import MemoryRangeClient  # noqa: E402


def build_tar() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        directory = tarfile.TarInfo("release/log/CAM_F0")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        image = tarfile.TarInfo("release/log/CAM_F0/fixture.jpg")
        payload = b"\xff\xd8fixture-jpeg\xff\xd9"
        image.size = len(payload)
        archive.addfile(image, io.BytesIO(payload))
    return stream.getvalue()


class NuplanRangeTarTest(unittest.TestCase):
    def test_iterates_and_extracts_a_regular_member(self) -> None:
        archive = RangeTarArchive.open("memory://fixture", client=MemoryRangeClient(build_tar()))
        members = list(archive.iter_members())
        self.assertEqual(
            [(member.name, member.typeflag, member.size) for member in members],
            [
                ("release/log/CAM_F0/", b"5", 0),
                ("release/log/CAM_F0/fixture.jpg", b"0", 16),
            ],
        )
        self.assertEqual(archive.read_member(members[1]), b"\xff\xd8fixture-jpeg\xff\xd9")
        with tempfile.TemporaryDirectory() as temporary:
            target = archive.extract_member(
                members[1],
                Path(temporary),
                relative_path="sensor_blobs/log/CAM_F0/fixture.jpg",
            )
            self.assertEqual(target.read_bytes(), b"\xff\xd8fixture-jpeg\xff\xd9")

    def test_rejects_a_corrupt_header_checksum(self) -> None:
        payload = bytearray(build_tar())
        payload[10] ^= 0x01
        archive = RangeTarArchive.open("memory://corrupt", client=MemoryRangeClient(bytes(payload)))
        with self.assertRaisesRegex(RangeTarError, "checksum mismatch"):
            list(archive.iter_members())

    def test_scan_chunk_reuses_fetched_target_payload(self) -> None:
        payload = build_tar()
        client = MemoryRangeClient(payload)
        archive = RangeTarArchive.open(
            "memory://chunked",
            client=client,
            scan_chunk_bytes=len(payload),
        )
        members = list(archive.iter_members())
        self.assertEqual(archive.read_member(members[1]), b"\xff\xd8fixture-jpeg\xff\xd9")
        # One single-byte probe plus one whole scan range is sufficient; the
        # target payload was safely inside that ETag-pinned scan response.
        self.assertEqual(len(client.requests), 2)


if __name__ == "__main__":
    unittest.main()
