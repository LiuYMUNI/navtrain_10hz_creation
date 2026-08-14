from __future__ import annotations

import binascii
import io
import os
import socket
import ssl
import struct
import sys
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from nuplan_range_zip import (  # noqa: E402
    _DirectPeerPool,
    _PinnedPeerHTTPSConnection,
    HttpRangeClient,
    HttpRangeResponse,
    RangeZipArchive,
    RangeZipError,
)


class MemoryRangeClient:
    def __init__(self, payload: bytes, *, etag: str = '"fixture-etag"') -> None:
        self.payload = bytearray(payload)
        self.etag = etag
        self.requests: list[tuple[int, int, str | None]] = []

    def fetch(self, url: str, start: int, end: int, *, etag: str | None = None) -> HttpRangeResponse:
        self.requests.append((start, end, etag))
        if etag is not None and etag != self.etag:
            raise RangeZipError("fixture ETag changed")
        if start < 0 or end < start or end >= len(self.payload):
            raise RangeZipError("fixture range outside object")
        return HttpRangeResponse(
            data=bytes(self.payload[start : end + 1]),
            start=start,
            end=end,
            total_size=len(self.payload),
            etag=self.etag,
        )


def build_zip(entries: list[tuple[str, bytes, int]]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload, compression in entries:
            archive.writestr(name, payload, compress_type=compression)
    return stream.getvalue()


def build_tiny_zip64() -> bytes:
    """Build a small valid ZIP64 archive with ZIP64 member fields.

    The member's real offsets/sizes fit in 32 bits, but ZIP64 sentinels and
    extra fields force the reader through exactly the code path used by the
    multi-gigabyte nuPlan database archives.
    """

    name = b"data/cache/train/fixture.db"
    payload = b"zip64 fixture database"
    crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    local_extra = struct.pack("<HHQQ", 0x0001, 16, len(payload), len(payload))
    local = (
        struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            45,
            0,
            0,
            0,
            0,
            crc32,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name),
            len(local_extra),
        )
        + name
        + local_extra
        + payload
    )
    central_offset = len(local)
    central_extra = struct.pack("<HHQQQ", 0x0001, 24, len(payload), len(payload), 0)
    central = (
        struct.pack(
            "<IHHHHHHIIIHHHHHII",
            0x02014B50,
            (3 << 8) | 45,
            45,
            0,
            0,
            0,
            0,
            crc32,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(name),
            len(central_extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + name
        + central_extra
    )
    zip64_eocd_offset = central_offset + len(central)
    zip64_eocd = struct.pack(
        "<IQHHIIQQQQ",
        0x06064B50,
        44,
        (3 << 8) | 45,
        45,
        0,
        0,
        1,
        1,
        len(central),
        central_offset,
    )
    locator = struct.pack("<IIQI", 0x07064B50, 0, zip64_eocd_offset, 1)
    eocd = struct.pack(
        "<IHHHHIIH",
        0x06054B50,
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    return local + central + zip64_eocd + locator + eocd


class FakeHttpResponse:
    def __init__(self, *, status: int, headers: dict[str, str], data: bytes) -> None:
        self.status = status
        self.headers = headers
        self._data = data

    def getcode(self) -> int:
        return self.status

    def read(self, amount: int | None = None) -> bytes:
        return self._data if amount is None else self._data[:amount]

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None


class RecordingSocket:
    def __init__(self) -> None:
        self.bound_address: tuple[str, int] | None = None
        self.connected_address: tuple[object, ...] | None = None
        self.sent: list[bytes] = []
        self.timeout: object | None = None
        self.closed = False

    def settimeout(self, value: object) -> None:
        self.timeout = value

    def bind(self, address: tuple[str, int]) -> None:
        self.bound_address = address

    def connect(self, address: tuple[object, ...]) -> None:
        self.connected_address = address

    def setsockopt(self, level: int, option: int, value: int) -> None:
        return None

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


class RecordingTlsContext:
    verify_mode = ssl.CERT_REQUIRED
    check_hostname = True

    def __init__(self) -> None:
        self.server_hostnames: list[str | None] = []

    def wrap_socket(self, connection: RecordingSocket, *, server_hostname: str | None) -> RecordingSocket:
        self.server_hostnames.append(server_hostname)
        return connection


class NuplanRangeZipTest(unittest.TestCase):
    def test_reads_stored_and_deflated_members(self) -> None:
        source = build_zip(
            [
                ("data/cache/train/stored.db", b"stored database", zipfile.ZIP_STORED),
                ("data/cache/train/deflated.db", b"deflated database" * 100, zipfile.ZIP_DEFLATED),
            ]
        )
        client = MemoryRangeClient(source)
        archive = RangeZipArchive.open("memory://fixture", client=client)

        self.assertFalse(archive.metadata.uses_zip64)
        self.assertEqual(
            [member.name for member in archive.members()],
            ["data/cache/train/stored.db", "data/cache/train/deflated.db"],
        )
        self.assertEqual(archive.read_member("data/cache/train/stored.db"), b"stored database")
        self.assertEqual(archive.read_member("data/cache/train/deflated.db"), b"deflated database" * 100)
        self.assertTrue(all(request[2] == '"fixture-etag"' for request in client.requests[1:]))

    def test_reads_zip64_end_records_and_member_extra_fields(self) -> None:
        client = MemoryRangeClient(build_tiny_zip64())
        archive = RangeZipArchive.open("memory://zip64", client=client)

        self.assertTrue(archive.metadata.uses_zip64)
        self.assertEqual(archive.metadata.entry_count, 1)
        self.assertEqual(
            archive.read_member("data/cache/train/fixture.db"),
            b"zip64 fixture database",
        )

    def test_crc_mismatch_fails_before_extraction(self) -> None:
        client = MemoryRangeClient(build_zip([("safe.db", b"crc payload", zipfile.ZIP_STORED)]))
        archive = RangeZipArchive.open("memory://crc", client=client)
        member = archive.member("safe.db")
        fixed_size = 30
        name_length, extra_length = struct.unpack_from("<HH", client.payload, member.local_header_offset + 26)
        payload_offset = member.local_header_offset + fixed_size + name_length + extra_length
        client.payload[payload_offset] ^= 0x01

        with self.assertRaisesRegex(RangeZipError, "CRC32 mismatch"):
            archive.read_member(member)

    def test_extraction_rejects_path_traversal_and_writes_atomically(self) -> None:
        client = MemoryRangeClient(
            build_zip(
                [
                    ("../escape.db", b"no", zipfile.ZIP_STORED),
                    ("safe/inside.db", b"yes", zipfile.ZIP_STORED),
                ]
            )
        )
        archive = RangeZipArchive.open("memory://paths", client=client)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            with self.assertRaisesRegex(RangeZipError, "unsafe"):
                archive.extract_member("../escape.db", root)
            output = archive.extract_member("safe/inside.db", root)
            self.assertEqual(output, (root / "safe" / "inside.db").resolve())
            self.assertEqual(output.read_bytes(), b"yes")
            with self.assertRaisesRegex(RangeZipError, "overwrite"):
                archive.extract_member("safe/inside.db", root)
            remapped = archive.extract_member(
                "safe/inside.db",
                root,
                relative_path="splits/trainval/remapped.db",
            )
            self.assertEqual(remapped.read_bytes(), b"yes")

    def test_streamed_extraction_uses_bounded_payload_ranges(self) -> None:
        payload = os.urandom(96 * 1024 + 17)
        client = MemoryRangeClient(
            build_zip([("data/cache/train/large.db", payload, zipfile.ZIP_DEFLATED)])
        )
        archive = RangeZipArchive.open(
            "memory://streamed-extraction",
            client=client,
            member_chunk_bytes=4096,
        )
        member = archive.member("data/cache/train/large.db")
        local_name_length, local_extra_length = struct.unpack_from(
            "<HH", client.payload, member.local_header_offset + 26
        )
        payload_start = member.local_header_offset + 30 + local_name_length + local_extra_length
        payload_end = payload_start + member.compressed_size
        client.requests.clear()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            with mock.patch.object(archive, "read_member", side_effect=AssertionError("must not buffer member")):
                output = archive.extract_member(member, root)
            self.assertEqual(output.read_bytes(), payload)

        payload_ranges = [
            request
            for request in client.requests
            if payload_start <= request[0] < payload_end
        ]
        self.assertGreater(len(payload_ranges), 1)
        self.assertEqual(payload_ranges[0][0], payload_start)
        self.assertEqual(payload_ranges[-1][1], payload_end - 1)
        self.assertEqual(sum(end - start + 1 for start, end, _ in payload_ranges), member.compressed_size)
        self.assertTrue(all(end - start + 1 <= 4096 for start, end, _ in payload_ranges))
        self.assertTrue(all(etag == '"fixture-etag"' for _, _, etag in client.requests))

    def test_streamed_deflate_output_is_bounded_and_complete(self) -> None:
        payload = b"a highly compressible database row\n" * 40_000
        archive = RangeZipArchive.open(
            "memory://bounded-output",
            client=MemoryRangeClient(build_zip([("large.db", payload, zipfile.ZIP_DEFLATED)])),
            member_chunk_bytes=1024,
        )
        chunks: list[bytes] = []

        def collect(chunk: bytes) -> None:
            self.assertLessEqual(len(chunk), 1024)
            chunks.append(chunk)

        archive._stream_member(archive.member("large.db"), collect)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(b"".join(chunks), payload)

    def test_streamed_extraction_does_not_publish_crc_mismatch(self) -> None:
        client = MemoryRangeClient(
            build_zip([("safe.db", b"streamed CRC payload" * 4096, zipfile.ZIP_STORED)])
        )
        archive = RangeZipArchive.open("memory://streamed-crc", client=client, member_chunk_bytes=1024)
        member = archive.member("safe.db")
        local_name_length, local_extra_length = struct.unpack_from(
            "<HH", client.payload, member.local_header_offset + 26
        )
        payload_offset = member.local_header_offset + 30 + local_name_length + local_extra_length
        client.payload[payload_offset] ^= 0x01

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            target = root / "safe.db"
            with self.assertRaisesRegex(RangeZipError, "CRC32 mismatch"):
                archive.extract_member(member, root)
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".safe.db.*.tmp")), [])

    def test_http_client_requires_exact_partial_content(self) -> None:
        good = FakeHttpResponse(
            status=206,
            headers={
                "Content-Range": "bytes 3-5/10",
                "Content-Length": "3",
                "ETag": '"same"',
            },
            data=b"abc",
        )
        client = HttpRangeClient()
        with mock.patch("nuplan_range_zip.urllib.request.urlopen", return_value=good) as urlopen:
            response = client.fetch("https://example.invalid/object", 3, 5, etag='"same"')
        self.assertEqual(response.data, b"abc")
        self.assertEqual(response.total_size, 10)
        self.assertIsNotNone(urlopen.call_args.kwargs["context"])

        wrong_status = FakeHttpResponse(
            status=200,
            headers={"Content-Range": "bytes 3-5/10", "Content-Length": "3"},
            data=b"abc",
        )
        with mock.patch("nuplan_range_zip.urllib.request.urlopen", return_value=wrong_status):
            with self.assertRaisesRegex(RangeZipError, "instead of 206"):
                client.fetch("https://example.invalid/object", 3, 5)

        wrong_range = FakeHttpResponse(
            status=206,
            headers={"Content-Range": "bytes 2-5/10", "Content-Length": "4"},
            data=b"abcd",
        )
        with mock.patch("nuplan_range_zip.urllib.request.urlopen", return_value=wrong_range):
            with self.assertRaisesRegex(RangeZipError, "Server returned bytes"):
                client.fetch("https://example.invalid/object", 3, 5)

    def test_http_client_retries_a_transient_transport_failure_with_the_same_range(self) -> None:
        good = FakeHttpResponse(
            status=206,
            headers={
                "Content-Range": "bytes 3-5/10",
                "Content-Length": "3",
                "ETag": '"same"',
            },
            data=b"abc",
        )
        client = HttpRangeClient(max_attempts=2, retry_backoff_seconds=0.25)
        with mock.patch(
            "nuplan_range_zip.urllib.request.urlopen",
            side_effect=[urllib.error.URLError("temporary network failure"), good],
        ) as urlopen:
            with mock.patch("nuplan_range_zip.time.sleep") as sleep:
                response = client.fetch("https://example.invalid/object", 3, 5, etag='"same"')
        self.assertEqual(response.data, b"abc")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(urlopen.call_args_list[0].args[0].get_header("Range"), "bytes=3-5")
        self.assertEqual(urlopen.call_args_list[1].args[0].get_header("Range"), "bytes=3-5")
        sleep.assert_called_once_with(0.25)

    def test_http_client_rejects_a_full_object_response_before_reading_its_body(self) -> None:
        class FullObjectResponse(FakeHttpResponse):
            def read(self, amount: int | None = None) -> bytes:
                raise AssertionError("The range client must not read a non-206 response body")

        response = FullObjectResponse(status=200, headers={}, data=b"would be an entire archive")
        client = HttpRangeClient()
        with mock.patch("nuplan_range_zip.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RangeZipError, "instead of 206"):
                client.fetch("https://example.invalid/object", 3, 5)

    def test_direct_peer_connection_rotates_tcp_ips_but_preserves_sni_and_host(self) -> None:
        hostname = "bucket.s3.example.test"
        addresses = [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("203.0.113.10", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("203.0.113.11", 443)),
        ]
        first_socket = RecordingSocket()
        second_socket = RecordingSocket()
        tls_context = RecordingTlsContext()
        with mock.patch("nuplan_range_zip.socket.getaddrinfo", return_value=addresses) as resolver:
            pool = _DirectPeerPool((hostname,))
            with mock.patch(
                "nuplan_range_zip.socket.socket",
                side_effect=[first_socket, second_socket],
            ):
                first = _PinnedPeerHTTPSConnection(
                    hostname,
                    peer_pool=pool,
                    context=tls_context,  # type: ignore[arg-type]
                    timeout=4.0,
                )
                first.connect()
                first.putrequest("GET", "/object")
                first.endheaders()

                second = _PinnedPeerHTTPSConnection(
                    hostname,
                    peer_pool=pool,
                    context=tls_context,  # type: ignore[arg-type]
                    timeout=4.0,
                )
                second.connect()

        self.assertEqual(resolver.call_count, 1)
        self.assertEqual(first_socket.connected_address, ("203.0.113.10", 443))
        self.assertEqual(second_socket.connected_address, ("203.0.113.11", 443))
        self.assertEqual(tls_context.server_hostnames, [hostname, hostname])
        self.assertIn(b"\r\nHost: bucket.s3.example.test\r\n", b"".join(first_socket.sent))

    def test_direct_peer_client_uses_no_proxy_opener_and_preserves_retry_behavior(self) -> None:
        hostname = "bucket.s3.example.test"
        good = FakeHttpResponse(
            status=206,
            headers={
                "Content-Range": "bytes 3-5/10",
                "Content-Length": "3",
                "ETag": '"same"',
            },
            data=b"abc",
        )
        client = HttpRangeClient(
            direct_resolve_hosts=(hostname,),
            max_attempts=2,
            retry_backoff_seconds=0.25,
        )
        self.assertIsNotNone(client._direct_opener)
        # ProxyHandler({}) contributes no scheme handlers, so the direct
        # opener must contain only our origin-preserving HTTPS handler.
        https_handlers = client._direct_opener.handle_open["https"]  # type: ignore[union-attr]
        self.assertEqual(len(https_handlers), 1)
        self.assertEqual(type(https_handlers[0]).__name__, "_DirectResolvedHTTPSHandler")

        with mock.patch.object(
            client._direct_opener,  # type: ignore[arg-type]
            "open",
            side_effect=[urllib.error.URLError("temporary failure"), good],
        ) as direct_open:
            with mock.patch("nuplan_range_zip.urllib.request.urlopen") as regular_open:
                with mock.patch("nuplan_range_zip.time.sleep") as sleep:
                    response = client.fetch(
                        f"https://{hostname}/object", 3, 5, etag='"same"'
                    )

        self.assertEqual(response.data, b"abc")
        self.assertEqual(direct_open.call_count, 2)
        self.assertEqual(regular_open.call_count, 0)
        self.assertEqual(direct_open.call_args_list[0].args[0].full_url, f"https://{hostname}/object")
        self.assertEqual(direct_open.call_args_list[0].args[0].get_header("Range"), "bytes=3-5")
        self.assertEqual(direct_open.call_args_list[1].args[0].get_header("Range"), "bytes=3-5")
        sleep.assert_called_once_with(0.25)

    def test_direct_peer_client_rejects_unverified_context_and_unconfigured_origin(self) -> None:
        hostname = "bucket.s3.example.test"
        with self.assertRaisesRegex(ValueError, "CERT_REQUIRED"):
            HttpRangeClient(
                ssl_context=ssl._create_unverified_context(),
                direct_resolve_hosts=(hostname,),
            )

        client = HttpRangeClient(direct_resolve_hosts=(hostname,))
        with mock.patch.object(client._direct_opener, "open") as direct_open:  # type: ignore[arg-type]
            with self.assertRaisesRegex(RangeZipError, "only permits configured HTTPS origins"):
                client.fetch("https://other.example.test/object", 0, 0)
        direct_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
