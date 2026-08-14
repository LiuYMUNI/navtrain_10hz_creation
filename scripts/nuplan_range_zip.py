#!/usr/bin/env python3
"""Read selected members from a normal or ZIP64 archive over HTTP ranges.

This module is intentionally limited to real ZIP archives.  It is suitable
for the nuPlan v1.1 database archives, whose members are DEFLATE-compressed
SQLite databases.  The nuPlan camera files have a ``.zip`` suffix but are TAR
streams, so they must not be passed to this reader.

Only the Python standard library is required.  The reader fails closed: it
requires an exact HTTP 206 response for every range, pins all later reads to
the ETag observed during the initial probe, verifies ZIP/ZIP64 metadata, and
checks the member's uncompressed size and CRC32 before returning or atomically
publishing data.
"""

from __future__ import annotations

import binascii
import http.client
import io
import os
import re
import ssl
import socket
import struct
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

from nuplan_tls import NuPlanTlsError, verified_https_context


EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
CENTRAL_DIRECTORY_DIGITAL_SIGNATURE = b"PK\x05\x05"
ZIP64_EXTRA_FIELD_ID = 0x0001
UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF
DEFAULT_MEMBER_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_DIRECT_PEER_REFRESH_SECONDS = 30.0
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

EOCD_STRUCT = struct.Struct("<IHHHHIIH")
ZIP64_LOCATOR_STRUCT = struct.Struct("<IIQI")
ZIP64_EOCD_FIXED_STRUCT = struct.Struct("<IQHHIIQQQQ")
CENTRAL_DIRECTORY_FIXED_STRUCT = struct.Struct("<IHHHHHHIIIHHHHHII")
LOCAL_FILE_FIXED_STRUCT = struct.Struct("<IHHHHHIIIHH")
CONTENT_RANGE_PATTERN = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)


class RangeZipError(RuntimeError):
    """An HTTP, archive-format, integrity, or safe-extraction failure."""


@dataclass(frozen=True)
class HttpRangeResponse:
    """A validated one-range HTTP response."""

    data: bytes
    start: int
    end: int
    total_size: int
    etag: Optional[str]


@dataclass(frozen=True)
class ZipMember:
    """Central-directory metadata for one archive member."""

    name: str
    compression_method: int
    flags: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


@dataclass(frozen=True)
class ZipArchiveMetadata:
    """Stable metadata discovered while opening a range-readable archive."""

    url: str
    size: int
    etag: Optional[str]
    central_directory_offset: int
    central_directory_size: int
    entry_count: int
    uses_zip64: bool


def _normalized_https_hostname(host: str) -> str:
    normalized = str(host).strip().lower().rstrip(".")
    if not normalized:
        raise ValueError("HTTPS hostname must be nonempty")
    return normalized


def _require_verified_https_context(context: ssl.SSLContext) -> None:
    """Keep direct numeric dialing from weakening TLS verification."""

    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise ValueError(
            "Direct resolved HTTPS requires CERT_REQUIRED and hostname verification"
        )


class _DirectPeerPool:
    """Round-robin reachable IPv4 peers while retaining the TLS hostname."""

    def __init__(
        self,
        hosts: Iterable[str],
        *,
        refresh_seconds: float = DEFAULT_DIRECT_PEER_REFRESH_SECONDS,
    ) -> None:
        if isinstance(hosts, str):
            raise ValueError("direct_resolve_hosts must be an iterable of hostnames, not one string")
        normalized = {_normalized_https_hostname(host) for host in hosts}
        if not normalized:
            raise ValueError("direct peer rotation needs at least one hostname")
        if refresh_seconds <= 0:
            raise ValueError("direct peer refresh interval must be positive")
        self._hosts = frozenset(normalized)
        self._refresh_seconds = float(refresh_seconds)
        self._peers: Dict[str, Tuple[Tuple[int, int, int, str, tuple], ...]] = {}
        self._refreshed_at: Dict[str, float] = {}
        self._next_index: Dict[str, int] = {}
        self._lock = threading.Lock()

    def permits(self, host: str, port: int) -> bool:
        try:
            return port == 443 and _normalized_https_hostname(host) in self._hosts
        except ValueError:
            return False

    def next_peer(self, host: str, port: int) -> Tuple[int, int, int, str, tuple]:
        normalized_host = _normalized_https_hostname(host)
        if normalized_host not in self._hosts or port != 443:
            raise RangeZipError(f"Direct peer rotation rejected unexpected HTTPS host: {host!r}:{port}")
        now = time.monotonic()
        with self._lock:
            peers = self._peers.get(normalized_host)
            if peers is None or now - self._refreshed_at.get(normalized_host, 0.0) >= self._refresh_seconds:
                try:
                    resolved = socket.getaddrinfo(
                        normalized_host,
                        443,
                        family=socket.AF_INET,
                        type=socket.SOCK_STREAM,
                    )
                except OSError as exc:
                    raise OSError(
                        f"Could not resolve direct HTTPS peers for {normalized_host}: {exc}"
                    ) from exc
                deduplicated: list[Tuple[int, int, int, str, tuple]] = []
                seen: set[tuple] = set()
                for peer in resolved:
                    if peer[0] != socket.AF_INET or peer[1] != socket.SOCK_STREAM:
                        continue
                    key = (peer[0], peer[1], peer[2], peer[4])
                    if key not in seen:
                        seen.add(key)
                        deduplicated.append(peer)
                if not deduplicated:
                    raise OSError(f"No IPv4 HTTPS peers resolved for {normalized_host}")
                peers = tuple(deduplicated)
                self._peers[normalized_host] = peers
                self._refreshed_at[normalized_host] = now
            index = self._next_index.get(normalized_host, 0)
            self._next_index[normalized_host] = index + 1
            return peers[index % len(peers)]


class _PinnedPeerHTTPSConnection(http.client.HTTPSConnection):
    """Dial one numeric peer but preserve the logical HTTPS hostname for TLS."""

    def __init__(
        self,
        host: str,
        *,
        peer_pool: _DirectPeerPool,
        **kwargs: object,
    ) -> None:
        super().__init__(host, **kwargs)
        _require_verified_https_context(self._context)
        self._direct_peer = peer_pool.next_peer(self.host, self.port)
        # Leave HTTPSConnection.connect() intact. It wraps this socket with
        # server_hostname=self.host and HTTPConnection emits Host from self.host.
        self._create_connection = self._connect_direct_peer

    def _connect_direct_peer(
        self,
        logical_address: tuple[str, int],
        timeout: object = socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        if logical_address != (self.host, self.port):
            raise OSError(f"Unexpected HTTPS logical address: {logical_address!r}")
        family, socktype, protocol, _canonname, sockaddr = self._direct_peer
        sock = socket.socket(family, socktype, protocol)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)  # type: ignore[arg-type]
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except BaseException:
            sock.close()
            raise


class _DirectResolvedHTTPSHandler(urllib.request.HTTPSHandler):
    """Use direct, rotating peers without changing SNI, Host, or CA checks."""

    def __init__(self, *, peer_pool: _DirectPeerPool, context: ssl.SSLContext) -> None:
        _require_verified_https_context(context)
        super().__init__(context=context)
        self._peer_pool = peer_pool

    def https_open(self, request: urllib.request.Request) -> object:
        host_headers = [
            value
            for key, value in (*request.headers.items(), *request.unredirected_hdrs.items())
            if key.lower() == "host"
        ]
        # urllib adds this Host header before HTTPS handlers run. It is safe
        # only when it still names the logical URL authority rather than the
        # numeric peer selected for TCP.
        if getattr(request, "_tunnel_host", None) or host_headers != [request.host]:
            raise RangeZipError(
                "Direct resolved HTTPS requires the logical URL Host header and refuses proxy tunnels"
            )

        def connection_factory(host: str, **kwargs: object) -> _PinnedPeerHTTPSConnection:
            return _PinnedPeerHTTPSConnection(
                host,
                peer_pool=self._peer_pool,
                **kwargs,
            )

        return self.do_open(connection_factory, request, context=self._context)


class _RejectDirectPeerRedirects(urllib.request.HTTPRedirectHandler):
    """Keep a direct resolved range request pinned to its configured origin."""

    def redirect_request(
        self,
        request: urllib.request.Request,
        response: object,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request:
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "Redirects are not permitted for direct resolved range requests",
            headers,
            response,
        )


class HttpRangeClient:
    """Fetch exact byte ranges with strict HTTP and mutation checks."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        user_agent: str = "nuplan-range-zip/1",
        ssl_context: Optional[ssl.SSLContext] = None,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
        direct_resolve_hosts: Optional[Iterable[str]] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be nonnegative")
        self._timeout_seconds = float(timeout_seconds)
        self._user_agent = str(user_agent)
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = float(retry_backoff_seconds)
        try:
            self._ssl_context = ssl_context or verified_https_context()
        except NuPlanTlsError as exc:
            raise RangeZipError(f"Could not configure verified HTTPS transport: {exc}") from exc
        self._direct_opener: urllib.request.OpenerDirector | None = None
        self._direct_peer_pool: _DirectPeerPool | None = None
        if direct_resolve_hosts is not None:
            peer_pool = _DirectPeerPool(direct_resolve_hosts)
            _require_verified_https_context(self._ssl_context)
            self._direct_peer_pool = peer_pool
            self._direct_opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _RejectDirectPeerRedirects(),
                _DirectResolvedHTTPSHandler(peer_pool=peer_pool, context=self._ssl_context),
            )

    def _open_range_request(self, request: urllib.request.Request) -> object:
        if self._direct_opener is None:
            return urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
                context=self._ssl_context,
            )

        parsed = urllib.parse.urlsplit(request.full_url)
        try:
            port = parsed.port or 443
        except ValueError as exc:
            raise RangeZipError(f"Invalid direct resolved HTTPS URL: {request.full_url!r}") from exc
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname is None
            or self._direct_peer_pool is None
            or not self._direct_peer_pool.permits(parsed.hostname, port)
        ):
            raise RangeZipError(
                "Direct resolved range client only permits configured HTTPS origins on port 443"
            )
        return self._direct_opener.open(request, timeout=self._timeout_seconds)

    def fetch(
        self,
        url: str,
        start: int,
        end: int,
        *,
        etag: Optional[str] = None,
    ) -> HttpRangeResponse:
        """Fetch exactly inclusive ``start`` through ``end`` from ``url``.

        The server must return HTTP 206 and an exact ``Content-Range``.  A
        normal HTTP 200 is deliberately rejected so a missing range capability
        cannot trigger an accidental full archive download.
        """

        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError("Range start must be a nonnegative integer")
        if not isinstance(end, int) or isinstance(end, bool) or end < start:
            raise ValueError("Range end must be an integer no smaller than start")

        headers = {
            "Range": "bytes={}-{}".format(start, end),
            "Accept-Encoding": "identity",
            "User-Agent": self._user_agent,
        }
        if etag:
            headers["If-Match"] = etag
        request = urllib.request.Request(url, headers=headers, method="GET")
        expected_length = end - start + 1
        for attempt in range(1, self._max_attempts + 1):
            try:
                response_context = self._open_range_request(request)
                with response_context as response:
                    status = getattr(response, "status", response.getcode())
                    response_headers = response.headers
                    if status in RETRYABLE_HTTP_STATUSES and attempt < self._max_attempts:
                        self._sleep_before_retry(attempt)
                        continue
                    if status != 206:
                        raise RangeZipError(
                            "Range server returned HTTP {} instead of 206 for {} bytes {}-{}".format(
                                status, url, start, end
                            )
                        )
                    content_range = response_headers.get("Content-Range")
                    if content_range is None:
                        raise RangeZipError("HTTP 206 response lacks Content-Range for {}".format(url))
                    match = CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
                    if match is None:
                        raise RangeZipError(
                            "Malformed Content-Range {!r} for {}".format(content_range, url)
                        )
                    actual_start, actual_end, total_size = (int(value) for value in match.groups())
                    if actual_start != start or actual_end != end:
                        raise RangeZipError(
                            "Server returned bytes {}-{} for requested {}-{} from {}".format(
                                actual_start, actual_end, start, end, url
                            )
                        )
                    if total_size <= end:
                        raise RangeZipError(
                            "Content-Range total {} cannot contain requested end {} for {}".format(
                                total_size, end, url
                            )
                        )
                    content_length = response_headers.get("Content-Length")
                    if content_length is not None:
                        try:
                            parsed_content_length = int(content_length)
                        except ValueError as exc:
                            raise RangeZipError(
                                "Malformed Content-Length {!r} for {}".format(content_length, url)
                            ) from exc
                        if parsed_content_length != expected_length:
                            raise RangeZipError(
                                "Content-Length {} does not match requested range length {} for {}".format(
                                    parsed_content_length, expected_length, url
                                )
                            )
                    # A compliant 206 has the exact declared length.  Reading
                    # one byte beyond it also protects the no-Content-Length
                    # case from an intermediary returning a full object.
                    data = response.read(expected_length + 1)
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_HTTP_STATUSES and attempt < self._max_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise RangeZipError(
                    "HTTP range request failed for {} bytes {}-{}: HTTP {}".format(url, start, end, exc.code)
                ) from exc
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                if attempt < self._max_attempts:
                    self._sleep_before_retry(attempt)
                    continue
                raise RangeZipError(
                    "HTTP range request failed for {} bytes {}-{} after {} attempt(s): {}".format(
                        url, start, end, attempt, exc
                    )
                ) from exc
            break

        if len(data) != expected_length:
            raise RangeZipError(
                "Received {} bytes for requested {} bytes from {}".format(len(data), expected_length, url)
            )

        response_etag = response_headers.get("ETag")
        if etag is not None and response_etag != etag:
            raise RangeZipError(
                "ETag changed or disappeared while reading {}: expected {!r}, got {!r}".format(
                    url, etag, response_etag
                )
            )
        return HttpRangeResponse(
            data=data,
            start=actual_start,
            end=actual_end,
            total_size=total_size,
            etag=response_etag,
        )

    def _sleep_before_retry(self, completed_attempts: int) -> None:
        delay = self._retry_backoff_seconds * (2 ** (completed_attempts - 1))
        if delay:
            time.sleep(delay)


MemberReference = Union[str, ZipMember]


class RangeZipArchive:
    """A validated normal/ZIP64 archive backed by exact HTTP range requests."""

    def __init__(
        self,
        *,
        client: object,
        metadata: ZipArchiveMetadata,
        members: Tuple[ZipMember, ...],
        max_compressed_member_bytes: int,
        max_uncompressed_member_bytes: int,
        member_chunk_bytes: int = DEFAULT_MEMBER_CHUNK_BYTES,
    ) -> None:
        self._client = client
        self.metadata = metadata
        self._members = members
        self._members_by_name: Dict[str, ZipMember] = {member.name: member for member in members}
        self._max_compressed_member_bytes = max_compressed_member_bytes
        self._max_uncompressed_member_bytes = max_uncompressed_member_bytes
        self._member_chunk_bytes = member_chunk_bytes

    @classmethod
    def open(
        cls,
        url: str,
        *,
        client: Optional[object] = None,
        eocd_search_bytes: int = 128 * 1024,
        max_central_directory_bytes: int = 64 * 1024 * 1024,
        max_entries: int = 1_000_000,
        max_compressed_member_bytes: int = 512 * 1024 * 1024,
        max_uncompressed_member_bytes: int = 2 * 1024 * 1024 * 1024,
        member_chunk_bytes: int = DEFAULT_MEMBER_CHUNK_BYTES,
    ) -> "RangeZipArchive":
        """Open a real ZIP archive without downloading its payload wholesale.

        ``client`` need only provide ``fetch(url, start, end, *, etag=None)``
        returning :class:`HttpRangeResponse`; it permits deterministic tests or
        a controlled transport without weakening archive validation.
        """

        if not url:
            raise ValueError("url must be nonempty")
        if eocd_search_bytes < 65_557:
            raise ValueError("eocd_search_bytes must be at least 65557 bytes")
        if max_central_directory_bytes < 0 or max_entries < 0:
            raise ValueError("central-directory limits must be nonnegative")
        if max_compressed_member_bytes < 0 or max_uncompressed_member_bytes < 0:
            raise ValueError("member-size limits must be nonnegative")
        if (
            not isinstance(member_chunk_bytes, int)
            or isinstance(member_chunk_bytes, bool)
            or member_chunk_bytes <= 0
        ):
            raise ValueError("member_chunk_bytes must be a positive integer")

        active_client = client if client is not None else HttpRangeClient()
        probe = _fetch_range(active_client, url, 0, 0, etag=None)
        size = probe.total_size
        if size <= 0:
            raise RangeZipError("Archive is empty: {}".format(url))
        etag = probe.etag

        tail_start = max(0, size - min(size, eocd_search_bytes))
        tail = _fetch_range(active_client, url, tail_start, size - 1, etag=etag).data
        eocd_relative_offset = _find_eocd(tail)
        if eocd_relative_offset is None:
            raise RangeZipError(
                "No valid ZIP end-of-central-directory record was found in the final {} bytes of {}; "
                "this is not a supported ZIP archive".format(len(tail), url)
            )
        eocd_offset = tail_start + eocd_relative_offset
        (
            _signature,
            disk_number,
            central_directory_disk,
            entries_on_disk,
            entry_count,
            central_directory_size,
            central_directory_offset,
            _comment_length,
        ) = EOCD_STRUCT.unpack_from(tail, eocd_relative_offset)

        uses_zip64 = _requires_zip64(
            entries_on_disk,
            entry_count,
            central_directory_size,
            central_directory_offset,
        )
        if disk_number != 0 or central_directory_disk != 0:
            raise RangeZipError("Multi-disk ZIP archives are not supported: {}".format(url))
        if uses_zip64:
            if eocd_offset < ZIP64_LOCATOR_STRUCT.size:
                raise RangeZipError("ZIP64 EOCD locator precedes archive start: {}".format(url))
            locator_offset = eocd_offset - ZIP64_LOCATOR_STRUCT.size
            locator = _fetch_range(
                active_client,
                url,
                locator_offset,
                eocd_offset - 1,
                etag=etag,
            ).data
            (
                locator_signature,
                zip64_eocd_disk,
                zip64_eocd_offset,
                zip64_total_disks,
            ) = ZIP64_LOCATOR_STRUCT.unpack(locator)
            if locator_signature != int.from_bytes(ZIP64_LOCATOR_SIGNATURE, "little"):
                raise RangeZipError("ZIP64 archive lacks a valid ZIP64 EOCD locator: {}".format(url))
            if zip64_eocd_disk != 0 or zip64_total_disks != 1:
                raise RangeZipError("Multi-disk ZIP64 archives are not supported: {}".format(url))
            if zip64_eocd_offset + ZIP64_EOCD_FIXED_STRUCT.size > locator_offset:
                raise RangeZipError("ZIP64 EOCD record overlaps its locator in {}".format(url))
            zip64_fixed = _fetch_range(
                active_client,
                url,
                zip64_eocd_offset,
                zip64_eocd_offset + ZIP64_EOCD_FIXED_STRUCT.size - 1,
                etag=etag,
            ).data
            (
                zip64_signature,
                zip64_record_size,
                _version_made_by,
                _version_needed,
                zip64_disk_number,
                zip64_central_directory_disk,
                zip64_entries_on_disk,
                zip64_entry_count,
                zip64_central_directory_size,
                zip64_central_directory_offset,
            ) = ZIP64_EOCD_FIXED_STRUCT.unpack(zip64_fixed)
            if zip64_signature != int.from_bytes(ZIP64_EOCD_SIGNATURE, "little") or zip64_record_size < 44:
                raise RangeZipError("Invalid ZIP64 EOCD record in {}".format(url))
            if zip64_eocd_offset + 12 + zip64_record_size > locator_offset:
                raise RangeZipError("ZIP64 EOCD record length is invalid in {}".format(url))
            if zip64_disk_number != 0 or zip64_central_directory_disk != 0:
                raise RangeZipError("Multi-disk ZIP64 archives are not supported: {}".format(url))
            if zip64_entries_on_disk != zip64_entry_count:
                raise RangeZipError("ZIP64 central directory spans multiple disks in {}".format(url))
            entry_count = zip64_entry_count
            central_directory_size = zip64_central_directory_size
            central_directory_offset = zip64_central_directory_offset
            central_directory_limit = zip64_eocd_offset
        else:
            if entries_on_disk != entry_count:
                raise RangeZipError("Central directory spans multiple disks in {}".format(url))
            central_directory_limit = eocd_offset

        _validate_central_directory_bounds(
            size=size,
            central_directory_offset=central_directory_offset,
            central_directory_size=central_directory_size,
            central_directory_limit=central_directory_limit,
            max_central_directory_bytes=max_central_directory_bytes,
            max_entries=max_entries,
            url=url,
        )
        if central_directory_size:
            central_directory = _fetch_range(
                active_client,
                url,
                central_directory_offset,
                central_directory_offset + central_directory_size - 1,
                etag=etag,
            ).data
        else:
            central_directory = b""
        members = _parse_central_directory(central_directory, entry_count, url=url)
        metadata = ZipArchiveMetadata(
            url=url,
            size=size,
            etag=etag,
            central_directory_offset=central_directory_offset,
            central_directory_size=central_directory_size,
            entry_count=entry_count,
            uses_zip64=uses_zip64,
        )
        return cls(
            client=active_client,
            metadata=metadata,
            members=members,
            max_compressed_member_bytes=max_compressed_member_bytes,
            max_uncompressed_member_bytes=max_uncompressed_member_bytes,
            member_chunk_bytes=member_chunk_bytes,
        )

    def members(self) -> Tuple[ZipMember, ...]:
        """Return central-directory members in archive order."""

        return self._members

    def member(self, name: str) -> ZipMember:
        """Return a member by its exact archive path."""

        try:
            return self._members_by_name[name]
        except KeyError as exc:
            raise RangeZipError("Archive does not contain member {!r}".format(name)) from exc

    def read_member(self, member: MemberReference) -> bytes:
        """Fetch, decompress, and integrity-check one archive member in memory.

        This convenience API necessarily retains the uncompressed result in
        memory.  Use :meth:`extract_member` for large members; it streams both
        HTTP ranges and decompressed output through a bounded buffer.
        """

        payload = io.BytesIO()
        self._stream_member(self._resolve_member(member), payload.write)
        return payload.getvalue()

    def extract_member(
        self,
        member: MemberReference,
        destination_root: Path,
        *,
        relative_path: Optional[Union[str, Path]] = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a verified member below ``destination_root`` atomically.

        The default relative output path is the archive member path.  Set a
        safe relative ``relative_path`` when the staging layout intentionally
        differs from the archive's internal layout.  Absolute paths,
        ``..`` components, Windows drive paths, and backslashes are rejected.
        Existing targets are retained unless ``overwrite=True`` is explicit.
        """

        selected = self._resolve_member(member)
        candidate = selected.name if relative_path is None else str(relative_path)
        safe_relative = _safe_relative_path(candidate)
        root = Path(destination_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / safe_relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RangeZipError("Extraction target escapes destination root: {!r}".format(candidate)) from exc
        if target.exists() and not overwrite:
            raise RangeZipError("Refusing to overwrite existing extraction target: {}".format(target))

        target.parent.mkdir(parents=True, exist_ok=True)
        parent_resolved = target.parent.resolve()
        parent_is_below_root = (
            parent_resolved.is_relative_to(root)
            if hasattr(Path, "is_relative_to")
            else _is_below(parent_resolved, root)
        )
        if not parent_is_below_root:
            raise RangeZipError("Extraction parent escapes destination root: {}".format(target.parent))
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".{}.".format(target.name), suffix=".tmp", dir=str(target.parent)
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                self._stream_member(selected, handle.write)
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists() and not overwrite:
                raise RangeZipError("Refusing to overwrite existing extraction target: {}".format(target))
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        return target

    def _stream_member(self, selected: ZipMember, write: Callable[[bytes], object]) -> None:
        """Validate and stream one member to ``write`` without full buffering."""

        payload_start, payload_end = self._member_payload_bounds(selected)
        written_size = 0
        crc32 = 0

        def write_verified(data: bytes) -> None:
            nonlocal written_size, crc32
            if not data:
                return
            remaining = selected.uncompressed_size - written_size
            if len(data) > remaining:
                raise RangeZipError(
                    "Member {!r} expands beyond declared size {}".format(
                        selected.name, selected.uncompressed_size
                    )
                )
            write_result = write(data)
            if write_result is not None and write_result != len(data):
                raise RangeZipError(
                    "Short write while extracting {!r}: wrote {}, expected {} bytes".format(
                        selected.name, write_result, len(data)
                    )
                )
            written_size += len(data)
            crc32 = binascii.crc32(data, crc32)

        if selected.compression_method == 0:
            if selected.compressed_size != selected.uncompressed_size:
                raise RangeZipError("Stored member has inconsistent sizes: {!r}".format(selected.name))
            for compressed in self._iter_member_payload_chunks(payload_start, payload_end):
                write_verified(compressed)
        else:
            self._stream_deflate_member(
                selected,
                payload_start=payload_start,
                payload_end=payload_end,
                write=write_verified,
            )

        if written_size != selected.uncompressed_size:
            raise RangeZipError(
                "Member {!r} has uncompressed size {}, expected {}".format(
                    selected.name, written_size, selected.uncompressed_size
                )
            )
        actual_crc32 = crc32 & UINT32_MAX
        if actual_crc32 != selected.crc32:
            raise RangeZipError(
                "CRC32 mismatch for {!r}: expected {:08x}, got {:08x}".format(
                    selected.name, selected.crc32, actual_crc32
                )
            )

    def _member_payload_bounds(self, selected: ZipMember) -> Tuple[int, int]:
        """Validate a member's local header and return its payload bounds."""

        if selected.flags & 0x1:
            raise RangeZipError("Encrypted ZIP members are not supported: {!r}".format(selected.name))
        if selected.compression_method not in (0, 8):
            raise RangeZipError(
                "Unsupported ZIP compression method {} for {!r}; only stored and DEFLATE are supported".format(
                    selected.compression_method, selected.name
                )
            )
        if selected.compressed_size > self._max_compressed_member_bytes:
            raise RangeZipError(
                "Compressed member {!r} is {} bytes, above configured limit {}".format(
                    selected.name, selected.compressed_size, self._max_compressed_member_bytes
                )
            )
        if selected.uncompressed_size > self._max_uncompressed_member_bytes:
            raise RangeZipError(
                "Uncompressed member {!r} is {} bytes, above configured limit {}".format(
                    selected.name, selected.uncompressed_size, self._max_uncompressed_member_bytes
                )
            )

        local_fixed = _fetch_range(
            self._client,
            self.metadata.url,
            selected.local_header_offset,
            selected.local_header_offset + LOCAL_FILE_FIXED_STRUCT.size - 1,
            etag=self.metadata.etag,
        ).data
        (
            local_signature,
            _local_version_needed,
            local_flags,
            local_compression_method,
            _local_modified_time,
            _local_modified_date,
            _local_crc32,
            _local_compressed_size,
            _local_uncompressed_size,
            local_name_length,
            local_extra_length,
        ) = LOCAL_FILE_FIXED_STRUCT.unpack(local_fixed)
        if local_signature != int.from_bytes(LOCAL_FILE_SIGNATURE, "little"):
            raise RangeZipError("Local header signature mismatch for {!r}".format(selected.name))
        if local_flags != selected.flags or local_compression_method != selected.compression_method:
            raise RangeZipError("Local header metadata mismatch for {!r}".format(selected.name))
        variable_header_size = local_name_length + local_extra_length
        payload_start = selected.local_header_offset + LOCAL_FILE_FIXED_STRUCT.size + variable_header_size
        if payload_start > self.metadata.central_directory_offset:
            raise RangeZipError("Local header exceeds central directory for {!r}".format(selected.name))
        variable_header = (
            _fetch_range(
                self._client,
                self.metadata.url,
                selected.local_header_offset + LOCAL_FILE_FIXED_STRUCT.size,
                payload_start - 1,
                etag=self.metadata.etag,
            ).data
            if variable_header_size
            else b""
        )
        local_name = _decode_name(variable_header[:local_name_length], selected.flags, context="local header")
        if local_name != selected.name:
            raise RangeZipError(
                "Local header name {!r} does not match central directory name {!r}".format(local_name, selected.name)
            )

        payload_end = payload_start + selected.compressed_size
        if payload_end > self.metadata.central_directory_offset:
            raise RangeZipError("Member payload exceeds central directory for {!r}".format(selected.name))
        return payload_start, payload_end

    def _iter_member_payload_chunks(self, payload_start: int, payload_end: int) -> Iterable[bytes]:
        """Yield exact, ETag-pinned payload ranges no larger than one chunk."""

        offset = payload_start
        while offset < payload_end:
            next_offset = min(payload_end, offset + self._member_chunk_bytes)
            yield _fetch_range(
                self._client,
                self.metadata.url,
                offset,
                next_offset - 1,
                etag=self.metadata.etag,
            ).data
            offset = next_offset

    def _stream_deflate_member(
        self,
        selected: ZipMember,
        *,
        payload_start: int,
        payload_end: int,
        write: Callable[[bytes], object],
    ) -> None:
        """Inflate a raw-DEFLATE member with bounded compressed/output chunks."""

        try:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            compressed_offset = payload_start
            for compressed in self._iter_member_payload_chunks(payload_start, payload_end):
                compressed_offset += len(compressed)
                pending = compressed
                while pending:
                    before = len(pending)
                    output = decompressor.decompress(pending, self._member_chunk_bytes)
                    pending = decompressor.unconsumed_tail
                    write(output)
                    if decompressor.eof:
                        if decompressor.unused_data or pending or compressed_offset != payload_end:
                            raise RangeZipError(
                                "DEFLATE stream is malformed or has trailing bytes for {!r}".format(selected.name)
                            )
                        break
                    if len(pending) >= before and not output:
                        raise RangeZipError("DEFLATE stream made no progress for {!r}".format(selected.name))
                if decompressor.eof:
                    break
            if not decompressor.eof or decompressor.unused_data:
                raise RangeZipError("DEFLATE stream is malformed or has trailing bytes for {!r}".format(selected.name))
            while True:
                output = decompressor.flush(self._member_chunk_bytes)
                if not output:
                    break
                write(output)
        except zlib.error as exc:
            raise RangeZipError("Could not DEFLATE-decompress {!r}: {}".format(selected.name, exc)) from exc

    def _resolve_member(self, member: MemberReference) -> ZipMember:
        if isinstance(member, str):
            return self.member(member)
        if not isinstance(member, ZipMember):
            raise TypeError("member must be a ZipMember or exact member name")
        known = self._members_by_name.get(member.name)
        if known != member:
            raise RangeZipError("ZipMember does not belong to this archive: {!r}".format(member.name))
        return known


def _fetch_range(client: object, url: str, start: int, end: int, *, etag: Optional[str]) -> HttpRangeResponse:
    try:
        response = client.fetch(url, start, end, etag=etag)  # type: ignore[attr-defined]
    except RangeZipError:
        raise
    except Exception as exc:
        raise RangeZipError(
            "Range client failed for {} bytes {}-{}: {}".format(url, start, end, exc)
        ) from exc
    if not isinstance(response, HttpRangeResponse):
        raise RangeZipError("Range client returned an invalid response object for {}".format(url))
    if response.start != start or response.end != end:
        raise RangeZipError(
            "Range client returned bytes {}-{} for requested {}-{} from {}".format(
                response.start, response.end, start, end, url
            )
        )
    if response.total_size <= end:
        raise RangeZipError("Range client reported invalid object size {} for {}".format(response.total_size, url))
    if len(response.data) != end - start + 1:
        raise RangeZipError("Range client returned an incorrect byte count for {}".format(url))
    if etag is not None and response.etag != etag:
        raise RangeZipError(
            "Range client ETag changed or disappeared for {}: expected {!r}, got {!r}".format(
                url, etag, response.etag
            )
        )
    return response


def _find_eocd(tail: bytes) -> Optional[int]:
    """Find an EOCD whose comment ends exactly at the object end."""

    search_end = len(tail)
    while True:
        offset = tail.rfind(EOCD_SIGNATURE, 0, search_end)
        if offset < 0:
            return None
        if offset + EOCD_STRUCT.size <= len(tail):
            comment_length = struct.unpack_from("<H", tail, offset + 20)[0]
            if offset + EOCD_STRUCT.size + comment_length == len(tail):
                return offset
        search_end = offset


def _requires_zip64(
    entries_on_disk: int,
    entry_count: int,
    central_directory_size: int,
    central_directory_offset: int,
) -> bool:
    return (
        entries_on_disk == UINT16_MAX
        or entry_count == UINT16_MAX
        or central_directory_size == UINT32_MAX
        or central_directory_offset == UINT32_MAX
    )


def _validate_central_directory_bounds(
    *,
    size: int,
    central_directory_offset: int,
    central_directory_size: int,
    central_directory_limit: int,
    max_central_directory_bytes: int,
    max_entries: int,
    url: str,
) -> None:
    if central_directory_offset < 0 or central_directory_size < 0:
        raise RangeZipError("Negative central-directory bounds in {}".format(url))
    if central_directory_size > max_central_directory_bytes:
        raise RangeZipError(
            "Central directory is {} bytes, above configured limit {} for {}".format(
                central_directory_size, max_central_directory_bytes, url
            )
        )
    if central_directory_offset + central_directory_size > central_directory_limit:
        raise RangeZipError("Central directory overlaps end records in {}".format(url))
    if central_directory_offset + central_directory_size > size:
        raise RangeZipError("Central directory exceeds archive size in {}".format(url))
    if max_entries and max_entries < 1:
        raise RangeZipError("max_entries must be zero or positive")


def _parse_central_directory(data: bytes, entry_count: int, *, url: str) -> Tuple[ZipMember, ...]:
    if entry_count < 0:
        raise RangeZipError("Negative central-directory entry count in {}".format(url))
    cursor = 0
    members = []
    names = set()
    for index in range(entry_count):
        if cursor + CENTRAL_DIRECTORY_FIXED_STRUCT.size > len(data):
            raise RangeZipError("Central directory ends before entry {} in {}".format(index, url))
        fields = CENTRAL_DIRECTORY_FIXED_STRUCT.unpack_from(data, cursor)
        if fields[0] != int.from_bytes(CENTRAL_DIRECTORY_SIGNATURE, "little"):
            raise RangeZipError("Invalid central-directory signature at entry {} in {}".format(index, url))
        (
            _signature,
            _version_made_by,
            _version_needed,
            flags,
            compression_method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = fields
        entry_end = cursor + CENTRAL_DIRECTORY_FIXED_STRUCT.size + name_length + extra_length + comment_length
        if entry_end > len(data):
            raise RangeZipError("Truncated central-directory entry {} in {}".format(index, url))
        name_start = cursor + CENTRAL_DIRECTORY_FIXED_STRUCT.size
        name_bytes = data[name_start : name_start + name_length]
        extra_start = name_start + name_length
        extra = data[extra_start : extra_start + extra_length]
        name = _decode_name(name_bytes, flags, context="central directory")
        if not name or "\x00" in name:
            raise RangeZipError("Invalid central-directory member name at entry {} in {}".format(index, url))
        (
            compressed_size,
            uncompressed_size,
            local_header_offset,
            disk_start,
        ) = _resolve_zip64_values(
            extra,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
            disk_start=disk_start,
            name=name,
        )
        if disk_start != 0:
            raise RangeZipError("Multi-disk member {!r} is not supported".format(name))
        if name in names:
            raise RangeZipError("Duplicate central-directory member {!r} in {}".format(name, url))
        names.add(name)
        members.append(
            ZipMember(
                name=name,
                compression_method=compression_method,
                flags=flags,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        cursor = entry_end
    trailing = data[cursor:]
    if trailing and not _is_valid_digital_signature(trailing):
        raise RangeZipError("Unexpected bytes after central-directory records in {}".format(url))
    return tuple(members)


def _resolve_zip64_values(
    extra: bytes,
    *,
    compressed_size: int,
    uncompressed_size: int,
    local_header_offset: int,
    disk_start: int,
    name: str,
) -> Tuple[int, int, int, int]:
    requires_zip64 = (
        compressed_size == UINT32_MAX
        or uncompressed_size == UINT32_MAX
        or local_header_offset == UINT32_MAX
        or disk_start == UINT16_MAX
    )
    if not requires_zip64:
        return compressed_size, uncompressed_size, local_header_offset, disk_start
    zip64_payload = None
    cursor = 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise RangeZipError("Malformed extra field for {!r}".format(name))
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        field_start = cursor + 4
        field_end = field_start + field_size
        if field_end > len(extra):
            raise RangeZipError("Truncated extra field for {!r}".format(name))
        if field_id == ZIP64_EXTRA_FIELD_ID:
            zip64_payload = extra[field_start:field_end]
            break
        cursor = field_end
    if zip64_payload is None:
        raise RangeZipError("ZIP64 values are missing their extra field for {!r}".format(name))
    cursor = 0

    def read_uint64(label: str) -> int:
        nonlocal cursor
        if cursor + 8 > len(zip64_payload):
            raise RangeZipError("ZIP64 {} is truncated for {!r}".format(label, name))
        value = struct.unpack_from("<Q", zip64_payload, cursor)[0]
        cursor += 8
        return value

    def read_uint32(label: str) -> int:
        nonlocal cursor
        if cursor + 4 > len(zip64_payload):
            raise RangeZipError("ZIP64 {} is truncated for {!r}".format(label, name))
        value = struct.unpack_from("<I", zip64_payload, cursor)[0]
        cursor += 4
        return value

    if uncompressed_size == UINT32_MAX:
        uncompressed_size = read_uint64("uncompressed size")
    if compressed_size == UINT32_MAX:
        compressed_size = read_uint64("compressed size")
    if local_header_offset == UINT32_MAX:
        local_header_offset = read_uint64("local header offset")
    if disk_start == UINT16_MAX:
        disk_start = read_uint32("disk start")
    return compressed_size, uncompressed_size, local_header_offset, disk_start


def _is_valid_digital_signature(data: bytes) -> bool:
    if len(data) < 6 or not data.startswith(CENTRAL_DIRECTORY_DIGITAL_SIGNATURE):
        return False
    signature_length = struct.unpack_from("<H", data, 4)[0]
    return len(data) == 6 + signature_length


def _decode_name(value: bytes, flags: int, *, context: str) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        return value.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise RangeZipError("Could not decode {} filename as {}".format(context, encoding)) from exc


def _safe_relative_path(value: str) -> Path:
    if not value or "\x00" in value or "\\" in value:
        raise RangeZipError("Extraction relative path is empty or unsafe: {!r}".format(value))
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise RangeZipError("Extraction relative path must not be absolute: {!r}".format(value))
    parts = posix_path.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise RangeZipError("Extraction relative path contains an unsafe component: {!r}".format(value))
    return Path(*parts)


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
