#!/usr/bin/env python3
"""Read selected entries from an uncompressed POSIX TAR object over HTTP ranges.

nuPlan v1.1 camera objects are named ``*.zip`` but are uncompressed TAR
streams.  TAR has no central directory, so this module deliberately walks
validated TAR headers.  Callers may choose a larger scan chunk to make a
sequential remote walk practical; requested payloads are still written only
for explicitly selected members.  It shares the strict HTTP-206 and
ETag-pinning transport used for the native database ZIP archives.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterator, Optional, Union

from nuplan_range_zip import HttpRangeClient, HttpRangeResponse, RangeZipError


TAR_BLOCK_SIZE = 512
TAR_REGULAR_TYPES = {b"\0", b"0"}
TAR_DIRECTORY_TYPE = b"5"
TAR_EXTENDED_TYPES = {b"x", b"g", b"L", b"K"}


class RangeTarError(RuntimeError):
    """An HTTP, TAR-format, integrity, or safe-extraction failure."""


@dataclass(frozen=True)
class TarArchiveMetadata:
    url: str
    size: int
    etag: str | None


@dataclass(frozen=True)
class TarMember:
    name: str
    typeflag: bytes
    size: int
    header_offset: int
    data_offset: int
    next_header_offset: int

    @property
    def is_regular_file(self) -> bool:
        return self.typeflag in TAR_REGULAR_TYPES

    @property
    def is_directory(self) -> bool:
        return self.typeflag == TAR_DIRECTORY_TYPE


MemberReference = Union[TarMember, str]


class RangeTarArchive:
    """A range-readable uncompressed single-volume POSIX TAR object."""

    def __init__(
        self,
        *,
        client: object,
        metadata: TarArchiveMetadata,
        max_member_bytes: int,
        scan_chunk_bytes: int,
    ) -> None:
        self._client = client
        self.metadata = metadata
        self._max_member_bytes = max_member_bytes
        self._scan_chunk_bytes = scan_chunk_bytes
        self._scan_cache_start: int | None = None
        self._scan_cache = b""

    @classmethod
    def open(
        cls,
        url: str,
        *,
        client: object | None = None,
        max_member_bytes: int = 64 * 1024 * 1024,
        scan_chunk_bytes: int = TAR_BLOCK_SIZE,
    ) -> "RangeTarArchive":
        if not url:
            raise ValueError("url must be nonempty")
        if max_member_bytes < 0:
            raise ValueError("max_member_bytes must be nonnegative")
        if (
            not isinstance(scan_chunk_bytes, int)
            or isinstance(scan_chunk_bytes, bool)
            or scan_chunk_bytes < TAR_BLOCK_SIZE
            or scan_chunk_bytes % TAR_BLOCK_SIZE
        ):
            raise ValueError("scan_chunk_bytes must be a positive multiple of 512")
        active_client = client if client is not None else HttpRangeClient(user_agent="nuplan-range-tar/1")
        probe = _fetch_range(active_client, url, 0, 0, etag=None)
        if probe.total_size < TAR_BLOCK_SIZE * 2:
            raise RangeTarError(f"TAR object is too small: {url}")
        return cls(
            client=active_client,
            metadata=TarArchiveMetadata(url=url, size=probe.total_size, etag=probe.etag),
            max_member_bytes=max_member_bytes,
            scan_chunk_bytes=scan_chunk_bytes,
        )

    def iter_members(self, *, start_header_offset: int = 0) -> Iterator[TarMember]:
        """Yield members sequentially from a validated TAR-header boundary."""

        if start_header_offset < 0 or start_header_offset % TAR_BLOCK_SIZE:
            raise ValueError("start_header_offset must be a nonnegative 512-byte boundary")
        offset = start_header_offset
        if offset >= self.metadata.size:
            raise RangeTarError("TAR start header offset is outside the object")
        while offset + TAR_BLOCK_SIZE <= self.metadata.size:
            block = self._scan_bytes(offset, TAR_BLOCK_SIZE)
            if block == b"\0" * TAR_BLOCK_SIZE:
                following = offset + TAR_BLOCK_SIZE
                if following + TAR_BLOCK_SIZE > self.metadata.size:
                    raise RangeTarError("TAR has only one zero termination block")
                second = self._scan_bytes(following, TAR_BLOCK_SIZE)
                if second != b"\0" * TAR_BLOCK_SIZE:
                    raise RangeTarError("TAR zero block is not followed by a second termination block")
                return
            member = _parse_header(block, offset=offset, archive_size=self.metadata.size)
            if member.typeflag in TAR_EXTENDED_TYPES:
                raise RangeTarError(
                    f"Unsupported PAX/GNU TAR extension {member.typeflag!r} at byte {offset} in {self.metadata.url}"
                )
            yield member
            offset = member.next_header_offset
        raise RangeTarError("TAR reaches object end without the required two zero termination blocks")

    def read_member(self, member: TarMember) -> bytes:
        if not isinstance(member, TarMember):
            raise TypeError("member must be a TarMember")
        if not member.is_regular_file:
            raise RangeTarError(f"Requested TAR member is not a regular file: {member.name!r}")
        if member.size > self._max_member_bytes:
            raise RangeTarError(
                f"TAR member {member.name!r} is {member.size} bytes, above configured limit {self._max_member_bytes}"
            )
        if member.size == 0:
            return b""
        cached = self._cached_bytes(member.data_offset, member.size)
        if cached is not None:
            return cached
        response = _fetch_range(
            self._client,
            self.metadata.url,
            member.data_offset,
            member.data_offset + member.size - 1,
            etag=self.metadata.etag,
        )
        if len(response.data) != member.size:
            raise RangeTarError(f"TAR member payload size mismatch for {member.name!r}")
        return response.data

    def _scan_bytes(self, start: int, size: int) -> bytes:
        """Read a small header range from a validated sequential scan cache."""

        cached = self._cached_bytes(start, size)
        if cached is not None:
            return cached
        end = min(self.metadata.size - 1, start + self._scan_chunk_bytes - 1)
        response = _fetch_range(
            self._client,
            self.metadata.url,
            start,
            end,
            etag=self.metadata.etag,
        )
        self._scan_cache_start = start
        self._scan_cache = response.data
        cached = self._cached_bytes(start, size)
        if cached is None:
            raise RangeTarError(
                f"Remote TAR scan cache is shorter than requested header range {start}+{size}"
            )
        return cached

    def _cached_bytes(self, start: int, size: int) -> bytes | None:
        """Return a byte range if it lies fully inside the current scan chunk."""

        cache_start = self._scan_cache_start
        if cache_start is None:
            return None
        relative_start = start - cache_start
        relative_end = relative_start + size
        if relative_start < 0 or relative_end > len(self._scan_cache):
            return None
        return self._scan_cache[relative_start:relative_end]

    def extract_member(
        self,
        member: TarMember,
        destination_root: Path,
        *,
        relative_path: str | Path,
        overwrite: bool = False,
    ) -> Path:
        safe_relative = _safe_relative_path(str(relative_path))
        root = Path(destination_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = (root / safe_relative).resolve()
        if not _is_below(target, root):
            raise RangeTarError(f"Extraction target escapes destination root: {relative_path!r}")
        if target.exists() and not overwrite:
            raise RangeTarError(f"Refusing to overwrite existing TAR extraction target: {target}")
        payload = self.read_member(member)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _is_below(target.parent.resolve(), root):
            raise RangeTarError(f"Extraction parent escapes destination root: {target.parent}")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists() and not overwrite:
                raise RangeTarError(f"Refusing to overwrite existing TAR extraction target: {target}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


def _fetch_range(client: object, url: str, start: int, end: int, *, etag: str | None) -> HttpRangeResponse:
    try:
        response = client.fetch(url, start, end, etag=etag)  # type: ignore[attr-defined]
    except RangeZipError as exc:
        raise RangeTarError(str(exc)) from exc
    except Exception as exc:
        raise RangeTarError(f"Range client failed for {url} bytes {start}-{end}: {exc}") from exc
    if not isinstance(response, HttpRangeResponse):
        raise RangeTarError(f"Range client returned an invalid response object for {url}")
    if response.start != start or response.end != end or len(response.data) != end - start + 1:
        raise RangeTarError(f"Range client returned an incorrect payload for {url} bytes {start}-{end}")
    if response.total_size <= end:
        raise RangeTarError(f"Range client reported invalid object size {response.total_size} for {url}")
    if etag is not None and response.etag != etag:
        raise RangeTarError(f"TAR ETag changed or disappeared for {url}")
    return response


def _parse_header(block: bytes, *, offset: int, archive_size: int) -> TarMember:
    if len(block) != TAR_BLOCK_SIZE:
        raise RangeTarError(f"TAR header at {offset} has length {len(block)}, not 512")
    expected_checksum = _parse_number(block[148:156], label="checksum", offset=offset)
    checksum_block = bytearray(block)
    checksum_block[148:156] = b" " * 8
    actual_checksum = sum(checksum_block)
    if actual_checksum != expected_checksum:
        raise RangeTarError(
            f"TAR checksum mismatch at byte {offset}: expected {expected_checksum}, got {actual_checksum}"
        )
    name = _decode_field(block[0:100], field="name", offset=offset)
    prefix = _decode_field(block[345:500], field="prefix", offset=offset)
    if prefix:
        name = f"{prefix}/{name}" if name else prefix
    _validate_member_name(name, offset=offset)
    size = _parse_number(block[124:136], label="size", offset=offset)
    typeflag = block[156:157]
    padded_size = ((size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE
    data_offset = offset + TAR_BLOCK_SIZE
    next_header_offset = data_offset + padded_size
    if next_header_offset > archive_size:
        raise RangeTarError(f"TAR member {name!r} at {offset} extends beyond the object")
    return TarMember(
        name=name,
        typeflag=typeflag,
        size=size,
        header_offset=offset,
        data_offset=data_offset,
        next_header_offset=next_header_offset,
    )


def _parse_number(raw_field: bytes, *, label: str, offset: int) -> int:
    if not raw_field:
        return 0
    if raw_field[0] & 0x80:
        # GNU/POSIX base-256 encoding. Negative values are invalid for TAR
        # sizes/checksums used here, so preserve only the positive magnitude.
        value = int.from_bytes(bytes([raw_field[0] & 0x7F]) + raw_field[1:], "big", signed=False)
        return value
    raw = raw_field.rstrip(b"\0 ").lstrip(b" ")
    if not raw:
        return 0
    if any(byte < ord("0") or byte > ord("7") for byte in raw):
        raise RangeTarError(f"Invalid TAR {label!r} field at byte {offset}: {raw!r}")
    return int(raw, 8)


def _decode_field(value: bytes, *, field: str, offset: int) -> str:
    raw = value.split(b"\0", 1)[0]
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RangeTarError(f"Could not decode TAR {field} at byte {offset}") from exc


def _validate_member_name(name: str, *, offset: int) -> None:
    path = PurePosixPath(name)
    if not name or "\0" in name or "\\" in name or path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RangeTarError(f"Unsafe TAR member name at byte {offset}: {name!r}")


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or "\0" in value
        or "\\" in value
        or path.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RangeTarError(f"Extraction relative path is unsafe: {value!r}")
    return Path(*path.parts)


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
