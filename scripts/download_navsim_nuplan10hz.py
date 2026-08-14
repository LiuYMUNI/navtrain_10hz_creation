#!/usr/bin/env python3
"""Download only native nuPlan v1.1 databases needed by NAVSIM ``navtrain``.

This is the database-only first stage of a derived 10 Hz NAVSIM camera
protocol.  It deliberately does not download camera images, LiDAR blobs,
maps, test data, or any full nuPlan archive.  The public nuPlan archives are
queried through HTTP byte ranges and only the selected ``.db`` ZIP members are
materialized under ``splits/trainval``.

The downloader is fail-closed:

* the SceneFilter must be the pinned official NAVSIM ``navtrain.yaml``;
* all ten public native train/validation archives must appear in the S3
  listing, while test and mini archives are ignored;
* each of the 1,192 requested logs must resolve to exactly one safe database
  member;
* every new or resumed file is checked against its ZIP CRC32 and byte count;
* completed files are published with an atomic rename, and the final JSON
  report is atomic as well.

The result is native nuPlan metadata used later to prove NAVSIM/OpenScene to
nuPlan identity and to plan native 10 Hz camera extraction.  It is not an
official replacement for NAVSIM's released 2 Hz sensor protocol.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import ssl
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from nuplan_range_zip import HttpRangeClient, RangeZipArchive, RangeZipError, ZipMember
from nuplan_tls import NuPlanTlsError, verified_https_context


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_FILTER = (
    REPO_ROOT / "reference/navtrain.yaml"
)
DEFAULT_NUPLAN_ROOT = REPO_ROOT / "dataset/raw/nuplan-v1.1-navtrain-10hz"
DEFAULT_REPORT = (
    REPO_ROOT
    / "dataset/manifests/navsim_nuplan10hz/"
    "navtrain_native_db_download_report.json"
)
DEFAULT_LISTING_URL = (
    "https://motional-nuplan.s3.dualstack.ap-northeast-1.amazonaws.com/"
    "?list-type=2&prefix=public%2Fnuplan-v1.1%2F&delimiter=%2F"
)

NUPLAN_PUBLIC_PREFIX = "public/nuplan-v1.1/"
OFFICIAL_NAVTRAIN_FILTER_SHA256 = "5f3c406a06c961e75d69d33bc127acd5704a3a440a7186bfa2212493ecf9fc01"
OFFICIAL_NAVTRAIN_LOG_COUNT = 1_192
REPORT_FORMAT = "navsim_nuplan10hz_native_db_download_report_v1"
MAX_NATIVE_DB_COMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_NATIVE_DB_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_NATIVE_DB_MEMBER_CHUNK_BYTES = 1 * 1024 * 1024

# nuPlan v1.1 publishes native train databases in nine geographic shards and
# native validation databases in one shard.  Names are checked against the
# live public listing instead of being treated as download URLs on their own.
EXPECTED_NATIVE_DB_ARCHIVE_FILENAMES = (
    "nuplan-v1.1_train_boston.zip",
    "nuplan-v1.1_train_pittsburgh.zip",
    "nuplan-v1.1_train_singapore.zip",
    "nuplan-v1.1_train_vegas_1.zip",
    "nuplan-v1.1_train_vegas_2.zip",
    "nuplan-v1.1_train_vegas_3.zip",
    "nuplan-v1.1_train_vegas_4.zip",
    "nuplan-v1.1_train_vegas_5.zip",
    "nuplan-v1.1_train_vegas_6.zip",
    "nuplan-v1.1_val.zip",
)


class DownloadError(RuntimeError):
    """A data contract or download failure that should not be retried blindly."""


@dataclass(frozen=True)
class ArchiveSpec:
    """One immutable entry from the public S3 object listing."""

    key: str
    url: str
    listing_size: int
    listing_etag: str | None


@dataclass(frozen=True)
class PlannedDatabase:
    """One selected remote ZIP member and its local database destination."""

    log_name: str
    archive: ArchiveSpec
    archive_handle: Any
    member: ZipMember


@dataclass(frozen=True)
class ArchivePlanSummary:
    """Non-serializable archive details plus serializable selection statistics."""

    archive: ArchiveSpec
    members_scanned: int
    selected_count: int
    selected_compressed_bytes: int
    selected_uncompressed_bytes: int


@dataclass(frozen=True)
class ExistingFileCheck:
    """Verification result for a possible resume target."""

    valid: bool
    reason: str
    observed_size: int | None = None
    observed_crc32: int | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-filter-yaml",
        type=Path,
        default=DEFAULT_FILTER,
        help="Pinned official NAVSIM navtrain SceneFilter YAML.",
    )
    parser.add_argument(
        "--nuplan-root",
        type=Path,
        default=DEFAULT_NUPLAN_ROOT,
        help=(
            "Destination native nuPlan root. Databases are written only to "
            "<nuplan-root>/splits/trainval/."
        ),
    )
    parser.add_argument(
        "--listing-url",
        default=DEFAULT_LISTING_URL,
        help="Public nuPlan v1.1 S3 ListObjectsV2 URL used to discover archives.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT,
        help="Atomic JSON report destination. Dry runs print the report and do not write it.",
    )
    parser.add_argument(
        "--listing-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for public S3 listing requests (default: 60 seconds).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent verified database-member transfers (default: 8; maximum: 32).",
    )
    parser.add_argument(
        "--range-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for one ETag-pinned archive range request (default: 120 seconds).",
    )
    parser.add_argument(
        "--range-max-attempts",
        type=int,
        default=6,
        help="Maximum attempts for a transient archive-range failure (default: 6; maximum: 10).",
    )
    parser.add_argument(
        "--range-retry-backoff-seconds",
        type=float,
        default=1.0,
        help="Initial exponential backoff between retryable archive ranges (default: 1 second).",
    )
    parser.add_argument(
        "--member-chunk-bytes",
        type=int,
        default=DEFAULT_NATIVE_DB_MEMBER_CHUNK_BYTES,
        help=(
            "Maximum compressed bytes in one range request while extracting a database "
            "(default: 1048576 / 1 MiB)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Discover and validate the exact member plan without extracting databases "
            "or writing the report. Existing databases are still CRC-checked read-only."
        ),
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_sequence_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_official_navtrain_log_names(path: Path) -> tuple[tuple[str, ...], str]:
    """Read log names from the exact, hash-pinned official filter.

    This intentionally avoids a PyYAML runtime dependency.  The SHA256 check
    happens before the narrow list parser, so the parser only accepts the
    already-pinned official YAML bytes rather than serving as a general YAML
    parser.
    """

    if not path.is_file():
        raise DownloadError(f"NAVSIM SceneFilter YAML does not exist: {path}")
    payload = path.read_bytes()
    filter_sha256 = sha256_bytes(payload)
    if filter_sha256 != OFFICIAL_NAVTRAIN_FILTER_SHA256:
        raise DownloadError(
            "Refusing a non-official NAVSIM SceneFilter: "
            f"sha256={filter_sha256}, expected={OFFICIAL_NAVTRAIN_FILTER_SHA256}"
        )

    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DownloadError(f"NAVSIM SceneFilter is not UTF-8: {path}") from exc

    in_log_names = False
    saw_tokens = False
    names: list[str] = []
    item_pattern = re.compile(
        r"^\s*-\s*(?:'([^']+)'|\"([^\"]+)\"|([A-Za-z0-9_.-]+))\s*$"
    )
    for line in lines:
        if not in_log_names:
            if line == "log_names:":
                in_log_names = True
            continue
        if line == "tokens:":
            saw_tokens = True
            break
        if not line.strip():
            continue
        match = item_pattern.fullmatch(line)
        if match is None:
            raise DownloadError(f"Unexpected official log_names YAML line: {line!r}")
        name = next(value for value in match.groups() if value is not None)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise DownloadError(f"Unsafe log name in official SceneFilter: {name!r}")
        names.append(name)

    if not in_log_names or not saw_tokens:
        raise DownloadError("Pinned NAVSIM SceneFilter has no bounded log_names list")
    if len(names) != OFFICIAL_NAVTRAIN_LOG_COUNT:
        raise DownloadError(
            "Pinned NAVSIM SceneFilter log count mismatch: "
            f"{len(names)} != {OFFICIAL_NAVTRAIN_LOG_COUNT}"
        )
    if len(names) != len(set(names)):
        raise DownloadError("Pinned NAVSIM SceneFilter contains duplicate log names")
    return tuple(names), filter_sha256


def _xml_child_text(element: ElementTree.Element, local_name: str) -> str | None:
    for child in element:
        if child.tag.rsplit("}", 1)[-1] == local_name:
            return child.text
    return None


def parse_s3_listing_page(payload: bytes) -> tuple[list[tuple[str, int, str | None]], bool, str | None]:
    """Parse one public S3 ListObjectsV2 XML page without accepting malformed rows."""

    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise DownloadError(f"Could not parse public S3 XML listing: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "ListBucketResult":
        raise DownloadError(f"Unexpected public S3 listing root: {root.tag!r}")

    records: list[tuple[str, int, str | None]] = []
    truncated_text: str | None = None
    continuation_token: str | None = None
    for child in root:
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name == "Contents":
            key = _xml_child_text(child, "Key")
            size_text = _xml_child_text(child, "Size")
            etag = _xml_child_text(child, "ETag")
            if not key or size_text is None:
                raise DownloadError("Public S3 listing has a Contents row without Key or Size")
            try:
                size = int(size_text)
            except ValueError as exc:
                raise DownloadError(f"Invalid public S3 object size {size_text!r} for {key!r}") from exc
            if size < 0:
                raise DownloadError(f"Negative public S3 object size for {key!r}")
            records.append((key, size, etag.strip('"') if etag else None))
        elif local_name == "IsTruncated":
            truncated_text = child.text
        elif local_name == "NextContinuationToken":
            continuation_token = child.text

    if truncated_text not in {"true", "false"}:
        raise DownloadError(f"Public S3 listing has invalid IsTruncated value: {truncated_text!r}")
    if truncated_text == "true" and not continuation_token:
        raise DownloadError("Public S3 listing is truncated without NextContinuationToken")
    return records, truncated_text == "true", continuation_token


def fetch_listing_page(
    url: str,
    timeout_seconds: float,
    *,
    ssl_context: ssl.SSLContext | None = None,
) -> bytes:
    if timeout_seconds <= 0:
        raise DownloadError("--listing-timeout-seconds must be positive")
    if ssl_context is None:
        try:
            ssl_context = verified_https_context()
        except NuPlanTlsError as exc:
            raise DownloadError(f"Could not configure verified HTTPS transport: {exc}") from exc
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/xml", "User-Agent": "navsim-nuplan10hz-db-downloader/1"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=float(timeout_seconds),
            context=ssl_context,
        ) as response:
            status = getattr(response, "status", response.getcode())
            if status != 200:
                raise DownloadError(f"Public S3 listing returned HTTP {status}: {url}")
            return response.read()
    except urllib.error.URLError as exc:
        raise DownloadError(f"Could not fetch public S3 listing {url}: {exc}") from exc


def continuation_listing_url(url: str, continuation_token: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered = [(key, value) for key, value in query if key != "continuation-token"]
    filtered.append(("continuation-token", continuation_token))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(filtered), parsed.fragment)
    )


def list_public_s3_objects(
    listing_url: str,
    timeout_seconds: float,
    *,
    ssl_context: ssl.SSLContext | None = None,
) -> list[tuple[str, int, str | None]]:
    """Retrieve every page from a public S3 ListObjectsV2 endpoint."""

    parsed = urllib.parse.urlsplit(listing_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise DownloadError("--listing-url must be an HTTPS S3 ListObjectsV2 URL")
    if ssl_context is None:
        try:
            ssl_context = verified_https_context()
        except NuPlanTlsError as exc:
            raise DownloadError(f"Could not configure verified HTTPS transport: {exc}") from exc

    page_url = listing_url
    all_records: list[tuple[str, int, str | None]] = []
    seen_page_urls: set[str] = set()
    while True:
        if page_url in seen_page_urls:
            raise DownloadError("Public S3 listing continuation loop detected")
        seen_page_urls.add(page_url)
        records, truncated, token = parse_s3_listing_page(
            fetch_listing_page(page_url, timeout_seconds, ssl_context=ssl_context)
        )
        all_records.extend(records)
        if not truncated:
            return all_records
        assert token is not None
        page_url = continuation_listing_url(listing_url, token)


def archive_url_from_listing_url(listing_url: str, key: str) -> str:
    parsed = urllib.parse.urlsplit(listing_url)
    quoted_key = urllib.parse.quote(key, safe="/")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, f"/{quoted_key}", "", ""))


def select_native_db_archives(
    records: Iterable[tuple[str, int, str | None]], listing_url: str
) -> tuple[ArchiveSpec, ...]:
    """Select only the ten native train/validation archives from a live listing."""

    expected_keys = {
        f"{NUPLAN_PUBLIC_PREFIX}{filename}" for filename in EXPECTED_NATIVE_DB_ARCHIVE_FILENAMES
    }
    matching: dict[str, ArchiveSpec] = {}
    for key, size, etag in records:
        if key not in expected_keys:
            continue
        if key in matching:
            raise DownloadError(f"Duplicate expected archive in public S3 listing: {key}")
        if size <= 0:
            raise DownloadError(f"Expected archive has non-positive listed size: {key} ({size})")
        matching[key] = ArchiveSpec(
            key=key,
            url=archive_url_from_listing_url(listing_url, key),
            listing_size=size,
            listing_etag=etag,
        )

    missing = sorted(expected_keys - set(matching))
    if missing:
        raise DownloadError(
            "Public nuPlan v1.1 listing is missing required native train/validation archives: "
            + ", ".join(missing)
        )
    return tuple(matching[f"{NUPLAN_PUBLIC_PREFIX}{filename}"] for filename in EXPECTED_NATIVE_DB_ARCHIVE_FILENAMES)


def safe_member_path(name: str) -> PurePosixPath:
    """Validate a remote member name before its basename is used for local output."""

    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise DownloadError(f"Unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DownloadError(f"Unsafe ZIP member path traversal: {name!r}")
    if name.endswith("/"):
        raise DownloadError(f"Expected a database file but found a ZIP directory: {name!r}")
    return path


def validate_selected_member(member: ZipMember, expected_log_name: str) -> None:
    path = safe_member_path(member.name)
    expected_filename = f"{expected_log_name}.db"
    if path.name != expected_filename or path.suffix != ".db":
        raise DownloadError(
            f"ZIP member {member.name!r} does not match requested database {expected_filename!r}"
        )
    if int(member.compressed_size) < 0 or int(member.uncompressed_size) < 0:
        raise DownloadError(f"ZIP member has invalid sizes: {member.name!r}")
    if int(member.flags) & 0x1:
        raise DownloadError(f"Refusing encrypted ZIP member: {member.name!r}")
    if int(member.compression_method) not in {0, 8}:
        raise DownloadError(
            f"Unsupported ZIP compression method {member.compression_method} for {member.name!r}"
        )


def plan_database_members(
    archives: Sequence[ArchiveSpec],
    log_names: Sequence[str],
    *,
    archive_opener: Callable[[str], Any] = RangeZipArchive.open,
) -> tuple[tuple[PlannedDatabase, ...], tuple[ArchivePlanSummary, ...]]:
    """Read ZIP central directories and resolve each requested log exactly once."""

    expected_logs = set(log_names)
    if not expected_logs or len(expected_logs) != len(log_names):
        raise DownloadError("Requested NAVSIM log name set is empty or duplicated")

    selected_by_log: dict[str, PlannedDatabase] = {}
    summaries: list[ArchivePlanSummary] = []
    for archive_spec in archives:
        archive_handle = archive_opener(archive_spec.url)
        try:
            members = tuple(archive_handle.members())
        except AttributeError as exc:
            raise DownloadError(f"Range ZIP archive has no members() API: {archive_spec.url}") from exc

        selected_members: list[ZipMember] = []
        for member in members:
            raw_name = str(member.name)
            # Fast path avoids applying path validation to unrelated archive
            # members.  A selected basename always receives full validation.
            candidate_basename = PurePosixPath(raw_name).name
            if not candidate_basename.endswith(".db"):
                continue
            candidate_log_name = candidate_basename[:-3]
            if candidate_log_name not in expected_logs:
                continue
            validate_selected_member(member, candidate_log_name)
            if candidate_log_name in selected_by_log:
                previous = selected_by_log[candidate_log_name]
                raise DownloadError(
                    "Requested native database appears in more than one ZIP member: "
                    f"{candidate_log_name} ({previous.archive.key}:{previous.member.name}, "
                    f"{archive_spec.key}:{member.name})"
                )
            selected = PlannedDatabase(
                log_name=candidate_log_name,
                archive=archive_spec,
                archive_handle=archive_handle,
                member=member,
            )
            selected_by_log[candidate_log_name] = selected
            selected_members.append(member)

        summaries.append(
            ArchivePlanSummary(
                archive=archive_spec,
                members_scanned=len(members),
                selected_count=len(selected_members),
                selected_compressed_bytes=sum(int(member.compressed_size) for member in selected_members),
                selected_uncompressed_bytes=sum(int(member.uncompressed_size) for member in selected_members),
            )
        )

    missing = [log_name for log_name in log_names if log_name not in selected_by_log]
    if missing:
        preview = ", ".join(missing[:10])
        raise DownloadError(
            f"{len(missing)} requested NAVSIM log database(s) are absent from the public native archives; "
            f"first entries: {preview}"
        )
    planned = tuple(selected_by_log[log_name] for log_name in log_names)
    if len(planned) != len(log_names):
        raise DownloadError("Internal selected native database count mismatch")
    return planned, tuple(summaries)


def open_native_database_archive(
    url: str,
    *,
    ssl_context: ssl.SSLContext,
    range_timeout_seconds: float,
    range_max_attempts: int,
    range_retry_backoff_seconds: float,
    member_chunk_bytes: int,
    direct_resolve_hosts: Sequence[str],
) -> RangeZipArchive:
    """Open a public native DB archive with bounds matching its real members.

    The generic range reader defaults to 512 MiB compressed members.  nuPlan's
    official geographic database shards contain legitimate single-log members
    above that threshold, so this narrowly scoped caller raises the cap while
    retaining archive, ETag, size, and CRC validation.
    """

    return RangeZipArchive.open(
        url,
        client=HttpRangeClient(
            ssl_context=ssl_context,
            timeout_seconds=range_timeout_seconds,
            max_attempts=range_max_attempts,
            retry_backoff_seconds=range_retry_backoff_seconds,
            direct_resolve_hosts=direct_resolve_hosts,
        ),
        max_compressed_member_bytes=MAX_NATIVE_DB_COMPRESSED_BYTES,
        max_uncompressed_member_bytes=MAX_NATIVE_DB_UNCOMPRESSED_BYTES,
        member_chunk_bytes=member_chunk_bytes,
    )


def database_directory(nuplan_root: Path) -> Path:
    return nuplan_root / "splits" / "trainval"


def database_destination(nuplan_root: Path, log_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", log_name):
        raise DownloadError(f"Unsafe local log name: {log_name!r}")
    return database_directory(nuplan_root) / f"{log_name}.db"


def crc32_file(path: Path) -> tuple[int, int]:
    """Return byte count and unsigned ZIP-style CRC32 for one regular local file."""

    checksum = 0
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            checksum = zlib.crc32(block, checksum)
    return size, checksum & 0xFFFFFFFF


def check_existing_database(destination: Path, member: ZipMember) -> ExistingFileCheck:
    """CRC-check an existing destination before using it as a resumable result."""

    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return ExistingFileCheck(valid=False, reason="missing")
    if stat.S_ISLNK(metadata.st_mode):
        raise DownloadError(f"Refusing symlink database destination: {destination}")
    if not stat.S_ISREG(metadata.st_mode):
        raise DownloadError(f"Expected regular database file at {destination}")
    expected_size = int(member.uncompressed_size)
    if metadata.st_size != expected_size:
        return ExistingFileCheck(
            valid=False,
            reason=f"size_mismatch(expected={expected_size}, actual={metadata.st_size})",
            observed_size=metadata.st_size,
        )
    observed_size, observed_crc32 = crc32_file(destination)
    if observed_size != expected_size:
        return ExistingFileCheck(
            valid=False,
            reason=f"read_size_mismatch(expected={expected_size}, actual={observed_size})",
            observed_size=observed_size,
            observed_crc32=observed_crc32,
        )
    expected_crc32 = int(member.crc32) & 0xFFFFFFFF
    if observed_crc32 != expected_crc32:
        return ExistingFileCheck(
            valid=False,
            reason=f"crc32_mismatch(expected={expected_crc32:08x}, actual={observed_crc32:08x})",
            observed_size=observed_size,
            observed_crc32=observed_crc32,
        )
    return ExistingFileCheck(
        valid=True,
        reason="validated",
        observed_size=observed_size,
        observed_crc32=observed_crc32,
    )


def ensure_safe_database_directory(destination_directory: Path) -> None:
    """Create the narrow target directory and reject a symlink at that boundary."""

    if destination_directory.exists() or destination_directory.is_symlink():
        if destination_directory.is_symlink():
            raise DownloadError(f"Refusing symlink database directory: {destination_directory}")
        if not destination_directory.is_dir():
            raise DownloadError(f"Database destination is not a directory: {destination_directory}")
        return
    destination_directory.mkdir(parents=True, exist_ok=True)
    if destination_directory.is_symlink() or not destination_directory.is_dir():
        raise DownloadError(f"Could not create a safe database directory: {destination_directory}")


def unexpected_database_files(destination_directory: Path, requested_logs: set[str]) -> list[str]:
    """Reject unrelated ``.db`` files so this root remains a minimal derivative."""

    if not destination_directory.exists():
        return []
    if destination_directory.is_symlink() or not destination_directory.is_dir():
        raise DownloadError(f"Database destination is not a safe directory: {destination_directory}")
    unexpected: list[str] = []
    for child in destination_directory.iterdir():
        if child.name.endswith(".db") and child.name[:-3] not in requested_logs:
            unexpected.append(child.name)
    return sorted(unexpected)


def extract_database_atomically(plan: PlannedDatabase, destination: Path) -> None:
    """Range-extract one member into a temporary sibling, verify it, then publish it."""

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise DownloadError(f"Refusing symlink database destination: {destination}")
        if not destination.is_file():
            raise DownloadError(f"Database destination is not a regular file: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    temporary = Path(temporary_name)
    os.close(descriptor)
    # The range utility owns file creation and validates the member path.  Map
    # the safe archive member to our private sibling filename, never the final
    # target, then validate and publish that sibling atomically below.
    temporary.unlink(missing_ok=True)
    try:
        written = plan.archive_handle.extract_member(
            plan.member,
            destination.parent,
            relative_path=temporary.name,
        )
        if written != temporary.resolve():
            raise DownloadError(
                f"Range extractor wrote an unexpected temporary database path for {plan.log_name}: {written}"
            )
        check = check_existing_database(temporary, plan.member)
        if not check.valid:
            raise DownloadError(
                f"Range-extracted database failed local validation for {plan.log_name}: {check.reason}"
            )
        if destination.is_symlink():
            raise DownloadError(f"Refusing symlink database destination: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def selection_sha256(plans: Sequence[PlannedDatabase]) -> str:
    records = (
        "\t".join(
            (
                plan.log_name,
                plan.archive.key,
                plan.member.name,
                str(int(plan.member.compression_method)),
                str(int(plan.member.crc32) & 0xFFFFFFFF),
                str(int(plan.member.compressed_size)),
                str(int(plan.member.uncompressed_size)),
            )
        )
        for plan in plans
    )
    return stable_sequence_sha256(records)


def report_archives(summaries: Sequence[ArchivePlanSummary]) -> list[dict[str, Any]]:
    return [
        {
            "key": summary.archive.key,
            "url": summary.archive.url,
            "listing_size_bytes": summary.archive.listing_size,
            "listing_etag": summary.archive.listing_etag,
            "central_directory_members_scanned": summary.members_scanned,
            "selected_database_count": summary.selected_count,
            "selected_compressed_bytes": summary.selected_compressed_bytes,
            "selected_uncompressed_bytes": summary.selected_uncompressed_bytes,
        }
        for summary in summaries
    ]


def build_report(
    *,
    status: str,
    args: argparse.Namespace,
    scene_filter_sha256: str,
    log_names: Sequence[str],
    plans: Sequence[PlannedDatabase],
    archive_summaries: Sequence[ArchivePlanSummary],
    resumed_count: int,
    downloaded_count: int,
    invalid_existing_count: int,
    output_verified_count: int,
) -> dict[str, Any]:
    compressed_bytes = sum(int(plan.member.compressed_size) for plan in plans)
    uncompressed_bytes = sum(int(plan.member.uncompressed_size) for plan in plans)
    return {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "status": status,
        "scope": "native_nuplan_v1.1_databases_only",
        "source": {
            "public_s3_listing_url": args.listing_url,
            "archive_count": len(archive_summaries),
            "excluded_data": ["camera_images", "lidar_blobs", "maps", "mini", "test"],
            "archive_range_transport": {
                "timeout_seconds": float(args.range_timeout_seconds),
                "max_attempts": int(args.range_max_attempts),
                "initial_retry_backoff_seconds": float(args.range_retry_backoff_seconds),
                "member_chunk_bytes": int(args.member_chunk_bytes),
            },
        },
        "navsim": {
            "scene_filter_yaml": str(args.scene_filter_yaml),
            "scene_filter_sha256": scene_filter_sha256,
            "requested_log_count": len(log_names),
            "requested_log_names_sha256": stable_sequence_sha256(log_names),
        },
        "destination": {
            "nuplan_root": str(args.nuplan_root),
            "database_directory": str(database_directory(args.nuplan_root)),
        },
        "selection": {
            "selected_database_count": len(plans),
            "selection_sha256": selection_sha256(plans),
            "selected_compressed_bytes": compressed_bytes,
            "selected_uncompressed_bytes": uncompressed_bytes,
            "archives": report_archives(archive_summaries),
        },
        "resume": {
            "crc_valid_existing_database_count": resumed_count,
            "invalid_existing_database_count": invalid_existing_count,
            "newly_downloaded_database_count": downloaded_count,
            "verified_output_database_count": output_verified_count,
        },
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix != ".json":
        raise DownloadError(f"--report-json must have a .json suffix: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Plan, optionally extract, and fully validate the database-only stage."""

    log_names, scene_filter_sha256 = load_official_navtrain_log_names(args.scene_filter_yaml)
    if args.workers < 1 or args.workers > 32:
        raise DownloadError("--workers must be between 1 and 32")
    if args.range_timeout_seconds <= 0:
        raise DownloadError("--range-timeout-seconds must be positive")
    if args.range_max_attempts < 1 or args.range_max_attempts > 10:
        raise DownloadError("--range-max-attempts must be between 1 and 10")
    if args.range_retry_backoff_seconds < 0:
        raise DownloadError("--range-retry-backoff-seconds must be nonnegative")
    if args.member_chunk_bytes < 64 * 1024 or args.member_chunk_bytes > 8 * 1024 * 1024:
        raise DownloadError("--member-chunk-bytes must be between 65536 and 8388608")
    try:
        ssl_context = verified_https_context()
    except NuPlanTlsError as exc:
        raise DownloadError(f"Could not configure verified HTTPS transport: {exc}") from exc
    listing = list_public_s3_objects(
        args.listing_url,
        float(args.listing_timeout_seconds),
        ssl_context=ssl_context,
    )
    listing_host = urllib.parse.urlsplit(args.listing_url).hostname
    if not listing_host:
        raise DownloadError("--listing-url has no hostname")
    archives = select_native_db_archives(listing, args.listing_url)
    plans, archive_summaries = plan_database_members(
        archives,
        log_names,
        archive_opener=lambda url: open_native_database_archive(
            url,
            ssl_context=ssl_context,
            range_timeout_seconds=float(args.range_timeout_seconds),
            range_max_attempts=int(args.range_max_attempts),
            range_retry_backoff_seconds=float(args.range_retry_backoff_seconds),
            member_chunk_bytes=int(args.member_chunk_bytes),
            direct_resolve_hosts=(listing_host,),
        ),
    )

    target_directory = database_directory(args.nuplan_root)
    requested_log_set = set(log_names)
    extras = unexpected_database_files(target_directory, requested_log_set)
    if extras:
        raise DownloadError(
            "Database destination already contains unexpected .db file(s); refusing to mix "
            "a minimal navtrain derivative with other data: "
            + ", ".join(extras[:10])
        )

    existing_checks: dict[str, ExistingFileCheck] = {}
    for plan in plans:
        existing_checks[plan.log_name] = check_existing_database(
            database_destination(args.nuplan_root, plan.log_name), plan.member
        )
    resumed_count = sum(check.valid for check in existing_checks.values())
    invalid_existing_count = sum(
        check.reason != "missing" and not check.valid for check in existing_checks.values()
    )

    if args.dry_run:
        return build_report(
            status="dry_run",
            args=args,
            scene_filter_sha256=scene_filter_sha256,
            log_names=log_names,
            plans=plans,
            archive_summaries=archive_summaries,
            resumed_count=resumed_count,
            downloaded_count=0,
            invalid_existing_count=invalid_existing_count,
            output_verified_count=resumed_count,
        )

    pending = [plan for plan in plans if not existing_checks[plan.log_name].valid]
    if pending:
        ensure_safe_database_directory(target_directory)
    downloaded_count = 0
    if pending:
        with ThreadPoolExecutor(max_workers=min(int(args.workers), len(pending))) as executor:
            futures = {
                executor.submit(
                    extract_database_atomically,
                    plan,
                    database_destination(args.nuplan_root, plan.log_name),
                ): plan
                for plan in pending
            }
            try:
                for future in as_completed(futures):
                    future.result()
                    downloaded_count += 1
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    extras_after = unexpected_database_files(target_directory, requested_log_set)
    if extras_after:
        raise DownloadError("Unexpected .db file(s) appeared during extraction: " + ", ".join(extras_after[:10]))

    verified_count = 0
    for plan in plans:
        check = check_existing_database(database_destination(args.nuplan_root, plan.log_name), plan.member)
        if not check.valid:
            raise DownloadError(f"Final database validation failed for {plan.log_name}: {check.reason}")
        verified_count += 1
    if verified_count != len(log_names):
        raise DownloadError(f"Final verified database count mismatch: {verified_count} != {len(log_names)}")

    return build_report(
        status="pass",
        args=args,
        scene_filter_sha256=scene_filter_sha256,
        log_names=log_names,
        plans=plans,
        archive_summaries=archive_summaries,
        resumed_count=resumed_count,
        downloaded_count=downloaded_count,
        invalid_existing_count=invalid_existing_count,
        output_verified_count=verified_count,
    )


def failure_report(args: argparse.Namespace, error: Exception) -> dict[str, Any]:
    return {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "status": "fail",
        "scope": "native_nuplan_v1.1_databases_only",
        "error": str(error),
        "source": {"public_s3_listing_url": args.listing_url},
        "destination": {
            "nuplan_root": str(args.nuplan_root),
            "database_directory": str(database_directory(args.nuplan_root)),
        },
    }


def print_report(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = execute(args)
    except (DownloadError, RangeZipError, OSError, ValueError) as exc:
        report = failure_report(args, exc)
        if not args.dry_run:
            try:
                atomic_write_json(args.report_json, report)
            except (DownloadError, OSError) as report_exc:
                report["report_write_error"] = str(report_exc)
        print_report(report)
        return 2

    if not args.dry_run:
        atomic_write_json(args.report_json, report)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
