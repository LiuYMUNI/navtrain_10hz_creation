#!/usr/bin/env python3
"""Export completed pack offsets; download exact native JPEGs without TAR scans.

Standard library only. Release assets are gzip JSONL shards and a latest.json
manifest. Existing-image hashes are deliberately excluded from native hashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import ssl
import tempfile
import time
import urllib.request
import urllib.parse

FORMAT = "navsim_direct_jpeg_v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ro(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def safe_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in value:
        raise ValueError(f"Unsafe destination: {value}")
    return p


def export(args):
    # A new output directory prevents accidental replacement of published assets.
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"format": FORMAT, "shards": [], "rows": 0}
    seen_archives = set()
    seen_destinations = set()
    with ro(args.target) as target:
        meta = dict(target.execute("SELECT key,value FROM metadata"))
        manifest["inventory_sha256"] = meta["inventory_sha256"]
        for source in sorted(args.source, key=lambda value: str(value)):
            with ro(source) as db:
                sm = dict(db.execute("SELECT key,value FROM metadata"))
                if sm.get("format") != "navsim_nuplan10hz_camera_pack_state_v1":
                    raise ValueError("Unsupported pack ledger")
                for key in ("inventory_sha256", "inventory_row_count", "navsim_scene_filter_sha256"):
                    if sm.get(key) != meta.get(key):
                        raise ValueError(f"Source/target metadata mismatch: {key}")
                for key, source_url, size, etag, status, total, done in db.execute(
                    "SELECT archive_key,source_url,listed_size,remote_etag,status,target_total,target_done "
                    "FROM archive_progress ORDER BY archive_key"
                ):
                    if key in seen_archives or status != "complete" or total != done:
                        raise ValueError(f"Duplicate or incomplete archive: {key}")
                    seen_archives.add(key)
                    name = f"{len(seen_archives):03d}.jsonl.gz"
                    count = 0
                    expected = iter(target.execute(
                        "SELECT tar_member_name,destination_relative_path FROM target WHERE archive_key=? ORDER BY tar_member_name", (key,)
                    ))
                    with gzip.open(out / name, "wt", encoding="utf-8") as f:
                        for member, dest, kind, length, sha, header, offset in db.execute(
                            "SELECT tar_member_name,destination_relative_path,storage_kind,payload_size,sha256,source_header_offset,source_data_offset FROM member_storage WHERE archive_key=? ORDER BY tar_member_name", (key,)
                        ):
                            if next(expected, None) != (member, dest):
                                raise ValueError("Target/member coverage mismatch")
                            safe_path(dest)
                            if dest in seen_destinations:
                                raise ValueError(f"Duplicate target destination: {dest}")
                            seen_destinations.add(dest)
                            if kind not in ("pack", "existing") or not (0 < length <= 64 * 1024 * 1024) or header < 0 or header % 512 or offset != header + 512 or offset + length > size:
                                raise ValueError("Invalid source byte range")
                            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                                raise ValueError("Invalid recorded hash")
                            row = {"path": dest, "offset": offset, "size": length,
                                   "sha256": sha if kind == "pack" else None}
                            f.write(json.dumps(row, separators=(",", ":")) + "\n")
                            count += 1
                    if next(expected, None) is not None or count != total:
                        raise ValueError("Archive coverage mismatch")
                    manifest["shards"].append({"asset": args.asset_base.rstrip("/") + "/" + name,
                        "sha256": file_digest(out / name), "rows": count,
                        "source_url": (args.source_base.rstrip("/") + "/" + key)
                        if args.source_base else source_url,
                        "source_size": size, "etag": etag})
                    manifest["rows"] += count
        if seen_archives != {r[0] for r in target.execute("SELECT DISTINCT archive_key FROM target")} or len(seen_destinations) != manifest["rows"] or manifest["rows"] != int(meta["inventory_row_count"]):
            raise ValueError("Global inventory coverage mismatch")
    (out / "latest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(out / "latest.json"), "rows": manifest["rows"]}))


def read_url(url, stats=None, context=None, headers=None):
    headers = dict(headers or {})
    if urllib.parse.urlparse(url).hostname not in {"api.github.com", "raw.githubusercontent.com"}:
        headers.pop("Authorization", None)
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            redirected = super().redirect_request(req, fp, code, msg, response_headers, newurl)
            if redirected is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
                redirected.remove_header("Authorization")
            return redirected
    opener = urllib.request.build_opener(SafeRedirect(), urllib.request.HTTPSHandler(context=context))
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=90) as response:
        data = response.read()
    if stats is not None:
        stats["metadata_requests"] += 1
        stats["metadata_bytes"] += len(data)
    return data


def download(args):
    stats = {"downloaded": 0, "skipped": 0, "payload_requests": 0,
             "payload_bytes": 0, "metadata_requests": 0, "metadata_bytes": 0,
             "retries": 0}
    ca_file = getattr(args, "ca_file", None)
    context = ssl.create_default_context(cafile=ca_file) if ca_file else ssl.create_default_context()
    token_env = getattr(args, "github_token_env", "GITHUB_TOKEN")
    github_token = os.environ.get(token_env)
    github_headers = {"Authorization": "Bearer " + github_token} if github_token else {}
    manifest = json.loads(read_url(args.manifest, stats, context, github_headers))
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported manifest")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Resume receipts use the actual native payload hash, including rows whose
    # original ledger hash described a reused OpenScene file instead.
    state = sqlite3.connect(root / ".navsim-download.sqlite")
    state.execute("CREATE TABLE IF NOT EXISTS receipt (path TEXT PRIMARY KEY, identity TEXT, sha TEXT)")

    def fetch(job):
        shard, row, destination, identity = job
        start, size = row["offset"], row["size"]
        attempts = 0
        for attempt in range(args.attempts):
            attempts += 1
            try:
                request = urllib.request.Request(shard["source_url"], headers={
                    "Range": f"bytes={start}-{start+size-1}", "If-Match": shard["etag"], "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=90, context=context) as response:
                    expected = f"bytes {start}-{start+size-1}/{shard['source_size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected or response.headers.get("ETag") != shard["etag"]:
                        raise ValueError("Range/source identity mismatch")
                    data = response.read(size + 1)
                sha = digest(data)
                if len(data) != size or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9") or (row["sha256"] and sha != row["sha256"]):
                    raise ValueError("JPEG length/markers/hash mismatch")
                destination.parent.mkdir(parents=True, exist_ok=True)
                fd, temporary = tempfile.mkstemp(prefix=".jpeg-", dir=destination.parent)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(data)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temporary, destination)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                return row["path"], identity, sha, size, attempts
            except ValueError:
                raise
            except OSError:
                if attempt + 1 == args.attempts:
                    raise
                time.sleep(min(8, 2 ** attempt))

    selected = 0
    try:
        selected_shards = getattr(args, "shard_index", [])
        selected_shards = set(selected_shards) if selected_shards else None
        provenance = getattr(args, "hash_provenance", "all")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for shard_index, shard in enumerate(manifest["shards"], start=1):
                if selected_shards is not None and shard_index not in selected_shards:
                    continue
                if args.limit and selected >= args.limit:
                    break
                asset_headers = dict(github_headers)
                if shard["asset"].startswith("https://api.github.com/repos/"):
                    asset_headers["Accept"] = "application/octet-stream"
                compressed = read_url(shard["asset"], stats, context, asset_headers)
                if digest(compressed) != shard["sha256"]:
                    raise ValueError("Index shard hash mismatch")
                rows = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
                if len(rows) != shard["rows"]:
                    raise ValueError("Shard count mismatch")
                jobs = []
                for row in rows:
                    if args.limit and selected >= args.limit:
                        break
                    if provenance == "native" and row["sha256"] is None:
                        continue
                    if provenance == "unverified" and row["sha256"] is not None:
                        continue
                    selected += 1
                    destination = root.joinpath(*safe_path(row["path"]).parts)
                    try:
                        destination.resolve().relative_to(root)
                    except ValueError:
                        raise ValueError("Destination escapes output directory")
                    if destination.is_symlink():
                        raise ValueError("Destination escapes output directory")
                    if not (isinstance(row["size"], int) and 0 < row["size"] <= 64*1024*1024 and isinstance(row["offset"], int) and 0 <= row["offset"] <= shard["source_size"] - row["size"]):
                        raise ValueError("Invalid download range")
                    identity = digest(json.dumps([shard["source_url"], shard["etag"], shard["source_size"], row], sort_keys=True).encode())
                    receipt = state.execute("SELECT identity,sha FROM receipt WHERE path=?", (row["path"],)).fetchone()
                    if destination.is_file() and receipt and receipt[0] == identity and destination.stat().st_size == row["size"] and file_digest(destination) == receipt[1]:
                        stats["skipped"] += 1
                        continue
                    jobs.append((shard, row, destination, identity))
                    if len(jobs) >= args.workers * 2:
                        for path, ident, sha, size, attempts in pool.map(fetch, jobs):
                            state.execute("INSERT OR REPLACE INTO receipt VALUES (?,?,?)", (path, ident, sha))
                            state.commit()
                            stats["downloaded"] += 1
                            stats["payload_bytes"] += size
                            stats["payload_requests"] += attempts
                            stats["retries"] += attempts - 1
                        jobs.clear()
                for path, ident, sha, size, attempts in pool.map(fetch, jobs):
                    state.execute("INSERT OR REPLACE INTO receipt VALUES (?,?,?)", (path, ident, sha))
                    state.commit()
                    stats["downloaded"] += 1
                    stats["payload_bytes"] += size
                    stats["payload_requests"] += attempts
                    stats["retries"] += attempts - 1
    finally:
        state.close()
        stats["elapsed_seconds"] = time.monotonic() - started
        print(json.dumps(stats))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("export")
    e.add_argument("--target", required=True)
    e.add_argument("--source", action="append", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--asset-base", required=True, help="Versioned GitHub Release asset base URL")
    e.add_argument("--source-base", default=None,
                   help="Optional replacement base URL for tests or a stable CDN mirror")
    d = sub.add_parser("download")
    d.add_argument("--manifest", required=True, help="HTTPS URL or file:// URL")
    d.add_argument("--output", required=True)
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--attempts", type=int, default=4)
    d.add_argument("--limit", type=int, default=0, help="Bounded test run; zero downloads all")
    d.add_argument("--shard-index", action="append", type=int, default=[],
                   help="One-based index shard to read; repeatable")
    d.add_argument("--hash-provenance", choices=("all", "native", "unverified"), default="all",
                   help="Select native-hashed records or reused records whose native hash is unverified")
    d.add_argument("--ca-file", type=Path,
                   help="PEM CA bundle for TLS verification when Python has no system trust store")
    d.add_argument("--github-token-env", default="GITHUB_TOKEN",
                   help="Environment variable holding a GitHub token for private manifests and release assets")
    a = p.parse_args()
    if a.command == "download" and (a.workers < 1 or a.attempts < 1 or a.limit < 0 or any(value < 1 for value in a.shard_index)):
        p.error("workers/attempts must be positive; limit must be nonnegative")
    export(a) if a.command == "export" else download(a)


if __name__ == "__main__":
    main()
