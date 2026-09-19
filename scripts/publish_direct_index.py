#!/usr/bin/env python3
"""Publish existing gzip indexes to this standalone repository's release.

Uses GITHUB_TOKEN or the configured git credential helper. Never prints secrets.
Verifies each asset by downloading and hashing it before publishing the pointer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import urllib.request
import urllib.error

from navsim_direct_jpeg import read_url

REPO = "LiuYMUNI/navtrain_10hz_creation"
TAG = "navsim-direct-jpeg-v1"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--manifest-output", type=Path, required=True)
    p.add_argument("--ca-file")
    a = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True, check=True)
        token = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)["password"]
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"}
    context = ssl.create_default_context(cafile=a.ca_file)
    base = "https://api.github.com/repos/" + REPO

    def api(url, body=None, content_type="application/json"):
        h = dict(headers)
        h["Content-Type"] = content_type
        request = urllib.request.Request(url, data=body, headers=h)
        with urllib.request.urlopen(request, context=context, timeout=180) as r:
            return json.load(r)

    try:
        release = api(base + "/releases/tags/" + TAG)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        release = api(base + "/releases", json.dumps({"tag_name": TAG,
            "target_commitish": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "name": "Direct native JPEG index v1", "draft": False,
            "body": "55 source-offset index shards for 5,625,150 NAVSIM camera JPEGs. Use scripts/navsim_direct_jpeg.py and reference/direct_jpeg/latest.json on main. Index metadata only; JPEGs are retrieved from the original nuPlan source."}).encode())
    assets = {x["name"]: x for x in api(base + "/releases/" + str(release["id"]) + "/assets?per_page=100")}
    manifest = json.loads((a.bundle / "latest.json").read_text())

    def publish(shard):
        name = shard["asset"].rsplit("/", 1)[-1]
        data = (a.bundle / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != shard["sha256"]:
            raise ValueError("Local shard digest mismatch: " + name)
        asset = assets.get(name)
        if asset is None:
            asset = api(release["upload_url"].split("{")[0] + "?name=" + name, data, "application/gzip")
        url = base + "/releases/assets/" + str(asset["id"])
        remote = read_url(url, context=context, headers={**headers, "Accept": "application/octet-stream"})
        if len(remote) != len(data) or hashlib.sha256(remote).hexdigest() != shard["sha256"]:
            raise ValueError("Published asset digest mismatch: " + name)
        print(name + " uploaded and download-hash verified", flush=True)
        return {**shard, "asset": url}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest["shards"] = list(pool.map(publish, manifest["shards"]))
    a.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    a.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": release["html_url"], "shards": len(manifest["shards"]), "rows": manifest["rows"]}))


if __name__ == "__main__":
    main()
