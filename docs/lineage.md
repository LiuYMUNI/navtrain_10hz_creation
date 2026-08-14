# NAVSIM-to-nuPlan 10 Hz Camera Lineage

## Scope

This repository's staged `dataset/navsim2` data is OpenScene/NAVSIM data.
OpenScene retains camera and LiDAR inputs at 2 Hz. Native nuPlan v1.1 stores
camera images at 10 Hz and LiDAR point clouds at 20 Hz. `navtrain` is a
scene-filtered subset of the OpenScene `trainval` logs, so it remains a 2 Hz
input protocol.

```text
native nuPlan v1.1: camera 10 Hz; LiDAR/annotations 20 Hz
                         |
                         v
OpenScene: retained 2 Hz observations
                         |
                         v
NAVSIM navtrain: selected 2 Hz anchor windows
```

The missing four camera images between two retained 2 Hz observations cannot
be reconstructed from OpenScene. They must come from the matching licensed
native nuPlan release.

## Identity Contract

`navtrain.yaml` selects 1,192 source log names and 103,288 anchor tokens.
The token list is not positionally paired with the log list. Resolve a token
through the OpenScene `.pkl` row, which retains:

- `log_name`, `log_token`, `scene_token`, `token`, and timestamp;
- `lidar_path`, whose filename is the retained LiDAR token;
- per-camera `cams[CHANNEL].data_path` and calibration.

For each anchor, the native data must pass all of these checks before a 10 Hz
record is accepted:

1. `<log_name>.db` exists under `nuplan-v1.1/splits/trainval/`, and both
   `log.logfile == log_name` and `log.token == OpenScene log_token`.
2. The native `lidar_pc` record matches the OpenScene token, timestamp, scene
   token, and point-cloud filename. A timestamp-plus-filename fallback is
   recorded explicitly if token preservation differs.
3. Every retained 2 Hz OpenScene camera filename resolves to exactly one
   native `image` record in the same channel within the configured +/-50 ms
   LiDAR-to-image window.
4. The matched first and current retained camera images bound an exact,
   contiguous native 10 Hz sequence, which includes all four retained NAVSIM
   history images for that channel. A clean four-row history is nominally
   sixteen frames; released NAVSIM timestamp gaps can legitimately yield a
   shorter or longer range. JPEG bytes are deliberately not read by the
   canonical metadata index.

The result is a derived `NAVSIM-derived-10Hz-camera-history` protocol. It is
not a change to official NAVSIM, and it cannot make the stock NAVSIM loader,
prediction anchors, labels, or test inputs run at 10 Hz.

## Prerequisite Layout

Obtain the matching, license-approved nuPlan v1.1 `trainval` databases through
the official nuPlan distribution. The metadata stage needs only:

```text
<nuplan-root>/
  splits/trainval/<log_name>.db
```

Native JPEGs are deferred. A later feature-building stage may materialize
them under:

```text
<nuplan-root>/
  sensor_blobs/<log_name>/CAM_F0/<image>.jpg
  sensor_blobs/<log_name>/CAM_L0/<image>.jpg
  ...
```

Do not mix a different nuPlan release with the OpenScene data. Test, mini,
maps, native LiDAR blobs, and camera JPEGs are not required to build this
metadata index.

## Canonical Metadata Index

Build the camera-time index after the required native `trainval` databases
are staged. This reads NAVSIM pickle metadata and native SQLite metadata only;
it does not require or inspect JPEGs.

```bash
.venv/bin/python \
  scripts/plan_navsim_nuplan10hz.py \
  --nuplan-root /path/to/nuplan-v1.1 \
  --index-sqlite dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_index.sqlite \
  --report-json dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_plan_report.json
```

The SQLite file is the canonical format. It contains `anchor` provenance,
deduplicated `native_image` records, and compact `anchor_camera` stream-index
ranges. An `anchor_camera` range is bounded by the matched native images for
the first and current retained NAVSIM 2 Hz camera frames, not by LiDAR
timestamps. All four retained camera identities must be inside the range.
The standard four-frame NAVSIM history normally resolves to sixteen native
images per channel. Native cameras are nominally 10 Hz, but the source has
rare real capture gaps. The index preserves the exact source timestamps and
the resulting variable frame count; it never duplicates, drops, interpolates,
or imputes an image. The plan report records both all-stream and
anchor-history gap counts and timestamp deltas.

For a later camera-archive stage that still accepts JSONL, request an explicit
compatibility export with `--inventory-jsonl <path>`. The legacy
`plan_navsim_nuplan10hz_images.py` command delegates to this canonical planner
and is deprecated. It does not provide a second data schema.

The index adds a 10 Hz camera history to the existing 2 Hz NAVSIM anchors. It
does not make new 10 Hz prediction anchors or recreate NAVSIM labels, routes,
or held-out evaluation data.

## Verified `navtrain` Metadata State

The full `navtrain` metadata plan was verified on August 7, 2026:

- 103,288 existing NAVSIM anchors match 1,192 native nuPlan databases.
- The normalized index contains 5,625,150 distinct native JPEG references and
  13,230,173 anchor-to-image references.
- The source has 62 genuine native camera capture gaps across 35 logs. They
  remain exact timestamps in the index; no source image was duplicated,
  interpolated, or silently discarded.
- Native JPEG blobs are intentionally a separate staging step. Until they are
  present, `Navsim10HzImageLoader` fails closed instead of returning a
  fabricated 10 Hz sample.

The published artifacts are:

```text
dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_index.sqlite
dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_inventory.jsonl
dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_plan_report.json
```

## Deferred Blob Audit

After the referenced JPEGs are staged, use the NAVSIM virtual environment to
check availability of the exact files named by the canonical SQLite index.
The auditor does not derive a new camera window from mutable source metadata,
does not read or decode JPEG bytes, and does not modify either source dataset.
This is not a prerequisite for metadata planning:

```bash
.venv/bin/python \
  scripts/audit_navsim_nuplan10hz_lineage.py \
  --nuplan-root /path/to/nuplan-v1.1 \
  --max-anchors 8 \
  --require-openscene-files \
  --dry-run
```

`pass` means every requested anchor was audited. `partial_pass` means an
intentional anchor-limited pilot passed. Any missing indexed native blob or
index-contract failure returns a nonzero status and does not produce a
manifest. Native databases are required and verified by the metadata planner,
not reopened by this later immutable-index audit.

The optional `--require-openscene-files` check additionally verifies all four
retained 2 Hz OpenScene camera files for every requested anchor. The native
10 Hz file list still comes only from the immutable canonical index, whose
camera-matched endpoints preserve the exact native frame count per channel.

## Exact JPEG Staging and Loading

After the planner publishes `--inventory-jsonl`, stage only those immutable
references, not nuPlan test, mini, map, or LiDAR data:

```bash
.venv/bin/python \
  scripts/download_navsim_nuplan10hz_cameras.py \
  --inventory-jsonl dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_inventory.jsonl \
  --nuplan-root /path/to/nuplan-v1.1 \
  --report-json dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_download_report.json
```

The downloader is resume-safe and writes exact files below
`sensor_blobs/<log>/CAM_*/`. It verifies each staged JPEG's TAR-member size,
SHA-256, and JPEG markers. The source camera objects are uncompressed TAR
streams despite their `.zip` suffix; staging therefore scans their member
order instead of assuming random ZIP lookup. It does not retain those large
source TARs, but a TAR has no central directory, so the network scan can read
more archive bytes than the final exact-JPEG footprint. Check available space
against the planner's unique-image inventory before starting the full run.

The persistent state database is
`dataset/cache/navsim_nuplan10hz/camera_download_state.sqlite`. It records
each exact target and archive-header checkpoint. It is protected against two
concurrent downloader processes and can be resumed without rebuilding its
multi-million-row target ledger. To validate or distribute a particular
archive, add its exact public key:

```bash
.venv/bin/python \
  scripts/download_navsim_nuplan10hz_cameras.py \
  --archive-key public/nuplan-v1.1/sensor_blobs/train_set/nuplan-v1.1_train_camera_0.zip \
  --max-headers-per-archive 100
```

## Target-Only TAR Offset Index

The public camera files are TAR streams without a central directory. Before
materializing JPEGs, a header-only pass can record where every selected
`navtrain` JPEG resides in its immutable upstream archive:

```bash
.venv/bin/python \
  scripts/download_navsim_nuplan10hz_cameras.py \
  --operation offset-index \
  --workers 8 \
  --state-sqlite dataset/cache/navsim_nuplan10hz/camera_download_state.sqlite \
  --tar-offset-index-sqlite \
    dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_tar_offsets.sqlite \
  --report-json \
    dataset/manifests/navsim_nuplan10hz/navtrain_camera_10hz_tar_offset_index_report.json
```

This operation does **not** download or write JPEG payloads. It records only
the source archive, pinned ETag, TAR header offset, image payload offset, and
payload size for the immutable 10 Hz `navtrain` inventory. The compact
offset-index SQLite file is published atomically only when all 5,625,150
selected JPEGs have been located. A later exact-range JPEG downloader can then
fetch selected image payloads without repeating the full TAR walk.

`NuPlan10HzCameraHistoryIndex` remains metadata-only. For pixels, use
`Navsim10HzImageLoader` with a staged nuPlan root. It returns each channel's
chronological image sequence, exact `native_image_timestamp_us`, and a
validity mask. Its optional padding method uses `False` mask values and
timestamp `-1` for non-source slots, so a model can batch variable-length
histories without treating padding as an observed frame.

## Low-Transfer Packed Staging

The offset-index operation above is useful when a reusable upstream TAR
member index is the desired artifact. It is not the fastest path when the
immediate goal is to obtain the selected JPEG bytes over a constrained
network link. A TAR header already declares the following payload length, so
the selective packer reads one validated 512-byte header, jumps over an
unwanted payload, and range-fetches payload bytes only when the member is in
the immutable target ledger:

```bash
.venv/bin/python \
  scripts/pack_navsim_nuplan10hz_cameras.py \
  --target-state-sqlite \
    dataset/cache/navsim_nuplan10hz/camera_download_state.sqlite \
  --pack-state-sqlite \
    dataset/cache/navsim_nuplan10hz/camera_pack_state.sqlite \
  --pack-root \
    dataset/raw/nuplan-v1.1-navtrain-10hz/camera_packs \
  --existing-camera-root \
    dataset/navsim2/sensor_blobs/trainval \
  --workers 8
```

The packer combines header indexing and payload staging in one resumable
pass. It produces one raw concatenated-JPEG pack per upstream archive plus a
SQLite index containing:

- immutable source archive key, size, ETag, TAR header offset, and payload
  offset;
- destination-relative JPEG path;
- storage kind (`pack` or an already-present OpenScene/NAVSIM JPEG);
- pack-relative path and byte offset when packed;
- exact payload size and SHA-256.

Pack bytes are fsynced before their SQLite offsets are committed. On resume,
uncommitted trailing bytes are truncated from the pack owned by this
operation. A source JPEG is reused from the existing 2 Hz OpenScene tree only
when the canonical path exists as a regular file, its byte size equals the
native TAR declaration, and it passes JPEG marker and SHA-256 validation.
Missing target JPEGs retain their exact upstream payload bytes. The compact
packs avoid millions of filesystem inodes and may be materialized into loose
JPEGs later.

On August 7, 2026, the verified full inventory contained 5,625,150 target
JPEGs. The staged OpenScene/NAVSIM tree already contained 1,219,960 of those
exact target paths (152,495 per camera channel), leaving 4,405,190 target
payloads for packed retrieval. This saves duplicate storage but does not
change the canonical 10 Hz timestamps or validity masks.

The archive URL used for payload reads may differ from the S3 listing URL
only when every object still matches the official listed byte size and ETag.
This permits a faster range-capable CDN or a source-adjacent relay without
weakening source identity. Test, mini, maps, LiDAR, and non-inventory camera
payloads remain excluded.

### Relay-to-HPC Offload

Completed relay packs are copied into the canonical HPC location:

```text
dataset/raw/nuplan-v1.1-navtrain-10hz/camera_packs/
```

The automatic offloader is designed for a space-constrained relay:

```bash
python3 scripts/offload_navsim_nuplan10hz_relay.py \
  --watch \
  --threshold-gb 200 \
  --poll-seconds 300
```

It triggers when completed immutable packs exceed 200 decimal GB. A smaller
final remainder is flushed only after the independent relay scheduler
atomically publishes `state/navsim10hz_packing_complete.json`; process
liveness is not used as a deletion condition. Transfer is pull-based from HPC
using a dedicated key with `BatchMode`; no password is stored. The
offloader uses resumable rsync, creates SQLite backup snapshots, copies batch
reports, and compares full SHA-256 hashes for every pack. Only exact regular
`train_set/nuplan-v1.1_train_camera_N.pack` or
`val_set/nuplan-v1.1_val_camera_N.pack` files whose size, modification time,
SHA-256, and complete SQLite record all agree are unlinked from the relay.
SQLite states, reports, manifests, partial packs, and unrelated files are
never deleted.

Each successful cycle publishes an immutable local manifest under:

```text
dataset/manifests/navsim_nuplan10hz/offload_manifests/
```

The local manifest is written and fsynced before relay deletion. Thus every
deleted relay pack has a complete verified copy on HPC plus the corresponding
SQLite state snapshot and report needed to interpret it.

The remaining relay packing is run separately by:

```bash
.venv/bin/python scripts/run_navtrain10hz_relay.py \
  --config configs/relay-producer.yaml
```

This scheduler never transfers or deletes data. It owns three bounded,
resumable SQLite batches covering the remaining 39 archives exactly once and
starts a batch only when at least 500 decimal GB is free. Completed batch
recognition comes from immutable SQLite status, so the scheduler remains
correct after the offloader removes an already-verified pack. Conversely, the
offloader operates only on finalized `.pack` files and never modifies a
partial pack or packer state. Packing and offloading may therefore overlap on
different immutable files without process-liveness coupling.

For the lowest-risk experiment, keep official 2 Hz `navtrain` anchors and
labels, and replace only their camera history with verified native 10 Hz
images. Creating new 10 Hz anchors is a separate dataset-construction task:
it requires regenerated labels, route eligibility, and evaluation, and must
be reported as a new benchmark protocol.
