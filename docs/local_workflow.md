# Fast single-machine workflow

The single-machine workflow is not a fallback slow path. It uses the same
low-transfer selective packer as relay mode and writes the final pack format
directly.

## Stages

1. `doctor` validates the pinned filter, NAVSIM layout, dependencies, and free
   space.
2. The metadata downloader range-extracts only the 1,192 required nuPlan DB
   members.
3. The planner proves NAVSIM/OpenScene-to-nuPlan identities and emits the
   canonical history index plus immutable image inventory.
4. `--operation initialize` creates the 5,625,150-row target ledger without
   scanning or downloading camera TAR payloads.
5. The packer processes camera archives concurrently through a range-capable
   CDN. It reads validated TAR headers, jumps over unwanted payloads, reuses
   official NAVSIM frames, and fetches only missing target JPEGs.
6. The finalizer renders deterministic random panels, builds the storage
   index, and decodes one real eight-camera anchor.

## Performance controls

`pack_workers` controls concurrent source archives. Use 16-32 on a fast link;
the allowed maximum is 55. More workers are not always faster if the CDN or
local link throttles requests.

The final packs contain about 946 GB. Allow at least 1 TB free for packs,
state, manifests, and the 4 GB storage index. Official NAVSIM camera files are
reused instead of copied.

The public camera objects are TAR streams without a central directory. A
truly fresh run must discover member headers at least once. State checkpoints
make that discovery resumable. A previously published, ETag-pinned target
offset index may accelerate a later recreation, but it is an optional
metadata cache and never changes source validation.

## Recovery

Rerun the same stage with the same configuration. The packer validates source
ETags, committed byte counts, and SQLite state, truncates only uncommitted
tails, and continues. Do not delete state SQLite files while retaining their
partial packs.
