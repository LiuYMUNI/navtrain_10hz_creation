# Relay workflow

Relay mode is useful when the destination has storage but poor internet
connectivity. The relay is a producer; the destination is the only transfer
and deletion authority.

## Producer

`run_navtrain10hz_relay.py` discovers all 55 archives from the immutable
target ledger and partitions them into declarative batches. No dated batch
names or previous-run assumptions are embedded in the code. Each batch has a
separate pack state, report, and log.

The current efficient workflow gives the relay read access to official
NAVSIM `sensor_blobs/trainval`. This allows it to mark the 1,219,960 official
files as trusted existing storage and avoid packing another ~262 GB.

The producer never runs rsync and never deletes data. It writes an atomic
completion marker only after all batches contain 55 unique complete archives.

## Destination offloader

The offloader uses a dedicated SSH key and `BatchMode`; passwords and keys are
never stored in this repository. It pulls only finalized regular `.pack`
files represented by complete relay SQLite rows.

Before deleting one exact relay pack, it requires:

- resumable rsync completion;
- local and remote byte-size equality;
- complete SHA-256 equality;
- a consistent SQLite backup snapshot;
- exact target/member coverage and valid offsets;
- an immutable local offload manifest durably written first.

Partial packs, SQLite states, reports, unrelated files, and changed packs are
never deleted.

After the final snapshot is present, run `finalize_navtrain10hz_relay.py` to
discover a 55-archive complete snapshot, render the visual audit, and publish
the unified storage index.
