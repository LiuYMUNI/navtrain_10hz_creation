from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LOCAL = load("run_navtrain10hz_local")
RELAY = load("run_navtrain10hz_relay")
FINALIZE = load("finalize_navtrain10hz_relay")
CAMERAS = load("download_navsim_nuplan10hz_cameras")


class PortableWorkflowTest(unittest.TestCase):
    def test_local_paths_are_derived_only_from_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = LOCAL.paths({
                "navsim_root": str(root / "navsim"),
                "work_root": str(root / "work"),
            })
            self.assertEqual(result["packs"], (root / "work/camera_packs").resolve())
            self.assertEqual(
                result["existing"],
                (root / "navsim/sensor_blobs/trainval").resolve(),
            )
            self.assertNotIn("2026", str(result))

    def test_relay_batches_cover_every_archive_once(self) -> None:
        keys = [f"archive-{number:02d}" for number in range(55)]
        groups = RELAY.batches(keys, 8)
        flattened = [key for group in groups for key in group]
        self.assertEqual(flattened, keys)
        self.assertEqual(len(set(flattened)), 55)
        self.assertEqual([len(group) for group in groups], [8, 8, 8, 8, 8, 8, 7])

    def test_initialize_operation_is_explicit(self) -> None:
        arguments = CAMERAS.parse_args(["--operation", "initialize"])
        self.assertEqual(arguments.operation, "initialize")

    def test_relay_finalizer_selects_a_complete_55_archive_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incomplete = root / "001"
            complete = root / "002"
            incomplete.mkdir(); complete.mkdir()
            self._write_state(incomplete / "camera_pack_state_000.sqlite", 0, 3, complete=False)
            self._write_state(complete / "camera_pack_state_000.sqlite", 0, 28, complete=True)
            self._write_state(complete / "camera_pack_state_001.sqlite", 28, 55, complete=True)
            selected = FINALIZE.complete_snapshot(root)
            self.assertEqual(len(selected), 2)
            self.assertEqual(selected[0].parent, complete)

    @staticmethod
    def _write_state(path: Path, start: int, end: int, *, complete: bool) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE archive_progress (archive_key TEXT PRIMARY KEY, "
                "status TEXT, target_total INTEGER, target_done INTEGER)"
            )
            connection.executemany(
                "INSERT INTO archive_progress VALUES (?,?,?,?)",
                [
                    (
                        f"archive-{number:02d}",
                        "complete" if complete else "running",
                        10,
                        10 if complete else 5,
                    )
                    for number in range(start, end)
                ],
            )
            connection.commit()
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
