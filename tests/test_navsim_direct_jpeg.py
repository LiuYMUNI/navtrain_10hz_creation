"""Exercise export, actual HTTP ranges, payload checks, and restart receipts."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("direct", Path(__file__).resolve().parents[1] / "scripts/navsim_direct_jpeg.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DirectTest(unittest.TestCase):
    def test_export_download_resume_and_corruption(self):
        jpeg = b"\xff\xd8fixture\xff\xd9"
        payload = bytes(512) + jpeg
        calls = []

        class Handler(BaseHTTPRequestHandler):
            status = 206
            bad_range = False

            def log_message(self, *args):
                pass

            def do_GET(self):
                calls.append(self.headers.get("Range"))
                self.send_response(self.status)
                self.send_header("ETag", '"test"')
                self.send_header("Content-Range", "bytes 0-0/1" if self.bad_range else f"bytes 512-{len(payload)-1}/{len(payload)}")
                self.end_headers()
                self.wfile.write(jpeg)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                target, source = root / "target.sqlite", root / "source.sqlite"
                meta = {"inventory_sha256": "inventory", "inventory_row_count": "1", "navsim_scene_filter_sha256": "filter"}
                for path in (target, source):
                    with sqlite3.connect(path) as db:
                        db.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
                        db.executemany("INSERT INTO metadata VALUES (?,?)", meta.items())
                with sqlite3.connect(target) as db:
                    db.execute("CREATE TABLE target(archive_key,tar_member_name,destination_relative_path)")
                    db.execute("INSERT INTO target VALUES ('archive','member','sensor_blobs/log/CAM_F0/image.jpg')")
                with sqlite3.connect(source) as db:
                    db.execute("INSERT INTO metadata VALUES ('format','navsim_nuplan10hz_camera_pack_state_v1')")
                    db.execute("CREATE TABLE archive_progress(archive_key,source_url,listed_size,remote_etag,status,target_total,target_done)")
                    db.execute("INSERT INTO archive_progress VALUES ('archive',?,?,'\"test\"','complete',1,1)", (f"http://127.0.0.1:{server.server_port}/archive", len(payload)))
                    db.execute("CREATE TABLE member_storage(archive_key,tar_member_name,destination_relative_path,storage_kind,payload_size,sha256,source_header_offset,source_data_offset)")
                    db.execute("INSERT INTO member_storage VALUES ('archive','member','sensor_blobs/log/CAM_F0/image.jpg','pack',?,?,0,512)", (len(jpeg), hashlib.sha256(jpeg).hexdigest()))
                assets = root / "assets"
                m.export(SimpleNamespace(output=str(assets), target=target, source=[source], asset_base=assets.as_uri(), source_base=f"http://127.0.0.1:{server.server_port}"))
                args = SimpleNamespace(manifest=(assets / "latest.json").as_uri(), output=root / "images", workers=2, attempts=1, limit=0)
                m.download(args)
                result = args.output / "sensor_blobs/log/CAM_F0/image.jpg"
                self.assertEqual(result.read_bytes(), jpeg)
                self.assertEqual(calls, [f"bytes=512-{len(payload)-1}"])
                m.download(args)
                self.assertEqual(len(calls), 1)
                result.write_bytes(b"bad")
                m.download(args)
                self.assertEqual(result.read_bytes(), jpeg)
                self.assertEqual(len(calls), 2)
                Handler.status = 200
                args.output = root / "ignored-range"
                with self.assertRaisesRegex(ValueError, "Range/source identity mismatch"):
                    m.download(args)
                self.assertFalse((args.output / row["path"] if 'row' in locals() else args.output / "sensor_blobs/log/CAM_F0/image.jpg").exists())
                Handler.status = 206
                Handler.bad_range = True
                with self.assertRaisesRegex(ValueError, "Range/source identity mismatch"):
                    m.download(args)
                Handler.bad_range = False
                # Wrong expected native hashes fail without publishing a JPEG.
                import gzip
                shard = assets / "001.jsonl.gz"
                row = json.loads(gzip.decompress(shard.read_bytes()))
                row["sha256"] = "0" * 64
                shard.write_bytes(gzip.compress((json.dumps(row) + "\n").encode()))
                manifest = json.loads((assets / "latest.json").read_text())
                manifest["shards"][0]["sha256"] = m.file_digest(shard)
                (assets / "latest.json").write_text(json.dumps(manifest))
                args.output = root / "bad-hash"
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    m.download(args)
                self.assertFalse((args.output / row["path"]).exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
