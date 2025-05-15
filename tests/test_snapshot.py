import tempfile
import pathlib
import json
import shutil
from ds_agent.snapshot import SnapshotManager


def test_compute_sha256():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = pathlib.Path(tmpdir) / "test.txt"
        f.write_text("hello world")
        snap = SnapshotManager(pathlib.Path(tmpdir))
        h = snap.compute_sha256(f)
        # Precomputed SHA-256 for 'hello world'
        assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_snapshot_outputs_and_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        epdir = pathlib.Path(tmpdir) / "DS-25-006"
        epdir.mkdir()
        # Create dummy artifact files
        nb = epdir / "analysis.ipynb"
        csv = epdir / "results.csv"
        nb.write_text("notebook content")
        csv.write_text("csv content")
        # Create episode.json
        epjson = epdir / "episode.json"
        epjson.write_text(json.dumps({"episode_id": "DS-25-006"}))
        snap = SnapshotManager(epdir)
        files = snap.snapshot_outputs()
        assert len(files) == 2
        # Should be in outputs/ subdir
        for f in files:
            assert "outputs" in str(f)
            assert f.exists()
        # Update manifest
        snap.update_manifest(files)
        data = json.loads(epjson.read_text())
        assert "artifacts" in data
        assert len(data["artifacts"]) == 2
        # Add again, should not duplicate
        snap.update_manifest(files)
        data2 = json.loads(epjson.read_text())
        assert len(data2["artifacts"]) == 2
