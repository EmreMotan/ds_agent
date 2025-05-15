import pathlib
import hashlib
import json
from typing import List
from datetime import datetime
import shutil


class SnapshotManager:
    """Handles artifact snapshotting and manifest updates for an episode."""

    ARTIFACT_EXTS = {".ipynb", ".csv", ".xlsx", ".pdf", ".html", ".png", ".jpg", ".jpeg"}

    def __init__(self, episode_dir: pathlib.Path):
        self.episode_dir = episode_dir

    def compute_sha256(self, file_path: pathlib.Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def snapshot_outputs(self) -> List[pathlib.Path]:
        """Copy executed notebooks and artifacts to outputs/DATE_HASH/. Return list of snapshot file paths."""
        # Find all artifact files (excluding outputs/)
        artifact_files = []
        for ext in self.ARTIFACT_EXTS:
            artifact_files.extend(self.episode_dir.glob(f"**/*{ext}"))
        # Exclude anything already in outputs/
        artifact_files = [f for f in artifact_files if "outputs" not in f.parts]
        if not artifact_files:
            return []
        # Create outputs/DATE_HASH/ dir
        now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        hash_input = "".join(str(f) for f in artifact_files).encode()
        short_hash = hashlib.sha256(hash_input).hexdigest()[:8]
        out_dir = self.episode_dir / "outputs" / f"{now}_{short_hash}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Copy files
        copied = []
        for src in artifact_files:
            dest = out_dir / src.name
            shutil.copy2(src, dest)
            copied.append(dest)
        return copied

    def update_manifest(self, files: List[pathlib.Path]) -> None:
        """Update episode.json manifest with file type, SHA-256, and path for each file."""
        episode_json = self.episode_dir / "episode.json"
        if not episode_json.exists():
            raise FileNotFoundError(f"episode.json not found in {self.episode_dir}")
        with open(episode_json, "r") as f:
            data = json.load(f)
        if "artifacts" not in data or not isinstance(data["artifacts"], list):
            data["artifacts"] = []
        # Build new artifact entries
        new_entries = []
        for file in files:
            file_hash = self.compute_sha256(file)
            entry = {
                "type": file.suffix.lstrip("."),
                "sha256": file_hash,
                "path": str(file.relative_to(self.episode_dir)),
            }
            # Avoid duplicates
            if not any(
                e["sha256"] == file_hash and e["path"] == entry["path"] for e in data["artifacts"]
            ):
                new_entries.append(entry)
        data["artifacts"].extend(new_entries)
        with open(episode_json, "w") as f:
            json.dump(data, f, indent=2)
