import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
from ds_agent.snapshot import SnapshotManager


def save_fig(fig: plt.Figure, episode_id: str, name: str) -> str:
    """Save PNG to episode outputs, register artifact, return relative path."""
    # 1. Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, bbox_inches="tight")
        tmp_path = Path(tmp.name)

    # 2. Compute SHA-256 hash
    episode_dir = Path("episodes") / episode_id
    outputs_dir = episode_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    sm = SnapshotManager(episode_dir)
    file_hash = sm.compute_sha256(tmp_path)

    # 3. Move/rename to outputs/<hash>.png
    out_path = outputs_dir / f"{file_hash}.png"
    shutil.move(str(tmp_path), out_path)

    # 4. Update manifest
    sm.update_manifest([out_path])

    # 5. Return relative path from episode_dir
    rel_path = out_path.relative_to(episode_dir)
    return str(rel_path)
