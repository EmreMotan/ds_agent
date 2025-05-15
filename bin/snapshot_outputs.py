#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from ds_agent.snapshot import SnapshotManager


def main():
    parser = argparse.ArgumentParser(
        description="Snapshot executed notebooks and artifacts for an episode."
    )
    parser.add_argument("--episode", required=True, help="Episode ID (e.g. DS-25-006)")
    args = parser.parse_args()

    episode_dir = Path("episodes") / args.episode
    if not episode_dir.exists():
        print(f"[ERROR] Episode directory not found: {episode_dir}")
        sys.exit(1)
    try:
        snap = SnapshotManager(episode_dir)
        files = snap.snapshot_outputs()
        if not files:
            print("[INFO] No artifacts found to snapshot.")
            return
        snap.update_manifest(files)
        print(f"[INFO] Snapshotted {len(files)} files and updated manifest in episode.json:")
        for f in files:
            print(f"  - {f.relative_to(episode_dir)}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
