import subprocess
import sys
import re
from pathlib import Path
import json
import os


def test_create_episode_cli(tmp_path):
    # Patch episodes dir to tmp_path
    venv_python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")
    script = str(Path(__file__).parent.parent / "bin" / "manage_episode.py")
    cwd = tmp_path
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    result = subprocess.run(
        [venv_python, script, "create", "Test Title", "Test Goal"],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    print("STDERR:", result.stderr)  # For debugging
    assert result.returncode == 0
    eid = re.search(r"DS-\d{2}-\d{3}", result.stdout).group()
    ep_path = Path(cwd) / "episodes" / eid / "episode.json"
    assert ep_path.exists()

    # Test --dataset update
    result2 = subprocess.run(
        [venv_python, script, "update", eid, "--dataset", "foo", "--dataset", "bar"],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
    )
    print("STDERR2:", result2.stderr)  # For debugging
    assert result2.returncode == 0
    data = json.loads(ep_path.read_text())
    assert data["datasets"] == ["foo", "bar"]
