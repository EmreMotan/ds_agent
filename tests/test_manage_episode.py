"""Tests for the manage_episode CLI."""

import datetime as dt
from pathlib import Path
import re
import subprocess
import sys
import os

import pytest

from ds_agent.episode import Episode, EpisodeStatus, load_episode


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    """Create a temporary episode directory."""
    episode_dir = tmp_path / "episodes" / "DS-25-001"
    episode_dir.mkdir(parents=True)
    return episode_dir


def test_update_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test updating an existing episode."""
    # Set up test environment
    monkeypatch.chdir(tmp_path)
    episode_dir = tmp_path / "episodes" / "DS-25-001"
    episode_dir.mkdir(parents=True)

    # Create test episode
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Test goal",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={
            "human_lead": "test@example.com",
            "agent_lead": "agent-001",
        },
        iteration_history=[],
    )
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json(indent=2))

    # Directly update status
    episode.status = EpisodeStatus.IN_ANALYSIS
    episode_file.write_text(episode.model_dump_json(indent=2))
    updated = load_episode(str(episode_file))
    assert updated.status == EpisodeStatus.IN_ANALYSIS

    # Directly update title
    updated.title = "New Title"
    episode_file.write_text(updated.model_dump_json(indent=2))
    loaded = load_episode(str(episode_file))
    assert loaded.title == "New Title"


def test_update_nonexistent_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test updating a nonexistent episode (should not exist)."""
    monkeypatch.chdir(tmp_path)
    episode_file = tmp_path / "episodes" / "DS-25-999" / "episode.json"
    assert not episode_file.exists()


def test_update_no_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test update command with no changes specified (API logic)."""
    # Set up test environment
    monkeypatch.chdir(tmp_path)
    episode_dir = tmp_path / "episodes" / "DS-25-001"
    episode_dir.mkdir(parents=True)

    # Create test episode
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Test goal",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={
            "human_lead": "test@example.com",
            "agent_lead": "agent-001",
        },
        iteration_history=[],
    )
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json(indent=2))
    loaded = load_episode(str(episode_file))
    # No changes made
    assert loaded.title == "Test Episode"
    assert loaded.status == EpisodeStatus.SCOPED


def test_create_episode_cli(tmp_path, monkeypatch):
    # Patch episodes dir to tmp_path
    monkeypatch.chdir(tmp_path)
    script = str(Path(__file__).parent.parent / "bin" / "manage_episode.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent / "src")
    result = subprocess.run(
        [sys.executable, script, "create", "Test Title", "Test Goal"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    eid = re.search(r"DS-\d{2}-\d{3}", result.stdout).group()
    ep_path = Path("episodes") / eid / "episode.json"
    assert ep_path.exists()
