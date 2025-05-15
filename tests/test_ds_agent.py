"""Tests for the DataScienceAgent class."""

import datetime as dt
from pathlib import Path

import pytest
import pandas as pd
import os
import shutil

from ds_agent.agents.ds_agent import DataScienceAgent
from ds_agent.episode import Episode, EpisodeStatus, create_episode


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    """Create a temporary episode directory."""
    episode_dir = tmp_path / "episodes" / "DS-25-001"
    episode_dir.mkdir(parents=True)
    return episode_dir


@pytest.fixture
def test_episode(episode_dir: Path) -> Episode:
    """Create a test episode."""
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
    return episode


def test_agent_initialization(episode_dir: Path) -> None:
    """Test agent initialization."""
    agent = DataScienceAgent(episode_dir)
    assert agent.episode_dir == episode_dir
    assert agent.history_file == episode_dir / "history.log"


def test_log_action(episode_dir: Path) -> None:
    """Test logging actions to history file."""
    agent = DataScienceAgent(episode_dir)
    agent._log_action("Test action")

    # Verify log file was created and contains the action
    history_file = episode_dir / "history.log"
    assert history_file.exists()
    content = history_file.read_text()
    assert "[AGENT] Test action" in content


def test_plan_analysis(episode_dir: Path, test_episode: Episode) -> None:
    """Test planning analysis."""
    agent = DataScienceAgent(episode_dir)
    agent.plan_analysis(test_episode)

    # Verify action was logged
    history_file = episode_dir / "history.log"
    content = history_file.read_text()
    assert f"Planning analysis for episode {test_episode.episode_id}" in content


def test_run_analysis(episode_dir: Path, test_episode: Episode) -> None:
    """Test running analysis."""
    agent = DataScienceAgent(episode_dir)
    agent.run_analysis(test_episode)

    # Verify action was logged
    history_file = episode_dir / "history.log"
    content = history_file.read_text()
    assert f"Running analysis for episode {test_episode.episode_id}" in content


def test_revise_analysis(episode_dir: Path, test_episode: Episode) -> None:
    """Test revising analysis."""
    agent = DataScienceAgent(episode_dir)
    agent.revise_analysis(test_episode)

    # Verify action was logged
    history_file = episode_dir / "history.log"
    content = history_file.read_text()
    assert f"Revising analysis for episode {test_episode.episode_id}" in content


@pytest.fixture
def tmp_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_episode(tmp_path):
    ep_dir = tmp_path / "ep1"
    ep_dir.mkdir(parents=True, exist_ok=True)
    episode = create_episode("Test", "Goal")
    ep_json = ep_dir / "episode.json"
    ep_json.write_text(episode.model_dump_json(indent=2))
    return ep_dir, episode


def test_agent_plan_and_run(tmp_csv, tmp_episode, monkeypatch):
    ep_dir, episode = tmp_episode
    # Patch data_sources.yaml to point to tmp_csv
    config_path = Path("config/data_sources.yaml")
    config_bak = None
    if config_path.exists():
        config_bak = config_path.read_text()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
sources:
  test:
    path: {tmp_csv}
    format: csv
    description: test csv
"""
    )
    try:
        agent = DataScienceAgent(ep_dir)
        # Plan: fill in table/source
        agent.plan_analysis(episode)
        plan = Path(ep_dir) / "plan.json"
        plan_data = plan.read_text()
        assert "load_table" in plan_data and "describe" in plan_data
        # Patch plan to use real table/source
        plan_json = plan.read_text()
        plan_json = plan_json.replace("<TABLE>", "test").replace("<SOURCE>", "test")
        plan.write_text(plan_json)
        # Run
        agent.run_analysis(episode)
        # Check history log
        hist = (ep_dir / "history.log").read_text()
        assert "Analysis complete." in hist
    finally:
        if config_bak is not None:
            config_path.write_text(config_bak)
        else:
            config_path.unlink(missing_ok=True)
        shutil.rmtree(ep_dir)
