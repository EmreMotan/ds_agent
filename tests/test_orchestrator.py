"""Tests for the Orchestrator class."""

import datetime as dt
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import subprocess

from ds_agent.episode import Episode, EpisodeStatus
from ds_agent.orchestrator import Orchestrator, main


@pytest.fixture
def episode_dir(tmp_path: Path) -> Path:
    """Create a temporary episode directory with a valid episode.json."""
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
    episode_dir = tmp_path / "episodes" / "DS-25-001"
    episode_dir.mkdir(parents=True)
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json(indent=2))
    return episode_dir


def test_orchestrator_initialization(episode_dir: Path) -> None:
    """Test that the Orchestrator initializes correctly."""
    orchestrator = Orchestrator(episode_dir)
    assert orchestrator.episode_dir == episode_dir
    assert orchestrator.episode_file == episode_dir / "episode.json"
    assert not orchestrator._should_stop


def test_determine_next_action(episode_dir: Path) -> None:
    """Test determining the next action based on episode status."""
    orchestrator = Orchestrator(episode_dir)
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

    action, next_status = orchestrator._determine_next_action(episode)
    assert action == "plan_analysis"
    assert next_status == EpisodeStatus.IN_ANALYSIS

    # Follow valid state transitions to reach FINAL
    episode.status = EpisodeStatus.IN_ANALYSIS
    episode.status = EpisodeStatus.REVIEW
    episode.status = EpisodeStatus.FINAL

    action, next_status = orchestrator._determine_next_action(episode)
    assert action is None
    assert next_status is None


def test_execute_action(episode_dir: Path) -> None:
    """Test executing an action and updating episode status."""
    orchestrator = Orchestrator(episode_dir)
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

    # Mock the agent's method
    orchestrator.agent.plan_analysis = MagicMock()

    # Execute the action
    orchestrator._execute_action(
        episode,
        "plan_analysis",
        EpisodeStatus.IN_ANALYSIS,
    )

    # Verify agent method was called
    orchestrator.agent.plan_analysis.assert_called_once_with(episode)

    # Verify episode status was updated
    updated = Episode.model_validate_json(orchestrator.episode_file.read_text())
    assert updated.status == EpisodeStatus.IN_ANALYSIS


def test_run_once(episode_dir: Path) -> None:
    """Test running one cycle of the orchestrator."""
    orchestrator = Orchestrator(episode_dir)
    orchestrator._execute_action = MagicMock()

    orchestrator.run_once()

    orchestrator._execute_action.assert_called_once()


def test_run_once_no_action_needed(episode_dir: Path) -> None:
    """Test running one cycle when no action is needed."""
    orchestrator = Orchestrator(episode_dir)
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Test goal",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.BLOCKED,
        owners={
            "human_lead": "test@example.com",
            "agent_lead": "agent-001",
        },
        iteration_history=[],
    )
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json(indent=2))

    orchestrator.run_once()  # Should not raise any errors


def test_run_once_file_not_found(episode_dir: Path) -> None:
    """Test running one cycle when episode file doesn't exist."""
    orchestrator = Orchestrator(episode_dir)
    (episode_dir / "episode.json").unlink()

    with pytest.raises(SystemExit):
        orchestrator.run_once()


def test_run_loop_complete(episode_dir: Path) -> None:
    """Test running the loop until episode is complete."""
    orchestrator = Orchestrator(episode_dir)
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Test goal",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.FINAL,
        owners={
            "human_lead": "test@example.com",
            "agent_lead": "agent-001",
        },
        iteration_history=[],
    )
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json(indent=2))

    orchestrator.run_loop(1)  # Should exit immediately


def test_run_loop_interrupt(episode_dir: Path) -> None:
    """Test running the loop with interrupt."""
    orchestrator = Orchestrator(episode_dir)
    orchestrator.run_once = MagicMock()

    # Mock signal handler
    signal.signal = MagicMock()

    # Run loop for a short time
    with patch("time.sleep", side_effect=KeyboardInterrupt):
        orchestrator.run_loop(1)

    # Verify signal handler was set up
    signal.signal.assert_called_once_with(
        signal.SIGINT,
        orchestrator._handle_sigint,
    )

    # Verify run_once was called
    orchestrator.run_once.assert_called()


def test_cli_main(episode_dir: Path) -> None:
    """Test the CLI main function."""
    result = subprocess.run(
        ["python", "-m", "ds_agent.orchestrator", "DS-25-001", "--once"],
        env={"PYTHONPATH": str(Path.cwd())},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_cli_main_dir_not_found() -> None:
    """Test the CLI main function when episode directory doesn't exist."""
    result = subprocess.run(
        ["python", "-m", "ds_agent.orchestrator", "DS-25-999", "--once"],
        env={"PYTHONPATH": str(Path.cwd())},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


def test_cli_main_loop(episode_dir: Path) -> None:
    """Test the CLI main function in loop mode."""
    result = subprocess.run(
        ["python", "-m", "ds_agent.orchestrator", "DS-25-001", "--loop", "30"],
        env={"PYTHONPATH": str(Path.cwd())},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
