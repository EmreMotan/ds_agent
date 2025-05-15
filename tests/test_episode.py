"""Tests for the Episode model and related functionality."""

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from ds_agent.episode import (
    Episode,
    EpisodeStateError,
    EpisodeStatus,
    IterationEvent,
    IterationEventType,
    create_episode,
    load_episode,
    update_episode,
)


def test_episode_creation() -> None:
    """Test basic episode creation with required fields."""
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Analyze test data",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={"human_lead": "test@example.com", "agent_lead": "agent-001"},
        iteration_history=[],
    )
    assert episode.episode_id == "DS-25-001"
    assert episode.title == "Test Episode"
    assert episode.status == EpisodeStatus.SCOPED


def test_episode_id_validation() -> None:
    """Test episode_id format validation."""
    with pytest.raises(ValidationError):
        Episode(
            episode_id="invalid-id",
            title="Test Episode",
            goal_statement="Analyze test data",
            created_at=dt.datetime.now(dt.timezone.utc),
            updated_at=dt.datetime.now(dt.timezone.utc),
            status=EpisodeStatus.SCOPED,
            owners={
                "human_lead": "test@example.com",
                "agent_lead": "agent-001",
            },
            iteration_history=[],
        )


def test_status_transitions() -> None:
    """Test valid and invalid status transitions."""
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Analyze test data",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={"human_lead": "test@example.com", "agent_lead": "agent-001"},
        iteration_history=[],
    )

    # Test valid transition
    episode.status = EpisodeStatus.IN_ANALYSIS
    assert episode.status == EpisodeStatus.IN_ANALYSIS

    # Test invalid transition
    with pytest.raises(EpisodeStateError):
        episode.status = EpisodeStatus.FINAL


def test_iteration_history() -> None:
    """Test adding iteration events."""
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Analyze test data",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={"human_lead": "test@example.com", "agent_lead": "agent-001"},
        iteration_history=[],
    )

    # Add a status change event
    episode.status = EpisodeStatus.IN_ANALYSIS
    assert len(episode.iteration_history) == 1
    assert episode.iteration_history[0].event_type == IterationEventType.STATUS_CHANGE


def test_create_episode_helper() -> None:
    """Test the create_episode helper function."""
    episode = create_episode(
        title="Test Episode",
        goal="Analyze test data",
        priority="HIGH",
    )
    assert episode.status == EpisodeStatus.SCOPED
    assert episode.title == "Test Episode"
    assert episode.goal_statement == "Analyze test data"


def test_episode_persistence(tmp_path: Path) -> None:
    """Test saving and loading episodes."""
    episode = create_episode(
        title="Test Episode",
        goal="Analyze test data",
    )

    # Save episode
    episode_dir = tmp_path / episode.episode_id
    episode_dir.mkdir()
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json())

    # Load episode
    loaded = load_episode(str(episode_file))
    assert loaded.episode_id == episode.episode_id
    assert loaded.title == episode.title


def test_update_episode() -> None:
    """Test the update_episode helper function."""
    episode = create_episode(
        title="Test Episode",
        goal="Analyze test data",
    )

    initial_history_len = len(episode.iteration_history)

    updated = update_episode(
        episode,
        title="Updated Title",
        status=EpisodeStatus.IN_ANALYSIS,
    )

    assert updated.title == "Updated Title"
    assert updated.status == EpisodeStatus.IN_ANALYSIS
    assert len(updated.iteration_history) > initial_history_len


def test_episode_post_init() -> None:
    """Test that iteration_history is initialized correctly."""
    # Use model_construct to bypass validation
    episode = Episode.model_construct(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Analyze test data",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={"human_lead": "test@example.com", "agent_lead": "agent-001"},
        iteration_history=None,
    )
    assert episode.iteration_history == []


def test_bad_episode_id_formats() -> None:
    """Test various invalid episode_id formats."""
    invalid_ids = [
        "DS-25",  # Missing sequence
        "DS-25-",  # Empty sequence
        "DS-25-ABC",  # Non-numeric sequence
        "DS-25-0000",  # Too long sequence
        "DS-25-00",  # Too short sequence
        "DS-25-1",  # Not padded
        "DS-25-001-extra",  # Extra characters
    ]

    for episode_id in invalid_ids:
        with pytest.raises(ValidationError):
            Episode(
                episode_id=episode_id,
                title="Test Episode",
                goal_statement="Analyze test data",
                created_at=dt.datetime.now(dt.timezone.utc),
                updated_at=dt.datetime.now(dt.timezone.utc),
                status=EpisodeStatus.SCOPED,
                owners={"human_lead": "test@example.com", "agent_lead": "agent-001"},
                iteration_history=[],
            )


def test_missing_owners() -> None:
    """Test validation of required owner fields."""
    # Missing human_lead
    with pytest.raises(ValidationError):
        Episode(
            episode_id="DS-25-001",
            title="Test Episode",
            goal_statement="Analyze test data",
            created_at=dt.datetime.now(dt.timezone.utc),
            updated_at=dt.datetime.now(dt.timezone.utc),
            status=EpisodeStatus.SCOPED,
            owners={"agent_lead": "agent-001"},
            iteration_history=[],
        )

    # Missing agent_lead
    with pytest.raises(ValidationError):
        Episode(
            episode_id="DS-25-001",
            title="Test Episode",
            goal_statement="Analyze test data",
            created_at=dt.datetime.now(dt.timezone.utc),
            updated_at=dt.datetime.now(dt.timezone.utc),
            status=EpisodeStatus.SCOPED,
            owners={"human_lead": "test@example.com"},
            iteration_history=[],
        )

    # Empty owners
    with pytest.raises(ValidationError):
        Episode(
            episode_id="DS-25-001",
            title="Test Episode",
            goal_statement="Analyze test data",
            created_at=dt.datetime.now(dt.timezone.utc),
            updated_at=dt.datetime.now(dt.timezone.utc),
            status=EpisodeStatus.SCOPED,
            owners={},
            iteration_history=[],
        )


def test_blocked_state_transitions() -> None:
    """Test blocked state transitions and unblocking."""
    episode = Episode(
        episode_id="DS-25-001",
        title="Test Episode",
        goal_statement="Analyze test data",
        created_at=dt.datetime.now(dt.timezone.utc),
        updated_at=dt.datetime.now(dt.timezone.utc),
        status=EpisodeStatus.SCOPED,
        owners={"human_lead": "test@example.com", "agent_lead": "agent-001"},
        iteration_history=[],
    )

    # Block from SCOPED
    episode.status = EpisodeStatus.BLOCKED
    assert episode.status == EpisodeStatus.BLOCKED
    assert len(episode.iteration_history) == 1
    assert episode.iteration_history[0].event_type == IterationEventType.STATUS_CHANGE

    # Try invalid unblock
    with pytest.raises(EpisodeStateError):
        episode.status = EpisodeStatus.FINAL

    # Valid unblock to previous state
    episode.status = EpisodeStatus.SCOPED
    assert episode.status == EpisodeStatus.SCOPED
    assert len(episode.iteration_history) == 2

    # Block from IN_ANALYSIS
    episode.status = EpisodeStatus.IN_ANALYSIS
    episode.status = EpisodeStatus.BLOCKED
    assert episode.status == EpisodeStatus.BLOCKED

    # Valid unblock to previous state
    episode.status = EpisodeStatus.IN_ANALYSIS
    assert episode.status == EpisodeStatus.IN_ANALYSIS
