"""Episode model and related functionality for the DS-Agent system."""

import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class EpisodeStateError(Exception):
    """Raised when an invalid state transition is attempted."""


class EpisodeStatus(str, Enum):
    """Valid states for an Episode."""

    SCOPED = "SCOPED"
    IN_ANALYSIS = "IN_ANALYSIS"
    REVIEW = "REVIEW"
    REVISION = "REVISION"
    BLOCKED = "BLOCKED"
    FINAL = "FINAL"
    ARCHIVED = "ARCHIVED"


class IterationEventType(str, Enum):
    """Types of events that can occur during an Episode."""

    STATUS_CHANGE = "STATUS_CHANGE"
    COMMENT = "COMMENT"
    BLOCKER_ADDED = "BLOCKER_ADDED"
    BLOCKER_RESOLVED = "BLOCKER_RESOLVED"


class IterationEvent(BaseModel):
    """Record of a change or comment during an Episode."""

    timestamp: dt.datetime
    event_type: IterationEventType
    description: str = Field(..., min_length=1, max_length=5000)


class Episode(BaseModel):
    """Core model representing a data science investigation."""

    episode_id: str = Field(..., pattern=r"^DS-[0-9]{2}-[0-9]{3}$")
    title: str = Field(..., min_length=1, max_length=100)
    goal_statement: str = Field(..., min_length=1, max_length=5000)
    created_at: dt.datetime
    updated_at: dt.datetime
    status: EpisodeStatus
    owners: Dict[str, Any] = Field(
        ...,
        json_schema_extra={
            "example": {
                "human_lead": "analyst@company.com",
                "agent_lead": "agent-001",
                "support": ["reviewer@company.com"],
            }
        },
    )
    iteration_history: List[IterationEvent] = Field(default_factory=list)
    artifacts: List[dict] = Field(default_factory=list)

    # Track previous status for validation
    _previous_status: Optional[EpisodeStatus] = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._previous_status = self.status

    @field_validator("owners")
    def validate_owners(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required owner fields are present."""
        if "human_lead" not in v or "agent_lead" not in v:
            raise ValueError("Both human_lead and agent_lead are required")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize with empty iteration history if none provided."""
        if self.iteration_history is None:
            self.iteration_history = []
        if self.artifacts is None:
            self.artifacts = []

    @property
    def valid_next_states(self) -> List[EpisodeStatus]:
        """Get list of valid next states based on current state."""
        transitions = {
            EpisodeStatus.SCOPED: [
                EpisodeStatus.IN_ANALYSIS,
                EpisodeStatus.BLOCKED,
            ],
            EpisodeStatus.IN_ANALYSIS: [
                EpisodeStatus.REVIEW,
                EpisodeStatus.BLOCKED,
            ],
            EpisodeStatus.REVIEW: [
                EpisodeStatus.REVISION,
                EpisodeStatus.FINAL,
                EpisodeStatus.BLOCKED,
            ],
            EpisodeStatus.REVISION: [
                EpisodeStatus.IN_ANALYSIS,
                EpisodeStatus.BLOCKED,
            ],
            EpisodeStatus.BLOCKED: ([self._previous_status] if self._previous_status else []),
            EpisodeStatus.FINAL: [EpisodeStatus.ARCHIVED],
            EpisodeStatus.ARCHIVED: [],
        }
        return transitions.get(self.status, [])

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to validate status transitions."""
        if name == "status" and hasattr(self, "status"):
            old_status = self.status
            if value not in self.valid_next_states:
                raise EpisodeStateError(
                    f"Invalid transition from {old_status} to {value}. "
                    f"Valid next states: {self.valid_next_states}"
                )
            self._previous_status = old_status

            # Record status change in iteration history
            self.iteration_history.append(
                IterationEvent(
                    timestamp=dt.datetime.now(dt.timezone.utc),
                    event_type=IterationEventType.STATUS_CHANGE,
                    description=f"Status changed from {old_status} to {value}",
                )
            )
        super().__setattr__(name, value)

    def register_artifact(self, artifact_type: str, path: str):
        self.artifacts.append(
            {
                "type": artifact_type,
                "path": path,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        self.updated_at = dt.datetime.now(dt.timezone.utc)
        self.save()


def _generate_episode_id() -> str:
    """Generate a unique episode ID in the format DS-YY-NNN."""
    year = dt.datetime.now().strftime("%y")
    # TODO: Check database for next available number in production
    return f"DS-{year}-001"


def create_episode(
    title: str,
    goal: str,
    priority: str = "MEDIUM",
    human_lead: Optional[str] = None,
    agent_lead: Optional[str] = None,
) -> Episode:
    """Create a new Episode with generated ID and timestamps.

    Args:
        title: One-line description
        goal: Definition of done
        priority: Priority level (default: MEDIUM)
        human_lead: Email of human lead (default: from config)
        agent_lead: ID of agent lead (default: from config)

    Returns:
        A new Episode instance in SCOPED state
    """
    now = dt.datetime.now(dt.timezone.utc)

    # TODO: Get these from config in production
    human_lead = human_lead or "default@company.com"
    agent_lead = agent_lead or "agent-001"

    return Episode(
        episode_id=_generate_episode_id(),
        title=title,
        goal_statement=goal,
        created_at=now,
        updated_at=now,
        status=EpisodeStatus.SCOPED,
        owners={
            "human_lead": human_lead,
            "agent_lead": agent_lead,
        },
        iteration_history=[
            IterationEvent(
                timestamp=now,
                event_type=IterationEventType.STATUS_CHANGE,
                description=(f"Episode created with {priority} priority"),
            )
        ],
    )


def load_episode(episode_path: str) -> Episode:
    """Load an Episode from its JSON file.

    Args:
        episode_path: Path to episode.json

    Returns:
        The loaded Episode instance

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid
    """
    with open(episode_path) as f:
        data = f.read()
    return Episode.model_validate_json(data)


def update_episode(episode: Episode, **changes: Any) -> Episode:
    """Update an Episode with changes and record in iteration history.

    Args:
        episode: Episode to update
        **changes: Attribute changes to apply

    Returns:
        Updated Episode instance

    Raises:
        EpisodeStateError: If an invalid status transition is attempted
    """
    now = dt.datetime.now(dt.timezone.utc)

    # Create events for all changes
    for attr, value in changes.items():
        old_value = getattr(episode, attr)
        if old_value != value:
            if attr == "status":
                event_type = IterationEventType.STATUS_CHANGE
                description = f"Status changed from {old_value} to {value}"
            else:
                event_type = IterationEventType.COMMENT
                description = f"Updated {attr} from '{old_value}' to '{value}'"

            episode.iteration_history.append(
                IterationEvent(
                    timestamp=now,
                    event_type=event_type,
                    description=description,
                )
            )

    # Apply all changes
    for attr, value in changes.items():
        setattr(episode, attr, value)

    episode.updated_at = now
    return episode
