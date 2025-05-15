"""Orchestrator module for DS-Agent system.

The Orchestrator is responsible for planning and executing data science
workflows, managing the lifecycle of Episodes, and coordinating between
different components.
"""

import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
import argparse
from rich.console import Console

from .agents.ds_agent import DataScienceAgent
from .episode import (
    Episode,
    EpisodeStatus,
    load_episode,
    update_episode,
)
from .exec_agent import ExecAgent

console = Console()


class Orchestrator:
    """Central component that plans and executes data science workflows."""

    def __init__(self, episode_dir: Path) -> None:
        """Initialize the Orchestrator.

        Args:
            episode_dir: Path to the episode directory.
        """
        self.episode_dir = episode_dir
        self.episode_file = episode_dir / "episode.json"
        self.agent = DataScienceAgent(episode_dir)
        self._should_stop = False

    def _handle_sigint(self, signum: int, frame: Optional[object]) -> None:
        """Handle SIGINT (Ctrl-C) gracefully.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        self._should_stop = True
        console.print("\n[bold yellow]Stopping gracefully...[/]")

    def _determine_next_action(
        self, episode: Episode
    ) -> Tuple[Optional[str], Optional[EpisodeStatus]]:
        """Determine the next action based on the current episode status.

        Args:
            episode: The current episode.

        Returns:
            Tuple of (action_name, next_status) or (None, None) if no action
            needed.
        """
        action_map = {
            EpisodeStatus.SCOPED: ("plan_analysis", EpisodeStatus.IN_ANALYSIS),
            EpisodeStatus.IN_ANALYSIS: ("run_analysis", EpisodeStatus.REVIEW),
            EpisodeStatus.REVISION: ("revise_analysis", EpisodeStatus.REVIEW),
            EpisodeStatus.BLOCKED: (None, None),
            EpisodeStatus.FINAL: (None, None),
            EpisodeStatus.ARCHIVED: (None, None),
        }
        return action_map.get(episode.status, (None, None))

    def _execute_action(self, episode: Episode, action: str, next_status: EpisodeStatus) -> None:
        """Execute the determined action and update episode status.

        Args:
            episode: The current episode.
            action: The action to execute.
            next_status: The status to set after the action.
        """
        # Execute the action
        getattr(self.agent, action)(episode)

        # Update episode status
        updated = update_episode(episode, status=next_status)

        # Save episode atomically
        temp_file = self.episode_file.with_suffix(".tmp")
        temp_file.write_text(updated.model_dump_json(indent=2))
        temp_file.replace(self.episode_file)

    def run_once(self) -> None:
        """Run one cycle of the orchestrator."""
        try:
            episode = load_episode(str(self.episode_file))
        except FileNotFoundError:
            msg = f"[red]Episode file not found: {self.episode_file}[/]"
            console.print(msg)
            sys.exit(1)

        # Integration: If IN_ANALYSIS or REVISION, call ExecAgent
        if episode.status in (EpisodeStatus.IN_ANALYSIS, EpisodeStatus.REVISION):
            console.print(f"[cyan]Invoking Execution Agent for episode {episode.episode_id}...[/]")
            agent = ExecAgent()
            agent.run_episode(episode.episode_id)
            # After execution, check for sanity_passed in executed notebook
            from .kernel_runner import extract_globals

            executed_nb = self.episode_dir / "analysis_executed.ipynb"
            globals_ = extract_globals(executed_nb) if executed_nb.exists() else {}
            print("DEBUG: Extracted globals:", globals_)
            print("DEBUG: sanity_passed value:", globals_.get("sanity_passed"))
            print("DEBUG: sanity_passed type:", type(globals_.get("sanity_passed")))
            if bool(globals_.get("sanity_passed")):
                console.print(f"[green]Sanity checks passed. Updating status to REVIEW.[/]")
                updated = update_episode(episode, status=EpisodeStatus.REVIEW)
                temp_file = self.episode_file.with_suffix(".tmp")
                temp_file.write_text(updated.model_dump_json(indent=2))
                temp_file.replace(self.episode_file)
            else:
                console.print(f"[yellow]Sanity checks not passed. Status unchanged.[/]")
            return

        action, next_status = self._determine_next_action(episode)
        if action is None:
            if episode.status in (EpisodeStatus.FINAL, EpisodeStatus.ARCHIVED):
                msg = f"[green]Episode {episode.episode_id} is complete[/]"
                console.print(msg)
            else:
                msg = f"[yellow]No action needed for status: {episode.status}[/]"
                console.print(msg)
            return

        msg = f"[blue]Executing {action} for episode {episode.episode_id}[/]"
        console.print(msg)
        self._execute_action(episode, action, next_status)
        console.print(f"[green]Updated status to {next_status}[/]")

    def run_loop(self, interval: int) -> None:
        """Run the orchestrator in a loop until episode is complete or interrupted.

        Args:
            interval: Seconds to wait between cycles.
        """
        # Set up signal handler
        signal.signal(signal.SIGINT, self._handle_sigint)

        while not self._should_stop:
            self.run_once()

            # Check if episode is complete
            episode = load_episode(str(self.episode_file))
            if episode.status in (EpisodeStatus.FINAL, EpisodeStatus.ARCHIVED):
                msg = f"[green]Episode {episode.episode_id} is complete[/]"
                console.print(msg)
                break

            # Wait for next cycle
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="DS-Agent Orchestrator CLI")
    parser.add_argument("episode_id", help="Episode ID (e.g. DS-25-001)")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument(
        "--loop", type=int, default=60, help="Run in a loop with given interval (seconds)"
    )
    args = parser.parse_args()
    episode_dir = Path("episodes") / args.episode_id
    if not episode_dir.exists():
        msg = f"[red]Episode directory not found: {episode_dir}[/]"
        console.print(msg)
        sys.exit(1)

    orchestrator = Orchestrator(episode_dir)
    if args.once:
        orchestrator.run_once()
    else:
        orchestrator.run_loop(args.loop)


if __name__ == "__main__":
    main()
