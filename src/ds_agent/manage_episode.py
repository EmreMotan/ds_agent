"""CLI for managing episodes (argparse version)."""

import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt

from ds_agent.episode import Episode, EpisodeStatus, create_episode, load_episode, update_episode

console = Console()


def create_cmd(args):
    title = args.title
    goal = args.goal
    priority = args.priority
    if goal is None:
        goal = Prompt.ask("Enter goal statement")
    episode = create_episode(
        title=title,
        goal=goal,
        priority=priority,
    )
    episode_dir = Path("episodes") / episode.episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_file = episode_dir / "episode.json"
    episode_file.write_text(episode.model_dump_json(indent=2))
    console.print(f"[green]Created episode {episode.episode_id}[/]")
    console.print(f"Title: {episode.title}")
    console.print(f"Goal: {episode.goal_statement}")
    console.print(f"Status: {episode.status}")


def update_cmd(args):
    episode_id = args.episode_id
    status = args.status
    title = args.title
    goal = args.goal
    episode_file = Path("episodes") / episode_id / "episode.json"
    try:
        episode = load_episode(str(episode_file))
    except FileNotFoundError:
        console.print(f"[red]Episode {episode_id} not found[/]")
        sys.exit(1)
    changes = {}
    if status:
        changes["status"] = status
    if title:
        changes["title"] = title
    if goal:
        changes["goal_statement"] = goal
    if not changes:
        console.print("[yellow]No changes specified[/]")
        return
    try:
        updated = update_episode(episode, **changes)
        episode_file.write_text(updated.model_dump_json(indent=2))
        console.print(f"[green]Updated episode {episode_id}[/]")
        for key, value in changes.items():
            console.print(f"{key}: {value}")
    except Exception as e:
        console.print(f"[red]Error updating episode: {e}[/]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Manage episodes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # create
    create_parser = subparsers.add_parser("create", help="Create a new episode")
    create_parser.add_argument("title", help="Episode title")
    create_parser.add_argument("--goal", "-g", help="Episode goal statement")
    create_parser.add_argument(
        "--priority", "-p", default="MEDIUM", help="Priority level (HIGH/MEDIUM/LOW)"
    )
    create_parser.set_defaults(func=create_cmd)
    # update
    update_parser = subparsers.add_parser("update", help="Update an existing episode")
    update_parser.add_argument("episode_id", help="Episode ID (e.g. DS-25-001)")
    update_parser.add_argument("--status", "-s", help="New status")
    update_parser.add_argument("--title", "-t", help="New title")
    update_parser.add_argument("--goal", "-g", help="New goal statement")
    update_parser.set_defaults(func=update_cmd)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
