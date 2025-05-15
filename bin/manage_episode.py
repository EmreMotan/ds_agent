#!/usr/bin/env python3
"""CLI for managing Episodes (argparse version)."""

import json
import sys
from pathlib import Path
from typing import Optional, List
import datetime as dt
import re
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ds_agent.episode import Episode, EpisodeStatus, IterationEvent, IterationEventType
from ds_agent.memory import Memory

console = Console()


def create_episode(args):
    try:
        year = dt.datetime.now().strftime("%y")
        episodes_dir = Path("episodes")
        episodes_dir.mkdir(exist_ok=True)
        existing = [
            p.name
            for p in episodes_dir.iterdir()
            if p.is_dir() and re.match(r"DS-\d{2}-\d{3}", p.name)
        ]
        nums = [int(x.split("-")[-1]) for x in existing if x.split("-")[-1].isdigit()]
        next_n = max(nums) + 1 if nums else 1
        episode_id = f"DS-{year}-{next_n:03d}"
        now = dt.datetime.now(dt.timezone.utc)
        owners = {"human_lead": args.human_lead, "agent_lead": args.agent_lead}
        iteration_history = [
            {
                "timestamp": now.isoformat(),
                "event_type": "STATUS_CHANGE",
                "description": "Episode created with MEDIUM priority",
            }
        ]
        episode = {
            "episode_id": episode_id,
            "title": args.title,
            "goal_statement": args.goal,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "status": "SCOPED",
            "owners": owners,
            "iteration_history": iteration_history,
            "artifacts": [],
        }
        episode_path = episodes_dir / episode_id
        episode_path.mkdir(parents=True, exist_ok=True)
        with open(episode_path / "episode.json", "w") as f:
            json.dump(episode, f, indent=2)
        console.print(
            Panel(
                f"[bold green]Episode created successfully![/]\n\n"
                f"[bold]ID:[/] {episode_id}\n"
                f"[bold]Title:[/] {args.title}\n"
                f"[bold]Status:[/] SCOPED\n"
                f"[bold]Path:[/] {episode_path}",
                title="Episode Created",
                expand=False,
            )
        )
        print(episode_id)  # For test extraction
        return 0
    except Exception as e:
        console.print(f"[bold red]Error creating episode:[/] {e}")
        return 1


def list_episodes(args):
    episodes_path = Path("episodes")
    if not episodes_path.exists():
        console.print("[yellow]No episodes directory found.[/]")
        return 0
    episodes = []
    for episode_file in episodes_path.glob("*/episode.json"):
        try:
            with open(episode_file, "r") as f:
                data = json.load(f)
                episodes.append(data)
        except Exception as e:
            console.print(f"[red]Error reading {episode_file}: {e}[/]")
    if not episodes:
        console.print("[yellow]No episodes found.[/]")
        return 0
    if args.status:
        episodes = [ep for ep in episodes if ep.get("status") == args.status]
        if not episodes:
            console.print(f"[yellow]No episodes with status '{args.status}' found.[/]")
            return 0
    table = Table(title="Episodes")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Created At", style="blue")
    table.add_column("Last Updated", style="blue")
    for episode in episodes:
        table.add_row(
            episode.get("episode_id", ""),
            episode.get("title", ""),
            episode.get("status", ""),
            episode.get("created_at", ""),
            episode.get("updated_at", ""),
        )
    console.print(table)
    return 0


def show_episode(args):
    episode_file = Path("episodes") / args.episode_id / "episode.json"
    if not episode_file.exists():
        console.print(f"[bold red]Episode {args.episode_id} not found![/]")
        return 1
    try:
        with open(episode_file, "r") as f:
            data = json.load(f)
        console.print(
            Panel(
                "\n".join(
                    f"[bold]{k}:[/] {v}"
                    for k, v in data.items()
                    if k not in ["history", "artifacts"]
                ),
                title=f"Episode {args.episode_id}",
                expand=False,
            )
        )
        if "history" in data and data["history"]:
            history_table = Table(title="History")
            history_table.add_column("Timestamp", style="blue")
            history_table.add_column("Action", style="green")
            for entry in data["history"]:
                history_table.add_row(
                    entry.get("timestamp", ""),
                    entry.get("action", ""),
                )
            console.print(history_table)
        return 0
    except Exception as e:
        console.print(f"[bold red]Error reading episode:[/] {e}")
        return 1


def update_episode(args):
    episode_file = Path("episodes") / args.episode_id / "episode.json"
    if not episode_file.exists():
        console.print(f"[bold red]Episode {args.episode_id} not found![/]")
        return 1
    try:
        with open(episode_file, "r") as f:
            data = json.load(f)
        changed = False
        if args.status:
            data["status"] = args.status
            changed = True
        if args.title:
            data["title"] = args.title
            changed = True
        if args.goal:
            data["goal_statement"] = args.goal
            changed = True
        if args.dataset and len(args.dataset) > 0:
            data["datasets"] = list(args.dataset)
            changed = True
            console.print(f"[green]Updated datasets: {args.dataset}[/]")
        if not changed:
            console.print("[yellow]No changes specified[/]")
            return 0
        with open(episode_file, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[bold green]Episode {args.episode_id} updated successfully![/]")
        return 0
    except Exception as e:
        console.print(f"[bold red]Error updating episode:[/] {e}")
        return 1


def memory_operations(args):
    memory_dir = Path("memory")
    memory = Memory(memory_dir)
    if args.command == "ingest":
        if not args.path:
            console.print("[bold red]Path is required for ingest command[/]")
            return 1
        try:
            doc_id = memory.ingest(args.path, episode_id=args.episode_id)
            console.print(f"[bold green]Successfully ingested document(s) with ID(s): {doc_id}[/]")
            return 0
        except Exception as e:
            console.print(f"[bold red]Error ingesting document:[/] {e}")
            return 1
    elif args.command == "query":
        if not args.query:
            console.print("[bold red]Query text is required for query command[/]")
            return 1
        try:
            chunks = memory.query(args.query, k=args.k)
            if not chunks:
                console.print("[yellow]No results found.[/]")
                return 0
            console.print(f"[bold green]Found {len(chunks)} results:[/]")
            for i, chunk in enumerate(chunks, 1):
                console.print(
                    Panel(
                        f"[bold]Score:[/] {chunk.score:.4f}\n"
                        f"[bold]Source:[/] {chunk.source_uri}\n"
                        + (f"[bold]Episode:[/] {chunk.episode_id}\n" if chunk.episode_id else "")
                        + f"\n{chunk.content[:200]}..."
                        + ("[dim]...(truncated)[/]" if len(chunk.content) > 200 else ""),
                        title=f"Result {i}",
                        expand=False,
                    )
                )
            return 0
        except Exception as e:
            console.print(f"[bold red]Error querying memory:[/] {e}")
            return 1
    else:
        console.print(f"[bold red]Unknown command: {args.command}[/]")
        console.print("[bold]Available commands:[/] ingest, query")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Manage DS-Agent episodes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_p = subparsers.add_parser("create", help="Create a new Episode")
    create_p.add_argument("title", help="Episode title")
    create_p.add_argument("goal", help="Episode goal statement")
    create_p.add_argument("--human-lead", default="user", help="Human lead username")
    create_p.add_argument("--agent-lead", default="ds-agent", help="Agent lead identifier")
    create_p.set_defaults(func=create_episode)

    # list
    list_p = subparsers.add_parser("list", help="List all Episodes")
    list_p.add_argument("--status", help="Filter episodes by status (e.g., 'IN_ANALYSIS')")
    list_p.set_defaults(func=list_episodes)

    # show
    show_p = subparsers.add_parser("show", help="Show details of a specific Episode")
    show_p.add_argument("episode_id", help="Episode ID")
    show_p.set_defaults(func=show_episode)

    # update
    update_p = subparsers.add_parser("update", help="Update an existing Episode")
    update_p.add_argument("episode_id", help="Episode ID")
    update_p.add_argument("--status", help="New status")
    update_p.add_argument("--title", help="New title")
    update_p.add_argument("--goal", help="New goal statement")
    update_p.add_argument(
        "--dataset", action="append", help="Datasets used (can specify multiple times)"
    )
    update_p.set_defaults(func=update_episode)

    # memory
    memory_p = subparsers.add_parser("memory", help="Interact with the Memory Backbone")
    memory_p.add_argument(
        "command", choices=["ingest", "query"], help="Memory command: 'ingest' or 'query'"
    )
    memory_p.add_argument("--path", type=Path, help="Path to file or directory to ingest")
    memory_p.add_argument("--query", help="Query text")
    memory_p.add_argument("--episode-id", help="Episode ID to associate with documents")
    memory_p.add_argument("-k", type=int, default=5, help="Number of results to return for query")
    memory_p.set_defaults(func=memory_operations)

    args = parser.parse_args()
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
