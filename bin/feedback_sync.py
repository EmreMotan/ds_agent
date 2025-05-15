#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path
import json
import time
from ds_agent.feedback import FeedbackCollector, FeedbackClassifier, BacklogUpdater
from ds_agent.memory import Memory

LOG_LEVEL = 1  # 0=quiet, 1=normal, 2=verbose


def log(msg, level=1):
    if LOG_LEVEL >= level:
        print(msg)


def run_sync(episode_dir, episode_file, backlog_file, episode_id, ingest_backlog):
    if not episode_file.exists():
        log(f"[ERROR] Episode file not found: {episode_file}", level=0)
        sys.exit(1)

    with open(episode_file) as f:
        episode_data = json.load(f)
    pr_url = episode_data.get("pr_url")
    if not pr_url:
        log("[ERROR] pr_url field missing in episode.json. Please add the PR URL.", level=0)
        sys.exit(1)

    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        log("[ERROR] GITHUB_TOKEN environment variable not set.", level=0)
        sys.exit(1)

    collector = FeedbackCollector(github_token)
    classifier = FeedbackClassifier()
    updater = BacklogUpdater(str(backlog_file))

    log(f"[INFO] Fetching comments from PR: {pr_url}", level=1)
    comments = collector.fetch_github_comments(pr_url)
    log(f"[INFO] {len(comments)} comments fetched.", level=1)

    new_tasks = 0
    for c in comments:
        task_type = classifier.classify(c["body"])
        task = {
            "id": f"T-{c['id']}",
            "type": task_type,
            "payload": c["body"].strip(),
            "status": "open",
            "source_comment": c["id"],
            "author": c["author"],
            "created_at": c["created_at"],
            "priority": classifier.get_priority(task_type),
        }
        updater.add_task(task)
        new_tasks += 1
        log(f"[DEBUG] Added task {task['id']} with priority {task['priority']}", level=2)

    log(
        f"[INFO] Sync complete. {new_tasks} tasks processed (duplicates skipped). Backlog: {backlog_file}",
        level=1,
    )

    if ingest_backlog:
        try:
            memory = Memory(Path("memory"))
            memory.ingest(backlog_file, episode_id=episode_id)
            log(f"[INFO] Ingested backlog.yaml into Memory for episode {episode_id}", level=1)
        except Exception as e:
            log(f"[ERROR] Failed to ingest backlog.yaml into Memory: {e}", level=0)


def main():
    global LOG_LEVEL
    parser = argparse.ArgumentParser(
        description="Sync feedback from GitHub PRs (and optionally Slack) into episode backlog."
    )
    parser.add_argument("--episode", required=True, help="Episode ID (e.g. DS-25-005)")
    parser.add_argument("--once", action="store_true", help="Run sync once and exit.")
    parser.add_argument(
        "--loop", type=int, default=None, help="Poll interval in seconds (if set, runs in a loop)."
    )
    parser.add_argument("--verbose", action="store_true", help="Show debug output.")
    parser.add_argument("--quiet", action="store_true", help="Suppress info output (only errors).")
    parser.add_argument(
        "--ingest-backlog", action="store_true", help="Ingest backlog.yaml into Memory after sync."
    )
    args = parser.parse_args()

    if args.quiet:
        LOG_LEVEL = 0
    elif args.verbose:
        LOG_LEVEL = 2
    else:
        LOG_LEVEL = 1

    episode_dir = Path("episodes") / args.episode
    episode_file = episode_dir / "episode.json"
    backlog_file = episode_dir / "backlog.yaml"

    if args.loop:
        log(
            f"[INFO] Starting feedback sync loop every {args.loop} seconds. Press Ctrl+C to stop.",
            level=1,
        )
        try:
            while True:
                run_sync(episode_dir, episode_file, backlog_file, args.episode, args.ingest_backlog)
                log(f"[INFO] Sleeping for {args.loop} seconds...", level=1)
                time.sleep(args.loop)
        except KeyboardInterrupt:
            log("[INFO] Exiting feedback sync loop.", level=1)
    else:
        run_sync(episode_dir, episode_file, backlog_file, args.episode, args.ingest_backlog)


if __name__ == "__main__":
    main()
