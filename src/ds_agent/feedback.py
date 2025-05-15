import os
import re
import requests
from typing import List, Dict
import yaml


class FeedbackCollector:
    """Collects feedback comments from GitHub PRs (and optionally Slack threads) for a given episode."""

    def __init__(self, github_token: str):
        self.github_token = github_token
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN is required for GitHub API access.")

    def fetch_github_comments(self, pr_url: str) -> List[Dict]:
        """Fetch comments from a GitHub PR. Returns a list of comment dicts."""
        # Example PR URL: https://github.com/owner/repo/pull/123
        m = re.match(r"https://github.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
        if not m:
            raise ValueError(f"Invalid PR URL: {pr_url}")
        owner, repo, pr_number = m.group(1), m.group(2), m.group(3)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        headers = {"Authorization": f"token {self.github_token}"}
        resp = requests.get(api_url, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"GitHub API error: {resp.status_code} {resp.text}")
        comments = resp.json()
        # Normalize output
        return [
            {
                "id": c["id"],
                "author": c["user"]["login"],
                "body": c["body"],
                "created_at": c["created_at"],
            }
            for c in comments
        ]


class FeedbackClassifier:
    """Classifies feedback comments using keyword/regex rules into types (scope_change, bug, polish, question)."""

    @staticmethod
    def get_priority(task_type: str) -> str:
        if task_type in ("bug", "scope_change"):
            return "high"
        if task_type == "question":
            return "medium"
        return "low"

    def classify(self, comment: str) -> str:
        """Classify a comment and return its type as a string.
        Types: scope_change, bug, polish, question.
        Actionable types (scope_change, bug) are prioritized over question.
        """
        text = comment.lower()
        # Scope change: contains 'scope', 'add', 'change', 'expand', 'segment', 'dimension', 'include', 'support'
        if re.search(r"\b(scope|add|change|expand|segment|dimension|include|support)\b", text):
            return "scope_change"
        # Bug: contains 'bug', 'fix', 'broken', 'error', 'issue', 'fail', 'fails', 'does not work'
        if re.search(r"\b(bug|fix|broken|error|issue|fail(s)?|does not work)\b", text):
            return "bug"
        # Question: contains a question mark or starts with 'why', 'how', etc.
        if "?" in text or re.match(r"\b(why|how|what|when|where|who)\b", text):
            return "question"
        # Polish: default if none of the above
        return "polish"


class BacklogUpdater:
    """Appends structured tasks to the episode's backlog.yaml, ensuring idempotency (no duplicates)."""

    def __init__(self, backlog_path: str):
        self.backlog_path = backlog_path

    def add_task(self, task: dict) -> None:
        """Add a task to the backlog if not already present (by source_comment)."""
        # Load existing backlog or start new
        try:
            with open(self.backlog_path, "r") as f:
                backlog = yaml.safe_load(f) or []
        except FileNotFoundError:
            backlog = []
        # Check for duplicate by source_comment
        if any(t.get("source_comment") == task.get("source_comment") for t in backlog):
            return  # Already present
        backlog.append(task)
        with open(self.backlog_path, "w") as f:
            yaml.safe_dump(backlog, f, sort_keys=False)
