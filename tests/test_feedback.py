import os
import tempfile
import yaml
import pytest
from ds_agent.feedback import FeedbackClassifier, BacklogUpdater, FeedbackCollector
from unittest.mock import patch, Mock


def test_placeholder():
    assert True


def test_classifier_types():
    clf = FeedbackClassifier()
    assert clf.classify("Can you add a new dimension?") == "scope_change"
    assert clf.classify("This is broken and needs a fix.") == "bug"
    assert clf.classify("Why did you choose this method?") == "question"
    assert clf.classify("Looks good, just polish the output.") == "polish"
    # Edge: ambiguous
    assert clf.classify("") == "polish"
    assert clf.classify("Please expand the scope.") == "scope_change"
    assert clf.classify("Fails on my machine.") == "bug"
    assert clf.classify("How does this work?") == "question"


def test_backlog_updater_add_and_idempotency():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "backlog.yaml")
        updater = BacklogUpdater(path)
        task1 = {
            "id": "T-1",
            "type": "bug",
            "payload": "Broken",
            "status": "open",
            "source_comment": 1,
            "priority": FeedbackClassifier.get_priority("bug"),
        }
        updater.add_task(task1)
        # Should be present
        with open(path) as f:
            data = yaml.safe_load(f)
        assert len(data) == 1
        assert data[0]["id"] == "T-1"
        assert data[0]["priority"] == "high"
        # Add duplicate
        updater.add_task(task1)
        with open(path) as f:
            data2 = yaml.safe_load(f)
        assert len(data2) == 1  # Still only one
        # Add another
        task2 = {
            "id": "T-2",
            "type": "question",
            "payload": "Why?",
            "status": "open",
            "source_comment": 2,
            "priority": FeedbackClassifier.get_priority("question"),
        }
        updater.add_task(task2)
        with open(path) as f:
            data3 = yaml.safe_load(f)
        assert len(data3) == 2
        ids = {t["id"] for t in data3}
        assert ids == {"T-1", "T-2"}
        priorities = {t["priority"] for t in data3}
        assert "high" in priorities and "medium" in priorities


def test_feedback_collector_valid_url_and_fetch():
    token = "dummy"
    collector = FeedbackCollector(token)
    pr_url = "https://github.com/octocat/Hello-World/pull/1347"
    mock_comments = [
        {
            "id": 1,
            "user": {"login": "alice"},
            "body": "Looks good",
            "created_at": "2025-05-03T00:00:00Z",
        },
        {
            "id": 2,
            "user": {"login": "bob"},
            "body": "Please fix",
            "created_at": "2025-05-03T01:00:00Z",
        },
    ]
    with patch("requests.get") as mock_get:
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_comments
        mock_get.return_value = mock_resp
        comments = collector.fetch_github_comments(pr_url)
        assert len(comments) == 2
        assert comments[0]["author"] == "alice"
        assert comments[1]["body"] == "Please fix"


def test_feedback_collector_invalid_url():
    token = "dummy"
    collector = FeedbackCollector(token)
    bad_url = "https://github.com/octocat/Hello-World/issue/1347"
    with pytest.raises(ValueError):
        collector.fetch_github_comments(bad_url)


def test_feedback_collector_missing_token():
    with pytest.raises(ValueError):
        FeedbackCollector("")


def test_feedback_collector_api_error():
    token = "dummy"
    collector = FeedbackCollector(token)
    pr_url = "https://github.com/octocat/Hello-World/pull/1347"
    with patch("requests.get") as mock_get:
        mock_resp = Mock()
        mock_resp.status_code = 404
        mock_resp.text = "Not Found"
        mock_get.return_value = mock_resp
        with pytest.raises(RuntimeError):
            collector.fetch_github_comments(pr_url)
