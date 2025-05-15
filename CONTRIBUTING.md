# Contributing to **DS‑Agent**

Welcome! 🎉  
This repository is a collaborative effort between:

- **Product Owner (PO)** – Emre  
- **Chief Architect** – ChatGPT (o3)  
- **Senior Developer** – Cursor Agent (Agent Mode)  
- *Future contributors are welcome—follow this guide to keep everything smooth.*

---

## Table of Contents
- [Contributing to **DS‑Agent**](#contributing-to-dsagent)
  - [Table of Contents](#tableofcontents)
  - [Project Setup](#projectsetup)
  - [Branching Model](#branching-model)
  - [Commit Message Convention](#commit-message-convention)
  - [Coding Standards](#coding-standards)
  - [Running Tests](#running-tests)
  - [Feedback-to-Task Pipeline](#feedback-to-task-pipeline)
  - [Artifact Snapshotting & Version-Control Hooks](#artifact-snapshotting-and-version-control-hooks)
  - [Pre‑commit Hooks](#precommit-hooks)
  - [Pull Request Process](#pull-request-process)
  - [Reviewer Checklist](#reviewer-checklist)
  - [Issue Templates](#issuetemplates)
  - [Post‑Merge Report](#postmerge-report)

---

## Project Setup

```bash
git clone <repo-url>
cd ds-agent
python -m venv .venv     # Python 3.11+
source .venv/bin/activate
pip install -r requirements.txt   # or `poetry install`
pre-commit install
pytest -q
```

---

## Branching Model

| Branch                 | Purpose                                                                   |
| ---------------------- | ------------------------------------------------------------------------- |
| `main`                 | Stable, passing tests; deployable at all times                            |
| `ep-<ID>-<short-desc>` | Feature branches tied to an Episode or PRD (e.g. `ep-001-episode-object`) |

> **Rule:** Only merge via PR — no direct pushes to `main`.

---

## Commit Message Convention

We follow a simplified **Conventional Commits** format:

```
<type>(<scope>): <short imperative message>
```

| `type`     | When to use                 |
| ---------- | --------------------------- |
| `feat`     | New functionality           |
| `fix`      | Bug fix                     |
| `docs`     | Documentation only          |
| `test`     | Adding or refactoring tests |
| `refactor` | Non‑functional code change  |
| `chore`    | Build, tooling, meta tasks  |

Examples:

```
feat(episode): add EpisodeStatus enum
fix(orchestrator): prevent status skip FINAL
docs(readme): add quick‑start section
```

---

## Coding Standards

* **Python 3.10+**  
* **Formatting:** `black` and `isort` (pre‑commit will auto‑format)  
* **Linting:** `flake8`  
* **Static typing:** `mypy --strict` (CI will fail otherwise)  
* **Docstrings:** Every public class/function must have a Google‑style docstring.  
* **Coverage:** ≥ 90 % for `src/` modules (`pytest --cov`).

---

## Running Tests

```bash
pytest -q           # fast run
pytest --cov        # with coverage
```

> Tests live in `/tests/` mirroring `/src/` structure.

---

## Feedback-to-Task Pipeline

The feedback-to-task pipeline automatically collects, classifies, and appends actionable tasks from GitHub PR comments to each episode's backlog.

- To enable feedback sync for an episode, add a `pr_url` field to its `episode.json`.
- You must set the `GITHUB_TOKEN` environment variable with access to the relevant repo.
- Tests for `src/ds_agent/feedback.py` must mock network calls (no real GitHub API calls in CI).
- Maintain ≥90% test coverage for the feedback pipeline (enforced in CI).
- `.yaml` is now a supported file type for Memory ingestion.
- The CLI supports `--verbose`, `--quiet`, and `--ingest-backlog` flags:
  - Use `--verbose` for debug output, `--quiet` to suppress info, and `--ingest-backlog` to auto-ingest backlog.yaml into Memory after sync.
- Before submitting changes, run all tests:
  ```bash
  pytest
  ```
- If your change affects CLI output or Memory ingestion, test with the relevant flags:
  ```bash
  python bin/feedback_sync.py --episode <EPISODE_ID> --once --verbose --ingest-backlog
  ```

---

## Artifact Snapshotting & Version-Control Hooks

- **Pre-push hook:**
  - Blocks direct pushes to `main` and enforces branch naming (`ep-<ID>-<desc>`)
  - To install:
    ```bash
    ln -sf ../../scripts/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    ```
- **Post-merge hook:**
  - After a merge, automatically snapshots artifacts for all episodes and updates their manifests
  - To install:
    ```bash
    ln -sf ../../scripts/post-merge .git/hooks/post-merge
    chmod +x .git/hooks/post-merge
    ```
- **Manual snapshotting:**
  - You can manually snapshot artifacts and update the manifest for a given episode:
    ```bash
    python bin/snapshot_outputs.py --episode <EPISODE_ID>
    ```
- **Review checklist:**
  - Ensure your branch name matches the required pattern before pushing
  - Confirm that artifacts are snapshotted and manifests updated after merges

---

## Pre‑commit Hooks

We use [pre‑commit](https://pre-commit.com/) to enforce style on every commit.

Installed by `pre-commit install` and includes:

1. `black`  
2. `isort`  
3. `flake8`  
4. `mypy --strict` (run as *check*, not type‑fail)  
5. `pytest -q` *(optional for fast commits; CI always runs full suite)*

---

## Pull Request Process

1. **Ensure branch is up to date with `main`.**  
2. **Verify checklist:**

   - [ ] Code compiles & `pytest -q` green  
   - [ ] `black`, `isort`, `flake8` clean  
   - [ ] Coverage ≥ 90 % (or justified)  
   - [ ] Linked to PRD / Episode (`Fixes #<issue>`)  
   - [ ] Updated docs if public behavior changed  
   - [ ] Added entry to `CHANGELOG.md` (if user‑visible)

3. **PR title** mirrors commit style (e.g. `feat(orchestrator): add skeleton loop`).  
4. **Description** includes:

   - Purpose / context  
   - How to test  
   - Acceptance checklist (copied and ticked)  

5. At least **one approval** (Architect or PO) required before merge.  
6. **Squash‑merge** preferred to keep `main` linear.

---

## Reviewer Checklist

- [ ] Scope matches linked PRD / episode  
- [ ] Design aligns with architecture overview  
- [ ] Tests cover happy + edge cases  
- [ ] No obvious security / perf issues  
- [ ] Docs updated  
- [ ] Commit messages readable & scoped

---

## Issue Templates

Create issues via GitHub's **"Task"** template (`.github/ISSUE_TEMPLATE/task.yaml`

---

## Post‑Merge Report

After your PR is merged, copy `docs/report_template.md` to  
`docs/reports/EP-<id>_report.md` (and `episodes/EP-<id>/report.md`)  
filled with the details of what shipped. Commit this as a docs‑only change.
