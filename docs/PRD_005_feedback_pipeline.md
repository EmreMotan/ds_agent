# PRD‑005: Feedback‑to‑Task Pipeline  
*Version 0.1 • May 3 2025*

---

## 1 | Overview  
This PRD equips DS‑Agent with a **closed feedback loop**: human comments in GitHub PRs or Slack threads are parsed, classified, and converted into actionable tasks inside each Episode.

Components:

1. **Feedback Collector** – polls PR reviews & Slack threads linked in `episode.json`.  
2. **NLP Classifier** – tags each comment (`scope_change`, `bug`, `polish`, `question`).  
3. **Backlog Updater** – appends structured tasks into `episodes/<id>/backlog.yaml`.  
4. **Orchestrator Integration** – reads backlog, prioritizes, and transitions Episode status when new blocking tasks appear.

---

## 2 | Problem / Goal  
Manual triage of feedback is slow and error‑prone. Automating it will:

* Shorten revision cycles.  
* Ensure no feedback is lost.  
* Provide traceability from comment → task → code change.

---

## 3 | Scope (MVP)

### 3.1 Feedback Collector  
* Sources: GitHub PR comments (REST API) and optional Slack thread URL.  
* Poll interval configurable (`--loop 60` default).  

### 3.2 NLP Classifier  
* Keyword + regex rules for MVP; upgradeable to ML later.  
* Output example:  

```yaml
id: cmt_123
author: "pm_amy"
ts: "2025-05-03T18:22:00Z"
type: scope_change
detail: "Can we segment by country?"
3.3 Backlog Updater
Creates/updates episodes/<id>/backlog.yaml

yaml
Always show details

Copy
- id: T-3
  type: scope_change
  payload: "Add country dimension"
  status: open
  source_comment: cmt_123
3.4 Orchestrator changes
While Episode status is REVIEW, Orchestrator checks backlog:

Open blocking tasks ⇒ set status REVISION.

No blocking tasks ⇒ allow FINAL.

3.5 CLI
bin/feedback_sync.py --episode DS-25-005 --once or --loop 60

4 | Success Criteria
Metric	Target
Comment → backlog latency	≤ 60 s
Classification precision (spot check)	≥ 80 %
Duplicate tasks created	0
Unit‑test coverage for src/feedback.py	≥ 90 %

5 | Non‑Functional Requirements
Env‑vars for GitHub & Slack tokens (GITHUB_TOKEN, SLACK_TOKEN).

Idempotent sync; re‑running does not duplicate tasks.

Python 3.10+; deps: requests, pyyaml, rich, filelock.

6 | Out of Scope
Jira/Linear integration (future).

ML classifier (future).

Sentiment analysis.

7 | Deliverables
Path	Artifact
src/ds_agent/feedback.py	Collector & classifier
bin/feedback_sync.py	CLI wrapper
tests/test_feedback.py	Unit tests
docs/feedback_flow.mmd	Mermaid sequence diagram

8 | Acceptance Checklist
 New comment classified & appended to backlog within 1 min.

 Duplicate runs do not add extra backlog entries.

 Backlog YAML validates against schema.

 Orchestrator moves Episode to REVISION when blocking task appears.

 Coverage ≥ 90 %.

 README updated with feedback sync instructions.

 Pre‑commit hooks clean.

9 | Sequence Diagram
Mermaid source docs/feedback_flow.mmd.

10 | Dependencies / References
PRD‑001 Episode Object (backlog file path).

PRD‑002 Orchestrator (state handling).

CONTRIBUTING for auth token setup.

End of PRD