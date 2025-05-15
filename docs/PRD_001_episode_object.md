# PRD‑001: Episode Object & Lifecycle  
*Version 0.1 • May 3 2025*

---

## 1 | Overview
This PRD specifies the **Episode Object**, its JSON schema, and the minimal tooling required to create, update, and audit an Episode.  
The Episode is the atomic unit of analytical work; every agent and human stakeholder will read or write this record throughout a project's lifecycle.

---

## 2 | Problem / Goal
Data‑science workflows become chaotic without a canonical "handle" for each investigation.  
We need a persistent Episode record so that:

* Agents can reliably load context, update status, and store artifacts.  
* Humans can track progress and reproduce results long after an engagement ends.  
* Future automation (metrics, dashboards) can query Episode metadata for reporting.

---

## 3 | Scope (MVP)
* Implement a **Pydantic model** for the Episode schema (see §9).  
* Provide **CRUD helpers**:

```python
create_episode(title: str, goal: str, priority: str = "MEDIUM") -> Episode
load_episode(episode_id: str) -> Episode
update_episode(ep: Episode, **changes) -> Episode
```

* Persist each Episode as `episodes/<episode_id>/episode.json`.  
* Maintain an **append‑only change‑log** at `episodes/<episode_id>/history.log`.  
* A minimal **CLI** (`bin/manage_episode.py`) for manual ops:

```bash
python manage_episode.py create "Signup Success drop"
python manage_episode.py update DS-25-001 --status IN_ANALYSIS
```

* Unit tests covering required‑field validation and illegal state transitions.

---

## 4 | Success Criteria
| Metric                                                           | Target              |
| ---------------------------------------------------------------- | ------------------- |
| Episode creation round‑trip                                      | ≤ 50 ms on a laptop |
| Unit‑test coverage (`src/episode.py`)                            | ≥ 90 %              |
| Illegal transition (`SCOPED → FINAL`) raises `EpisodeStateError` | 100 %               |

---

## 5 | Non‑Functional Requirements
* **Python 3.10**.  
* Dependencies limited to `pydantic`, `argparse`, `pytest`, `rich` (optional CLI output).  
* Atomic file writes (use temp file + rename).  
* Cross‑platform (Mac, Linux, Windows).  
* Thread‑safe within a single process (future multi‑thread orchestrator).

---

## 6 | Out of Scope
* Database or REST storage back‑ends.  
* GUI tooling.  
* Git hooks (handled in later tiers).

---

## 7 | Deliverables
| Path                         | File / Artifact                   |
| ---------------------------- | --------------------------------- |
| `src/episode.py`             | Pydantic model + helper functions |
| `bin/manage_episode.py`      | CLI wrapper                       |
| `tests/test_episode.py`      | Unit tests                        |
| `docs/episode_schema.yaml`   | Field‑level schema reference      |
| `docs/episode_lifecycle.mmd` | Mermaid state diagram             |
| README snippet               | Usage example                     |

---

## 8 | Acceptance Checklist
- [ ] All unit tests pass (`pytest -q`).  
- [ ] `manage_episode.py create` produces folder with JSON + log.  
- [ ] Status transitions adhere to lifecycle diagram; illegal moves raise `EpisodeStateError`.  
- [ ] JSON fields match `episode_schema.yaml`.  
- [ ] Docstrings & type hints present on all public methods.  
- [ ] Pre‑commit hooks (`black`, `isort`, `flake8`) show clean diff.  

---

## 9 | Schema (embedded quick reference)
See `docs/episode_schema.yaml` for the authoritative YAML.  
Key required fields:

* `episode_id: str` — unique slug (`DS-25-001`)  
* `title: str` — one‑line description  
* `goal_statement: str` — definition of done  
* `created_at, updated_at: datetime`  
* `status: enum<EpisodeStatus>` — `SCOPED`, `IN_ANALYSIS`, `REVIEW`, `REVISION`, `BLOCKED`, `FINAL`, `ARCHIVED`  
* `owners: object` — `human_lead`, `agent_lead`, optional `support` list  
* `iteration_history: list<IterationEvent>` (append‑only)

---

## 10 | Lifecycle Diagram
The Mermaid source is in `docs/episode_lifecycle.mmd`.  
States:

```
SCOPED → IN_ANALYSIS → REVIEW → (REVISION ↔ IN_ANALYSIS) → FINAL → ARCHIVED
```

`BLOCKED` may occur from any active state and returns to the previous state upon unblocking.

---

## 11 | Dependencies / References
* Architecture overview document (`docs/architecture_overview.md`).  
* Glossary and coding standards in `CONTRIBUTING.md`.  

---

*End of PRD*