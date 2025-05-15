# DS‑Agent – Architecture Overview  
*Version 0.2 • June 2025*

---

## 1 | Vision
> **Build an autonomous (yet human‑supervised) agent that can execute the entire data‑science workflow for a product team—from context gathering through rigorous analysis to decision briefs—while preserving reproducibility, transparency, and security.**

The system should:

- **Reduce analyst toil** (boilerplate code, documentation, re‑work)  
- **Shorten time‑to‑insight** through parallelized agent execution  
- **Learn continuously** by retaining episodic memory and feedback  
- **Ensure robust, reproducible, and transparent analysis**  
- **Support extensible, LLM-driven planning and execution**  

---

## 2 | High‑Level Component Diagram
<details>
<summary>Mermaid source</summary>

```mermaid
flowchart TD
  ceo([Product Owner (you)])
  architect([Chief Architect (me)])
  dev([Cursor Agent - Senior Developer])
  orchestrator[[Orchestrator (Planner + Executor)]]
  episodestore[(Episode JSON & History)]
  memory
  agents[/DataScienceAgent (multi-mode)/]
  repo((Git Repo))

  ceo -->|PRDs, feedback| architect
  architect -->|PRDs, tasks| dev
  dev -->|Code & PRs| repo
  orchestrator --> episodestore
  orchestrator --> agents
  agents --> memory
  memory --> orchestrator
  repo -->|deploy| orchestrator

  classDef memory fill:#f9f,stroke:#333,stroke-width:1px,color:#333;
```
</details>

---

## 3 | Layered Road‑Map
| Tier                       | Purpose                             | Key Deliverables                                                                                     |
| -------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **0 Foundations**          | Canonical work unit & control plane | *PRD‑001* Episode Object & Lifecycle<br>*PRD‑002* Orchestrator skeleton<br>*PRD‑003* Memory backbone |
| **1 Workflow Plumbing**    | Reproducible analysis loop          | Notebook template, auto‑tests, feedback‑to‑task pipeline                                             |
| **2 Guardrails**           | Prod‑ready robustness               | Security layer, cost controls, observability, robust plan validation, error handling                 |
| **3 UX & Adoption**        | Delight humans                      | Slack digests, approval gates, ROI dashboards                                                        |
| **4 Advanced Automations** | Stretch goals                       | Auto gap‑detection, anomaly hunters, CI/CD data pipelines                                            |

*Current focus ➜ **Tier 1–2: Workflow Plumbing & Guardrails***

---

## 4 | Repository Layout
```
/docs/          ← architecture, PRDs, specs
/src/           ← Python package (`ds_agent/`)
/tests/         ← pytest suites
/episodes/      ← episodic work folders (JSON, notebooks, outputs)
/memory/        ← raw docs, vector DB, future graph store
tasks.md        ← ordered checklist of open engineering tasks
CONTRIBUTING.md ← coding & review rules
README.md       ← project intro & quick‑start
```

---

## 5 | Development Workflow
1. **Product Owner** writes or prioritizes a feature idea  
2. **Architect** converts idea → PRD (in `/docs/`) and appends an unchecked item in `tasks.md`  
3. **Cursor Agent** (Agent Mode) selects the first unchecked task, reads the PRD, writes code & tests, and opens a PR  
4. **Review** follows the checklist in `CONTRIBUTING.md`; on approval, PR merges to `main`  
5. **Runtime** artifacts deploy automatically (future CI), and a new Episode may be spawned  

---

## 6 | Coding Standards (Summary)
- **Python 3.10+**, [black] + [isort] formatting  
- Type annotations mandatory (`mypy --strict` target)  
- Commit style: `type(scope): short imperative message`  
  - example: `feat(episode): add EpisodeStatus enum`  
- Unit‑test coverage target ≥ 90 % for `src/`  
- Linting and static analysis enforced in CI

(Full details in `CONTRIBUTING.md`.)

---

## 7 | Glossary
| Term                 | Meaning                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| **Episode**          | The atomic unit of analytical work, persisted as `episode.json`                                  |
| **Orchestrator**     | Central planner/executor that drives task sequencing and state updates                           |
| **DataScienceAgent** | LLM-driven planner that generates YAML plans and drafts reports                                  |
| **ExecAgent**        | Executes YAML plans, edits notebooks, validates plans, and runs code                             |
| **Memory**           | Composite of cold object storage + vector database (Chroma) used for retrieval‑augmented prompts |
| **SnapshotManager**  | Handles artifact snapshotting and manifest updates for episodes                                  |

---

## 8 | Open Questions (parking lot)
1. When do we introduce a symbolic graph over features ↔ metrics ↔ owners?  
2. What cost ceiling (tokens / $) do we enforce during the pilot?  
3. Will the orchestrator eventually become multi‑tenant (support parallel Episodes)?  
4. How do we best support human-in-the-loop review and feedback at scale?  

*(Track answers in future PRDs or decision logs.)*

---

## 9 | YAML-Based Plan Format

The agent now generates and parses plans as YAML, not JSON. Each plan is a list of steps, each with a unique id, tool, args, and (optionally) depends_on. Step references use ids, not numeric indices.

**Example:**
```yaml
steps:
  - id: load_teams
    tool: load_table
    args:
      table: team
      cols: [id, name]
      source: team
  - id: filter_active
    tool: filter_rows
    args:
      df: load_teams
      condition: "active == True"
    depends_on: [load_teams]
  - id: plot
    tool: plot_bar
    args:
      df: filter_active
      x: name
      y: score
    depends_on: [filter_active]
```

All code, validation, and execution now use these ids for step references and dependencies.

---

## 10 | Key System Features (2025)

- **LLM-driven planning:**  
  - Plans are generated as YAML, validated for schema and column correctness, and patched for robustness.
  - Only columns present in the schema or created in previous steps are allowed.
  - Composite columns must be created with `assign_column` before use.
  - Fallback plans are disabled; planning errors halt execution for transparency.

- **Robust plan validation and patching:**  
  - All columns and dependencies are tracked stepwise.
  - Aggregation, concatenation, and merging logic is robust to schema changes.
  - Debug output shows available columns at each step for troubleshooting.

- **Extensible tool registry:**  
  - Tools are registered in `tools.yaml` and implemented in `analysis/`.
  - New tools (statistical tests, regression, custom metrics, robust joins) are supported.

- **Memory and retrieval:**  
  - All ingested files are chunked, embedded, and stored in ChromaDB for semantic search.
  - Memory is used for retrieval-augmented prompting and episodic context.

- **Artifact management:**  
  - All outputs (notebooks, plots, CSVs) are snapshotted and tracked per episode.
  - Hashes and manifests ensure reproducibility.

- **Human-in-the-loop review:**  
  - All episodes move through SCOPED → IN_ANALYSIS → REVIEW → (REVISION/FINAL/BLOCKED/ARCHIVED).
  - Feedback is collected and classified for future improvement.

---

This document is the single source of truth for DS-Agent architecture and should be kept up to date as the system evolves.

---