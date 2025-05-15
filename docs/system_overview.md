# DS-Agent System: End-to-End Audit & Technical Documentation

> **For practical onboarding and usage, see:**
> - [Data Source Onboarding](data_source_onboarding.md)
> - [Analysis Workflow](analysis_workflow.md)
> - [Extending Tools](extending_tools.md)
> - [Agent Workflow](agent_workflow.md)

## 1. High-Level Workflow

### Intended Flow (from Architecture)
1. **Product Owner** proposes a feature/analysis (PRD).
2. **Architect** formalizes it as a PRD and adds a task.
3. **Cursor Agent** (AI/automation) picks up the task, creates an "Episode" (atomic work unit).
4. **Orchestrator** plans and executes the analysis, coordinating agents and tracking state.
5. **Agents** (DataScienceAgent, ExecAgent) perform planning, execution, and revision.
6. **Human Review**: Humans review, provide feedback, and approve or request revisions.
7. **Artifacts** (notebooks, outputs) are snapshotted and tracked.
8. **Memory**: Episodic and vector memory are used for retrieval and context.
9. **Deployment**: On approval, artifacts are deployed and new episodes may be spawned.

---

## 2. Technical Flow (Codebase Mapping)

### A. Episode Lifecycle

- **Episode**: The atomic unit of work, tracked as a JSON file in `/episodes/`.
  - Created with metadata: ID, title, goal, owners, status, iteration history, artifacts.
  - Status transitions: SCOPED → IN_ANALYSIS → REVIEW → (REVISION/FINAL/BLOCKED/ARCHIVED).
  - Iteration history logs all status changes and comments.

- **Episode Management**:  
  - `create_episode`, `load_episode`, `update_episode` in `episode.py`.
  - Status transitions are validated; artifacts and feedback are tracked.

### B. Orchestration

- **Orchestrator** (`orchestrator.py`):
  - Central controller for episode lifecycle.
  - Determines next action based on episode status.
  - Invokes the appropriate agent method (`plan_analysis`, `run_analysis`, `revise_analysis`).
  - Integrates with ExecAgent for notebook execution and sanity checks.
  - Updates episode status and persists changes.

### C. Agents

- **DataScienceAgent** (`agents/ds_agent.py`):
  - Placeholder for planning, running, and revising analysis.
  - Logs actions to episode history.
  - Intended to be replaced with a more intelligent agent in future tiers.

- **ExecAgent** (`exec_agent.py`):
  - Handles execution of analysis plans, notebook editing, and running.
  - Loads schema, validates plans, and applies tool-based workflows.
  - Integrates with LLMs for plan generation (future).
  - Executes analysis notebooks and checks for "sanity_passed" flag.

### D. Analysis & Tools

- **Analysis Utilities** (`analysis/`):
  - `analytics_tools.py`: Column selection, sorting, top-N, pivot, plotting, describe, value counts.
  - `segmentation.py`, `cohort.py`, `funnel.py`: Advanced analytics (group comparison, retention, funnel).
  - Plots and markdown summaries are supported.

- **Data Loading** (`data.py`):
  - `load_table`: Loads data from CSV/SQL/Parquet as configured in `data_sources.yaml`.
  - `profile_table`: Basic profiling (row count, null %, min/max).
  - All columns are stripped of whitespace after loading, columns are logged, and before applying a pandas query, all referenced columns are checked for existence. If any are missing, a clear error is raised listing missing and available columns. This ensures agent/LLM workflows fail fast and transparently.

### E. Memory

- **Memory Backbone** (`memory.py`):
  - Stores and retrieves documents (episodic and vector memory).
  - Uses ChromaDB for vector search and SentenceTransformers for embeddings.
  - Ingests files/folders, chunks text, and stores embeddings for retrieval.

### F. Artifact Management

- **Snapshotting** (`snapshot.py`):
  - Snapshots artifacts (notebooks, CSVs, plots) and updates episode manifest.
  - Computes hashes and ensures reproducibility.

### G. Human-in-the-Loop & Feedback

- **Feedback Collection** (`feedback.py`):
  - Fetches comments from GitHub PRs (and optionally Slack).
  - Classifies feedback (scope change, bug, polish, question).
  - Updates episode backlog with actionable tasks.

- **Review & Iteration**:
  - Human review is integrated via status transitions (REVIEW, REVISION).
  - Iteration history logs all changes and comments.

---

## 3. Human-in-the-Loop Points

- **Episode Ownership**: Each episode has a human lead and agent lead.
- **Review Status**: Analysis moves to REVIEW for human approval.
- **Feedback Integration**: Human feedback (via PRs, comments) is collected, classified, and can trigger new tasks or revisions.
- **Iteration History**: All changes, comments, and status transitions are logged for transparency.

---

## 4. Gaps & Ambiguities

- **Agent Intelligence**: DataScienceAgent is currently a stub; real planning and analysis logic is not yet implemented.
- **Automated Data Joins/Aggregations**: No reusable join/merge utilities; must be scripted per analysis.
- **End-to-End Templates**: No standardized pipeline for common analyses (e.g., career longevity).
- **Config Management**: Data sources must be registered in YAML config to be accessible.
- **Reporting**: While plots and markdown summaries exist, there is no standardized report template for analyses.
- **Multi-Agent Coordination**: Only a single agent is active per episode; multi-agent workflows are not yet implemented.

---

## 5. End-to-End Example (Current State)

1. **Create Episode**: User or agent creates an episode with a goal and owners.
2. **Orchestrator**: Picks up the episode, determines next action.
3. **Agent**: (Stub) logs planning/running/revision actions.
4. **ExecAgent**: Executes analysis notebook, checks for sanity, updates status.
5. **Human Review**: If in REVIEW, human can approve or request revision.
6. **Feedback**: Collected from PRs, classified, and added to backlog.
7. **Artifacts**: Snapshotted and tracked in episode manifest.
8. **Memory**: Documents and outputs are ingested for future retrieval.

---

## 6. Reference Diagram

```
User/Owner → Architect → Cursor Agent → Orchestrator → [DataScienceAgent, ExecAgent] → Analysis/Artifacts
      ↑                                                                                  ↓
   Feedback/Review  ←------------------- Human-in-the-Loop (REVIEW, REVISION) ←--------- 
```

---

## 7. Recommendations

- **Implement real agent logic** for planning and running analyses.
- **Add reusable data join/aggregation utilities**.
- **Standardize analysis/report templates** for common workflows.
- **Expand multi-agent and parallel episode support**.
- **Document data source configuration and onboarding steps**.
- **Schema validation and robust error handling** are now implemented in `load_table`, ensuring agent/LLM workflows fail fast and transparently if a query references missing columns.

---

This documentation can be saved as `docs/system_overview.md` or similar for ongoing reference and onboarding.  
Would you like me to create this file in your docs directory?

---

## 8. Documentation Maintenance

- **Keep this document updated** as new features, workflows, or architectural changes are introduced.
- **Reference this overview** in PRDs, onboarding guides, and technical discussions to ensure alignment and continuity.
- **Link to this document** from the project README and CONTRIBUTING guidelines for easy discoverability.

---

## 9. Appendix: Key File/Module Roles

| File/Module                   | Purpose/Role                                                                |
| ----------------------------- | --------------------------------------------------------------------------- |
| `orchestrator.py`             | Central workflow controller, manages episode lifecycle and agent actions    |
| `agents/ds_agent.py`          | (Stub) Data science agent, to be replaced with real planning/analysis logic |
| `exec_agent.py`               | Executes analysis plans, manages notebooks, integrates with LLMs            |
| `episode.py`                  | Episode model, status transitions, artifact and iteration tracking          |
| `analysis/analytics_tools.py` | Core analysis utilities: selection, aggregation, plotting, describing       |
| `analysis/segmentation.py`    | Group comparison and lift analysis                                          |
| `analysis/cohort.py`          | Retention and cohort analysis                                               |
| `analysis/funnel.py`          | Funnel conversion analysis                                                  |
| `data.py`                     | Data loading from CSV/SQL/Parquet, profiling                                |
| `memory.py`                   | Episodic and vector memory, document ingestion and retrieval                |
| `snapshot.py`                 | Artifact snapshotting and manifest management                               |
| `feedback.py`                 | Human feedback collection, classification, backlog updating                 |

---

## 10. How to Extend or Modify

- **To add a new analysis type**:  
  - Implement the logic in `analysis/` as a reusable function or class.
  - Update agent logic to call this function as part of the episode workflow.
  - Add reporting/visualization as needed.

- **To support new data sources**:  
  - Register the source in `config/data_sources.yaml`.
  - Ensure schema is cached in `schema_cache.yaml` for agent/LLM use.

- **To improve agent intelligence**:  
  - Replace the stub in `agents/ds_agent.py` with real planning, code generation, and review logic.
  - Integrate with LLMs or other automation as needed.

---

**For onboarding, practical usage, and extending the system, see:**
- [Data Source Onboarding](data_source_onboarding.md)
- [Analysis Workflow](analysis_workflow.md)
- [Extending Tools](extending_tools.md)
- [Agent Workflow](agent_workflow.md)

This living document is the single source of truth for how DS-Agent operates and should be referenced for all future development, review, and onboarding activities. 