# PRD‑008: Analysis‑Executor Agent & Tool Registry  
*Version 0.1 • May 3 2025*

---

## 1 | Overview  
Add an autonomous **Execution Agent** that converts high‑level analysis goals into runnable notebook code by orchestrating the Analysis Function Library (PRD‑007).

Components:

1. **Tool Registry** – static YAML mapping intents → callable functions.  
2. **Notebook Editor** – programmatically edits the analysis notebook.  
3. **Kernel Runner** – executes via Papermill, inspects globals.  
4. **Iteration Loop** – LLM plans, calls tools, re‑runs up to `max_iter`.

---

## 2 | Problem / Goal  
Enable the agent to:
* Write code, not prose, to solve data questions.  
* Iterate on results without human intervention.  
* Produce a notebook that passes sanity tests and is ready for review.

---

## 3 | Scope (MVP)

### 3.1 Tool Registry  
`src/ds_agent/tools.py` loads `tools.yaml`:

```yaml
cohort_retention:
  fn: ds_agent.analysis.cohort.run_retention
  args: [signup_date, event_date, user_id]
segmentation_lift:
  fn: ds_agent.analysis.segmentation.lift_table
  args: [metric_col, dim_col]
```

Resolver returns callable + param schema.

### 3.2 Execution Agent  
`src/ds_agent/exec_agent.py`

* Methods: `plan()`, `apply_plan()`, `run_episode(episode_id)`.  
* Uses GPT‑4o (env `OPENAI_MODEL`).  
* Stops when `notebook.globals['sanity_passed']` is `True` or `max_iter` hit.

### 3.3 Notebook Editor  
`src/ds_agent/notebook_editor.py`  
Helpers: `add_code`, `add_markdown`, `tag_cell`.

### 3.4 Orchestrator integration  
When Episode is `IN_ANALYSIS` or `REVISION`, call `ExecAgent.run_episode()`.

---

## 4 | Success Criteria  

| Metric                                   | Target |
| ---------------------------------------- | ------ |
| Notebook passes sanity tests first cycle | 100 %  |
| Iterations per cycle                     | ≤ 3    |
| Coverage (non‑LLM code)                  | ≥ 90 % |
| Avg token usage                          | ≤ 20 k |

---

## 5 | Deliverables  

| Path                                  | Artifact              |
| ------------------------------------- | --------------------- |
| `src/ds_agent/tools.py`, `tools.yaml` | Registry              |
| `src/ds_agent/exec_agent.py`          | Agent                 |
| `src/ds_agent/notebook_editor.py`     | Notebook utils        |
| `tests/exec_agent/*.py`               | Unit tests (mock LLM) |
| `docs/exec_agent_flow.mmd`            | Sequence diagram      |

---

## 6 | Acceptance Checklist  

- [ ] Agent calls at least two library functions in pilot Episode.  
- [ ] Notebook sets `sanity_passed=True`.  
- [ ] Iteration cap enforced.  
- [ ] Coverage ≥ 90 %.  
- [ ] README updated with ExecAgent usage.

---

## 7 | Dependencies / References  
* PRD‑004 Notebook engine  
* PRD‑007 Analysis library  
* OPENAI_MODEL default `gpt-4o`.

---

*End of PRD*