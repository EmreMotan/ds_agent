# Agent Workflow Guide

This guide explains how the DS-Agent plans and executes analysis workflows, and how to debug or extend agent behavior.

---

## 1. Agent Planning & Execution Overview

- The agent plans analysis steps based on the episode goal and schema.
- The plan is saved as `plan.json` in the episode directory.
- Each step specifies a tool and arguments (see `tools.yaml`).
- The agent executes each step in order, chaining results as needed.
- Progress and results are logged in `history.log`.

---

## 2. Debugging Agent Workflows

- Check `history.log` for step-by-step execution, errors, and results.
- If a step fails, the error and arguments are logged.
- You can manually edit `plan.json` to fix or rerun steps.

---

## 3. Customizing or Extending Agent Planning

- The current agent uses a simple rule-based planner (see `plan_analysis` in `ds_agent/agents/ds_agent.py`).
- To add smarter planning (e.g., LLM-based), extend or override `plan_analysis`.
- You can add new planning rules, parse goals, or use schema context.

---

## 4. References

- See [analysis_workflow.md](analysis_workflow.md) for how to run analyses.
- See [system_overview.md](system_overview.md) for architecture and workflow context. 