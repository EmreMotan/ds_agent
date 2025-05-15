# Analysis Workflow Guide

This guide explains how to run new analyses using DS-Agent, from specifying a goal to interpreting results.

> **Tip:** Before running analyses, validate your data for quality issues using `validate_table`. See [data_source_onboarding.md](data_source_onboarding.md) for details.

---

## 1. Specify Your Analysis Goal

- Define what you want to analyze (e.g., "Summarize the columns in my_csv").
- Register your data source (see [data_source_onboarding.md](data_source_onboarding.md)).

---

## 2. Create an Episode

- Use the CLI or Python to create a new episode:

```python
from ds_agent.episode import create_episode

episode = create_episode("Describe my_csv", "Get summary stats for my_csv")
# Save episode to episodes/<episode_id>/episode.json
```

---

## 3. Plan the Analysis

- Use the agent to plan the workflow:

```python
from ds_agent.agents.ds_agent import DataScienceAgent
agent = DataScienceAgent(<episode_dir>)
agent.plan_analysis(episode)
```

- This creates `plan.json` in the episode directory.
- Edit `plan.json` to fill in your table/source (e.g., replace `<TABLE>` with `my_csv`).

---

## 4. Run the Analysis

- Use the agent to execute the plan:

```python
agent.run_analysis(episode)
```

- Progress and results are logged in `history.log`.
- Outputs (plots, tables) are saved in the episode directory.

---

## 5. Interpret Results

- Check `history.log` for step-by-step execution and any errors.
- Review outputs in the episode directory (e.g., summary tables, plots).
- Use the `describe_table` or `profile_table` tools for further exploration.

---

## 6. Next Steps

- See [agent_workflow.md](agent_workflow.md) for details on agent planning and execution.
- See [extending_tools.md](extending_tools.md) to add new analysis tools.

## Schema Validation and Error Handling

- When using `load_table`, all column names are stripped of whitespace after loading.
- The columns are logged for debugging.
- Before applying a pandas query (the `where` argument), all referenced columns are checked for existence in the DataFrame.
- If any referenced column is missing, a clear error is raised listing the missing and available columns, along with the offending query.
- This ensures that agent/LLM workflows fail fast and transparently, making debugging and plan correction much easier. 