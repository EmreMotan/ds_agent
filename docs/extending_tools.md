# Extending DS-Agent with New Tools

This guide explains how to add and register new analysis tools for agent/LLM use in DS-Agent workflows.

---

## 1. Implement Your Utility

- Write your function in the appropriate module (e.g., `src/ds_agent/analysis/analytics_tools.py`).
- Use clear type hints, a detailed docstring, and robust error handling.

Example:
```python
def my_tool(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Example tool: Return rows where col > 0.
    Args:
        df: Input DataFrame.
        col: Column to filter.
    Returns:
        pd.DataFrame: Filtered DataFrame.
    Raises:
        ValueError: If col is missing.
    """
    if col not in df.columns:
        raise ValueError(f"Column {col} not found.")
    return df[df[col] > 0]
```

---

## 2. Register the Tool

- Add an entry to `tools.yaml`:

```yaml
my_tool:
  fn: ds_agent.analysis.analytics_tools.my_tool
  args: [df, col]
```

- The `fn` path should be importable from your codebase.
- The `args` list should match the function signature.

---

## 3. Write Tests

- Add or update a test file (e.g., `tests/analysis/test_analytics_tools.py`).
- Test normal, edge, and error cases using pytest.

---

## 4. Make It Agent/LLM Friendly

- Use clear, descriptive docstrings (these are shown to the agent/LLM).
- Validate input and provide helpful error messages.
- Keep function signatures simple and explicit.

---

## 5. Next Steps

- See [system_overview.md](system_overview.md) for architecture and workflow context.
- See [analysis_workflow.md](analysis_workflow.md) for how tools are used in agent-driven analysis. 