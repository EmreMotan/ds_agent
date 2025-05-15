# DS-Agent Analysis Library

## Overview

The analysis library provides a set of tested, reusable Python functions for common data science tasks. These primitives are designed for use by agents and humans in notebooks, pipelines, and automated workflows. Each function is unit-tested, returns both numeric results and visualizations, and supports artifact saving and registration.

## Installation & Requirements

- Python 3.10+
- numpy
- pandas
- matplotlib
- scipy

Install requirements:
```bash
pip install numpy pandas matplotlib scipy
```

## Module & Function Catalog

### 1. Cohort Analysis (`cohort.py`)

```python
def run_retention(df: pd.DataFrame, signup_date: str, event_date: str, user_id: str, period: str = "W", episode_id: str = None) -> tuple[pd.DataFrame, plt.Figure, str]:
    """Return retention pivot, heatmap figure, markdown summary."""
```

**Example:**
```python
pivot, fig, summary = run_retention(df, "signup_date", "event_date", "user_id", period="W", episode_id="DS-25-001")
print(summary)
fig.show()
```

### 2. Segmentation (`segmentation.py`)

```python
def lift_table(df: pd.DataFrame, metric_col: str, dim_col: str) -> tuple[pd.DataFrame, str]:
    """Compute group means, lift vs. baseline, p-values; returns DataFrame + markdown."""
```

**Example:**
```python
result, summary = lift_table(df, "metric", "group")
print(result)
print(summary)
```

### 3. Funnel Analysis (`funnel.py`)

```python
def funnel_conversion(df: pd.DataFrame, step_cols: list[str], episode_id: str = None) -> tuple[pd.DataFrame, plt.Figure, str]:
    """Compute step-to-step conversion, bar plot, markdown summary."""
```

**Example:**
```python
funnel_df, fig, summary = funnel_conversion(df, ["step1", "step2", "step3"], episode_id="DS-25-001")
print(funnel_df)
print(summary)
fig.show()
```

### 4. Statistical Tests (`stat_tests.py`)

```python
def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, ci: float = 0.95, two_tailed: bool = True) -> dict:
    """Bootstrap mean diff CI + p; returns dict."""
```

**Example:**
```python
result = bootstrap_mean_diff(a, b)
print(result)
```

### 5. Visuals & Artifact Saving (`visuals.py`)

```python
def save_fig(fig: plt.Figure, episode_id: str, name: str) -> str:
    """Save PNG to episode outputs, register artifact, return relative path."""
```

- Saves the figure as a PNG in `episodes/<episode_id>/outputs/<hash>.png`.
- Computes SHA-256 and updates the episode manifest.
- Returns the relative path for markdown embedding.

**Example:**
```python
fig_path = save_fig(fig, "DS-25-001", "my_plot")
print(f"Figure saved at: {fig_path}")
```

## Integration Example

```python
import pandas as pd
from ds_agent.analysis.cohort import run_retention
from ds_agent.analysis.funnel import funnel_conversion

# Load your data...
# df = pd.read_csv(...)

pivot, fig, summary = run_retention(df, "signup_date", "event_date", "user_id", episode_id="DS-25-001")
print(summary)
fig.show()
```

## Artifact Saving & Registration

- All figures can be saved and registered using `visuals.save_fig`.
- Artifacts are tracked in `episode.json` and can be referenced in markdown summaries.

## Testing & Coverage

Run all tests (≥95% coverage expected):
```bash
pytest --cov=src/ds_agent/analysis
```

## References
- [PRD_007_analysis_library.md](PRD_007_analysis_library.md)
- [PRD_004_notebook_template.md](PRD_004_notebook_template.md)
- [PRD_006_version_control_hooks.md](PRD_006_version_control_hooks.md)
- [DS-Agent Architecture Overview](architecture_overview.md) 