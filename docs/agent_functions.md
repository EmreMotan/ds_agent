# DS-Agent Function Reference

This document provides a comprehensive list of all functions available to the DS-Agent system, organized by category. Each function is documented with its purpose, parameters, and example usage.

## Data Loading & Profiling

### `load_table`
Loads data from CSV/SQL/Parquet as configured in `data_sources.yaml`.
```python
load_table(table: str, cols: list[str] = None, where: str = None, source: str = None) -> pd.DataFrame
```

### `describe_table`
Returns JSON metadata from schema cache.
```python
describe_table(table: str, source: str = None) -> dict
```

### `profile_table`
Performs basic profiling (row count, null %, min/max).
```python
profile_table(table: str, source: str = None) -> dict
```

### `validate_table`
Validates table for data quality issues.
```python
validate_table(table: str, source: str = None) -> dict
```

## Data Manipulation

### `select_columns`
Selects specific columns from a DataFrame.
```python
select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame
```

### `rename_columns`
Renames columns in a DataFrame.
```python
rename_columns(df: pd.DataFrame, rename_dict: dict) -> pd.DataFrame
```

### `filter_rows`
Filters rows based on a condition.
```python
filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame
```

### `assign_column`
Creates a new column using a pandas expression.
```python
assign_column(df: pd.DataFrame, column: str, expr: str) -> pd.DataFrame
```

### `merge`
Joins two DataFrames on specified keys.
```python
merge(df1: pd.DataFrame, df2: pd.DataFrame, on: list[str] = None, how: str = 'inner', 
      left_on: list[str] = None, right_on: list[str] = None, suffixes: tuple = ('_x', '_y'),
      validate: str = None, indicator: bool = False) -> pd.DataFrame
```

### `concat`
Stacks DataFrames vertically.
```python
concat(dfs: list[pd.DataFrame], axis: int = 0, ignore_index: bool = True) -> pd.DataFrame
```

### `robust_merge`
Performs a robust merge operation on multiple DataFrames.
```python
robust_merge(dfs: list[pd.DataFrame], join_keys: list[str], how: str = 'inner',
             suffixes: tuple = ('_x', '_y'), validate: str = None, indicator: bool = False) -> pd.DataFrame
```

## Aggregation & Analysis

### `groupby_aggregate`
Performs groupby operations with aggregation.
```python
groupby_aggregate(df: pd.DataFrame, groupby_cols: list[str], agg_dict: dict,
                 dropna: bool = True, as_index: bool = True) -> pd.DataFrame
```

### `sort_values`
Sorts DataFrame by specified columns.
```python
sort_values(df: pd.DataFrame, by: list[str], ascending: list[bool] = None) -> pd.DataFrame
```

### `top_n`
Gets top N rows for each group.
```python
top_n(df: pd.DataFrame, groupby_cols: list[str], sort_col: str, n: int) -> pd.DataFrame
```

### `pivot_table`
Creates a pivot table.
```python
pivot_table(df: pd.DataFrame, index: list[str], columns: list[str], values: str,
            aggfunc: str = 'mean') -> pd.DataFrame
```

### `value_counts`
Counts unique values in a column.
```python
value_counts(df: pd.DataFrame, column: str) -> pd.Series
```

## Statistical Analysis

### `correlation`
Computes correlation between two columns.
```python
correlation(df: pd.DataFrame, col1: str, col2: str, method: str = 'pearson') -> float
```

### `regression`
Performs regression analysis.
```python
regression(df: pd.DataFrame, y: str, X: list[str], model_type: str = 'linear',
           params: dict = None) -> dict
```

### `t_test`
Performs t-test between two groups.
```python
t_test(df: pd.DataFrame, group_col: str, value_col: str, group1: str, group2: str) -> dict
```

### `chi_square`
Performs chi-square test.
```python
chi_square(df: pd.DataFrame, col1: str, col2: str) -> dict
```

### `anova`
Performs ANOVA test.
```python
anova(df: pd.DataFrame, group_col: str, value_col: str) -> dict
```

### `bootstrap_mean_diff`
Computes bootstrap confidence interval for mean difference.
```python
bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, ci: float = 0.95,
                    two_tailed: bool = True) -> dict
```

## Advanced Analytics

### `cohort_retention`
Analyzes user retention by cohort.
```python
cohort_retention(signup_date: str, event_date: str, user_id: str) -> tuple[pd.DataFrame, plt.Figure, str]
```

### `segmentation_lift`
Computes lift analysis for segments.
```python
segmentation_lift(df: pd.DataFrame, metric_col: str, dim_col: str) -> tuple[pd.DataFrame, str]
```

### `funnel_conversion`
Analyzes step-to-step conversion in a funnel.
```python
funnel_conversion(step_cols: list[str]) -> tuple[pd.DataFrame, plt.Figure, str]
```

### `custom_metric`
Computes a custom metric using a pandas expression.
```python
custom_metric(df: pd.DataFrame, metric_name: str, expr: str) -> float
```

## Visualization

### `plot_time_series`
Creates a time series plot.
```python
plot_time_series(df: pd.DataFrame, x: str, y: str, title: str, episode_id: str,
                 hue: str = None) -> plt.Figure
```

### `plot_bar`
Creates a bar plot.
```python
plot_bar(df: pd.DataFrame, x: str, y: str = None, title: str, episode_id: str) -> plt.Figure
```

### `plot_scatter`
Creates a scatter plot.
```python
plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, episode_id: str) -> plt.Figure
```

## Event & Time Analysis

### `event_filter`
Filters events based on event values.
```python
event_filter(df: pd.DataFrame, event_col: str, event_values: list[str]) -> pd.DataFrame
```

### `time_window_filter`
Filters data within a time window.
```python
time_window_filter(df: pd.DataFrame, time_col: str, start: str, end: str) -> pd.DataFrame
```

## Reporting & Output

### `generate_report`
Generates a comprehensive analysis report.
```python
generate_report(goal: str, top_n_table: pd.DataFrame = None, value_counts: pd.Series = None,
                describe_stats: dict = None, segmentation: pd.DataFrame = None,
                episode_id: str = None) -> str
```

### `set_sanity_passed`
Marks analysis as passing sanity checks.
```python
set_sanity_passed() -> None
```

## Usage Notes

1. All functions that return DataFrames preserve the original index unless specified otherwise.
2. Functions that create new columns (e.g., `assign_column`) must be used before referencing those columns in other functions.
3. When using `groupby_aggregate`, output columns are flattened (e.g., `win_sum`, `win_count`).
4. All string values in expressions must be properly quoted.
5. For binary columns (e.g., `is_home`), use integer values (1/0) rather than strings.

## Example Workflow

```python
# Load and prepare data
df = load_table('game_summary', cols=['date', 'team', 'points', 'is_home'])

# Create derived column
df = assign_column(df, 'home_points', 'df[df["is_home"] == 1]["points"]')

# Analyze
result = groupby_aggregate(df, ['team'], {'points': ['mean', 'count']})
result = assign_column(result, 'avg_points', 'df["points_mean"]')

# Visualize
fig = plot_bar(result, x='team', y='avg_points', title='Average Points by Team')
```

## References

- [Analysis Library Documentation](analysis_library.md)
- [System Overview](system_overview.md)
- [Data Source Onboarding](data_source_onboarding.md) 