import pandas as pd
from typing import Tuple
from scipy import stats
import numpy as np


def lift_table(df: pd.DataFrame, metric_col: str, dim_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Computes lift vs. baseline and p-values for each group in dim_col.
    Returns a DataFrame and a markdown summary.
    """
    # Input validation
    for col in [metric_col, dim_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    # Compute group means
    group_means = df.groupby(dim_col)[metric_col].mean()
    baseline = group_means.iloc[0]
    lift = (group_means - baseline) / baseline * 100
    # Compute p-values vs. baseline
    pvals = {}
    base_group = df[df[dim_col] == group_means.index[0]][metric_col]
    for group in group_means.index:
        if group == group_means.index[0]:
            pvals[group] = np.nan
        else:
            test_group = df[df[dim_col] == group][metric_col]
            _, p = stats.ttest_ind(base_group, test_group, equal_var=False)
            pvals[group] = p
    # Build result DataFrame
    result = pd.DataFrame(
        {
            "mean": group_means,
            "lift_vs_baseline_%": lift,
            "p_value": pd.Series(pvals),
        }
    )
    # Markdown summary
    summary = f"""**Segmentation Lift Table**\n\n- Baseline group: {group_means.index[0]}\n- Groups: {len(group_means)}\n"""
    for group in group_means.index:
        if group == group_means.index[0]:
            continue
        summary += f"- {group}: lift={lift[group]:.1f}%, p={pvals[group]:.3g}\n"
    return result, summary
