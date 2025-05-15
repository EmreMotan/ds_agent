import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from ds_agent.analysis.visuals import save_fig
import numpy as np


def run_retention(
    df: pd.DataFrame,
    signup_date: str,
    event_date: str,
    user_id: str,
    period: str = "W",
    episode_id: str = None,
) -> Tuple[pd.DataFrame, plt.Figure, str]:
    """
    Compute retention matrix (pivot), plot heatmap, and return markdown summary.
    Args:
        df: DataFrame with user events
        signup_date: column name for user signup date
        event_date: column name for event date
        user_id: column name for user id
        period: period for cohorting (default 'W' for week)
        episode_id: optional, for artifact saving
    Returns:
        (pivot_df, heatmap_fig, markdown_summary)
    """
    # Input validation
    for col in [signup_date, event_date, user_id]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    df = df.copy()
    df[signup_date] = pd.to_datetime(df[signup_date])
    df[event_date] = pd.to_datetime(df[event_date])
    # Assign cohort (period of signup)
    df["cohort"] = df[signup_date].dt.to_period(period).dt.start_time
    periods = (df[event_date] - df[signup_date]).dt.days // 7
    df["period_number"] = periods.astype("Int64")
    # Pivot: index=cohort, columns=period_number, values=unique users
    cohort_pivot = (
        df.groupby(["cohort", "period_number"])[user_id]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    # Normalize by cohort size
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0)
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(retention, aspect="auto", cmap="Blues")
    ax.set_title("User Retention Heatmap")
    ax.set_xlabel("Weeks Since Signup")
    ax.set_ylabel("Cohort (Signup Week)")
    ax.set_xticks(np.arange(retention.shape[1]))
    ax.set_xticklabels(retention.columns)
    ax.set_yticks(np.arange(retention.shape[0]))
    ax.set_yticklabels([str(idx.date()) for idx in retention.index])
    plt.colorbar(im, ax=ax, label="Retention Rate")
    plt.tight_layout()
    # Save figure if episode_id is provided
    fig_path = None
    if episode_id:
        fig_path = save_fig(fig, episode_id, "retention_heatmap")
    # Markdown summary
    first_week = retention.iloc[:, 0].mean() * 100
    second_week = retention.iloc[:, 1].mean() * 100 if retention.shape[1] > 1 else None
    summary = f"""**Retention Analysis**\n\n- Average week 1 retention: {first_week:.1f}%\n"""
    if second_week is not None:
        summary += f"- Average week 2 retention: {second_week:.1f}%\n"
    summary += f"- Cohorts analyzed: {len(retention)}\n"
    if fig_path:
        summary += f"\n![Retention Heatmap]({fig_path})"
    return retention, fig, summary
