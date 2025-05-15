import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from ds_agent.analysis.visuals import save_fig


def funnel_conversion(
    df: pd.DataFrame, step_cols: List[str], episode_id: str = None
) -> Tuple[pd.DataFrame, plt.Figure, str]:
    """
    Calculates step-to-step conversion rates and bar plot.
    Returns DataFrame, matplotlib Figure, and markdown summary.
    """
    # Input validation
    if not step_cols or len(step_cols) < 2:
        raise ValueError("At least two step columns required")
    for col in step_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    # Compute counts at each step
    counts = [df[col].notnull().sum() for col in step_cols]
    rates = [100.0]
    for i in range(1, len(counts)):
        rate = counts[i] / counts[i - 1] * 100 if counts[i - 1] > 0 else 0.0
        rates.append(rate)
    funnel_df = pd.DataFrame(
        {
            "step": step_cols,
            "count": counts,
            "conversion_%": rates,
        }
    )
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(step_cols, rates, color="skyblue")
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Funnel Conversion Rates")
    ax.set_ylim(0, 100)
    for i, v in enumerate(rates):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center")
    plt.tight_layout()
    # Save figure if episode_id is provided
    fig_path = None
    if episode_id:
        fig_path = save_fig(fig, episode_id, "funnel_conversion")
    # Markdown summary
    summary = f"""**Funnel Conversion**\n\n- Steps: {', '.join(step_cols)}\n"""
    for i, step in enumerate(step_cols):
        summary += f"- {step}: {counts[i]} users, {rates[i]:.1f}%\n"
    if fig_path:
        summary += f"\n![Funnel Conversion]({fig_path})"
    return funnel_df, fig, summary
