import pandas as pd
import numpy as np
from ds_agent.analysis.cohort import run_retention
import matplotlib.figure


def test_run_retention_basic(tmp_path):
    # Create synthetic event data
    n_users = 100
    np.random.seed(42)
    signup_dates = pd.date_range("2024-01-01", periods=10, freq="7D")
    data = []
    for i in range(n_users):
        signup = np.random.choice(signup_dates)
        user_id = f"user_{i}"
        # Simulate up to 3 weeks of activity
        for week in range(np.random.randint(1, 4)):
            event_date = signup + pd.Timedelta(days=7 * week)
            data.append(
                {
                    "user_id": user_id,
                    "signup_date": signup,
                    "event_date": event_date,
                }
            )
    df = pd.DataFrame(data)
    # Run retention
    pivot, fig, summary = run_retention(df, "signup_date", "event_date", "user_id", period="W")
    # Check pivot shape and values
    assert isinstance(pivot, pd.DataFrame)
    assert pivot.shape[0] > 0 and pivot.shape[1] > 0
    assert (pivot.values >= 0).all()
    # Check figure
    assert isinstance(fig, matplotlib.figure.Figure)
    # Check summary
    assert "Retention Analysis" in summary
    assert "week 1 retention" in summary
