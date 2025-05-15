import pandas as pd
import numpy as np
from ds_agent.analysis.segmentation import lift_table


def test_lift_table_basic():
    # Create synthetic data
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "group": np.repeat(["A", "B", "C"], 50),
            "metric": np.concatenate(
                [
                    np.random.normal(10, 2, 50),
                    np.random.normal(12, 2, 50),
                    np.random.normal(15, 2, 50),
                ]
            ),
        }
    )
    result, summary = lift_table(df, "metric", "group")
    # Check DataFrame
    assert isinstance(result, pd.DataFrame)
    assert set(["mean", "lift_vs_baseline_%", "p_value"]).issubset(result.columns)
    assert result.shape[0] == 3
    # Check summary
    assert "Segmentation Lift Table" in summary
    assert "lift=" in summary
    assert "p=" in summary
