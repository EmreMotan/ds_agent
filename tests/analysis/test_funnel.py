import pandas as pd
from ds_agent.analysis.funnel import funnel_conversion
import matplotlib.figure


def test_funnel_conversion_basic():
    # Create synthetic funnel data
    df = pd.DataFrame(
        {
            "step1": [1, 1, 1, 1, 1, None, None, None, None, None],
            "step2": [1, 1, 1, None, None, None, None, None, None, None],
            "step3": [1, 1, None, None, None, None, None, None, None, None],
        }
    )
    funnel_df, fig, summary = funnel_conversion(df, ["step1", "step2", "step3"])
    # Check DataFrame
    assert isinstance(funnel_df, pd.DataFrame)
    assert set(["step", "count", "conversion_%"]).issubset(funnel_df.columns)
    assert funnel_df.shape[0] == 3
    # Check figure
    assert isinstance(fig, matplotlib.figure.Figure)
    # Check summary
    assert "Funnel Conversion" in summary
    assert "step1" in summary and "step2" in summary and "step3" in summary
