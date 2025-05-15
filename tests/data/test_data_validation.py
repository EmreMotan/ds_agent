import pytest
import pandas as pd
import numpy as np
from ds_agent.data import validate_table


@pytest.fixture
def tmp_csv(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2, 2, np.nan, np.nan, np.nan],
            "b": [10, 20, 20, 30, 40, 10000],
            "c": ["x", "y", "y", "z", "z", "z"],
        }
    )
    csv_path = tmp_path / "val.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def patch_data_sources(tmp_csv, monkeypatch):
    import yaml
    from pathlib import Path

    config_path = Path("config/data_sources.yaml")
    config_bak = None
    if config_path.exists():
        config_bak = config_path.read_text()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
sources:
  val:
    path: {tmp_csv}
    format: csv
    description: test csv
"""
    )
    yield
    if config_bak is not None:
        config_path.write_text(config_bak)
    else:
        config_path.unlink(missing_ok=True)


def test_validate_table(patch_data_sources):
    res = validate_table("val", source="val")
    # Nulls
    assert "a" in res["null_pct"] and res["null_pct"]["a"] > 0.4
    # Duplicates
    assert res["n_duplicates"] > 0
    # Types
    assert res["dtypes"]["a"] == "float64"
    # Outliers
    assert "b" in res["outliers"]
    # Issues summary
    issues = res["issues"]
    assert any("null" in s.lower() for s in issues)
    assert any("duplicate" in s.lower() for s in issues)
    assert any("outlier" in s.lower() for s in issues)
