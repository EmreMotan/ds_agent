import pytest
import pandas as pd
import os
from pathlib import Path
import shutil
from ds_agent.analysis import analytics_tools as at


def setup_episode_dir(episode_id):
    ep_dir = Path(f"episodes/{episode_id}")
    ep_dir.mkdir(parents=True, exist_ok=True)
    ep_json = ep_dir / "episode.json"
    if not ep_json.exists():
        ep_json.write_text("{}")
    return ep_dir


def teardown_episode_dir(episode_id):
    ep_dir = Path(f"episodes/{episode_id}")
    if ep_dir.exists():
        shutil.rmtree(ep_dir)


def test_merge():
    df1 = pd.DataFrame({"id": [1, 2], "val1": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "val2": [100, 200]})
    # Basic inner join
    merged = at.merge(df1, df2, on="id")
    assert "val1" in merged.columns and "val2" in merged.columns
    # Left join with suffixes
    df3 = pd.DataFrame({"id": [2, 3], "val1": [30, 40]})
    merged2 = at.merge(df1, df3, on="id", how="left", suffixes=("_a", "_b"))
    assert "val1_a" in merged2.columns or "val1_b" in merged2.columns
    # Error: bad column
    with pytest.raises(ValueError):
        at.merge(df1, df2, on="bad_col")


def test_groupby_aggregate():
    df = pd.DataFrame({"g": ["a", "a", "b"], "x": [1, 2, 3], "y": [10, 20, 30]})
    # Single agg
    out = at.groupby_aggregate(df, ["g"], {"x": "sum"})
    assert "x_sum" in out.columns or "x" in out.columns
    # Multiple aggs
    out2 = at.groupby_aggregate(df, ["g"], {"x": ["sum", "mean"]})
    assert any("x_sum" in c or "x_mean" in c for c in out2.columns)
    # Note: Named aggregations (nested dicts) are not supported by pandas .agg()
    # Error: bad groupby col
    with pytest.raises(Exception):
        at.groupby_aggregate(df, ["badcol"], {"x": "sum"})


def test_select_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = at.select_columns(df, ["a"])
    assert list(out.columns) == ["a"]
    with pytest.raises(ValueError):
        at.select_columns(df, ["c"])


def test_rename_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = at.rename_columns(df, {"a": "x"})
    assert "x" in out.columns
    with pytest.raises(ValueError):
        at.rename_columns(df, {"c": "y"})


def test_assign_column():
    df = pd.DataFrame({"a": [1, 2]})
    out = at.assign_column(df, "b", "df['a'] * 2")
    assert all(out["b"] == df["a"] * 2)
    # Error: bad expr
    with pytest.raises(ValueError):
        at.assign_column(df, "c", "df['notacol'] * 2")


def test_describe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    desc = at.describe(df)
    assert "a" in desc.columns and "b" in desc.columns
    # Empty DataFrame should raise
    with pytest.raises(ValueError):
        at.describe(pd.DataFrame())


def test_value_counts():
    df = pd.DataFrame({"a": ["x", "y", "x"]})
    vc = at.value_counts(df, "a")
    assert set(vc["a"]) == {"x", "y"}
    # Missing column
    with pytest.raises(ValueError):
        at.value_counts(df, "b")
    # Empty DataFrame
    with pytest.raises(ValueError):
        at.value_counts(pd.DataFrame(), "a")


def test_plot_time_series():
    episode_id = "ep1"
    setup_episode_dir(episode_id)
    try:
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "y": [1, 2, 3]})
        # Should not raise
        at.plot_time_series(df, "date", "y", "Test Time Series", episode_id)
        # Missing column
        with pytest.raises(ValueError):
            at.plot_time_series(df, "bad", "y", "Test", episode_id)
        # Empty DataFrame
        with pytest.raises(ValueError):
            at.plot_time_series(pd.DataFrame(), "date", "y", "Test", episode_id)
    finally:
        teardown_episode_dir(episode_id)


def test_plot_bar():
    episode_id = "ep1"
    setup_episode_dir(episode_id)
    try:
        df = pd.DataFrame({"cat": ["a", "b"], "val": [1, 2]})
        at.plot_bar(df, "cat", "val", "Test Bar", episode_id)
        with pytest.raises(ValueError):
            at.plot_bar(df, "bad", "val", "Test", episode_id)
        with pytest.raises(ValueError):
            at.plot_bar(pd.DataFrame(), "cat", "val", "Test", episode_id)
    finally:
        teardown_episode_dir(episode_id)


def test_plot_scatter():
    episode_id = "ep1"
    setup_episode_dir(episode_id)
    try:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        at.plot_scatter(df, "x", "y", "Test Scatter", episode_id)
        with pytest.raises(ValueError):
            at.plot_scatter(df, "bad", "y", "Test", episode_id)
        with pytest.raises(ValueError):
            at.plot_scatter(pd.DataFrame(), "x", "y", "Test", episode_id)
    finally:
        teardown_episode_dir(episode_id)
