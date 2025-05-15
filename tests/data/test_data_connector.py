import pandas as pd
import yaml
from pathlib import Path
from ds_agent.data import load_table, describe_table, profile_table


def test_load_table_csv(tmp_path, monkeypatch):
    # Create a temp CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    # Patch config to use this CSV
    config = {"sources": {"local_csv": {"path": str(csv_path), "format": "csv"}}}
    config_path = tmp_path / "data_sources.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    monkeypatch.setattr("ds_agent.data.CONFIG_PATH", config_path)
    out = load_table(table=None, source="local_csv")
    pd.testing.assert_frame_equal(out, df)


def test_describe_table(tmp_path, monkeypatch):
    # Create a temp schema cache
    cache = {
        "sources": {
            "local_csv": {
                "tables": {
                    "my_table": {
                        "table_desc": "desc",
                        "columns": {"a": {"dtype": "int", "col_desc": ""}},
                    }
                }
            }
        }
    }
    cache_path = tmp_path / "schema_cache.yaml"
    with open(cache_path, "w") as f:
        yaml.safe_dump(cache, f)
    monkeypatch.setattr("ds_agent.data.SCHEMA_CACHE_PATH", cache_path)
    out = describe_table("my_table", source="local_csv")
    assert out["table_desc"] == "desc"
    assert "a" in out["columns"]


def test_profile_table(tmp_path, monkeypatch):
    # Create a temp CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    # Patch config
    config = {"sources": {"local_csv": {"path": str(csv_path), "format": "csv"}}}
    config_path = tmp_path / "data_sources.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    monkeypatch.setattr("ds_agent.data.CONFIG_PATH", config_path)
    out = profile_table(table=None, source="local_csv")
    assert out["row_count"] == 3
    assert set(out["min"].keys()) == {"a", "b"}
    assert set(out["max"].keys()) == {"a", "b"}


def test_load_table_query_valid(monkeypatch, tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    config = {"sources": {"local_csv": {"path": str(csv_path), "format": "csv"}}}
    config_path = tmp_path / "data_sources.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    monkeypatch.setattr("ds_agent.data.CONFIG_PATH", config_path)
    out = load_table(table=None, source="local_csv", where="a == 2")
    assert out.shape[0] == 1
    assert out.iloc[0]["a"] == 2


def test_load_table_query_missing_column(monkeypatch, tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    config = {"sources": {"local_csv": {"path": str(csv_path), "format": "csv"}}}
    config_path = tmp_path / "data_sources.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    monkeypatch.setattr("ds_agent.data.CONFIG_PATH", config_path)
    try:
        load_table(table=None, source="local_csv", where="not_a_col == 1")
        assert False, "Should have raised ValueError for missing column"
    except ValueError as e:
        assert "missing columns" in str(e)
        assert "not_a_col" in str(e)


def test_load_table_query_whitespace_columns(monkeypatch, tmp_path):
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({" a ": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_path, index=False)
    config = {"sources": {"local_csv": {"path": str(csv_path), "format": "csv"}}}
    config_path = tmp_path / "data_sources.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    monkeypatch.setattr("ds_agent.data.CONFIG_PATH", config_path)
    out = load_table(table=None, source="local_csv", where="a == 1")
    assert out.shape[0] == 1
    assert out.iloc[0]["a"] == 1
