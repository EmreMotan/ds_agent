import pandas as pd
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path
import os
from sqlalchemy import create_engine, text
import warnings
import numpy as np
import re
import keyword

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "data_sources.yaml"
SCHEMA_CACHE_PATH = Path(__file__).parent.parent.parent / "schema_cache.yaml"


def load_table(
    table: str,
    cols: Optional[List[str]] = None,
    where: Optional[str] = None,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a table from a registered data source (SQL, CSV, Parquet).

    Args:
        table: For SQL, the table name. For file-based sources, the source name as registered in config.
        cols: List of columns to select (optional).
        where: For SQL, a SQL WHERE clause (e.g., "season = 2020"). For file-based, a pandas query string (e.g., "season == 2020").
        source: The source name as registered in config. Required if multiple sources are present.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        ValueError: If source/table is missing, file is not found, or columns/query are invalid.
            - If a column referenced in 'where' does not exist, a ValueError is raised listing the missing and available columns.

    Notes:
        - For SQL sources, 'where' should be a valid SQL WHERE clause (do not include the 'WHERE' keyword).
        - For file-based sources, 'where' should be a valid pandas query string.
        - For large files, consider specifying 'cols' and/or 'where' for performance.
        - All column names are stripped of leading/trailing whitespace after loading.
        - After loading, the columns are logged for debugging.
        - Before applying a pandas query, all referenced columns are checked for existence and a clear error is raised if any are missing.
    """
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    sources = config["sources"]

    # Require explicit source if multiple sources
    if source is None:
        if len(sources) == 1:
            src_name = next(iter(sources))
        else:
            raise ValueError("Multiple sources registered. Please specify the 'source' argument.")
    else:
        src_name = source
    if src_name not in sources:
        raise ValueError(f"Source '{src_name}' not found in config.")
    src = sources[src_name]

    # SQL source
    if "url" in src:
        url = os.path.expandvars(src["url"])
        engine = create_engine(url)
        cols_str = ", ".join(cols) if cols else "*"
        query = f"SELECT {cols_str} FROM {table}"
        if where:
            query += f" WHERE {where}"
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
        except Exception as e:
            raise ValueError(f"SQL query failed: {e}\nQuery: {query}")
        return df

    # File-based source (CSV/Parquet)
    file_format = src.get("format")
    file_path = Path(src.get("path", ""))
    if not file_path.exists():
        raise ValueError(f"File not found for source '{src_name}': {file_path}")

    # Warn if file is very large and no filtering is applied
    try:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 500 and not cols and not where:
            warnings.warn(
                f"Loading large file ({file_size_mb:.1f} MB) without column selection or filtering may be slow."
            )
    except Exception:
        pass  # Ignore file size errors

    # Load DataFrame
    try:
        if file_format == "csv":
            read_args = {"filepath_or_buffer": file_path}
            if cols:
                read_args["usecols"] = cols
            df = pd.read_csv(**read_args)
        elif file_format == "parquet":
            read_args = {"path": file_path}
            if cols:
                read_args["columns"] = cols
            df = pd.read_parquet(**read_args)
        else:
            raise ValueError(f"Unknown or unsupported file format: {file_format}")
    except Exception as e:
        raise ValueError(f"Failed to load file '{file_path}': {e}")

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Log columns after loading
    print(f"[load_table] Columns loaded: {list(df.columns)}")

    # Validate columns if cols specified (for CSV, usecols already does this)
    if cols and file_format == "parquet":
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in file: {missing}")

    # Apply pandas query if where specified
    if where:
        # Remove quoted strings (single or double quotes) before extracting column names
        where_no_strings = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", "", where)
        referenced_cols = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", where_no_strings))
        # Filter out Python keywords and logical operators
        logical_operators = {"and", "or", "not", "in", "is", "True", "False", "None"}
        referenced_cols = {
            col
            for col in referenced_cols
            if not keyword.iskeyword(col) and col not in logical_operators
        }
        missing_cols = [col for col in referenced_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Query references missing columns: {missing_cols}\nAvailable columns: {list(df.columns)}\nQuery: {where}"
            )
        try:
            df = df.query(where)
        except Exception as e:
            raise ValueError(f"Failed to apply pandas query: {e}\nQuery: {where}")

    return df


def describe_table(table: str, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Return schema and metadata for a table from the schema cache.
    """
    with open(SCHEMA_CACHE_PATH, "r") as f:
        cache = yaml.safe_load(f)
    sources = cache.get("sources", {})
    src_name = source or next(iter(sources))
    src = sources.get(src_name)
    if not src:
        raise ValueError(f"Source '{src_name}' not found in schema cache.")
    tables = src.get("tables", {})
    tbl = tables.get(table)
    if not tbl:
        raise ValueError(f"Table '{table}' not found in source '{src_name}'.")
    return tbl


def profile_table(table: str, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Return basic profile (row count, null %, min/max) for a table.
    """
    df = load_table(table, source=source)
    profile = {"row_count": len(df)}
    null_pct = df.isnull().mean().to_dict()
    profile["null_pct"] = {k: float(f"{v:.3f}") for k, v in null_pct.items()}
    # Min/max for numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns
    profile["min"] = {col: float(df[col].min()) for col in num_cols}
    profile["max"] = {col: float(df[col].max()) for col in num_cols}
    return profile


def validate_table(table: str, source: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate a table for data quality issues.
    Checks:
      - Null percentage per column
      - Duplicate rows
      - Numeric outliers (IQR method)
      - Basic type info
    Args:
        table: Table name (as registered in config).
        source: Optional source name.
    Returns:
        Dict with validation results and issues.
    """
    df = load_table(table, source=source)
    results = {}
    # Nulls
    null_pct = df.isnull().mean().to_dict()
    results["null_pct"] = {k: float(f"{v:.3f}") for k, v in null_pct.items()}
    # Duplicates
    n_dupes = df.duplicated().sum()
    results["n_duplicates"] = int(n_dupes)
    # Types
    results["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    # Outliers (IQR method for numeric columns)
    outlier_cols = {}
    for col in df.select_dtypes(include=["number"]).columns:
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_out = int(((vals < lower) | (vals > upper)).sum())
        if n_out > 0:
            outlier_cols[col] = n_out
    results["outliers"] = outlier_cols
    # Issues summary
    issues = []
    for col, pct in results["null_pct"].items():
        if pct >= 0.5:
            issues.append(f"Column '{col}' has >=50% nulls.")
    if n_dupes > 0:
        issues.append(f"Table has {n_dupes} duplicate rows.")
    for col, n_out in outlier_cols.items():
        issues.append(f"Column '{col}' has {n_out} outliers (IQR method).")
    results["issues"] = issues
    return results
