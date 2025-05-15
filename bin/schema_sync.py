#!/usr/bin/env python3
import yaml
from sqlalchemy import create_engine, inspect
import os
from pathlib import Path
import re
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ds_agent.data import load_table

CONFIG_PATH = Path(__file__).parent.parent / "config" / "data_sources.yaml"
SCHEMA_CACHE_PATH = Path(__file__).parent.parent / "schema_cache.yaml"


def reflect_sql_schema():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    sources = config["sources"]
    cache = {"sources": {}}
    for src_name, src in sources.items():
        if "url" not in src:
            continue
        url = os.path.expandvars(src["url"])
        engine = create_engine(url)
        insp = inspect(engine)
        tables = {}
        for table_name in insp.get_table_names():
            columns = {}
            for col in insp.get_columns(table_name):
                columns[col["name"]] = {"dtype": str(col["type"]), "col_desc": ""}
            tables[table_name] = {"table_desc": "", "columns": columns}
        cache["sources"][src_name] = {"tables": tables}
    with open(SCHEMA_CACHE_PATH, "w") as f:
        yaml.safe_dump(cache, f, sort_keys=False)
    print(f"[schema_sync] Wrote schema cache to {SCHEMA_CACHE_PATH}")


def parse_markdown_data_sources(docs_dir: Path) -> dict:
    """
    Scan markdown files in docs_dir for 'Data Sources' tables and extract table descriptions.
    Returns a dict: {table_name: {table_desc: str}}
    """
    table_descs = {}
    for md_file in docs_dir.glob("**/*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        # Look for a 'Data Sources' section with a markdown table
        match = re.search(r"##+\s*Data Sources\s*\n([\s\S]+?)(?:\n##|\Z)", content, re.IGNORECASE)
        if not match:
            continue
        table_block = match.group(1)
        # Find markdown table rows
        rows = re.findall(r"^\|([^\n]+)\|([^\n]+)\|$", table_block, re.MULTILINE)
        for row in rows:
            table, desc = row
            table = table.strip().strip("`")
            desc = desc.strip()
            if table and desc and table != "Table":
                table_descs[table] = {"table_desc": desc}
    return table_descs


def infer_file_schema(csv_path: Path, table_name: str) -> dict:
    """
    Infer schema from a CSV file using pandas. Returns a dict:
    {table_name: {table_desc: '', columns: {col: {dtype, col_desc}}}}
    """
    df = pd.read_csv(csv_path, nrows=100)  # sample first 100 rows for type inference
    dtype_map = {
        "int64": "int",
        "float64": "float",
        "object": "string",
        "bool": "bool",
        "datetime64[ns]": "date",
    }
    columns = {}
    for col in df.columns:
        pdtype = str(df[col].dtype)
        dtype = dtype_map.get(pdtype, "string")
        columns[col] = {"dtype": dtype, "col_desc": ""}
    return {table_name: {"table_desc": "", "columns": columns}}


def merge_file_and_markdown_schema(file_schema: dict, md_descs: dict) -> dict:
    """
    Merge file-inferred schema and markdown metadata for a table.
    Uses markdown for table/column descriptions if available.
    """
    merged = {}
    for table, tbl_meta in file_schema.items():
        # Table description from markdown if available
        table_desc = md_descs.get(table, {}).get("table_desc", "")
        columns = {}
        for col, col_meta in tbl_meta["columns"].items():
            # Column description from markdown if available (not implemented yet, so fallback)
            col_desc = ""  # Could extend parser to get column descs from markdown
            columns[col] = {"dtype": col_meta["dtype"], "col_desc": col_desc}
        merged[table] = {"table_desc": table_desc, "columns": columns}
    return merged


def process_all_sources(config_path: Path, docs_dir: Path) -> dict:
    """
    Process all sources in config, merge with markdown docs, return full catalog dict.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    sources = config["sources"]
    md_descs = parse_markdown_data_sources(docs_dir)
    catalog = {"sources": {}}
    for src_name, src in sources.items():
        tables = {}
        # SQL source
        if "url" in src:
            try:
                url = os.path.expandvars(src["url"])
                engine = create_engine(url)
                insp = inspect(engine)
                for table_name in insp.get_table_names():
                    columns = {}
                    for col in insp.get_columns(table_name):
                        columns[col["name"]] = {"dtype": str(col["type"]), "col_desc": ""}
                    # Merge with markdown descs if available
                    table_desc = md_descs.get(table_name, {}).get("table_desc", "")
                    tables[table_name] = {"table_desc": table_desc, "columns": columns}
            except Exception as e:
                print(f"[schema_sync] WARNING: Could not reflect SQL source '{src_name}': {e}")
        # CSV/Parquet source
        elif src.get("format") in ("csv", "parquet"):
            path = Path(src["path"])
            if not path.exists():
                print(f"[schema_sync] WARNING: File not found for source '{src_name}': {path}")
                continue
            file_schema = infer_file_schema(path, src_name)
            merged = merge_file_and_markdown_schema(file_schema, md_descs)
            tables.update(merged)
        else:
            print(f"[schema_sync] WARNING: Unknown or unsupported source type for '{src_name}'")
        if tables:
            catalog["sources"][src_name] = {"tables": tables}
    return catalog


def enrich_schema_with_value_samples(catalog, max_sample=1000, max_values=10):
    """
    For each table/column in the catalog, add a 'values' field with up to max_values unique values for string/categorical columns.
    """
    for src_name, src_meta in catalog["sources"].items():
        for tbl_name, tbl_meta in src_meta["tables"].items():
            try:
                df = load_table(tbl_name, source=src_name)
                for col, col_meta in tbl_meta["columns"].items():
                    dtype = col_meta.get("dtype", "")
                    if dtype in ("string", "object"):
                        # Get up to max_values unique values
                        values = df[col].dropna().unique()
                        if len(values) > max_values:
                            values = values[:max_values]
                        col_meta["values"] = [str(v) for v in values]
            except Exception as e:
                print(
                    f"[schema_sync] WARNING: Could not profile values for {src_name}.{tbl_name}: {e}"
                )
    return catalog


def main():
    docs_dir = Path(__file__).parent.parent / "docs"
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    catalog = process_all_sources(config_path, docs_dir)
    # Enrich with value samples
    catalog = enrich_schema_with_value_samples(catalog)
    cache_path = Path(__file__).parent.parent / "schema_cache.yaml"
    with open(cache_path, "w") as f:
        yaml.safe_dump(catalog, f, sort_keys=False)
    print(f"[schema_sync] Wrote merged schema catalog to {cache_path}")
    print("[schema_sync] Catalog summary:")
    for src, src_meta in catalog["sources"].items():
        print(f"  Source: {src}")
        for tbl, tbl_meta in src_meta["tables"].items():
            print(f"    Table: {tbl} | {tbl_meta['table_desc']}")
            for col, cmeta in tbl_meta["columns"].items():
                values_str = f" | values: {cmeta['values']}" if "values" in cmeta else ""
                print(f"      {col}: {cmeta['dtype']} | {cmeta['col_desc']}{values_str}")


if __name__ == "__main__":
    main()
