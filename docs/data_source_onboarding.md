# Data Source Onboarding Guide

This guide explains how to register new data sources (CSV, Parquet, SQL) for use in DS-Agent workflows.

---

## 1. Register Your Data Source

Edit `config/data_sources.yaml` and add an entry for your new dataset. Supported formats:

### CSV Example
```yaml
sources:
  my_csv:
    path: /absolute/or/relative/path/to/data.csv
    format: csv
    description: My CSV dataset
```

### Parquet Example
```yaml
sources:
  my_parquet:
    path: /absolute/or/relative/path/to/data.parquet
    format: parquet
    description: My Parquet dataset
```

### SQL Example
```yaml
sources:
  my_sql:
    url: postgresql://user:password@host:5432/dbname
    format: sql
    description: My SQL table
```

- The `description` field is optional but recommended.
- For SQL, use environment variables for secrets if needed (e.g., `${DB_PASSWORD}`).

---

## 2. Sync and Cache the Schema

After editing `data_sources.yaml`, run:

```bash
python bin/schema_sync.py
```

This will:
- Introspect your data sources
- Update `schema_cache.yaml` with column names, types, and sample values

---

## 3. Verify Registration

- Check `schema_cache.yaml` for your new source and table.
- Use the `describe_table` or `profile_table` tool to verify access:

```python
from ds_agent.data import describe_table, profile_table
describe_table('my_csv')
profile_table('my_csv')
```

- If you see errors, check:
  - File paths are correct and accessible
  - SQL credentials are valid
  - The format matches the file type

---

## 4. Validate Your Data (Recommended)

Before running analyses, check your data for quality issues:

```python
from ds_agent.data import validate_table
results = validate_table('my_csv')
print(results['issues'])  # List of detected issues (nulls, duplicates, outliers, etc.)
```

- The output includes null percentages, duplicate counts, type info, outlier counts, and a summary of issues.
- Review and address any issues before proceeding with analysis.

---

## 5. Next Steps

- See [analysis_workflow.md](analysis_workflow.md) for how to run analyses on your new data source.
- See [extending_tools.md](extending_tools.md) to add new analysis tools. 