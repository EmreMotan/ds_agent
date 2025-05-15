# PRD‑009: Data Connector & Schema Registry  
*Version 0.1 • May 3 2025*

---

## 1 | Overview  
Provide DS‑Agent with **first‑class data access and rich schema metadata** so the Execution Agent can autonomously load tables and understand their structure.

Deliverables:

1. **Data Connector Module** – unified API for SQLAlchemy URLs and local CSV/Parquet.  
2. **Schema Loader** – merges DB reflection and markdown docs into a YAML registry.  
3. **Schema Registry & Cache** – lightweight catalog `schema_cache.yaml`.  
4. **Metadata Tools** – `load_table`, `describe_table`, `profile_table` added to tool registry.  
5. **Credential Handling** – connections defined in `config/data_sources.yaml`, secrets via env vars.

---

## 2 | Problem / Goal  
Agents currently can’t discover or load data without hard‑coded code.  
We need a layer that tells them *what* data exists and *how* to fetch it safely.

---

## 3 | Scope (MVP)

### 3.1 Config files  

* `config/data_sources.yaml` – DB URLs, driver, read‑only creds.  
* `docs/schema_docs/<source>/<table>.md` – human descriptions.  
* `schema_cache.yaml` – auto‑generated catalog.

### 3.2 Data Connector (`src/ds_agent/data.py`)

```python
def load_table(table: str,
               cols: list[str] | None = None,
               where: str | None = None,
               source: str | None = None) -> pd.DataFrame
```

### 3.3 Schema Loader (`bin/schema_sync.py`)

* Reflects SQL schema via SQLAlchemy.  
* Parses markdown docs with column descriptions.  
* Writes merged YAML:

```yaml
sources:
  nba_mysql:
    tables:
      nba.games:
        table_desc: "One row per NBA game"
        columns:
          pts_home:
            dtype: INT
            col_desc: "Points scored by home team"
```

### 3.4 Tool additions

| Intent           | Implementation                   |
| ---------------- | -------------------------------- |
| `load_table`     | Wrapper around `data.load_table` |
| `describe_table` | Return JSON from cache           |
| `profile_table`  | Basic row count, null %, min/max |

### 3.5 Execution Agent changes

Before planning, insert table metadata from cache into LLM context.

---

## 4 | Success Criteria  

| Metric                                  | Target |
| --------------------------------------- | ------ |
| Schema sync ≤ 2 min for 100 tables      | ✔︎      |
| `load_table` ≤ 10 s for 1 M rows        | ✔︎      |
| ≥ 90 % columns have descriptions        | ✔︎      |
| Coverage (data & schema modules) ≥ 90 % | ✔︎      |

---

## 5 | Non‑Functional Requirements  

* Secrets only via environment variables.  
* SQL connections are read‑only.  
* Works offline with CSV fallback.  
* Python 3.10+; deps: `sqlalchemy`, `pymysql`, `pyyaml`, `pandas`.

---

## 6 | Deliverables  

| Path                       | Artifact          |
| -------------------------- | ----------------- |
| `config/data_sources.yaml` | Source config     |
| `docs/schema_docs/`        | Markdown docs     |
| `bin/schema_sync.py`       | Loader CLI        |
| `schema_cache.yaml`        | Generated catalog |
| `src/ds_agent/data.py`     | Connector         |
| `src/ds_agent/tools.py`    | Registry updates  |
| `tests/data/test_*.py`     | Unit tests        |
| `docs/data_access.md`      | Usage guide       |

---

## 7 | Acceptance Checklist  

- [ ] Sync merges DB + markdown descriptions.  
- [ ] Agent loads table in pilot without manual code.  
- [ ] Coverage ≥ 90 %.  
- [ ] README updated with credential setup.  
- [ ] Pre‑commit hooks clean.

---

*End of PRD*