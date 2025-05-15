# PRD-004: Notebook Template & Auto-Tests
*Version 0.1 • May 3 2025*

## 1 | Overview
This PRD adds a reproducible analysis engine to DS-Agent:

1. Parameterized notebook template – skeleton notebook agents fill with SQL, Python, narrative, and charts.
2. Execution harness – runs the template via Papermill, injecting parameters and exporting artifacts.
3. Sanity-test suite – asserts basic data-quality checks so bad numbers never reach stakeholders.
4. CLI wrappers – one-command execution for humans or the Orchestrator.

## 2 | Problem / Goal
Analyses must be reproducible, testable, and parameterizable to support rapid iteration.

## 3 | Scope (MVP)

### 3.1 Notebook template
Path: templates/analysis_template.ipynb  
Sections: Title & Parameters, Data Extraction, Data Checks, Visualization, Findings & Next Steps.  
Parameters: episode_id, start_date, end_date, metric_name, sql_query.

### 3.2 Execution harness
File: src/ds_agent/run_notebook.py

def run_analysis(template_path: Path, params: dict, output_path: Path) -> Path:
"""Execute notebook via Papermill, save executed .ipynb and .html, return notebook path."""

Uses Papermill & nbconvert; updates artifacts in episode.json.

### 3.3 Auto-tests
tests/test_notebook_sanity.py validates:
* All code cells executed.
* Parameters substituted.
* sanity_passed == True variable exists.
* CLI exit 1 on assert failure.

### 3.4 CLI
bin/run_notebook.py (argparse) executes notebook with params; returns 0 on success.

### 3.5 Orchestrator hook
Orchestrator state IN_ANALYSIS calls run_analysis() and logs notebook & HTML.

## 4 | Success Criteria
Execution ≤ 10s (1k rows), notebook passes tests, ≥ 90% coverage, CLI exit accurate.

## 5 | Non-Functional Requirements
Headless execution, Python 3.10+, deps: papermill, nbformat, nbconvert, matplotlib.

## 6 | Out of Scope
SQL auto-gen, advanced viz styling, deep statistical tests.

## 7 | Deliverables
- templates/analysis_template.ipynb
- src/ds_agent/run_notebook.py
- bin/run_notebook.py
- tests/test_notebook_sanity.py
- docs/notebook_workflow.mmd

## 8 | Acceptance Checklist
- Parameters render in title.
- Executed ipynb & html saved.
- CLI returns exit 1 on assert fail.
- Artifact registered in episode.json.
- Coverage ≥ 90%.
- README updated.
- Pre-commit hooks clean.

## 9 | Sequence Diagram
Mermaid source docs/notebook_workflow.mmd.

## 10 | Dependencies / References
PRD-001, PRD-002, PRD-003.

*End of PRD*