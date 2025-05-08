# DS-Agent: Autonomous, Human-Supervised Data Science Workflows

[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/yourname/ds-agent)

> **Vision:**
> Build an autonomous (yet human-supervised) agent that can execute the entire data science workflow—from context gathering and robust analysis planning to artifact snapshotting and human review—while preserving reproducibility, transparency, and security.

---

## Why DS-Agent?

- **Accelerate analysis:** Automate boilerplate, reduce time-to-insight, and let agents handle the busywork.
- **Human-in-the-loop:** Every workflow is reviewable, auditable, and can be iterated with human feedback.
- **LLM-powered planning:** Uses large language models to generate, validate, and patch analysis plans—auto-fixing common errors and surfacing actionable feedback for unsatisfiable plans.
- **Reproducibility & Transparency:** Every step, artifact, and decision is tracked and versioned.

---

## Key Features

- **Episode Management:** Track and manage atomic units of work ("episodes") with full state transitions and iteration history.
- **Orchestration:** Central controller coordinates agents, humans, and artifact flows.
- **LLM-Driven Planning:** Generate and patch robust, dependency-satisfying analysis plans with automatic error correction and clear feedback.
- **Memory Backbone:** Ingest, chunk, and semantically search documents and outputs for context-aware analysis.
- **Analysis Library:** Built-in, unit-tested functions for cohort, segmentation, funnel, statistical tests, and plotting.
- **Artifact Snapshotting:** Auto-snapshots notebooks, plots, and outputs; updates manifest for full reproducibility.
- **Feedback-to-Task Pipeline:** Turns PR comments into actionable tasks, closing the loop between code and review.
- **Extensible:** Add new tools, data sources, or agent logic with minimal friction.

---

## How It Works

1. **Create an Episode:** Define a goal and context for your analysis.
2. **Agent Planning:** LLM generates a step-by-step plan (YAML), auto-patched for dependency and column correctness.
3. **Execution:** The ExecAgent runs the plan, edits notebooks, and applies robust error handling.
4. **Human Review:** All outputs and plans are reviewable; feedback is ingested and can trigger new tasks.
5. **Artifact Tracking:** Every output is snapshotted, hashed, and registered for reproducibility.
6. **Memory & Retrieval:** All docs and outputs are ingested for future context-aware analysis.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/ds-agent.git
cd ds-agent

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First Analysis in 5 Minutes

```python
from ds_agent.episode import create_episode
from ds_agent.agents.ds_agent import DataScienceAgent

# 1. Create an episode
episode = create_episode("Churn Analysis", "Analyze customer churn factors")

# 2. Plan the analysis
agent = DataScienceAgent(<episode_dir>)
agent.plan_analysis(episode)
# Edit plan.json to specify your data source if needed

# 3. Run the analysis
agent.run_analysis(episode)

# 4. Review outputs in the episode directory (notebooks, plots, logs)
```

Or use the CLI:

```bash
python bin/manage_episode.py create "Churn Analysis" "Analyze churn"
python -m ds_agent.orchestrator --once
```

---

## Analysis Library: Built-in, Reusable Functions

- **Cohort Analysis:** `run_retention(df, signup_date, event_date, user_id)` → retention pivot, heatmap, markdown
- **Segmentation:** `lift_table(df, metric_col, dim_col)` → group means, lift, p-values
- **Funnel Analysis:** `funnel_conversion(df, step_cols)` → conversion rates, bar plot
- **Statistical Tests:** `bootstrap_mean_diff(a, b)`, `t_test`, `chi_square`, `anova`
- **Visualization:** `plot_bar`, `plot_time_series`, `plot_scatter`, and more
- **Artifact Saving:** `save_fig(fig, episode_id, name)` auto-registers outputs

See [Analysis Library Documentation](docs/analysis_library.md) for full catalog and examples.

---

## Robustness & Guardrails

- **Plan Validation & Patching:**
  - Tracks column lineage and DataFrame dependencies
  - Auto-inserts missing columns, reorders steps, and fails fast with actionable errors
  - Prevents cycles and unsatisfiable dependencies
- **Human-in-the-Loop:**
  - All plans and outputs are reviewable and auditable
  - Feedback from PRs is classified and turned into tasks
- **Artifact Versioning:**
  - Snapshots all outputs, updates manifest, and enforces branch policy via Git hooks

---

## Extending DS-Agent

- **Add new analysis tools:** Implement in `src/ds_agent/analysis/` and register in the tool registry
- **Support new data sources:** Register in `config/data_sources.yaml` and update schema cache
- **Improve agent intelligence:** Swap in new LLMs, update prompt engineering, or add custom plan patching logic

See [System Overview](docs/system_overview.md) and [Architecture Overview](docs/architecture_overview.md) for details.

---

## Learn More

- [System Overview](docs/system_overview.md): End-to-end workflow, technical mapping, and onboarding
- [Analysis Library](docs/analysis_library.md): Function catalog and usage examples
- [Architecture Overview](docs/architecture_overview.md): Vision, roadmap, and repo layout
- [Analysis Workflow Guide](docs/analysis_workflow.md): Step-by-step usage
- [Data Source Onboarding](docs/data_source_onboarding.md): How to register and validate new data
- [Agent Functions](docs/agent_functions.md): Full list of agent tools and signatures

---

## Development & Testing

- **Run all tests:**
  ```bash
  pytest --cov=src/ds_agent
  ```
- **Coverage target:** ≥90%
- **Pre-commit hooks:** Enforced for formatting and linting

---

## Feedback & Contribution

- PR comments are auto-classified and turned into actionable tasks
- See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and review process

---

*DS-Agent is designed for teams who want to move fast, automate the boring parts, and keep humans in the loop for what matters. Welcome to the future of data science workflows.*

## Features

- **Episode Management**: Track and manage data science workflows with atomic state transitions
- **Orchestration**: Coordinate workflows between human and agent actors
- **Memory Backbone**: Store and retrieve documents with semantic search capabilities
- **Reproducibility**: Each analysis is fully documented and reproducible

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourname/ds-agent.git
cd ds-agent

# Set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Episode Management

```bash
# Create a new episode
python bin/manage_episode.py create "Customer Churn Analysis" "Analyze factors affecting customer churn rates"

# List all episodes
python bin/manage_episode.py list

# Show episode details
python bin/manage_episode.py show <EPISODE_ID>

# Update episode status
python bin/manage_episode.py update <EPISODE_ID> --status IN_ANALYSIS
```

#### Memory Operations

```bash
# Ingest a document into memory
python bin/memory.py ingest --path data/customer_data.csv --episode-id <EPISODE_ID>

# Query memory
python bin/memory.py query --query "customer retention strategies" --k 5
```

#### Running the Orchestrator

```bash
# Run once
python -m ds_agent.orchestrator --once

# Run in loop mode (until Ctrl-C)
python -m ds_agent.orchestrator --loop
```

#### Notebook Analysis CLI Example

Run a parameterized analysis notebook and export HTML:

```bash
python bin/run_notebook.py \
  --template templates/analysis_template.ipynb \
  --output episodes/DS-25-001/analysis.ipynb \
  --episode-id DS-25-001 \
  --start-date 2024-01-01 \
  --end-date 2024-01-03 \
  --metric-name "test_metric" \
  --sql-query "SELECT * FROM test_table"
```

---

## Feedback-to-Task Pipeline

Automatically triage human feedback from GitHub PRs into actionable tasks for each episode.

- **Collects** PR comments via the GitHub API
- **Classifies** feedback (scope change, bug, polish, question)
- **Appends** tasks to `episodes/<EPISODE_ID>/backlog.yaml` (idempotent, no duplicates)
- **Each task includes a `priority` field** (`high`, `medium`, `low`) based on type
- **Supports** one-off or continuous polling
- **Optionally auto-ingests** `backlog.yaml` into Memory for contextual retrieval (see `--ingest-backlog`)

### Requirements
- Python 3.10+
- `requests`, `pyyaml`, `chromadb`, `sentence-transformers`, `filelock` (see requirements.txt)
- Set the `GITHUB_TOKEN` environment variable with access to your repo
- `.yaml` is now a supported file type for Memory ingestion

### Setup
1. Add a `pr_url` field to your `episode.json`:
   ```json
   {
     "episode_id": "DS-25-005",
     ...
     "pr_url": "https://github.com/yourname/yourrepo/pull/123"
   }
   ```
2. Ensure your token has access to the repo (private: `repo` scope, public: `public_repo`)

### Usage

**Run once:**
```bash
python bin/feedback_sync.py --episode DS-25-005 --once
```

**Run in loop mode (poll every 60 seconds):**
```bash
python bin/feedback_sync.py --episode DS-25-005 --loop 60
```

**Control log output:**
- `--verbose` for debug output
- `--quiet` to suppress info output (only errors)

**Auto-ingest backlog into Memory:**
```bash
python bin/feedback_sync.py --episode DS-25-005 --once --ingest-backlog
```

- Tasks are appended to `episodes/DS-25-005/backlog.yaml`.
- Duplicate runs do not create duplicate tasks.
- Output is valid YAML, ready for orchestrator or manual review.
- If `--ingest-backlog` is set, the backlog is ingested into Memory for contextual retrieval by agents.

---

## Artifact Snapshotting & Version-Control Hooks

Automatically snapshot executed notebooks and artifacts for each episode, and enforce branch policy for safe collaboration.

- **Snapshot CLI:**
  - Snapshots all `.ipynb`, `.csv`, `.xlsx`, `.pdf`, `.html`, `.png`, `.jpg`, `.jpeg` files in an episode directory (excluding outputs/)
  - Copies them to `episodes/<EPISODE_ID>/outputs/<DATE_HASH>/`
  - Updates the `artifacts` manifest in `episode.json` with type, SHA-256, and path

**Manual usage:**
```bash
python bin/snapshot_outputs.py --episode DS-25-006
```

- **Post-merge hook:**
  - After a merge, automatically snapshots artifacts for all episodes
  - To install:
    ```bash
    ln -sf ../../scripts/post-merge .git/hooks/post-merge
    chmod +x .git/hooks/post-merge
    ```

- **Branch policy pre-push hook:**
  - Blocks direct pushes to `main`
  - Enforces branch naming: `ep-<ID>-<desc>` (e.g., `ep-006-snapshot-artifacts`)
  - To install:
    ```bash
    ln -sf ../../scripts/pre-push .git/hooks/pre-push
    chmod +x .git/hooks/pre-push
    ```

---

## Development

### Running Tests

```
