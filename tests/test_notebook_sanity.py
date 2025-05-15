import subprocess
import tempfile
import nbformat
from pathlib import Path
import pytest
import os

NOTEBOOK_TEMPLATE = Path("templates/analysis_template.ipynb")
CLI_PATH = Path("bin/run_notebook.py")


@pytest.fixture
def test_params():
    return {
        "episode_id": "EP-TEST-001",
        "start_date": "2024-01-01",
        "end_date": "2024-01-03",
        "metric_name": "test_metric",
        "sql_query": "SELECT * FROM test_table",
    }


def run_cli(output_path: Path, params: dict):
    args = [
        "python",
        str(CLI_PATH),
        "--template",
        str(NOTEBOOK_TEMPLATE),
        "--output",
        str(output_path),
        "--episode-id",
        params["episode_id"],
        "--start-date",
        params["start_date"],
        "--end-date",
        params["end_date"],
        "--metric-name",
        params["metric_name"],
        "--sql-query",
        params["sql_query"],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(args, capture_output=True, text=True, env=env)


def test_notebook_execution_and_sanity(test_params):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "executed.ipynb"
        result = run_cli(output_path, test_params)
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Load executed notebook
        nb = nbformat.read(str(output_path), as_version=4)
        # All code cells executed
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert cell.execution_count is not None, "Unexecuted code cell found"
        # Parameters substituted
        param_cell = next(
            (c for c in nb.cells if "parameters" in c.get("metadata", {}).get("tags", [])), None
        )
        assert param_cell is not None, "Parameter cell missing"
        for key, value in test_params.items():
            assert str(value) in param_cell.source or str(value) in str(
                nb
            ), f"Parameter {key} not substituted"
        # Sanity check variable
        found = False
        for cell in nb.cells:
            if "sanity_passed" in cell.source and "True" in cell.source:
                found = True
        assert found, "sanity_passed == True not found in notebook"


def test_cli_exit_on_sanity_fail(test_params):
    # Patch the template to force sanity_passed = False
    import shutil
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_template = Path(tmpdir) / "bad_template.ipynb"
        shutil.copy(NOTEBOOK_TEMPLATE, bad_template)
        # Overwrite the data check cell to force fail
        nb = nbformat.read(str(bad_template), as_version=4)
        for cell in nb.cells:
            if "sanity_passed = True" in cell.source:
                cell.source = cell.source.replace(
                    "sanity_passed = True",
                    "sanity_passed = False\nprint('❌ Data sanity checks failed.')",
                )
        nbformat.write(nb, str(bad_template))
        output_path = Path(tmpdir) / "fail.ipynb"
        args = [
            "python",
            str(CLI_PATH),
            "--template",
            str(bad_template),
            "--output",
            str(output_path),
            "--episode-id",
            test_params["episode_id"],
            "--start-date",
            test_params["start_date"],
            "--end-date",
            test_params["end_date"],
            "--metric-name",
            test_params["metric_name"],
            "--sql-query",
            test_params["sql_query"],
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        result = subprocess.run(args, capture_output=True, text=True, env=env)
        assert result.returncode == 1, "CLI did not exit 1 on sanity fail"
