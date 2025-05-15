#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
from typing import Optional
from ds_agent.run_notebook import run_analysis
import nbformat
import re


def main():
    parser = argparse.ArgumentParser(
        description="Run a parameterized analysis notebook via Papermill."
    )
    parser.add_argument(
        "--template", type=Path, required=True, help="Path to notebook template (.ipynb)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save executed notebook (.ipynb)"
    )
    parser.add_argument("--episode-id", required=True, help="Episode ID")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--metric-name", required=True, help="Metric name")
    parser.add_argument("--sql-query", required=True, help="SQL query")
    args = parser.parse_args()
    params = {
        "episode_id": args.episode_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "metric_name": args.metric_name,
        "sql_query": args.sql_query,
    }
    try:
        executed_path = run_analysis(args.template, params, args.output)
        html_path = args.output.with_suffix(".html")
        print(f"Notebook executed: {executed_path}")
        print(f"HTML exported: {html_path}")
        # Check for sanity_passed in the executed notebook by output
        nb = nbformat.read(str(args.output), as_version=4)
        found_sanity_passed = False
        for cell in nb.cells:
            if cell.cell_type == "code" and "outputs" in cell:
                for output in cell.outputs:
                    if hasattr(output, "text") and "✅ Data sanity checks passed." in output.text:
                        found_sanity_passed = True
        if not found_sanity_passed:
            print("Sanity checks failed in notebook.", file=sys.stderr)
            sys.exit(1)
        # Register artifacts in episode.json if episode_id is provided
        try:
            from ds_agent.episode import load_episode

            episode_dir = Path("episodes") / params["episode_id"]
            episode_file = episode_dir / "episode.json"
            if episode_file.exists():
                episode = load_episode(str(episode_file))
                episode.register_artifact("NOTEBOOK", str(args.output))
                episode.register_artifact("DOC", str(html_path))
                print(f"Artifacts registered in {episode_file}")
        except Exception as e:
            print(f"[warn] Could not register artifacts: {e}")
        return
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
