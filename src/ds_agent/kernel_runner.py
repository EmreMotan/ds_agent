from typing import Any, Dict
from pathlib import Path
import papermill as pm
import nbformat


def run_notebook(notebook_path: Path, parameters: Dict[str, Any], output_path: Path) -> Path:
    """
    Execute the notebook at notebook_path with the given parameters using Papermill.
    Save the executed notebook to output_path. Return the output_path.
    """
    pm.execute_notebook(
        str(notebook_path),
        str(output_path),
        parameters=parameters,
        kernel_name=None,  # Use default kernel
        progress_bar=False,
        log_output=False,
    )
    return output_path


def extract_globals(notebook_path: Path) -> Dict[str, Any]:
    """
    Extract global variables from the executed notebook (e.g., sanity_passed).
    Return a dict of variable names to values.
    """
    nb = nbformat.read(str(notebook_path), as_version=4)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and cell.get("outputs"):
            print(f"DEBUG: Cell {idx} source:\n{cell.source}")
            for output in cell.outputs:
                print(f"  Output type: {output.output_type}")
                if hasattr(output, "data"):
                    print(f"    data: {getattr(output, 'data', None)}")
                if hasattr(output, "text"):
                    print(f"    text: {getattr(output, 'text', None)}")
                if output.output_type == "execute_result" and isinstance(output.data, dict):
                    text = output.data.get("text/plain", "")
                    try:
                        import ast

                        d = ast.literal_eval(text)
                        if "sanity_passed" in d:
                            return d
                    except Exception as e:
                        print(
                            f"DEBUG: Failed to parse execute_result as dict: {e}\nRaw text: {text[:500]}"
                        )
                        continue
    return {}
