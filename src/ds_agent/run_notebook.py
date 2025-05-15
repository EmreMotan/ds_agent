import logging
from pathlib import Path
import papermill as pm
import nbformat
from nbconvert import HTMLExporter

logger = logging.getLogger(__name__)


def run_analysis(template_path: Path, params: dict, output_path: Path) -> Path:
    """
    Execute notebook via Papermill, save executed .ipynb and .html, return notebook path.
    Args:
        template_path: Path to the notebook template (.ipynb)
        params: Dictionary of parameters to inject
        output_path: Path to save the executed notebook (.ipynb)
    Returns:
        Path to the executed notebook (.ipynb)
    Raises:
        RuntimeError if execution or conversion fails
    """
    try:
        logger.info(f"Running analysis notebook: {template_path} -> {output_path}")
        # Execute notebook with Papermill
        pm.execute_notebook(
            input_path=str(template_path),
            output_path=str(output_path),
            parameters=params,
            log_output=True,
        )
        logger.info(f"Notebook executed: {output_path}")
    except Exception as e:
        logger.error(f"Papermill execution failed: {e}")
        raise RuntimeError(f"Papermill execution failed: {e}")

    # Convert to HTML
    html_path = output_path.with_suffix(".html")
    try:
        nb = nbformat.read(str(output_path), as_version=4)
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(nb)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(body)
        logger.info(f"Notebook HTML exported: {html_path}")
    except Exception as e:
        logger.error(f"HTML export failed: {e}")
        raise RuntimeError(f"HTML export failed: {e}")

    return output_path
