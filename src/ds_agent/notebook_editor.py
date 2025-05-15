from nbformat import NotebookNode, v4 as nbf
from typing import List, Optional


def add_code(nb: NotebookNode, code: str, tag: Optional[str] = None) -> None:
    """
    Add a code cell to the notebook. Optionally tag the cell.
    """
    cell = nbf.new_code_cell(code)
    if tag:
        tag_cell(cell, tag)
    nb.cells.append(cell)


def add_markdown(nb: NotebookNode, markdown: str, tag: Optional[str] = None) -> None:
    """
    Add a markdown cell to the notebook. Optionally tag the cell.
    """
    cell = nbf.new_markdown_cell(markdown)
    if tag:
        tag_cell(cell, tag)
    nb.cells.append(cell)


def tag_cell(cell: NotebookNode, tag: str) -> None:
    """
    Add a tag to a notebook cell's metadata.
    """
    if "tags" not in cell.metadata:
        cell.metadata["tags"] = []
    if tag not in cell.metadata["tags"]:
        cell.metadata["tags"].append(tag)
