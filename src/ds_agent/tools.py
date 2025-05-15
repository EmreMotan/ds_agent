import yaml
import importlib
from pathlib import Path
from typing import Any, Callable, Dict

TOOLS_YAML = Path(__file__).parent.parent.parent / "tools.yaml"


def load_tools(yaml_path: Path = TOOLS_YAML) -> Dict[str, dict]:
    """
    Load the tool registry from tools.yaml, resolving function references to callables.
    Returns a dict: {intent: {fn: callable, args: [str, ...]}}
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    tools = {}
    for intent, entry in raw.items():
        fn_path = entry["fn"]
        module_name, fn_name = fn_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        tools[intent] = {"fn": fn, "args": entry.get("args", [])}
    return tools
