"""Layered YAML configuration (no Isaac Lab import; usable before the app starts).

Resolution, lowest -> highest precedence::

    configs/defaults.yaml
      < defaults.modes[mode]
        < configs/tasks/<task>.yaml
          < <task>.modes[mode]
            < overrides (CLI)

Nested dicts merge key by key; scalars and lists replace. The robot YAML is
read separately with :func:`load_robot`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_ROBOT = "rebot_b601_bimanual"
DEFAULT_TASK = "stationery_sorting"
MODES = ("generate", "view", "evaluation")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``override`` merged in (dicts recursively, the rest replaced)."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def load_task(
    task: str = DEFAULT_TASK, mode: str = "generate", overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve the task config for ``mode``; ``task`` is a name in configs/tasks or a YAML path."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    task_path = Path(task) if task.endswith((".yaml", ".yml")) else CONFIG_DIR / "tasks" / f"{task}.yaml"
    if not task_path.is_file():
        raise FileNotFoundError(f"task config not found: {task_path}")
    defaults = _read(CONFIG_DIR / "defaults.yaml")
    task_cfg = _read(task_path)
    resolved = deep_merge({k: v for k, v in defaults.items() if k != "modes"},
                          defaults.get("modes", {}).get(mode, {}))
    resolved = deep_merge(resolved, {k: v for k, v in task_cfg.items() if k != "modes"})
    resolved = deep_merge(resolved, task_cfg.get("modes", {}).get(mode, {}))
    if overrides:
        resolved = deep_merge(resolved, overrides)
    resolved["mode"] = mode
    return resolved


def load_robot(robot: str = DEFAULT_ROBOT) -> dict[str, Any]:
    """Robot facts; ``robot`` is a name in configs/robot or a YAML path."""
    path = Path(robot) if robot.endswith((".yaml", ".yml")) else CONFIG_DIR / "robot" / f"{robot}.yaml"
    return _read(path)


def find_rebot_project(explicit: str | None = None) -> Path:
    """The single-arm reBot project that holds the robot and stationery assets."""
    from ..cli import find_project  # stdlib only

    return find_project(explicit)
