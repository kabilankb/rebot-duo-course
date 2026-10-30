"""``rebot-sim-generate``: collect scripted-expert demonstrations without a person.

Runs the reBot Isaac Sim project (``rebot-arm-dli-isaacsim``) in ``--scripted``
mode: every episode is a freshly randomized scene, the expert puts both pens and
the eraser in the cup, successful episodes are written as LeRobot episodes, and
failed attempts are discarded and retried. Afterwards the dataset is read back
and the episode count reported, so the result does not depend on how Isaac Sim
exits.

Standard library only: it launches Isaac Sim's own Python, and can run from any
interpreter (the host's, a venv, ``python -m rebot_sim_pipeline``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# The checkout run.sh puts on Isaac Sim's PYTHONPATH (<root>/src). Only known for a
# source-layout install (pip install -e, or PYTHONPATH=<root>/src).
_PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = _PACKAGE_DIR.parents[1] if _PACKAGE_DIR.parent.name == "src" else None
_PROJECT_NAME = "rebot-arm-dli-isaacsim"
_DEFAULT_DATASET_ROOT = "datasets/rebot_stationery_scripted_dr"
_DEFAULT_REPO_ID = "local/rebot_stationery_scripted_dr"


def find_project(explicit: str | None) -> Path:
    """The reBot project: --project, $REBOT_HOME, a sibling checkout, or ~/<name>."""
    candidates = [explicit, os.environ.get("REBOT_HOME")]
    if PACKAGE_ROOT is not None:
        candidates.append(str(PACKAGE_ROOT.parent / _PROJECT_NAME))
    candidates.append(str(Path.home() / _PROJECT_NAME))
    for candidate in candidates:
        if candidate and (Path(candidate).expanduser() / "run.sh").is_file():
            return Path(candidate).expanduser().resolve()
    raise SystemExit(
        f"[generate] {_PROJECT_NAME} not found; pass --project or set REBOT_HOME "
        f"(looked in: {', '.join(c for c in candidates if c)})"
    )


def find_isaacsim(explicit: str | None) -> Path:
    """Isaac Sim root holding python.sh: a stock install or a source build's release dir."""
    candidates = [explicit, os.environ.get("ISAACSIM_ROOT"), "/isaac-sim", str(Path.home() / "isaacsim")]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        for option in (root, root / "_build" / "linux-x86_64" / "release"):
            if (option / "python.sh").is_file():
                return option.resolve()
    raise SystemExit(
        "[generate] Isaac Sim not found (no python.sh); pass --isaacsim-root or set ISAACSIM_ROOT"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rebot-sim-generate",
        description="Collect scripted-expert reBot demonstrations as a LeRobot dataset, unattended.",
        epilog="Arguments after -- go to run.sh unchanged, e.g. -- --strict-physics",
    )
    parser.add_argument("-n", "--episodes", type=int, default=20, help="successful episodes to collect (default 20)")
    parser.add_argument(
        "--dataset-root",
        default=_DEFAULT_DATASET_ROOT,
        help=f"LeRobot dataset folder; relative paths are taken from the project (default {_DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument("--repo-id", default=_DEFAULT_REPO_ID, help=f"dataset repo id (default {_DEFAULT_REPO_ID})")
    parser.add_argument("--task-name", help="task text stored with each episode (default: the project's)")
    parser.add_argument("--seed", type=int, help="randomization seed; vary it to extend a dataset with new scenes")
    parser.add_argument("--gui", action="store_true", help="show the Isaac Sim window (default: headless)")
    parser.add_argument("--no-dr", action="store_true", help="fixed scene, no domain randomization")
    parser.add_argument("--keep-failed", action="store_true", help="also save episodes that do not end in success")
    parser.add_argument("--no-recording", action="store_true", help="dry run: run the expert, write nothing")
    parser.add_argument("--project", help=f"{_PROJECT_NAME} checkout (default: $REBOT_HOME, sibling or ~/)")
    parser.add_argument("--isaacsim-root", help="Isaac Sim root (default: $ISAACSIM_ROOT, /isaac-sim or ~/isaacsim)")
    parser.add_argument("--print-command", action="store_true", help="print the run.sh command and exit")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.extra = extra
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    return args


def build_command(args: argparse.Namespace, project: Path) -> list[str]:
    command = [str(project / "run.sh"), "--scripted", "--episodes", str(args.episodes)]
    if not args.gui:
        command.append("--headless")
    if args.no_recording:
        command.append("--no-recording")
    else:
        command += ["--dataset-root", str(dataset_path(args, project)), "--repo-id", args.repo_id]
        if args.task_name:
            command += ["--task-name", args.task_name]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.no_dr:
        command.append("--no-dr")
    if args.keep_failed:
        command.append("--keep-failed")
    return command + list(args.extra)


def dataset_path(args: argparse.Namespace, project: Path) -> Path:
    root = Path(args.dataset_root).expanduser()
    return root if root.is_absolute() else project / root


def count_episodes(root: Path) -> int | None:
    """Episodes recorded in a LeRobot dataset (meta/info.json), None if there is none."""
    try:
        return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])
    except (OSError, ValueError, KeyError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project = find_project(args.project)
    isaacsim = find_isaacsim(args.isaacsim_root)
    command = build_command(args, project)
    environment = dict(os.environ, ISAACSIM_ROOT=str(isaacsim))
    if PACKAGE_ROOT is not None:
        environment["REBOT_SIM_PIPELINE"] = str(PACKAGE_ROOT)
    if args.print_command:
        pipeline = environment.get("REBOT_SIM_PIPELINE", "<run.sh default>")
        print(f"ISAACSIM_ROOT={isaacsim} REBOT_SIM_PIPELINE={pipeline} {shlex.join(command)}")
        return 0

    dataset = None if args.no_recording else dataset_path(args, project)
    before = count_episodes(dataset) if dataset else None
    print(f"[generate] {args.episodes} episode(s) -> {dataset or 'nothing (dry run)'}")
    print(f"[generate] {shlex.join(command)}", flush=True)
    started = time.monotonic()
    try:
        return_code = subprocess.call(command, cwd=project, env=environment)
    except KeyboardInterrupt:
        return_code = 130
    minutes = (time.monotonic() - started) / 60.0

    if dataset is None:
        print(f"[generate] Dry run finished in {minutes:.1f} min (run.sh exit {return_code})")
        return return_code
    after = count_episodes(dataset)
    added = (after or 0) - (before or 0)
    print(
        f"[generate] {added} new episode(s) in {minutes:.1f} min; dataset now has "
        f"{after or 0} episode(s) at {dataset} (run.sh exit {return_code})"
    )
    if return_code == 130:
        return 130
    return 0 if added >= args.episodes else (return_code or 2)


if __name__ == "__main__":
    sys.exit(main())
