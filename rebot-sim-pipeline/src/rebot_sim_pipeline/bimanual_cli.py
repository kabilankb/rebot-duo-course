"""``rebot-bimanual-generate``: bimanual reBot demonstrations in one command.

1. Runs ``rebot_sim_pipeline.lab.generate`` with the Isaac Lab Python: the
   scripted expert drives both arms, successful episodes are staged.
2. Runs ``rebot_sim_pipeline.write_dataset`` with the LeRobot venv's Python:
   staged episodes are appended to a LeRobot v3 dataset.
3. Reads the dataset back and reports how many episodes were added.

Standard library only, so it runs from any Python. Exit code 0 only if every
requested episode was collected and written.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .cli import PACKAGE_ROOT

_DEFAULT_NAME = "rebot_bimanual_stationery"


def _first_python(explicit: str | None, env_var: str, defaults: list[Path], what: str) -> Path:
    for candidate in [explicit, os.environ.get(env_var), *map(str, defaults)]:
        if candidate and Path(candidate).expanduser().is_file():
            return Path(candidate).expanduser()
    raise SystemExit(f"[bimanual] {what} Python not found; pass it explicitly or set {env_var}")


def _total_episodes(dataset_root: Path) -> int:
    info = dataset_root / "meta" / "info.json"
    if not info.is_file():
        return 0
    return int(json.loads(info.read_text(encoding="utf-8"))["total_episodes"])


def _env(extra_pythonpath: Path | None, clean_conda: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    if clean_conda:  # keep a conda env's libraries out of the LeRobot venv
        for key in ("PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "LD_LIBRARY_PATH"):
            env.pop(key, None)
    if extra_pythonpath is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(extra_pythonpath), env.get("PYTHONPATH", "")]))
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # numpy 2 OpenBLAS threads crash Kit's startup fork.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")  # progress lines appear as they happen
    return env


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--episodes", type=int, default=10, help="Successful episodes to collect.")
    parser.add_argument("--seed", type=int, default=0, help="Layout seed of the first attempt (use a new one to add).")
    parser.add_argument("--task", default="stationery_sorting")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help=f"Default: <checkout>/datasets/{_DEFAULT_NAME} (or ./datasets/...).")
    parser.add_argument("--repo-id", default=f"local/{_DEFAULT_NAME}")
    parser.add_argument("--gui", action="store_true",
                        help="Show the Isaac Sim window in real time; it stays open until closed.")
    parser.add_argument("--no-recording", action="store_true", help="Dry run: nothing staged or written.")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--isaaclab-python", default=None, help="Default: $REBOT_ISAACLAB_PYTHON or env_yamlab6.")
    parser.add_argument("--lerobot-python", default=None, help="Default: $REBOT_LEROBOT_PYTHON or ~/rebot_lerobot/.venv.")
    parser.add_argument("--rebot-project", default=None, help="Single-arm reBot checkout (assets).")
    parser.add_argument("--print-command", action="store_true", help="Only print the generate command.")
    args = parser.parse_args(argv)

    base = PACKAGE_ROOT if PACKAGE_ROOT is not None else Path.cwd()
    dataset_root = (args.dataset_root or base / "datasets" / _DEFAULT_NAME).expanduser().resolve()
    staging = dataset_root.parent / f".{dataset_root.name}_staging"
    src = PACKAGE_ROOT / "src" if PACKAGE_ROOT is not None else None
    home = Path.home()

    isaaclab_python = _first_python(args.isaaclab_python, "REBOT_ISAACLAB_PYTHON",
                                    [home / "miniconda3/envs/env_yamlab6/bin/python"], "Isaac Lab")
    generate = [str(isaaclab_python), "-m", "rebot_sim_pipeline.lab.generate",
                "--task", args.task, "--episodes", str(args.episodes), "--seed", str(args.seed),
                "--staging", str(staging)]
    if args.no_recording:
        generate.append("--no-recording")
    if args.keep_failed:
        generate.append("--keep-failed")
    if args.rebot_project:
        generate += ["--rebot-project", args.rebot_project]
    generate += ["--viz", "kit", "--realtime", "--hold-open", "--camera-views"] if args.gui else []
    generate += extra
    print("[bimanual] " + shlex.join(generate))
    if args.print_command:
        return 0

    before = _total_episodes(dataset_root)
    generated = subprocess.run(generate, env=_env(src)).returncode
    if args.no_recording:
        return generated

    lerobot_python = _first_python(args.lerobot_python, "REBOT_LEROBOT_PYTHON",
                                   [home / "rebot_lerobot/.venv/bin/python"], "LeRobot")
    write = [str(lerobot_python), "-m", "rebot_sim_pipeline.write_dataset", "--staging", str(staging),
             "--dataset-root", str(dataset_root), "--repo-id", args.repo_id]
    print("[bimanual] " + shlex.join(write))
    written = subprocess.run(write, env=_env(src, clean_conda=True)).returncode
    added = _total_episodes(dataset_root) - before
    print(f"[bimanual] dataset {dataset_root}: {added} episode(s) added, "
          f"{_total_episodes(dataset_root)} total (generate exit {generated}, writer exit {written})")
    return 0 if generated == 0 and written == 0 and added >= args.episodes else 1


if __name__ == "__main__":
    raise SystemExit(main())
