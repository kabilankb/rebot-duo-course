"""``rebot-vla-serve``: vla-edge's ``vla-edge-serve`` with the bimanual reBot registered.

Every vla-edge server option is passed through. Defaults here: the replay
policy and backend, and port 8202 as in vla-edge::

    rebot-vla-serve --checkpoint datasets/rebot_bimanual_stationery --episode 0
    rebot-vla-serve --policy <your-rebot-policy> --backend torch ...
"""

from __future__ import annotations

import argparse
import os
import sys

from .embodiment import REPLAY_POLICY_NAME, register
from .replay import EPISODE_ENV


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episode", type=int, default=None, help="Replay policy: dataset episode index.")
    ours, rest = parser.parse_known_args(argv)
    if ours.episode is not None:
        os.environ[EPISODE_ENV] = str(ours.episode)
    if not any(arg == "--policy" or arg.startswith("--policy=") for arg in rest):
        rest = ["--policy", REPLAY_POLICY_NAME] + rest
        if not any(arg == "--backend" or arg.startswith("--backend=") for arg in rest):
            rest += ["--backend", "replay"]
    register()
    from vla_edge.serving.server import main as serve

    return serve(rest)


if __name__ == "__main__":
    raise SystemExit(main())
