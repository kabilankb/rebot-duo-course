"""The data-generation CLI: argument handling and the run.sh command it builds."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rebot_sim_pipeline import cli

PROJECT = Path("/opt/rebot-arm-dli-isaacsim")


class CommandTest(unittest.TestCase):
    def test_defaults_are_headless_scripted_recording(self) -> None:
        command = cli.build_command(cli.parse_args([]), PROJECT)
        self.assertEqual(command[:4], [str(PROJECT / "run.sh"), "--scripted", "--episodes", "20"])
        self.assertIn("--headless", command)
        root = command[command.index("--dataset-root") + 1]
        self.assertEqual(root, str(PROJECT / "datasets/rebot_stationery_scripted_dr"))
        self.assertNotIn("--no-recording", command)

    def test_options_and_passthrough(self) -> None:
        args = cli.parse_args(
            ["-n", "3", "--gui", "--seed", "7", "--no-dr", "--keep-failed",
             "--dataset-root", "/data/out", "--", "--strict-physics"]
        )
        command = cli.build_command(args, PROJECT)
        self.assertNotIn("--headless", command)
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--dataset-root") + 1], "/data/out")
        self.assertEqual(command[-1], "--strict-physics")
        for flag in ("--no-dr", "--keep-failed"):
            self.assertIn(flag, command)

    def test_dry_run_writes_nothing(self) -> None:
        command = cli.build_command(cli.parse_args(["--no-recording"]), PROJECT)
        self.assertIn("--no-recording", command)
        self.assertNotIn("--dataset-root", command)

    def test_rejects_zero_episodes(self) -> None:
        with self.assertRaises(SystemExit):
            cli.parse_args(["--episodes", "0"])


class DiscoveryTest(unittest.TestCase):
    def test_finds_project_and_source_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp, "proj")
            project.mkdir()
            (project / "run.sh").touch()
            release = Path(tmp, "isaacsim", "_build", "linux-x86_64", "release")
            release.mkdir(parents=True)
            (release / "python.sh").touch()
            self.assertEqual(cli.find_project(str(project)), project.resolve())
            self.assertEqual(cli.find_isaacsim(str(Path(tmp, "isaacsim"))), release.resolve())

    def test_counts_dataset_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(cli.count_episodes(Path(tmp)))
            Path(tmp, "meta").mkdir()
            Path(tmp, "meta", "info.json").write_text(json.dumps({"total_episodes": 12}))
            self.assertEqual(cli.count_episodes(Path(tmp)), 12)


if __name__ == "__main__":
    unittest.main()
