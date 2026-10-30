"""Tests for the simulator-free parts of the bimanual pipeline."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rebot_sim_pipeline import conventions
from rebot_sim_pipeline.bimanual import arm_bases, sample_bimanual_layout, work_order
from rebot_sim_pipeline.chain import SerialChain, base_pose_from_xy_yaw
from rebot_sim_pipeline.workspace import footprint_points

try:
    import yaml  # noqa: F401
    from rebot_sim_pipeline.lab import config
    from rebot_sim_pipeline.lab.geometry import look_at_opengl, matrix_to_quat_wxyz
    from rebot_sim_pipeline.kinematics import quat_to_matrix
except ImportError:  # PyYAML missing: skip the config tests
    config = None

ROBOT_YAML = Path(__file__).resolve().parents[1] / "src/rebot_sim_pipeline/lab/configs/robot/rebot_b601_bimanual.yaml"

# Link poses authored in the reBot USD at q = 0 (base frame): the FK must reproduce them.
USD_LINK_POSITIONS = {
    1: (-0.000343, -0.000987, 0.075),
    2: (0.02, 0.02625, 0.145),
    3: (-0.216, 0.02625, 0.145),
    4: (0.012, 0.03075, 0.217746),
    5: (0.099, 0.000001, 0.265746),
    6: (0.1355, 0.0, 0.217746),
}
USD_LINK6_ROTATION = np.asarray([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

# The YAMLab station (configs/robot + configs/tasks/stationery_sorting.yaml).
CUP_XY = (0.45, 0.0)
BASES = arm_bases({
    "left_arm": {"base_xy_m": [0.2525, 0.305], "heading_xy": [1.0, 0.0]},
    "right_arm": {"base_xy_m": [0.2525, -0.305], "heading_xy": [1.0, 0.0]},
})
LAYOUT = {
    "center_radius_m": [0.18, 0.35],
    "bounds_x_m": [0.33, 0.97],
    "bounds_y_m": [-0.52, 0.52],
    "robot_base_clearance_m": 0.13,
    "keep_out_circles": [{"center_xy_m": list(CUP_XY), "radius_m": 0.085}],
    "min_separation_m": 0.05,
    "footprint_half_length_m": {"pen_red": 0.068, "pen_blue": 0.078, "eraser": 0.0223},
    "yaw_deg": [-180.0, 180.0],
    "max_attempts": 500,
}
NAMES = ["pen_red", "pen_blue", "eraser"]
ASSIGNMENT = {"pen_red": "left_arm", "pen_blue": "right_arm", "eraser": "any"}


def _chain_spec() -> list[dict]:
    import yaml

    return yaml.safe_load(ROBOT_YAML.read_text())["chain"]


@unittest.skipIf(config is None, "PyYAML not installed")
class ChainTest(unittest.TestCase):
    def test_zero_pose_matches_usd(self):
        frames = SerialChain(_chain_spec()).link_transforms([0.0] * 6)
        for link, position in USD_LINK_POSITIONS.items():
            np.testing.assert_allclose(frames[link][:3, 3], position, atol=2e-6)
        np.testing.assert_allclose(frames[6][:3, :3], USD_LINK6_ROTATION, atol=1e-4)

    def test_base_pose_turns_the_arm(self):
        chain = SerialChain(_chain_spec(), base_pose_from_xy_yaw((0.18, 0.455, 0.0), math.radians(-90.0)))
        position, _ = chain.fk([0.0] * 6)
        # At q = 0 link6 sits 0.1355 m along the base's +X, which faces world -Y.
        np.testing.assert_allclose(position, (0.18, 0.455 - 0.1355, 0.217746), atol=1e-5)

    def test_joint1_rotates_about_vertical(self):
        chain = SerialChain(_chain_spec())
        p0, _ = chain.fk([0.0] * 6)
        p1, _ = chain.fk([math.pi / 2] + [0.0] * 5)
        self.assertAlmostEqual(p0[2], p1[2], places=6)
        self.assertAlmostEqual(np.linalg.norm(p0[:2] - (-0.00034283, -0.00098683)),
                               np.linalg.norm(p1[:2] - (-0.00034283, -0.00098683)), places=6)


class ConventionsTest(unittest.TestCase):
    def test_action_round_trip(self):
        q = np.radians([10.0, -30.0, -60.0, 20.0, -15.0, 90.0])
        action = conventions.arm_action(q, 0.02)
        q_back, gripper = conventions.action_to_command(action)
        np.testing.assert_allclose(q_back, q, atol=1e-6)
        self.assertAlmostEqual(gripper, 0.02, places=6)

    def test_single_arm_signs(self):
        q = np.radians([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        np.testing.assert_allclose(conventions.arm_action(q, 0.05)[:6], [-10, -20, 30, 40, 50, -60], atol=1e-4)
        np.testing.assert_allclose(conventions.arm_state(q, 0.05)[:6], [-10, -20, -30, -40, -50, -60], atol=1e-4)
        self.assertAlmostEqual(float(conventions.arm_action(q, 0.05)[6]), 45.0, places=4)
        self.assertAlmostEqual(float(conventions.arm_state(q, 0.025)[6]), 135.0, places=4)

    def test_feature_names(self):
        names = conventions.feature_names(["left", "right"])
        self.assertEqual(len(names), 14)
        self.assertEqual(names[0], "left_shoulder_pan.pos")
        self.assertEqual(names[-1], "right_gripper.pos")


class LayoutTest(unittest.TestCase):
    def test_layouts_respect_constraints(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            result = sample_bimanual_layout(rng, NAMES, ASSIGNMENT, BASES, LAYOUT)
            self.assertIsNotNone(result)
            layout, arms = result
            self.assertEqual(arms["pen_red"], "left_arm")
            self.assertEqual(arms["pen_blue"], "right_arm")
            footprints = {}
            for name, (xy, yaw) in layout.items():
                own = BASES[arms[name]].base_xy
                other = [b.base_xy for n, b in BASES.items() if n != arms[name]][0]
                self.assertLess(np.linalg.norm(xy - own), np.linalg.norm(xy - other))
                radius = np.linalg.norm(xy - own)
                self.assertTrue(0.18 - 1e-9 <= radius <= 0.35 + 1e-9)
                points = footprint_points(xy, yaw, LAYOUT["footprint_half_length_m"][name])
                self.assertGreaterEqual(np.min(np.linalg.norm(points - CUP_XY, axis=1)), 0.085)
                self.assertTrue(np.all((points[:, 0] >= 0.33) & (points[:, 0] <= 0.97)))
                footprints[name] = points
            for a in NAMES:
                for b in NAMES:
                    if a < b:
                        gap = np.min(np.linalg.norm(footprints[a][:, None] - footprints[b][None], axis=2))
                        self.assertGreaterEqual(gap, 0.05)

    def test_same_seed_same_layout(self):
        first = sample_bimanual_layout(np.random.default_rng(7), NAMES, ASSIGNMENT, BASES, LAYOUT)
        second = sample_bimanual_layout(np.random.default_rng(7), NAMES, ASSIGNMENT, BASES, LAYOUT)
        for name in NAMES:
            np.testing.assert_allclose(first[0][name][0], second[0][name][0])
        self.assertEqual(first[1], second[1])

    def test_work_order_groups_turns(self):
        arms = {"pen_red": "left_arm", "pen_blue": "right_arm", "eraser": "right_arm"}
        self.assertEqual(work_order(NAMES, arms), [("left_arm", ["pen_red"]), ("right_arm", ["pen_blue", "eraser"])])
        arms["eraser"] = "left_arm"
        self.assertEqual(len(work_order(NAMES, arms)), 3)


@unittest.skipIf(config is None, "PyYAML not installed")
class ConfigTest(unittest.TestCase):
    def test_layers(self):
        generate = config.load_task(mode="generate")
        evaluation = config.load_task(mode="evaluation")
        self.assertEqual(generate["sim"]["device"], "cpu")
        self.assertEqual(evaluation["sim"]["device"], "cuda:0")
        self.assertEqual(evaluation["sim"]["episode_length_s"], 90.0)  # task modes layer
        self.assertEqual(generate["sim"]["decimation"], 4)  # from defaults
        overridden = config.load_task(mode="generate", overrides={"sim": {"num_envs": 3}})
        self.assertEqual(overridden["sim"]["num_envs"], 3)
        self.assertEqual(overridden["sim"]["dt"], generate["sim"]["dt"])

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            config.load_task(mode="teleop")

    def test_task_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mine.yaml"
            path.write_text("task_name: Mine-v0\nsim: {num_envs: 2}\n")
            resolved = config.load_task(str(path), mode="view")
            self.assertEqual(resolved["task_name"], "Mine-v0")
            self.assertEqual(resolved["sim"]["num_envs"], 2)

    def test_wrist_camera_matches_rebot_mount(self):
        cam = config.load_robot()["cameras"]["left_wrist"]
        rotation = look_at_opengl(cam["eye"], cam["target"], cam["up"])
        # Columns (right, up, -forward) of the single-arm app's FrontCamera xform.
        expected = np.asarray([[0.0, 0.9048, 0.4258], [1.0, 0.0, 0.0], [0.0, 0.4258, -0.9048]])
        np.testing.assert_allclose(rotation, expected, atol=2e-3)
        np.testing.assert_allclose(quat_to_matrix(matrix_to_quat_wxyz(rotation)), rotation, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
