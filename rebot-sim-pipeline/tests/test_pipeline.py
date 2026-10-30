from __future__ import annotations

import math
import unittest

import numpy as np

from rebot_sim_pipeline import (
    GripperGeometry,
    SceneGeometry,
    ScriptedPickPlace,
    footprint_points,
    point_residual,
    quat_to_matrix,
    sample_workspace_layout,
    solve_ik,
)
from toy_world import (
    FINGERTIP,
    GRASP_OFFSET,
    LOWER,
    UPPER,
    FakeWorld,
    toy_fk,
    yaw_rotation,
)

WORKSPACE = {
    "robot_base_xy_m": [0.0, 0.455],
    "center_radius_m": [0.18, 0.35],
    "bounds_x_m": [-0.34, 0.34],
    "bounds_y_m": [-0.09, 0.53],
    "robot_base_clearance_m": 0.13,
    "keep_out_circles": [{"center_xy_m": [-0.16, 0.12], "radius_m": 0.085}],
    "min_separation_m": 0.05,
    "footprint_half_length_m": {"pen_red": 0.068, "pen_blue": 0.078, "eraser": 0.0223},
    "yaw_deg": [-180.0, 180.0],
    "max_attempts": 500,
}
NAMES = ["pen_red", "pen_blue", "eraser"]
PEN_HALF_LENGTHS = {"pen_red": 0.068, "pen_blue": 0.078}
CUP_XY = np.asarray([-0.16, 0.12])


class QuaternionTest(unittest.TestCase):
    def test_identity_and_z_rotation(self) -> None:
        np.testing.assert_allclose(quat_to_matrix([1, 0, 0, 0]), np.eye(3))
        half = math.radians(90) / 2
        rotation = quat_to_matrix([math.cos(half), 0, 0, math.sin(half)])
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_normalizes(self) -> None:
        np.testing.assert_allclose(quat_to_matrix([2, 0, 0, 0]), np.eye(3))


class InverseKinematicsTest(unittest.TestCase):
    def test_reaches_table_point_pointing_down(self) -> None:
        target = np.asarray([0.12, 0.2, 0.01])
        point = point_residual(FINGERTIP, target)

        def residual(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
            down = rotation @ [0, 0, 1] + [0, 0, 1]
            return np.concatenate((point(position, rotation), 0.3 * down))

        seeds = [np.radians([20, -60, -60, 0, 0, 0]), np.radians([20, -30, -90, 0, 0, 0])]
        q, error = solve_ik(toy_fk, residual, seeds, LOWER, UPPER)
        self.assertLess(error, 1e-3)
        self.assertTrue(np.all(q >= LOWER) and np.all(q <= UPPER))

    def test_respects_joint_limits_when_unreachable(self) -> None:
        residual = point_residual(FINGERTIP, np.asarray([0.0, -1.0, 0.0]))
        q, error = solve_ik(toy_fk, residual, [np.zeros(6)], LOWER, UPPER)
        self.assertGreater(error, 0.1)
        self.assertTrue(np.all(q >= LOWER - 1e-12) and np.all(q <= UPPER + 1e-12))


class WorkspaceSamplingTest(unittest.TestCase):
    def test_layouts_satisfy_constraints(self) -> None:
        base = np.asarray(WORKSPACE["robot_base_xy_m"])
        for seed in range(300):
            layout = sample_workspace_layout(np.random.default_rng(seed), NAMES, WORKSPACE)
            self.assertIsNotNone(layout, f"seed {seed}")
            points = {
                name: footprint_points(xy, yaw, WORKSPACE["footprint_half_length_m"][name])
                for name, (xy, yaw) in layout.items()
            }
            for name, (xy, _yaw) in layout.items():
                radius = np.linalg.norm(xy - base)
                self.assertGreaterEqual(radius, 0.18 - 1e-9)
                self.assertLessEqual(radius, 0.35 + 1e-9)
                self.assertLess(xy[1], base[1], "objects are in front of the robot")
                self.assertGreaterEqual(np.min(np.linalg.norm(points[name] - CUP_XY, axis=1)), 0.085)
            for index, first in enumerate(NAMES):
                for second in NAMES[index + 1 :]:
                    gap = np.min(np.linalg.norm(points[first][:, None] - points[second][None], axis=2))
                    self.assertGreaterEqual(gap, 0.05)

    def test_deterministic_per_seed(self) -> None:
        first = sample_workspace_layout(np.random.default_rng(7), NAMES, WORKSPACE)
        second = sample_workspace_layout(np.random.default_rng(7), NAMES, WORKSPACE)
        for name in NAMES:
            np.testing.assert_array_equal(first[name][0], second[name][0])

    def test_impossible_constraints_return_none(self) -> None:
        config = dict(WORKSPACE, min_separation_m=1.0, max_attempts=20)
        self.assertIsNone(sample_workspace_layout(np.random.default_rng(0), NAMES, config))


class ExpertEndToEndTest(unittest.TestCase):
    """Runs the whole program against the toy arm and a fake carrying world."""

    def _run(self, seed: int) -> tuple[FakeWorld, ScriptedPickPlace]:
        layout = sample_workspace_layout(np.random.default_rng(seed), NAMES, WORKSPACE)
        heights = {"pen_red": 0.004, "pen_blue": 0.004, "eraser": 0.005}
        world = FakeWorld(
            {
                name: (np.asarray([xy[0], xy[1], heights[name]]), yaw_rotation(yaw))
                for name, (xy, yaw) in layout.items()
            }
        )
        expert = ScriptedPickPlace(
            fk=toy_fk,
            object_pose=world.object_pose,
            is_attached=world.is_attached,
            arm_lower=LOWER,
            arm_upper=UPPER,
            gripper=GripperGeometry(
                grasp_offset=GRASP_OFFSET,
                fingertip=FINGERTIP,
                finger_axis=np.asarray([0.0, 1.0, 0.0]),
            ),
            geometry=SceneGeometry(
                robot_base_xy=np.asarray([0.0, 0.455]),
                cup_center_xy=CUP_XY,
                cup_top_z=0.084,
                cup_floor_z=0.008,
                pen_half_lengths=PEN_HALF_LENGTHS,
            ),
            order=["pen_blue", "pen_red", "eraser"],
            open_m=0.05,
            closed_m=0.0,
            dt=1.0 / 60.0,
            start_q=np.zeros(6),
            log=lambda _message: None,
        )
        steps = 0
        while (command := expert.next_command()) is not None:
            q, gripper = command
            self.assertTrue(np.all(q >= LOWER - 1e-9) and np.all(q <= UPPER + 1e-9))
            world.step(q, gripper)
            steps += 1
            self.assertLess(steps, 20_000, "program did not terminate")
        return world, expert

    def test_places_everything_in_the_cup(self) -> None:
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                world, expert = self._run(seed)
                self.assertEqual(sorted(expert.grasped), sorted(NAMES))
                self.assertEqual(sorted(world.released_poses), sorted(NAMES))
                for name, (position, rotation) in world.released_poses.items():
                    self.assertLess(np.linalg.norm(position[:2] - CUP_XY), 0.03, name)
                    if name in PEN_HALF_LENGTHS:
                        # Released upright, bottom just above the cup floor.
                        self.assertGreater(abs(rotation[2, 0]), 0.95, name)
                        bottom = position[2] - PEN_HALF_LENGTHS[name] * abs(rotation[2, 0])
                        self.assertLess(abs(bottom - 0.012), 0.01, name)
                    else:
                        self.assertGreater(position[2], 0.084, name)


if __name__ == "__main__":
    unittest.main()
