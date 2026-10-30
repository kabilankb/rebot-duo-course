"""Scripted-expert demonstrations for the reBot stationery-sorting task.

Pure numpy; the simulator is reached only through callbacks, so the expert can
be tested without Isaac Sim and reused with other arms.
"""

from .expert import ExpertTuning, GripperGeometry, SceneGeometry, ScriptedPickPlace
from .kinematics import point_residual, quat_to_matrix, solve_ik
from .workspace import footprint_points, sample_workspace_layout

__all__ = [
    "ExpertTuning",
    "GripperGeometry",
    "SceneGeometry",
    "ScriptedPickPlace",
    "footprint_points",
    "point_residual",
    "quat_to_matrix",
    "sample_workspace_layout",
    "solve_ik",
]
__version__ = "0.1.0"
