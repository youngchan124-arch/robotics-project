"""Standalone validation for grasp_ik.py - no camera, no robot, same
test-before-hardware convention as the sibling sim_dry_run.py files. Uses
kinematics.py's real URDF-backed RobotKinematics for forward-kinematics
checks (read-only - never calls anything that touches a live robot).

Run: `uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin/test_grasp_ik.py`
from ~/lerobot.
"""

from __future__ import annotations

import numpy as np

import config
from grasp_ik import solve_fixed_roll_ik, solved_position_error_m
from kinematics import build_kinematics

# A handful of targets roughly in the real working area this project has
# actually used (see config.py's SEARCH_HOVER_XYZ / homography robot_points
# corners: x in [0.18, 0.28], y in [-0.08, 0.08]) plus TABLE_Z-ish heights.
TARGETS_XYZ = [
    (0.20, 0.00, 0.05),
    (0.25, 0.05, 0.03),
    (0.22, -0.06, 0.08),
]
TARGET_ROLLS_DEG = [-90.0, -30.0, 0.0, 45.0, 90.0]
POS_TOL_M = 0.005  # looser than the solver's own internal POS_TOL_M - end-to-end acceptance bound


def run() -> bool:
    kin = build_kinematics()
    seed = np.zeros(6)  # neutral start, real callers would seed with the arm's actual current pose
    all_ok = True

    for xyz in TARGETS_XYZ:
        for roll in TARGET_ROLLS_DEG:
            joints5 = solve_fixed_roll_ik(kin, seed, xyz, roll)
            err = solved_position_error_m(kin, joints5, xyz)
            roll_exact = abs(joints5[config.ARM_JOINTS.index("wrist_roll")] - roll) < 1e-9
            ok = err <= POS_TOL_M and roll_exact
            all_ok &= ok
            status = "OK  " if ok else "FAIL"
            print(f"{status} target={xyz} roll={roll:6.1f} -> pos_err={err*1000:.2f}mm roll_exact={roll_exact}")

    print("PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
