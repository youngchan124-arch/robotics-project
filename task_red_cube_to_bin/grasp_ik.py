"""Fixed-roll 4-DOF position IK - a fresh, standalone solver for the
"position + grasp yaw" problem, per the user's 2026-08-28 explicit
direction: NOT a post-hoc override of wrist_roll on top of the existing
5-DOF solve_ik() (kinematics.py) result, and NOT touching kinematics.py,
config.py, or any script actually run against real hardware so far - this
is a new, separate module.

Why the override idea was rejected: measured live via kin.forward_kinematics
(read-only use of kinematics.py's build_kinematics(), no edits) that
wrist_roll is NOT a pure in-place rotation of the gripper - sweeping
wrist_roll -90..135deg at a fixed arm pose moved the gripper's y/z position
by up to ~13mm (x stayed ~fixed, ~0.03mm swing). The gripper's TCP sits
offset from the wrist_roll joint's own rotation axis, so it traces a small
arc as roll changes. Silently overwriting wrist_roll after position-only
IK would reintroduce exactly this kind of position error - unacceptable
against PHYSICAL_TOLERANCE_M=0.004 (config.py) that the existing
visual-servo work already needed for a real grasp.

Why not just raise kinematics.py's IK_ORIENTATION_WEIGHT instead: already
tried and documented as a real failure (kinematics.py's own docstring) -
any nonzero weight blew position error up to 16-253mm across the
workspace. That's placo's 5-joint solver being asked to satisfy a FULL
3-DOF orientation target (roll+pitch+yaw all at once) simultaneously with
3-DOF position on only 5 joints - overconstrained (6 targets, 5 DOF) almost
everywhere, so the least-squares compromise sacrifices position badly.

This module sidesteps that by only ever asking for 1 DOF of orientation
(yaw, i.e. wrist_roll itself, held CONSTANT rather than solved for) plus
3-DOF position, solved on the 4 remaining joints
(shoulder_pan/shoulder_lift/elbow_flex/wrist_flex) via a from-scratch
finite-difference-Jacobian damped-least-squares loop - not placo's own
solver at all, and not its orientation_weight machinery. 4 joints for 3
position constraints is comfortably determined (1 redundant DOF, no
overconstraint), and wrist_roll being fixed throughout means the known
roll-offset effect above is handled exactly (it's baked into the FK used
at every iteration), not approximated or ignored.
"""

from __future__ import annotations

import numpy as np

import config
from kinematics import build_kinematics, gripper_position

FREE_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]  # wrist_roll held fixed, not solved
_FREE_IDX = [config.ARM_JOINTS.index(j) for j in FREE_JOINTS]
_ROLL_IDX = config.ARM_JOINTS.index("wrist_roll")

JAC_EPS_DEG = 0.5  # finite-difference perturbation size
DLS_DAMPING = 0.02  # Levenberg-Marquardt-style damping, avoids blow-ups near singularities
MAX_ITERS = 25
POS_TOL_M = 0.001

# 2026-08-28: a real live run seeded from the arm's actual resting pose
# converged to a 35mm-off solution with elbow_flex pinned exactly at its
# 98deg limit - a single-seed gradient/DLS solve can get stuck against a
# joint clamp and never recover, since the finite-difference Jacobian at a
# clamped boundary doesn't see a way out. Reproduced systematically (not a
# one-off): 21/60 random within-limits seeds failed to converge under 5mm
# for one fixed real target, several by 100-300mm. See RESTART_SEEDS below.
CONVERGENCE_TOL_M = 0.005
# A handful of spread-out, hand-picked seeds (not random - deterministic
# and reproducible) tried as fallbacks when the caller's own seed doesn't
# converge. All-zeros plus seeds leaning into each extreme of
# shoulder_lift/elbow_flex (the two joints observed pinning most often)
# covers the failure modes actually seen in the stress test above.
RESTART_SEEDS_DEG = [
    (0.0, 0.0, 0.0, 0.0),
    (0.0, -60.0, 80.0, 0.0),
    (0.0, 60.0, -80.0, 0.0),
    (0.0, -30.0, 40.0, -20.0),
]


def _full_joints(free_deg: np.ndarray, roll_deg: float) -> np.ndarray:
    """free_deg is the 4 FREE_JOINTS values, in FREE_JOINTS order -
    reassembles the full 5-vector kinematics.py's ARM_JOINTS order expects."""
    full = np.zeros(5)
    full[_FREE_IDX] = free_deg
    full[_ROLL_IDX] = roll_deg
    return full


def _solve_from_seed(kin, free_seed: np.ndarray, target: np.ndarray, target_roll_deg: float,
                      max_iters: int = MAX_ITERS) -> np.ndarray:
    """Single-seed DLS solve (the original solve_fixed_roll_ik body) -
    factored out so solve_fixed_roll_ik below can retry it from several
    seeds without duplicating the loop."""
    free = free_seed.copy()

    def fk(free_vec: np.ndarray) -> np.ndarray:
        full5 = _full_joints(free_vec, target_roll_deg)
        return gripper_position(kin, np.concatenate([full5, [0.0]]))  # [0.0]: unused gripper slot, see gripper_position

    for _ in range(max_iters):
        pos = fk(free)
        err = target - pos
        if np.linalg.norm(err) <= POS_TOL_M:
            break
        J = np.zeros((3, 4))
        for i in range(4):
            perturbed = free.copy()
            perturbed[i] += JAC_EPS_DEG
            J[:, i] = (fk(perturbed) - pos) / np.radians(JAC_EPS_DEG)
        JT = J.T
        step = JT @ np.linalg.solve(J @ JT + DLS_DAMPING * np.eye(3), err)
        free = free + np.degrees(step)
        for i, name in enumerate(FREE_JOINTS):
            lo, hi = config.JOINT_LIMITS_DEG[name]
            free[i] = np.clip(free[i], lo, hi)

    return free


def solve_fixed_roll_ik(kin, current_joint_deg: np.ndarray, target_xyz: tuple[float, float, float],
                         target_roll_deg: float, max_iters: int = MAX_ITERS) -> np.ndarray:
    """Returns joint degrees (5,) in config.ARM_JOINTS order: wrist_roll is
    exactly target_roll_deg (not solved-for, not overridden after the
    fact), the other 4 joints are numerically solved to place the gripper
    at target_xyz given that fixed roll. current_joint_deg seeds the free
    joints' starting guess.

    Multi-restart: if that seed's solve doesn't converge within
    CONVERGENCE_TOL_M (a real live run showed this genuinely happens -
    ~1/3 of realistic starting poses in a stress test, from a joint getting
    stuck against its own limit - see CONVERGENCE_TOL_M's comment), retries
    from RESTART_SEEDS_DEG and keeps whichever result has the lowest
    residual, including the original seed's own result. Does NOT raise on
    persistent failure - always returns its best attempt; callers that need
    to refuse an unreachable move should check the residual themselves via
    solved_position_error_m (this module's own convention, e.g.
    zeroshot_pick.py aborts before moving if it's too large)."""
    target = np.array(target_xyz, dtype=float)
    best_free = _solve_from_seed(kin, current_joint_deg[_FREE_IDX], target, target_roll_deg, max_iters)
    best_err = np.linalg.norm(target - gripper_position(
        kin, np.concatenate([_full_joints(best_free, target_roll_deg), [0.0]])))

    if best_err > CONVERGENCE_TOL_M:
        for seed in RESTART_SEEDS_DEG:
            free = _solve_from_seed(kin, np.array(seed), target, target_roll_deg, max_iters)
            err = np.linalg.norm(target - gripper_position(
                kin, np.concatenate([_full_joints(free, target_roll_deg), [0.0]])))
            if err < best_err:
                best_free, best_err = free, err
            if best_err <= CONVERGENCE_TOL_M:
                break

    return _full_joints(best_free, target_roll_deg)


def solved_position_error_m(kin, joint_deg_5: np.ndarray, target_xyz: tuple[float, float, float]) -> float:
    pos = gripper_position(kin, np.concatenate([joint_deg_5, [0.0]]))
    return float(np.linalg.norm(np.array(target_xyz) - pos))
