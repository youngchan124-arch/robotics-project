"""Gripper open/close and grasp-success verification.

Per the spec's risk note (§6): this hardware has no current/torque feedback
readable through the driver in use here, so "성공 판정이 사실상 불가"
doesn't actually apply - position feedback alone is enough. Closing on
nothing reaches (near) the same fully-closed reading every time; closing on
a real object stops well short of that, wedged against it. This was a real
false positive earlier today (a run reported success purely because descend
raised no collision, when the user watching live confirmed nothing was
grasped) - "did descend hit something" and "is something actually between
the jaws right now" are different facts, and only checking the gripper's own
final resting position answers the second one.

Wrist-cam color-presence as a second check (the spec's suggested backup) is
NOT implemented - it hasn't been validated against real hardware, and the
gripper-position check alone was enough to catch the one real false positive
seen today. Left as a clearly-marked extension point, not built speculatively.
"""

from __future__ import annotations

import time

import config
from kinematics import SOArm101


def set_pct(arm: SOArm101, pct: float, steps: int = 10, step_delay_s: float = 0.03) -> None:
    """pct: 0-100, this joint's actual unit (100=open, 0=closed - confirmed
    against real hardware). Refuses outright if the single-call delta
    exceeds MAX_MOVE_DELTA_DEG - use set_pct_converge for a large jump
    (e.g. fully closed -> fully open, a ~100-point delta)."""
    current = arm.get_joint_deg()
    delta = abs(pct - current[-1])
    if delta > config.MAX_MOVE_DELTA_DEG:
        raise RuntimeError(f"gripper.set_pct refused: {delta:.1f} delta exceeds {config.MAX_MOVE_DELTA_DEG} cap.")
    target = current.copy()
    target[-1] = pct
    for i in range(1, steps + 1):
        interp = current + (target - current) * (i / steps)
        arm.send_joint_deg(interp)
        time.sleep(step_delay_s)


def set_pct_converge(arm: SOArm101, pct: float, tolerance: float = 3.0, max_iters: int = 15,
                      steps: int = 8, step_delay_s: float = 0.03) -> float:
    """Retry-then-recheck for a target more than MAX_MOVE_DELTA_DEG away -
    re-reads the actual position each iteration and re-issues a fresh
    (<=35pt) delta. Returns the final actual position."""
    for _ in range(max_iters):
        current = arm.get_joint_deg()[-1]
        if abs(current - pct) <= tolerance:
            return current
        step_target = current + max(-35.0, min(35.0, pct - current))
        set_pct(arm, step_target, steps=steps, step_delay_s=step_delay_s)
        time.sleep(0.1)
    return arm.get_joint_deg()[-1]


def open_gripper(arm: SOArm101) -> float:
    return set_pct_converge(arm, 100.0)


def close_gripper(arm: SOArm101) -> float:
    return set_pct_converge(arm, 0.0)


def is_grasp_success(final_pct: float) -> bool:
    """final_pct: the return value of close_gripper() - the gripper's actual
    resting position after commanding fully closed. See module docstring."""
    return final_pct > config.GRIPPER_EMPTY_CLOSED_PCT + config.GRASP_DETECT_MARGIN_PCT
