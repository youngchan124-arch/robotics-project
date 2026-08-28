"""MuJoCo-backed stand-in for kinematics.SOArm101, real physics + real rendered
camera frames instead of the synthetic pixel-math FakeSOArm101 used by
../sim_dry_run.py. Implements the exact same duck-typed surface
task_state_machine.py's search/coarse_center/estimate_jacobian/fine_servo/
descend_and_grasp call (gripper_xyz, get_joint_deg, send_joint_deg,
move_to_xyz, move_to_xyz_converge, nudge_xy, move_z) plus render_wrist()/
render_astra()/render_onlooker() so those real functions can run completely
unmodified against this instead of the real robot - see mujoco_dry_run.py.

Deliberately NOT using the real kinematics.py's placo/URDF-based solve_ik:
that URDF's own base frame and this MJCF scene's world frame are two
different, uncalibrated coordinate systems (this scene's `base` body sits at
MJCF world pos (0.1, 0, 0.2), not necessarily anywhere close to where the
URDF puts its own origin) - feeding one's IK solution into the other would
silently target the wrong point in space. Instead this class does its own
Cartesian (position-only, same as the real IK_ORIENTATION_WEIGHT=0.0 policy)
damped-least-squares IK directly against THIS model's own `graspframe` site
Jacobian (mj_jacSite) - fully self-consistent within one frame, and it's real
forward kinematics (MuJoCo's, not a synthetic model) either way.

Joint-degree convention: get_joint_deg()/send_joint_deg() use THIS class's
own self-consistent units (arm joints in degrees of ITS OWN qpos, gripper in
0-100% matching the real convention gripper.py expects: 100=open, 0=closed).
These never need to match the real robot's own degree calibration - every
caller (gripper.py, in particular) only ever reads a value back from
get_joint_deg() and later re-sends it via send_joint_deg() unchanged, or
moves the gripper element by a relative amount - it's never compared against
a real-hardware-calibrated absolute number.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

import config
from kinematics import CollisionDetected

XML_PATH = Path(__file__).resolve().parent / "scene.xml"

RENDER_W, RENDER_H = config.FRAME_W, config.FRAME_H  # 640x480 - matches perception.py's assumptions

# DLS IK tuning - found by direct experiment against this scene (see the
# session notes/memory, not blind guesses): a 300-iteration, 0.05rad-per-step
# solver converges to <3mm within 20-25 iterations for targets across this
# workspace (cube/bin both tested), with no joint anywhere near its range
# limit - see this file's own dev-time probe script for the numbers.
IK_DAMPING = 0.02
IK_MAX_STEP_RAD = 0.05
IK_POS_TOL_M = 0.002

# Real-hardware STALL_CHECK_EVERY/STALL_CONSECUTIVE cadence (config.py)
# reused directly for the collision/stall check's timing - see
# move_to_xyz's docstring for why the SIGNAL itself (real contact + no
# progress) had to differ from the real robot's joint-lag proxy, even though
# the check cadence carries over fine.
STALL_CHECK_EVERY = config.STALL_CHECK_EVERY
STALL_CONSECUTIVE = config.STALL_CONSECUTIVE

GRIPPER_CLOSED_RAD = -0.17453  # matches scene.xml's gripper actuator ctrlrange lower bound
GRIPPER_OPEN_RAD = 1.74533  # ... and its upper bound


class MujocoSOArm101:
    def __init__(self, xml_path: Path = XML_PATH, ready_xyz=(0.28, 0.0, 0.34)):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=RENDER_H, width=RENDER_W)

        self.site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "graspframe")
        arm_joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in config.ARM_JOINTS]
        self.arm_qposadr = np.array([self.model.jnt_qposadr[j] for j in arm_joint_ids])
        self.arm_dofadr = np.array([self.model.jnt_dofadr[j] for j in arm_joint_ids])
        self.arm_range = np.array([self.model.jnt_range[j] for j in arm_joint_ids])
        self.arm_actuator_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in config.ARM_JOINTS])

        gripper_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self.gripper_qposadr = self.model.jnt_qposadr[gripper_joint_id]
        self.gripper_actuator_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")

        self.marker_mocap_id = self.model.body_mocapid[self._id(mujoco.mjtObj.mjOBJ_BODY, "grasp_marker")]

        # Geoms belonging to the arm/gripper itself (not the cube, bin, table
        # or floor) - move_to_xyz's stall check needs this to tell "the
        # gripper hit something" apart from "the cube happens to be resting
        # on the table" (data.ncon alone is >0 essentially always, since that
        # resting contact is there from frame 1 - found by actually printing
        # contact geom names against a real descend, see that method's
        # docstring).
        robot_body_names = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]
        self.robot_geom_ids = set()
        for name in robot_body_names:
            bid = self._id(mujoco.mjtObj.mjOBJ_BODY, name)
            start = self.model.body_geomadr[bid]
            self.robot_geom_ids.update(range(start, start + self.model.body_geomnum[bid]))

        # Start from a real IK solution for ready_xyz (a hover pose above the
        # table with clearance) rather than the model's all-zero default,
        # which sits with the arm stretched straight up - see this file's
        # dev-time probe for why (0.28, 0, 0.34) was picked: roughly centered
        # between the scene's cube and bin positions, well clear of the
        # table surface (z=0.2).
        self._solve_ik_inplace(np.array(ready_xyz), max_iters=300)
        self.data.ctrl[self.arm_actuator_ids] = self.data.qpos[self.arm_qposadr]
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_OPEN_RAD
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    def _id(self, kind, name: str) -> int:
        i = mujoco.mj_name2id(self.model, kind, name)
        if i < 0:
            raise ValueError(f"MuJoCo scene missing {kind}/{name!r}")
        return i

    # --- Cartesian IK -------------------------------------------------------
    def _jac_pos(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return jacp[:, self.arm_dofadr]

    def _solve_ik_inplace(self, target_xyz: np.ndarray, max_iters: int = 60) -> bool:
        """Moves data.qpos directly (no physics, no actuators) to a
        position-only IK solution - used only for the one-shot initial pose
        in __init__. Real moves go through move_to_xyz below, which drives
        actuators and steps physics instead of teleporting qpos."""
        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            err = target_xyz - self.data.site_xpos[self.site_id]
            if float(np.linalg.norm(err)) < IK_POS_TOL_M:
                return True
            J = self._jac_pos()
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING**2 * np.eye(3), err)
            dq = np.clip(dq, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)
            q = np.clip(self.data.qpos[self.arm_qposadr] + dq, self.arm_range[:, 0], self.arm_range[:, 1])
            self.data.qpos[self.arm_qposadr] = q
        return False

    # --- SOArm101-compatible surface ----------------------------------------
    def gripper_xyz(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def get_joint_deg(self) -> np.ndarray:
        arm_deg = np.degrees(self.data.qpos[self.arm_qposadr])
        gripper_rad = self.data.qpos[self.gripper_qposadr]
        gripper_pct = 100.0 * (gripper_rad - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
        return np.concatenate([arm_deg, [np.clip(gripper_pct, 0.0, 100.0)]])

    def send_joint_deg(self, joint_deg: np.ndarray) -> None:
        arm_rad = np.clip(np.radians(joint_deg[: len(config.ARM_JOINTS)]), self.arm_range[:, 0], self.arm_range[:, 1])
        self.data.ctrl[self.arm_actuator_ids] = arm_rad
        pct = float(np.clip(joint_deg[-1], 0.0, 100.0))
        self.data.ctrl[self.gripper_actuator_id] = GRIPPER_CLOSED_RAD + (pct / 100.0) * (
            GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD
        )
        for _ in range(5):  # lets the PD controller start catching up, same role as a real send_action's async motion
            mujoco.mj_step(self.model, self.data)

    def move_to_xyz(self, xyz, steps=25, step_delay_s=0.0, enforce_cap=True, stall_check=True) -> np.ndarray:
        """Velocity-resolved DLS IK toward a fixed target, `steps` iterations
        max, each one driving position actuators (not teleporting qpos) and
        stepping real physics - genuinely resisted by real contacts, unlike
        the numpy-only sim_dry_run.py's collide_after_n_moves flag.

        Collision signal: NOT joint-command lag (tried first, reusing config's
        real STALL_THRESHOLD_DEG/STALL_CONSECUTIVE directly since that's the
        real robot's own proxy for "hit something") - measured directly
        against a real table/cube contact in this scene and it never tripped.
        Root cause: this solver recomputes a fresh, already-small step from
        the CURRENT actual position every iteration (unlike the real robot's
        fixed-target interpolation), so even fully blocked by contact, next
        iteration's commanded delta stays tiny - lag never accumulates past a
        few degrees even while genuinely stuck (measured ~3-4deg max against
        a real contact, well under the 10deg threshold). MuJoCo knows
        whether a genuine contact exists (`data.ncon`), so that's used
        instead: no progress in Cartesian error for STALL_CONSECUTIVE checks
        AND an active contact INVOLVING THE ROBOT (self.robot_geom_ids) =
        stuck against something real. Plain `data.ncon > 0` alone isn't
        enough either - the cube resting on the table is itself a permanent
        contact from frame 1, unrelated to anything the gripper is doing,
        and made every move look "in contact" regardless of whether the
        gripper was anywhere near it (found by printing contact geom names
        against a real descend). No-progress alone isn't enough either (a
        target just outside reach with no real contact - e.g. near a joint
        limit - should end quietly at its best-effort position, not fake a
        collision)."""
        target = np.array(xyz, dtype=float)
        stall_count = 0
        best_err = float("inf")
        last_good = self.data.qpos[self.arm_qposadr].copy()
        for i in range(1, steps + 1):
            mujoco.mj_forward(self.model, self.data)
            cur = self.data.site_xpos[self.site_id]
            err = target - cur
            err_norm = float(np.linalg.norm(err))
            if err_norm < IK_POS_TOL_M:
                break
            J = self._jac_pos()
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING**2 * np.eye(3), err)
            dq = np.clip(dq, -IK_MAX_STEP_RAD, IK_MAX_STEP_RAD)
            q_target = np.clip(self.data.qpos[self.arm_qposadr] + dq, self.arm_range[:, 0], self.arm_range[:, 1])
            self.data.ctrl[self.arm_actuator_ids] = q_target
            for _ in range(8):
                mujoco.mj_step(self.model, self.data)

            if not stall_check or i % STALL_CHECK_EVERY != 0:
                continue
            in_contact = any(
                c.geom1 in self.robot_geom_ids or c.geom2 in self.robot_geom_ids
                for c in self.data.contact[: self.data.ncon]
            )
            if err_norm < best_err - 0.001:  # real progress (>1mm) since the last check
                best_err, stall_count, last_good = err_norm, 0, self.data.qpos[self.arm_qposadr].copy()
            elif in_contact:
                stall_count += 1
            else:
                stall_count = 0  # not improving, but nothing is actually blocking it either - keep trying
            if stall_count >= STALL_CONSECUTIVE:
                self.data.ctrl[self.arm_actuator_ids] = last_good
                for _ in range(15):
                    mujoco.mj_step(self.model, self.data)
                raise CollisionDetected(
                    f"[mujoco] move_to_xyz aborted: {self.data.ncon} active contact(s), no progress "
                    f"(best {best_err * 1000:.1f}mm) for {STALL_CONSECUTIVE} consecutive checks "
                    "(real contact, not simulated) - retreated to last known-good pose."
                )
        return self.gripper_xyz()

    def move_to_xyz_converge(self, xyz, tolerance_m=0.005, max_iters=15) -> np.ndarray:
        target = np.array(xyz, dtype=float)
        for _ in range(max_iters):
            cur = self.gripper_xyz()
            if float(np.linalg.norm(target - cur)) <= tolerance_m:
                return cur
            self.move_to_xyz(xyz, steps=18, enforce_cap=False)
        return self.gripper_xyz()

    def nudge_xy(self, dx: float, dy: float, steps=6, step_delay_s=0.0, stall_check=True) -> np.ndarray:
        cur = self.gripper_xyz()
        return self.move_to_xyz((cur[0] + dx, cur[1] + dy, cur[2]), steps=steps, stall_check=stall_check)

    def move_z(self, dz: float, steps=10, step_delay_s=0.0, stall_check=True) -> np.ndarray:
        cur = self.gripper_xyz()
        return self.move_to_xyz((cur[0], cur[1], cur[2] + dz), steps=steps, stall_check=stall_check)

    # --- Rendering (real pixels - perception.py's actual cv2 HSV code runs on these) ---
    def _render(self, camera_name: str) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[self.marker_mocap_id] = self.data.site_xpos[self.site_id]
        self._renderer.update_scene(self.data, camera=camera_name)
        rgb = self._renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # perception.py assumes cv2's BGR convention

    def render_wrist(self) -> np.ndarray:
        return self._render("wrist_cam")

    def render_astra(self) -> np.ndarray:
        return self._render("astra_cam")

    def render_onlooker(self) -> np.ndarray:
        return self._render("onlooker_cam")
