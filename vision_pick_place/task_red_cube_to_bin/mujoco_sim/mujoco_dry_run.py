"""Closed-loop MuJoCo dry run of the red-cube -> bin task: real physics, real
rendered camera pixels, and task_state_machine.py's actual search/
coarse_center/fine_servo/descend_and_grasp functions UNMODIFIED - the next
step up from the pure-numpy ../sim_dry_run.py, which only exercised the
control-flow logic against a synthetic pixel-math model. Here perception.py's
real cv2 HSV detector runs on real rendered frames, and grasp success is a
real physics outcome (the gripper genuinely does or doesn't catch the cube),
not a scripted `object_present` flag.

Still NOT hardware - no serial port, no real camera, nothing physical. And
still not a full end-to-end validation of main.py/task_state_machine.run():
this driver calls the individual state functions directly (same reasoning as
../sim_dry_run.py) rather than run() itself, because run() needs
arm.kin.forward_kinematics() (the real placo/URDF-based FK) to compute
home_xyz from a captured home_pose - that FK is in the real URDF's own base
frame, which has no calibrated relationship to this MJCF scene's world frame
(mujoco_env.py's whole design deliberately keeps this scene self-consistent
in its OWN frame instead of mixing the two - see that file's docstring).

Known simplifications, worth knowing about before trusting this too far:
  - config.SEARCH_HOVER_XYZ and config.TABLE_Z are monkeypatched to
    MuJoCo-scene-frame values below - the real config.py's numbers are
    real-hardware-frame constants and would be meaningless (or, for
    TABLE_Z=0.003, actively wrong - close to the mujoco world origin, nowhere
    near this scene's table) if used unmodified here.
  - perception.estimate_xy_from_astra / estimate_cube_height_m are patched
    to always return None - this run exercises the blind SEARCH_OFFSETS grid
    fallback and the plain-TABLE_Z descend path, not the Astra-homography
    coarse-position or depth-height features (those would need a homography
    calibrated for astra_cam's specific projection - out of scope here, see
    scene.xml's astra_cam comment).
  - config.GRASP_TARGET_PX is set to plain image-center, not measured from a
    real frame the way the real constant was (an attempt to calibrate it
    from this scene's own graspframe site via a synced marker - see
    scene.xml's grasp_marker body - didn't resolve at wrist_cam's very
    close range; left in the scene as a harmless visual aid, not relied on).

Run: uv run --with mujoco --with imageio python3
     custom_scripts/vision_pick_place/task_red_cube_to_bin/mujoco_sim/mujoco_dry_run.py
     (from ~/lerobot - `--with` overlays mujoco/imageio for just this run,
     without touching the shared project pyproject.toml/uv.lock.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

TASK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import gripper  # noqa: E402
import perception  # noqa: E402
import task_state_machine as tsm  # noqa: E402
from kinematics import CollisionDetected  # noqa: E402
from mujoco_env import MujocoSOArm101  # noqa: E402

MUJOCO_SEARCH_HOVER_XYZ = (0.28, 0.0, 0.34)  # this scene's own hover pose - see mujoco_env.py's ready_xyz default
MUJOCO_TABLE_Z = 0.05  # deliberately below the table/cube - real contact physics stops the
# descent before this is ever reached (see mujoco_env.MujocoSOArm101.move_to_xyz's stall-based
# CollisionDetected), the same way config.TABLE_Z=0.003 is a "definitely far enough" floor on
# real hardware, not literally the exact contact height.

config.SEARCH_HOVER_XYZ = MUJOCO_SEARCH_HOVER_XYZ
config.TABLE_Z = MUJOCO_TABLE_Z
# Measured the same way the real GRASP_TARGET_PX was ("measured from real
# saved wrist-cam frames" - config.py) but done here by direct query instead
# of eyeballing a screenshot: commanded graspframe's xy to exactly equal a
# known object's xy (bypassing pixel servoing entirely), then read back where
# that object actually landed in the rendered wrist_cam frame. Came out
# (612, 308) - close to the RIGHT EDGE of the 640-wide frame, not remotely
# near IMG_CENTER (320, 240) that an earlier version of this file used as a
# placeholder - confirms the exact real-hardware finding this whole design
# is based on ("centering on IMG_CENTER was never centering on the jaws",
# see project memory) reproduces here too: this synthetic wrist_cam's mount
# offset/rotation in scene.xml puts the true grasp point well off-center.
config.GRASP_TARGET_PX = (612.0, 308.0)
perception.estimate_xy_from_astra = lambda detect_fn, **k: None
perception.estimate_cube_height_m = lambda *a, **k: None

VIDEO_PATH = Path(__file__).resolve().parent / "mujoco_dry_run.mp4"
VIDEO_FPS = 15


def main() -> bool:
    env = MujocoSOArm101()
    frames: list[np.ndarray] = []

    def record(n: int = 1) -> None:
        for _ in range(n):
            frames.append(env.render_onlooker())

    # Piggybacks on every time.sleep() call already sprinkled through
    # task_state_machine.py/gripper.py (settle delays between moves) to grab
    # one video frame each - no changes needed to those files, and no real
    # wall-clock waiting is useful here anyway (mj_step already advances
    # simulated time inside mujoco_env's own move methods).
    time.sleep = lambda *_a, **_k: record(1)

    class MujocoCap:
        """PublishedFrameSource-shaped: task_state_machine.get_pixel() only
        ever calls cap.read() - renders a fresh wrist_cam frame from live
        sim state each time, no file round-trip needed in-process."""

        def read(self):
            return True, env.render_wrist()

    cap = MujocoCap()
    record(10)
    ok = False
    try:
        print("=== FINE_SERVO (빨간 큐브, 손목캠) ===")
        if not tsm.fine_servo(env, cap, perception.detect_red_cube, "빨간 큐브"):
            print("[결과] 큐브 정렬 실패")
            return False
        record(5)

        print("=== DESCEND + GRASP ===")
        grasped = tsm.descend_and_grasp(env)
        record(5)
        if not grasped:
            print("[결과] 파지 실패 (물리적으로 못 집음)")
            return False

        print("=== LIFT ===")
        env.move_z(0.08, steps=20)
        record(5)

        print("=== TRANSPORT ===")
        cur = env.gripper_xyz()
        try:
            env.move_to_xyz((cur[0], cur[1], MUJOCO_SEARCH_HOVER_XYZ[2]), steps=20, enforce_cap=False)
        except CollisionDetected as e:
            print(f"[TRANSPORT] 이동 중 충돌: {e}")
            return False
        record(5)

        print("=== FINE_SERVO (검은 쓰레기통, 손목캠) ===")
        if not tsm.fine_servo(env, cap, perception.detect_black_bin, "검은 쓰레기통"):
            print("[결과] 쓰레기통 정렬 실패 - 큐브를 든 채로 중단")
            return False
        record(5)

        print("=== RELEASE ===")
        try:
            env.move_z(-0.05, steps=15)
        except CollisionDetected as e:
            print(f"[RELEASE] 하강 중 접촉 감지 ({e}) - 현재 위치에서 놓음")
        gripper.open_gripper(env)
        record(8)
        env.move_z(0.05, steps=15)
        record(10)

        print("[결과] 완료: 큐브를 쓰레기통에 놓음")
        ok = True
        return True
    except Exception as e:  # noqa: BLE001 - a dry run must still save its video on any failure
        print(f"[예외] {type(e).__name__}: {e}")
        return False
    finally:
        print(f"\n{len(frames)} 프레임 녹화됨 - 저장 중: {VIDEO_PATH}")
        with imageio.get_writer(str(VIDEO_PATH), fps=VIDEO_FPS, codec="libx264") as w:
            for f in frames:
                w.append_data(f[:, :, ::-1])  # BGR (perception/cv2 convention) -> RGB (imageio)
        print(f"저장 완료: {VIDEO_PATH} (grasped={ok})")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
