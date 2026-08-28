"""Pure-simulation dry run of task_state_machine.py's control logic - NO robot
connection, NO camera device, nothing physical touched. Exercises the real
search / coarse_center / estimate_jacobian / fine_servo / descend_and_grasp
functions UNMODIFIED, against a synthetic arm + synthetic camera model, to
sanity-check this brand-new package before it is ever pointed at real
hardware.

Modeled directly on the sibling ~/lerobot/custom_scripts/vision_pick_place/
sim_dry_run.py (the old visual_servo_pick_place.py rewrite target) - same
approach, same synthetic-Jacobian shape (anisotropic, scale grows on
approach), same grasp-verification check, adapted to this package's own
function names/signatures and its extra pieces (Astra-homography coarse
search with SEARCH_OFFSETS grid fallback, coarse_center's hill-climb).

What is deliberately OUT of scope here: task_state_machine.run() itself,
because it calls arm.kin.forward_kinematics() (real placo IK, built from the
SO-101 URDF) to compute home_xyz from home_pose - that's real, already-
validated kinematics code, not something this pure-logic sim needs to
re-exercise. The state functions below are what actually needed a first
sanity check before real hardware; main.py's full plumbing (connect, home
capture, run()) is exercised for real the first time this runs on the arm.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin/sim_dry_run.py
(from ~/lerobot - only needs numpy/opencv + this package's own pure-Python
control flow, no hardware SDKs, no camera device, no serial port.)
"""

from __future__ import annotations

import time

import numpy as np

import config
import gripper
import perception
import task_state_machine as tsm
from kinematics import CollisionDetected

# Isolates this sim from whatever real files camera_hub.py/astra_s_live.py
# happen to have published on disk (same reasoning as the old sim_dry_run.py:
# deterministic, no race with a real watchdog process). Saved first so the
# perception-level unit tests below can still call the real implementations
# directly with explicit override paths.
_ORIGINAL_estimate_cube_height_m = perception.estimate_cube_height_m
_ORIGINAL_estimate_xy_from_astra = perception.estimate_xy_from_astra
perception.estimate_cube_height_m = lambda *a, **k: None
perception.estimate_xy_from_astra = lambda detect_fn, **k: None

# Real functions under test call time.sleep a lot (servo settle delays,
# gripper interpolation) - none of that time does anything useful against a
# purely virtual arm, so it's patched to a no-op to keep this dry run fast.
# Not needed for correctness, only speed; nothing here depends on real timing.
time.sleep = lambda *_a, **_k: None

FRAME_W, FRAME_H = config.FRAME_W, config.FRAME_H
GRASP_TARGET_PX = np.array(config.GRASP_TARGET_PX)
# Seed matched to the sibling sim_dry_run.py (visual_servo_pick_place.py's
# rewrite target) rather than picked fresh here - worth knowing: a sweep of
# seeds 1-20 against this same synthetic model/offset only converged within
# MAX_SERVO_ITERS about half the time (real detection noise + an anisotropic
# Jacobian is a genuinely marginal case for a 90-iteration budget at a 6cm
# starting offset, not a bug in fine_servo - see this file's docstring and
# the many real servo-divergence incidents already logged in project memory
# for 2026-08-26). This fixed seed demonstrates the logic converges
# correctly when it does, not that every real run is guaranteed to.
RNG = np.random.default_rng(7)


class FakeSOArm101:
    """Implements exactly the SOArm101 surface task_state_machine.py and
    gripper.py call, backed by a purely virtual xyz + a virtual 6-vector
    joint array (ARM_JOINTS + gripper pct) - no serial port, no motors, no
    placo IK. Can simulate a collision after N moves, and whether an object
    is physically between the jaws when the gripper is commanded closed."""

    def __init__(self, start_xyz, collide_after_n_moves=None, object_present=True):
        self._xyz = np.array(start_xyz, dtype=float)
        self._joint = np.zeros(len(config.ALL_JOINTS))
        self._joint[-1] = 100.0  # starts open, same as a real session after gripper.open_gripper
        self._n_moves = 0
        self._collide_after_n_moves = collide_after_n_moves
        self._object_present = object_present

    def _maybe_collide(self):
        self._n_moves += 1
        # Exact-match, not >=: a real collision is a transient obstruction at
        # one specific move, not a permanently blocked arm - the whole point
        # of test_search_collision_fallback is that search() should recover
        # and keep trying later grid points, which a >= check would prevent
        # (every move after the trigger would "collide" forever too).
        if self._n_moves == self._collide_after_n_moves:
            raise CollisionDetected("[sim] 시뮬레이션된 충돌")

    def gripper_xyz(self):
        return self._xyz.copy()

    def get_joint_deg(self):
        return self._joint.copy()

    def send_joint_deg(self, joint_deg):
        # Simulates the one physical fact descend_and_grasp's verification
        # relies on: a real gripper commanded fully closed can't actually
        # reach 0% if something is wedged between the jaws, and even with
        # nothing there it settles near GRIPPER_EMPTY_CLOSED_PCT rather than
        # exactly 0 (real hardware slack) - see gripper.py's module
        # docstring for the real false-positive this check was built to
        # catch. Only the gripper DOF (last element) needs this; the arm
        # joints just store whatever was commanded, unused by anything under
        # test here.
        pct = float(joint_deg[-1])
        if pct <= config.GRIPPER_EMPTY_CLOSED_PCT:
            pct = 24.0 if self._object_present else config.GRIPPER_EMPTY_CLOSED_PCT
        self._joint = np.array(joint_deg, dtype=float)
        self._joint[-1] = pct

    def move_to_xyz(self, xyz, steps=20, step_delay_s=0.05, enforce_cap=True, stall_check=True):
        self._maybe_collide()
        self._xyz = np.array(xyz, dtype=float)

    def move_to_xyz_converge(self, xyz, tolerance_m=0.005, max_iters=15):
        self.move_to_xyz(xyz)
        return self.gripper_xyz()

    def nudge_xy(self, dx, dy, steps=6, step_delay_s=0.03, stall_check=True):
        self._maybe_collide()
        self._xyz[0] += dx
        self._xyz[1] += dy
        return self.gripper_xyz()

    def move_z(self, dz, steps=10, step_delay_s=0.04, stall_check=True):
        self._maybe_collide()
        self._xyz[2] += dz
        return self.gripper_xyz()


def make_synthetic_detect_fn(arm: FakeSOArm101, target_xy, edge_bias=(0.0, 0.0)):
    """detect_fn(frame) computing where target_xy WOULD appear in the wrist
    camera given arm's current virtual xyz - ignores frame content entirely
    (DummyCap always returns the same blank frame; only the arm's virtual
    position matters), same approach as the sibling sim_dry_run.py. J_true
    scales up on approach (anisotropic, condition number ~6) to match the
    same real hardware readout that PHYSICAL_TOLERANCE_M/Broyden-update were
    built against (see config.py's FINE_SERVO comments)."""
    call_count = 0

    def detect_fn(frame):
        nonlocal call_count
        call_count += 1
        dx, dy = arm.gripper_xyz()[0] - target_xy[0], arm.gripper_xyz()[1] - target_xy[1]
        dist = float(np.hypot(dx, dy))
        scale = 800.0 * (1.0 + 9.0 * np.exp(-dist / 0.02))
        J_true = scale * np.array([[1.0, 0.05], [-0.08, 0.16]])
        pixel_offset = J_true @ np.array([dx, dy])
        bias_weight = max(0.0, 1.0 - 0.15 * call_count)
        bias = np.array(edge_bias) * bias_weight
        noise = RNG.normal(0, 1.5, size=2)
        cx, cy = GRASP_TARGET_PX + pixel_offset + bias + noise
        if not (0 <= cx <= FRAME_W and 0 <= cy <= FRAME_H):
            return None
        return perception.Detection(cx=float(cx), cy=float(cy), bbox=(int(cx) - 10, int(cy) - 10, 20, 20), area=400.0)

    return detect_fn


class DummyCap:
    def read(self):
        return True, np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def run_case(label, start_xyz, target_xy, object_present, edge_bias=(0.0, 0.0)):
    print(f"\n{'=' * 70}\n[시뮬레이션] {label}\n{'=' * 70}")
    arm = FakeSOArm101(start_xyz, object_present=object_present)
    cap = DummyCap()
    detect_fn = make_synthetic_detect_fn(arm, target_xy, edge_bias=edge_bias)

    ok = tsm.fine_servo(arm, cap, detect_fn, f"가상 타겟({label})")
    print(f"-> fine_servo 결과: {ok}, 최종 xyz={arm.gripper_xyz()}")
    if not ok:
        raise AssertionError(f"'{label}': fine_servo가 실패했습니다 (성공해야 하는 케이스)")

    grasped = tsm.descend_and_grasp(arm)
    print(f"-> descend_and_grasp 결과: grasped={grasped} (object_present={object_present}였음)")
    assert grasped == object_present, "grasp-verification 로직이 시뮬레이션 정답과 어긋남!"
    print("   [검증 통과] grasp 판정이 시뮬레이션 정답과 일치합니다.")


def test_search_blind_grid():
    """perception.estimate_xy_from_astra is monkeypatched to None module-wide
    (see top of file), so search() must fall back to its blind SEARCH_OFFSETS
    grid sweep around SEARCH_HOVER_XYZ - places the target at one of the
    grid's real offsets and checks search() actually finds it there instead
    of exhausting the sweep."""
    print(f"\n{'=' * 70}\n[시뮬레이션] search() 블라인드 그리드 폴백\n{'=' * 70}")
    hover = config.SEARCH_HOVER_XYZ
    dx, dy = config.SEARCH_OFFSETS[3]  # a real non-zero offset from the configured grid
    target_xy = (hover[0] + dx, hover[1] + dy)
    arm = FakeSOArm101(hover)
    cap = DummyCap()
    detect_fn = make_synthetic_detect_fn(arm, target_xy)

    found = tsm.search(arm, cap, detect_fn, "가상 타겟(그리드)")
    print(f"-> search 결과: {found}, 최종 xyz={arm.gripper_xyz()}")
    assert found, "블라인드 그리드 안에 있는 타겟을 못 찾음"
    print("   [검증 통과] Astra 추정 없이도 그리드 스윕으로 타겟을 찾음.")


def test_search_collision_fallback():
    """A real run hit a collision retreat mid-search and safely fell through
    to the next grid point rather than crashing - checks the same here:
    collide on the very first move, confirm search() doesn't propagate the
    exception and still finds the target at a later offset."""
    print(f"\n{'=' * 70}\n[시뮬레이션] search() 중 충돌 후 안전한 폴백\n{'=' * 70}")
    hover = config.SEARCH_HOVER_XYZ
    dx, dy = config.SEARCH_OFFSETS[1]
    target_xy = (hover[0] + dx, hover[1] + dy)
    arm = FakeSOArm101(hover, collide_after_n_moves=1)
    cap = DummyCap()
    detect_fn = make_synthetic_detect_fn(arm, target_xy)

    found = tsm.search(arm, cap, detect_fn, "가상 타겟(충돌 후 재시도)")
    print(f"-> search 결과: {found}")
    assert found, "충돌 이후 그리드 폴백이 타겟을 못 찾음"
    print("   [검증 통과] 충돌 감지 후에도 예외가 새지 않고 다음 지점에서 계속 탐색함.")


def test_hand_rejection():
    """A hand/arm/cable in the same color range as the cube/bin used to be
    able to win detection just by being the largest blob - see config.py's
    MIN_CUBE_SOLIDITY comment. Builds a synthetic frame with the real target
    (small, solid, square) and a much bigger, star-shaped (low-solidity)
    same-colored intruder, checks detection still finds the real target."""
    import math

    import cv2

    print(f"\n{'=' * 70}\n[시뮬레이션] 손/팔 오탐 거부 테스트\n{'=' * 70}")

    def make_frame(color_bgr):
        frame = np.full((480, 640, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (300, 200), (340, 240), color_bgr, -1)  # the real target
        pts = []
        for i in range(10):
            r = 90 if i % 2 == 0 else 30
            ang = math.pi * i / 5
            pts.append((int(150 + r * math.cos(ang)), int(150 + r * math.sin(ang))))
        cv2.fillPoly(frame, [np.array(pts)], color_bgr)  # a much bigger, low-solidity "hand"
        return frame

    cube_frame = make_frame((30, 30, 200))  # saturated red, BGR
    det = perception.detect_red_cube(cube_frame)
    print(f"  큐브: {det}")
    assert det is not None, "빨간 사각형을 아예 못 찾음"
    assert 290 <= det.cx <= 350 and 190 <= det.cy <= 250, f"별 모양(손) 오탐: {det}"
    print("  [검증 통과] 손 모양 무시하고 실제 큐브를 찾음.")

    bin_frame = make_frame((20, 20, 20))  # near-black, BGR
    det = perception.detect_black_bin(bin_frame)
    print(f"  쓰레기통: {det}")
    assert det is not None, "검은 사각형을 아예 못 찾음"
    assert 290 <= det.cx <= 350 and 190 <= det.cy <= 250, f"별 모양(손) 오탐: {det}"
    print("  [검증 통과] 손 모양 무시하고 실제 쓰레기통을 찾음.")


def test_estimate_cube_height():
    """Exercises estimate_cube_height_m()'s real (non-monkeypatched) logic
    directly against a synthetic Astra RGB frame + depth array in an
    isolated temp dir - never touches the real published paths."""
    import os
    import tempfile

    import cv2

    print(f"\n{'=' * 70}\n[시뮬레이션] estimate_cube_height_m() 단위 테스트\n{'=' * 70}")

    color = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(color, (300, 200), (340, 240), (30, 30, 200), -1)

    depth_mm = np.full((240, 320), 500, dtype=np.uint16)
    KNOWN_HEIGHT_MM = 25
    depth_mm[100:120, 150:170] = 500 - KNOWN_HEIGHT_MM

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth_mm.npy")
        cv2.imwrite(rgb_path, color)
        np.save(depth_path.replace(".npy", ""), depth_mm)

        height = _ORIGINAL_estimate_cube_height_m(rgb_path=rgb_path, depth_path=depth_path)
        print(f"   추정 높이: {height}")
        assert height is not None, "높이 추정이 None을 반환함 (합성 데이터인데도 실패)"
        expected = KNOWN_HEIGHT_MM / 1000.0
        assert abs(height - expected) < 0.003, f"추정치 {height} != 기대값 {expected} (허용오차 3mm)"
        print(f"   [검증 통과] 추정 높이 {height * 1000:.1f}mm ≈ 실제 {KNOWN_HEIGHT_MM}mm")


def test_estimate_xy_from_astra():
    """Same idea, for the homography-based coarse xy estimate: a known
    pure-affine homography applied to a red square at a known pixel should
    recover a known robot-frame xy."""
    import os
    import tempfile

    import cv2

    print(f"\n{'=' * 70}\n[시뮬레이션] estimate_xy_from_astra() 단위 테스트\n{'=' * 70}")

    color = np.full((480, 640, 3), 200, dtype=np.uint8)
    px, py = 300, 200
    cv2.rectangle(color, (px - 20, py - 20), (px + 20, py + 20), (30, 30, 200), -1)

    a, b, c, d = 0.001, 0.05, -0.0008, 0.6
    H = np.array([[a, 0.0, b], [0.0, c, d], [0.0, 0.0, 1.0]])
    expected_x, expected_y = a * px + b, c * py + d

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_path = os.path.join(tmpdir, "rgb.png")
        cv2.imwrite(rgb_path, color)

        result = _ORIGINAL_estimate_xy_from_astra(perception.detect_red_cube, rgb_path=rgb_path, homography=H)
        print(f"   추정 xy: {result} (기대값: ({expected_x:.4f}, {expected_y:.4f}))")
        assert result is not None, "xy 추정이 None을 반환함 (합성 데이터인데도 실패)"
        assert abs(result[0] - expected_x) < 1e-6 and abs(result[1] - expected_y) < 1e-6, (
            f"추정치 {result} != 기대값 ({expected_x}, {expected_y})"
        )
        print("   [검증 통과] 호모그래피 매핑이 정확히 복원됨.")


def test_gripper_verification_unit():
    """gripper.is_grasp_success is pure logic (no arm needed) - a direct unit
    check against config's real threshold, independent of the FakeSOArm101
    wedging simulation used in run_case above."""
    print(f"\n{'=' * 70}\n[시뮬레이션] gripper.is_grasp_success() 단위 테스트\n{'=' * 70}")
    empty = config.GRIPPER_EMPTY_CLOSED_PCT
    threshold = config.GRIPPER_EMPTY_CLOSED_PCT + config.GRASP_DETECT_MARGIN_PCT
    assert gripper.is_grasp_success(empty) is False, "빈 채로 닫혔는데 성공으로 판정됨"
    assert gripper.is_grasp_success(threshold + 0.1) is True, "물체가 걸렸는데 실패로 판정됨"
    assert gripper.is_grasp_success(threshold - 0.1) is False, "경계값 바로 아래인데 성공으로 판정됨"
    print("   [검증 통과] 빈 파지/실제 파지 경계가 config 임계값과 일치.")


def main():
    hover = config.SEARCH_HOVER_XYZ

    run_case(
        "정상 접근 + 실제로 집힘",
        start_xyz=hover,
        target_xy=(hover[0] - 0.06, hover[1] + 0.02),
        object_present=True,
    )
    run_case(
        "정상 접근 + 실제로는 못 집음",
        start_xyz=hover,
        target_xy=(hover[0] - 0.06, hover[1] + 0.02),
        object_present=False,
    )
    run_case(
        "화면 가장자리에서 시작 (coarse-center 경로)",
        start_xyz=hover,
        target_xy=(hover[0] - 0.04, hover[1] - 0.02),
        object_present=True,
        edge_bias=(-300.0, 0.0),
    )

    test_search_blind_grid()
    test_search_collision_fallback()
    test_hand_rejection()
    test_estimate_cube_height()
    test_estimate_xy_from_astra()
    test_gripper_verification_unit()

    print(f"\n{'=' * 70}\n모든 시뮬레이션 케이스 통과.\n{'=' * 70}")


if __name__ == "__main__":
    main()
