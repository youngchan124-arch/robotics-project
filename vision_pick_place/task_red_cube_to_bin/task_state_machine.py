"""Fixed-sequence state machine: SEARCH -> APPROACH -> FINE_SERVO -> DESCEND
-> GRASP -> VERIFY -> (retry up to MAX_GRASP_ATTEMPTS) -> LIFT -> TRANSPORT
-> FINE_SERVO(bin) -> RELEASE -> HOME.

No LLM re-planning (per the spec's §0 - this is Phase 1, a single fixed
sequence run start to finish on one command, not a multi-step agent loop).
Every state transition is logged. home_pose is captured live at session
start (read_joint_positions, never hardcoded) and guaranteed to be returned
to on every exit path - success, a failed retry budget, or any exception -
via try/finally.
"""

from __future__ import annotations

import enum
import time

import numpy as np

import config
import gripper
import perception
from kinematics import CollisionDetected, SOArm101


class TaskState(enum.Enum):
    HOME_CAPTURED = enum.auto()
    SEARCH = enum.auto()
    APPROACH = enum.auto()
    FINE_SERVO = enum.auto()
    DESCEND = enum.auto()
    GRASP = enum.auto()
    VERIFY = enum.auto()
    LIFT = enum.auto()
    TRANSPORT = enum.auto()
    RELEASE = enum.auto()
    HOME = enum.auto()
    FAILED = enum.auto()
    DONE = enum.auto()


def _log(state: TaskState, msg: str = "") -> None:
    print(f"[{state.name}]{' ' + msg if msg else ''}")


def get_pixel(cap: perception.PublishedFrameSource, detect_fn, tries: int = 8):
    for _ in range(tries):
        ret, frame = cap.read()
        if not ret or frame is None or perception.is_frame_corrupted(frame):
            time.sleep(0.03)
            continue
        det = detect_fn(frame)
        if det is not None:
            return det, frame
        time.sleep(0.03)
    return None, None


def search(arm: SOArm101, cap, detect_fn, name: str) -> bool:
    """Astra-homography coarse guess first (move straight there, confirm
    with the wrist cam); falls back to a bounded blind grid sweep around
    SEARCH_HOVER_XYZ if that's unavailable, unreachable, or unconfirmed - see
    perception.estimate_xy_from_astra's docstring for why this is only a
    coarse guess, not trusted on its own."""
    _log(TaskState.SEARCH, f"{name} 탐색 시작")
    astra_xy = perception.estimate_xy_from_astra(detect_fn)
    if astra_xy is not None:
        target = (astra_xy[0], astra_xy[1], config.SEARCH_HOVER_XYZ[2])
        _log(TaskState.APPROACH, f"Astra 추정 위치로 직접 이동: ({astra_xy[0]:.3f}, {astra_xy[1]:.3f})")
        try:
            arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=10)
            time.sleep(0.2)
            det, _ = get_pixel(cap, detect_fn, tries=10)
            if det is not None:
                _log(TaskState.APPROACH, f"손목캠 확인됨: pixel=({det.cx:.0f},{det.cy:.0f})")
                return True
            _log(TaskState.APPROACH, "손목캠에서 확인 안 됨 - 탐색 그리드로 대체")
        except CollisionDetected as e:
            _log(TaskState.APPROACH, f"이동 중 충돌 감지 ({e}) - 탐색 그리드로 대체")

    for dx, dy in config.SEARCH_OFFSETS:
        target = (config.SEARCH_HOVER_XYZ[0] + dx, config.SEARCH_HOVER_XYZ[1] + dy, config.SEARCH_HOVER_XYZ[2])
        try:
            arm.move_to_xyz_converge(target, tolerance_m=0.015, max_iters=10)
        except CollisionDetected as e:
            # 2026-08-26 (found via sim_dry_run.py, before ever touching real
            # hardware): unlike the Astra-estimate branch above, this loop
            # used to leave a collision during the grid sweep uncaught -
            # it would propagate out of search()/fine_servo()/run() (past
            # run()'s finally, which still returns home, but straight
            # through main.py's except KeyboardInterrupt, crashing with a
            # raw traceback instead of a clean failure log). A blocked
            # single grid offset should mean "skip it, try the next one",
            # not "abort the whole task" - same reasoning as the Astra
            # branch's own except just above.
            _log(TaskState.SEARCH, f"offset=({dx:+.2f},{dy:+.2f}) 이동 중 충돌 감지 ({e}) - 다음 지점으로")
            continue
        time.sleep(0.2)
        det, _ = get_pixel(cap, detect_fn, tries=10)
        if det is not None:
            _log(TaskState.SEARCH, f"찾음: offset=({dx:+.2f},{dy:+.2f}) pixel=({det.cx:.0f},{det.cy:.0f})")
            return True
    _log(TaskState.SEARCH, f"{name} 탐색 범위 내에서 못 찾음")
    return False


def coarse_center(arm: SOArm101, cap, detect_fn) -> bool:
    """Direction-agnostic hill-climb for a target found near a frame edge,
    where the small finite-difference Jacobian probe would lose it off-
    frame before ever computing a Jacobian. No-op (one distance check) if
    already close."""
    axis, sign = 0, 1.0
    target_px = np.array(config.GRASP_TARGET_PX)
    for _ in range(config.COARSE_MAX_ITERS):
        det, _ = get_pixel(cap, detect_fn, tries=6)
        if det is None:
            return False
        dist = float(np.linalg.norm(target_px - np.array([det.cx, det.cy])))
        if dist <= config.COARSE_TARGET_PX:
            return True
        dx = config.COARSE_STEP_M * sign if axis == 0 else 0.0
        dy = config.COARSE_STEP_M * sign if axis == 1 else 0.0
        try:
            arm.nudge_xy(dx, dy)
        except CollisionDetected:
            return False
        time.sleep(0.15)
        det2, _ = get_pixel(cap, detect_fn, tries=6)
        if det2 is None:
            arm.nudge_xy(-dx, -dy)
            time.sleep(0.15)
            sign, axis = (-1.0, axis) if sign > 0 else (1.0, 1 - axis)
            continue
        new_dist = float(np.linalg.norm(target_px - np.array([det2.cx, det2.cy])))
        if new_dist < dist:
            continue
        arm.nudge_xy(-dx, -dy)
        time.sleep(0.15)
        sign, axis = (-1.0, axis) if sign > 0 else (1.0, 1 - axis)
    return False


def estimate_jacobian(arm: SOArm101, cap, detect_fn) -> np.ndarray | None:
    """3-probe self-calibration (nudge +x, then +y, watch the pixel shift
    each time) - the only "calibration" this needs, since the wrist camera
    is eye-in-hand: whichever way the mount is actually oriented, this
    measures it fresh every time rather than assuming one."""
    base, _ = get_pixel(cap, detect_fn)
    if base is None:
        return None
    base_px = np.array([base.cx, base.cy])
    arm.nudge_xy(config.PROBE_DELTA_M, 0.0)
    time.sleep(0.2)
    dx_det, _ = get_pixel(cap, detect_fn)
    arm.nudge_xy(-config.PROBE_DELTA_M, config.PROBE_DELTA_M)
    time.sleep(0.2)
    dy_det, _ = get_pixel(cap, detect_fn)
    arm.nudge_xy(0.0, -config.PROBE_DELTA_M)
    time.sleep(0.2)
    if dx_det is None or dy_det is None:
        return None
    dx_px = np.array([dx_det.cx, dx_det.cy]) - base_px
    dy_px = np.array([dy_det.cx, dy_det.cy]) - base_px
    J = np.column_stack([dx_px / config.PROBE_DELTA_M, dy_px / config.PROBE_DELTA_M])
    if abs(np.linalg.det(J)) < 1e-3:
        return None
    return J


def fine_servo(arm: SOArm101, cap, detect_fn, name: str) -> bool:
    """Closed-loop pixel-error servo onto GRASP_TARGET_PX (the gripper's own
    jaw position in-frame, not the raw image center - see config.py).
    Convergence gate is a REAL-WORLD distance (PHYSICAL_TOLERANCE_M), not
    pixel count: a real run converged to <11px and still missed the cube
    entirely, because the image Jacobian is anisotropic (measured singular
    values 8152 vs 1233 px/m on real hardware, ~6.6x) - a small pixel error
    in the low-sensitivity direction can still be several mm off. J is
    refined every iteration via Broyden's rank-one update using the step
    just taken and the pixel shift it produced (skipped below
    BROYDEN_MIN_STEP_M - a small step amplifies ordinary detection noise
    into a wrong correction), with a full re-probe as a fallback whenever
    progress stalls (STALL_ITERS) or clearly diverges (DIVERGE_PX)."""
    if not search(arm, cap, detect_fn, name):
        return False
    if not coarse_center(arm, cap, detect_fn):
        return False

    J = estimate_jacobian(arm, cap, detect_fn)
    if J is None:
        if not search(arm, cap, detect_fn, name) or not coarse_center(arm, cap, detect_fn):
            return False
        J = estimate_jacobian(arm, cap, detect_fn)
        if J is None:
            return False
    J_inv = np.linalg.inv(J)

    _log(TaskState.FINE_SERVO, f"{name} 중앙 정렬 시작")
    target_px = np.array(config.GRASP_TARGET_PX)
    stable = lost_streak = stall_count = reestimates = 0
    best_err = float("inf")
    prev_px = prev_step = None

    for it in range(config.MAX_SERVO_ITERS):
        det, _ = get_pixel(cap, detect_fn, tries=6)
        if det is None:
            lost_streak += 1
            prev_px = prev_step = None
            if lost_streak >= 5:
                _log(TaskState.FINE_SERVO, "타겟을 계속 놓침 - 실패")
                return False
            continue
        lost_streak = 0
        cur_px = np.array([det.cx, det.cy])

        if prev_step is not None and prev_px is not None:
            step_norm2 = float(prev_step @ prev_step)
            if step_norm2 > config.BROYDEN_MIN_STEP_M**2:
                predicted = J @ prev_step
                J_candidate = J + np.outer(cur_px - prev_px - predicted, prev_step) / step_norm2
                if abs(np.linalg.det(J_candidate)) > 1e-3:
                    J, J_inv = J_candidate, np.linalg.inv(J_candidate)
        prev_px = cur_px

        err_px = target_px - cur_px
        err_norm = float(np.linalg.norm(err_px))
        dxy = J_inv @ err_px
        physical_err_m = float(np.linalg.norm(dxy))

        if err_norm <= config.PIXEL_TOLERANCE and physical_err_m <= config.PHYSICAL_TOLERANCE_M:
            stable += 1
            best_err = min(best_err, err_norm)
            stall_count = 0
            prev_step = None
            _log(TaskState.FINE_SERVO, f"iter {it}: 정렬됨 ({err_norm:.1f}px / {physical_err_m*1000:.1f}mm, {stable}/{config.CENTER_STABLE_FRAMES})")
            if stable >= config.CENTER_STABLE_FRAMES:
                return True
            continue
        stable = 0

        if err_norm < best_err - 2.0:
            best_err, stall_count = err_norm, 0
        else:
            stall_count += 1
        diverging = err_norm > best_err + config.DIVERGE_PX
        if (stall_count >= config.STALL_ITERS or diverging) and reestimates < config.MAX_REESTIMATES:
            reason = "발산" if diverging else "정체"
            _log(TaskState.FINE_SERVO, f"iter {it}: {reason} (최소 {best_err:.1f}px, 현재 {err_norm:.1f}px) - 재추정")
            new_J = estimate_jacobian(arm, cap, detect_fn)
            if new_J is not None:
                J, J_inv = new_J, np.linalg.inv(new_J)
                reestimates += 1
                best_err = err_norm
            stall_count = 0
            prev_px = prev_step = None
            continue

        step = config.SERVO_GAIN * dxy
        effective_max_step = config.MAX_STEP_M if err_norm > 30.0 else config.MAX_STEP_M * config.CLOSE_MAX_STEP_FRAC
        mag = float(np.linalg.norm(step))
        if mag > effective_max_step:
            step = step * (effective_max_step / mag)
        try:
            arm.nudge_xy(float(step[0]), float(step[1]))
        except CollisionDetected:
            _log(TaskState.FINE_SERVO, "서보잉 중 충돌 감지")
            return False
        prev_step = step
        time.sleep(0.1)

    _log(TaskState.FINE_SERVO, f"{config.MAX_SERVO_ITERS}회 내에 정렬 실패")
    return False


def descend_and_grasp(arm: SOArm101) -> bool:
    cur = arm.gripper_xyz()
    cube_height_m = perception.estimate_cube_height_m()
    if cube_height_m is not None:
        target_z = min(config.TABLE_Z + cube_height_m - config.DESCEND_MARGIN_M, cur[2])
        _log(TaskState.DESCEND, f"Astra 높이 추정 {cube_height_m*1000:.1f}mm -> 1차 목표 z={target_z:.4f}")
    else:
        target_z = config.TABLE_Z
        _log(TaskState.DESCEND, "높이 추정 실패 - TABLE_Z로 하강")

    contacted = False
    try:
        arm.move_to_xyz((cur[0], cur[1], target_z), steps=25, step_delay_s=0.05, enforce_cap=False, stall_check=True)
        _log(TaskState.DESCEND, "1차 목표 도달 (접촉 없음)")
    except CollisionDetected:
        _log(TaskState.DESCEND, "접촉 감지 (큐브로 판단)")
        contacted = True

    # A depth estimate can undershoot - contact detection, not the estimate,
    # is what actually decides "found it". Keep easing down to TABLE_Z
    # (the measured real table contact point + margin) if phase 1 found
    # nothing, instead of accepting "reached the estimate, nothing there".
    if not contacted and target_z > config.TABLE_Z + 1e-4:
        cur2 = arm.gripper_xyz()
        try:
            arm.move_to_xyz((cur2[0], cur2[1], config.TABLE_Z), steps=25, step_delay_s=0.06, enforce_cap=False, stall_check=True)
            _log(TaskState.DESCEND, "TABLE_Z까지 도달 (접촉 없음)")
        except CollisionDetected:
            _log(TaskState.DESCEND, "2차 하강 중 접촉 감지 (큐브로 판단)")
            contacted = True

    time.sleep(0.2)
    _log(TaskState.GRASP, "그리퍼 닫는 중")
    final_pct = gripper.close_gripper(arm)
    time.sleep(0.3)
    grasped = gripper.is_grasp_success(final_pct)
    _log(TaskState.VERIFY, f"최종 그리퍼 위치 {final_pct:.1f}% -> {'집힘' if grasped else '못 집음'}")
    return grasped


def run(arm: SOArm101, cap: perception.PublishedFrameSource) -> bool:
    """The whole fixed sequence. home_pose is captured live (never
    hardcoded) and its return is guaranteed via try/finally regardless of
    how this function exits - success, exhausting MAX_GRASP_ATTEMPTS, or any
    exception."""
    home_pose = arm.get_joint_deg()
    home_xyz = tuple(arm.kin.forward_kinematics(home_pose[: len(config.ARM_JOINTS)])[:3, 3])
    _log(TaskState.HOME_CAPTURED, f"joints={home_pose} xyz={home_xyz}")

    try:
        gripper.open_gripper(arm)
        grasped = False
        for attempt in range(1, config.MAX_GRASP_ATTEMPTS + 1):
            print(f"\n=== 시도 {attempt}/{config.MAX_GRASP_ATTEMPTS} ===")
            arm.move_to_xyz_converge(config.SEARCH_HOVER_XYZ, tolerance_m=0.015, max_iters=20)
            if not fine_servo(arm, cap, perception.detect_red_cube, "빨간 큐브"):
                _log(TaskState.FAILED, f"시도 {attempt}: 큐브 정렬 실패")
                continue
            if descend_and_grasp(arm):
                grasped = True
                break
            _log(TaskState.FAILED, f"시도 {attempt}: 파지 실패")
            gripper.open_gripper(arm)

        if not grasped:
            _log(TaskState.FAILED, f"{config.MAX_GRASP_ATTEMPTS}회 모두 실패")
            return False

        _log(TaskState.LIFT, f"{config.LIFT_M*100:.0f}cm 상승")
        arm.move_z(config.LIFT_M, steps=20, step_delay_s=0.05)

        cur = arm.gripper_xyz()
        _log(TaskState.TRANSPORT, "쓰레기통 쪽으로 이동")
        arm.move_to_xyz((cur[0], cur[1], config.SEARCH_HOVER_XYZ[2]), steps=20, step_delay_s=0.05, enforce_cap=False)
        if not fine_servo(arm, cap, perception.detect_black_bin, "검은 쓰레기통"):
            _log(TaskState.FAILED, "쓰레기통 정렬 실패 - 큐브를 든 채로 홈 복귀")
            return False

        _log(TaskState.RELEASE, f"쓰레기통 위에서 {config.BIN_DESCEND_M*100:.0f}cm 하강")
        try:
            arm.move_z(-config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)
        except CollisionDetected:
            _log(TaskState.RELEASE, "하강 중 접촉 감지 (쓰레기통 벽/바닥) - 현재 위치에서 놓음")
        gripper.open_gripper(arm)
        time.sleep(0.3)
        arm.move_z(config.BIN_DESCEND_M, steps=15, step_delay_s=0.05)

        _log(TaskState.DONE, "완료")
        return True
    finally:
        # A raw one-shot send_joint_deg(home_pose) would skip interpolated
        # stepping and stall/collision checking entirely (lerobot's own
        # max_relative_target would silently truncate a large jump anyway,
        # not safely retreat from one) - move there the same safety-checked
        # way as every other move in this file, via the xyz captured from
        # home_pose's forward kinematics.
        _log(TaskState.HOME, f"홈 포즈로 복귀: {home_xyz}")
        try:
            arm.move_to_xyz_converge(home_xyz, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[HOME] 복귀 중 충돌 감지, 안전 위치에서 정지: {e}")
        except Exception as e:  # noqa: BLE001 - this is the last-resort safety return, must not itself throw uncaught
            print(f"[HOME] 복귀 실패: {e}")
