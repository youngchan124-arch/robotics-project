"""Orchestration: text prompt -> Grounding DINO+SAM detection + grasp yaw
(perception_zeroshot.py) -> table-plane homography xy + Astra depth-delta z
(same math perception.py already has real-hardware-validated, reimplemented
here read-only to avoid a second GPU detection pass - see _xy_from_pixel's
docstring) -> fixed-roll 4-DOF grasp IK (grasp_ik.py) -> move + grasp via
kinematics.py's SOArm101.

This is the "Vision -> 3D Grasp Pose -> IK/Motion Planning" pipeline the
user asked for on 2026-08-28 (open-loop, single-shot - no per-object
demonstrations, no wrist-cam closed-loop visual servo). Per their explicit
instruction: a NEW file, not an edit of anything run against real hardware
before this pivot (red_cube_calib_pick.py, visual_servo_pick_place.py,
kinematics.py, config.py, perception.py are all imported read-only, never
modified).

2026-08-28, same day, added pick-AND-place (execute_pick_and_place) for the
actual task the user is running: grasp a red cube, carry it to and release
it at a black bin/box - see that function's docstring for the two-target
(pick prompt + place prompt) design and why place doesn't reuse GraspPlan.

STATUS: perception_zeroshot.py, grasp_ik.py, and this file's plan/solve
logic are each validated standalone (real saved photo / pure FK math /
dry-run against a fake-but-real-photo-backed fixture - see
test_zeroshot_pick.py). First live run against the real Astra S + robot is
today, 2026-08-28, per the user's explicit go-ahead ("지금 ... 진행할거야").

Two real gaps, flagged rather than silently guessed at (same "don't trust
an unmeasured number" convention as GRIPPER_EMPTY_CLOSED_PCT/
GRASP_TARGET_PX elsewhere in this project):
  1. IMAGE_YAW_TO_WORLD_ROLL_OFFSET_DEG - no real calibration yet mapping
     perception_zeroshot's image-plane PCA yaw to the robot's wrist_roll
     frame. Currently an identity placeholder (0.0). To calibrate: place an
     elongated object at a few different known orientations, compare the
     estimated yaw_deg to the wrist_roll that actually produces a
     jaws-aligned top-down grasp on real hardware.
  2. Grasp verification reuses config.GRIPPER_EMPTY_CLOSED_PCT /
     GRASP_DETECT_MARGIN_PCT, which per config.py's own comment are still
     untuned placeholders for this gripper (run manual_grasp_calibration.py
     first if this hasn't been done since).
  3. xy accuracy is only as good as homography.json's existing calibration
     (4/5 points, see orbbec-astra-s-lerobot.md) - this script does not add
     any wrist-cam fine-servo refinement on top, unlike
     visual_servo_pick_place.py, by design (this is the open-loop
     "Vision -> 3D Grasp Pose -> IK" architecture the user asked for, not
     visual servoing) - real grasp accuracy on an arbitrary new object has
     not been measured yet.

2026-08-28, later: relabeled every class/function/log line below with a
[계층 태그] per the user's 5-layer household-assistant-robot architecture
spec, LLM/SLAM excluded (no mobile base or natural-language planner exists
in this project) - documentation only, NO behavior change:
  [YOLO/비전 계층]  - detection, 3D pose estimation, coordinate transforms.
    Actually implemented via Grounding DINO + SAM + CLIP (perception_zeroshot.py),
    not YOLO - kept as "YOLO/비전 계층" per the spec's own layer NAME (an
    organizational label, not a mandate to swap detectors) since zero-shot
    YOLO was tried earlier this project and couldn't see the red cube at
    all (see orbbec-astra-s-lerobot.md) - flagged here so the tag is never
    mistaken for "this actually runs YOLO".
  [IK/조작 계층]    - grasp_ik.py's fixed-roll solve + all physical motion.
    Per the user's explicit 2026-08-28 decision, this keeps the roll-fixed
    4-DOF solver as-is rather than adopting the spec's literal
    "θ_pitch + θ_elbow + θ_wrist = -180°" full-orientation constraint - that
    exact style of constraint was tried on this arm (kinematics.py's
    IK_ORIENTATION_WEIGHT) and blew position error up to 16-253mm; the
    fixed-roll approach measures <1mm across a 200-case stress test instead.
  [Feedback 계층]   - grasp success verification (gripper-% proxy for
    torque/physical sensing - no true force sensor on this gripper) and
    place-completion reporting back up to whatever calls this module.
No [LLM 에이전트 계층] or [SLAM/Navigation 계층] tags appear anywhere below -
out of scope per the user's "일단 llm과 slam은 제외하고 진행해줘".
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass

import numpy as np

import config
from grasp_ik import solve_fixed_roll_ik, solved_position_error_m
from kinematics import CollisionDetected, SOArm101, build_kinematics, gripper_position
from perception import Detection, PublishedFrameSource
from perception_zeroshot import detect_all_objects, detect_and_segment, detect_zeroshot, estimate_height_m

# [YOLO/비전 계층] 2026-08-28: unvalidated placeholder - see module docstring gap #1.
IMAGE_YAW_TO_WORLD_ROLL_OFFSET_DEG = 0.0
DEFAULT_ROLL_DEG = 0.0  # used when no yaw estimate is available (symmetric object, or PCA below YAW_EVAL_RATIO_MIN)

# [IK/조작 계층]
HOVER_CLEARANCE_M = 0.05  # height above a target xyz to approach from before descending

# [Feedback 계층] 2026-08-27 finding on this same physical gripper (see
# orbbec-astra-s-lerobot.md, red_cube_calib_pick.py's grip-force work):
# staying pinned at a full/extreme close command for the whole lift/carry
# tripped the servo's own overload protection and dropped an already-
# successful grasp. That work used raw scservo ticks; this script uses
# lerobot's normalized 0-100% gripper units instead, so the fix is
# reapplied here in percent terms rather than imported directly.
GRIP_HOLD_MARGIN_PCT = 8.0

# [IK/조작 계층] 2026-08-28: defense in depth on top of grasp_ik.py's own
# multi-restart fix (see that module's CONVERGENCE_TOL_M) - even a solver
# that mostly converges should never have a bad/unreachable solution
# silently sent to the real arm. Slightly looser than grasp_ik's own 5mm
# target-convergence bound to leave room for its worst-case-but-still-
# acceptable residual (measured 4.46mm across a 60-seed stress test)
# without false-aborting.
IK_SAFETY_TOL_M = 0.008

# [IK/조작 계층] 2026-08-28: added after a real live run - the red cube (a
# near-symmetric small cube) got a PCA yaw estimate of 90deg from Gemini's
# bbox-rectangle "mask" (see perception_zeroshot.py's module docstring on
# why that's an approximation, not a true segmentation), and target_roll_deg
# was applied blindly - the arm's actual resting wrist_roll was far enough
# from 90deg that reaching it required a huge, slow, resistance-heavy sweep
# (lerobot's own max_relative_target safety clamp fired 8 times in a row on
# wrist_roll alone), which tripped CollisionDetected (65.8deg lag) before
# ever reaching the hover pose. The arm was NOT damaged (verified via both
# cameras post-trip, resting normally) - CollisionDetected did its job -
# but forcing a large wrist rotation for essentially no benefit on a
# near-symmetric object is the real bug, not the safety trip itself. Fix:
# cap how far wrist_roll is ever allowed to travel from wherever the arm
# ACTUALLY is when a task starts - beyond this, keep the current roll
# unchanged rather than commit to an orientation estimate that's likely
# arbitrary anyway (see estimate_grasp_yaw_deg/YAW_EVAL_RATIO_MIN - a
# yaw IS computed for near-square objects too, just less reliably).
MAX_ROLL_EXCURSION_DEG = 90.0


def _cap_roll_excursion(plan: GraspPlan, current_wrist_roll_deg: float, label: str) -> None:
    """Mutates plan.target_roll_deg in place if reaching it from
    current_wrist_roll_deg would exceed MAX_ROLL_EXCURSION_DEG - see that
    constant's docstring. Prints when it fires, so an overridden roll is
    visible in the log rather than silently different from what
    perception estimated."""
    delta = abs(plan.target_roll_deg - current_wrist_roll_deg)
    if delta > MAX_ROLL_EXCURSION_DEG:
        print(f"[IK/조작 계층] {label}: target_roll_deg={plan.target_roll_deg:.1f} is {delta:.1f}deg from the "
              f"arm's current wrist_roll ({current_wrist_roll_deg:.1f}) - exceeds the {MAX_ROLL_EXCURSION_DEG:.0f}deg "
              "safety cap, keeping current roll unchanged instead of forcing the sweep.")
        plan.target_roll_deg = current_wrist_roll_deg


def _require_converged(err_m: float, label: str) -> None:
    """[IK/조작 계층] 목표 자세로 이동하기 전 IK 수렴 여부를 확인하는 안전 게이트."""
    if err_m > IK_SAFETY_TOL_M:
        raise RuntimeError(
            f"[IK/조작 계층] Refusing to move: {label} IK residual {err_m * 1000:.1f}mm exceeds the "
            f"{IK_SAFETY_TOL_M * 1000:.0f}mm safety bound - target is likely unreachable at this orientation, "
            "or the solver got stuck (see grasp_ik.py's CONVERGENCE_TOL_M note)."
        )


@dataclass
class GraspPlan:
    """[YOLO/비전 계층] 비전 파이프라인(검출+3D 좌표+yaw 추정)의 출력 - IK/조작
    계층(_solve_for_xyz/solve_plan)이 이 데이터를 받아 실제 팔 동작으로 변환한다."""

    text_prompt: str
    detection: Detection
    xy: tuple[float, float]
    height_m: float | None
    target_xyz: tuple[float, float, float]
    yaw_deg: float | None
    target_roll_deg: float


def _load_homography() -> np.ndarray | None:
    """[YOLO/비전 계층] 카메라 픽셀 -> 로봇 베이스 xy 평면 호모그래피 로드."""
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=float)


def _xy_from_pixel(homography: np.ndarray, cx: float, cy: float) -> tuple[float, float] | None:
    """[YOLO/비전 계층] Same math as perception.py's estimate_xy_from_astra -
    reimplemented here (not calling that function) purely so callers can
    reuse a single already-computed Detection instead of triggering a
    second, redundant Grounding DINO pass (each inference is ~2s on this
    GPU)."""
    mapped = homography @ np.array([cx, cy, 1.0])
    if abs(mapped[2]) < 1e-9:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def _load_depth_mm(depth_path: str = config.ASTRA_DEPTH_MM_PATH) -> np.ndarray | None:
    """[YOLO/비전 계층] Astra S 뎁스(mm) 원본 로드 - object 높이 추정에 사용."""
    if not os.path.exists(depth_path):
        return None
    if (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        return np.load(depth_path)
    except (OSError, ValueError):
        return None


def check_port_not_busy(port: str = config.FOLLOWER_PORT) -> bool:
    """[IK/조작 계층] 로봇 시리얼 포트 연결 전 안전 가드. This follower arm is
    shared with other people (see orbbec-astra-s-lerobot.md's 2026-08-27
    note) - refuses to connect if something else already has the port open
    rather than fighting for it. Reimplemented here (not imported from
    red_cube_calib_pick.py) since importing that script would run its own
    module-level camera/robot setup as a side effect - this is a self-
    contained lsof check only."""
    try:
        result = subprocess.run(["lsof", port], capture_output=True, text=True, timeout=3)
        return result.returncode != 0  # lsof exits nonzero when nothing has the file open
    except FileNotFoundError:
        return True  # lsof unavailable - can't check, don't block a real run on that alone


def plan_grasp(text_prompt: str, box_threshold: float = None, rgb_path: str = config.ASTRA_RGB_FRAME_PATH,
                depth_path: str = config.ASTRA_DEPTH_MM_PATH) -> GraspPlan | None:
    """[YOLO/비전 계층] Runs the full perception -> 3D grasp pose pipeline for
    text_prompt against whatever Astra RGB(+depth) is currently published.
    Returns None (printing why) at any stage that can't proceed - never
    raises for an ordinary "nothing found"/"no fresh frame" case, matching
    this project's existing estimate_xy_from_astra/estimate_cube_height_m
    convention of None-on-unavailable rather than an exception."""
    from perception_zeroshot import BOX_THRESHOLD as _default_thr

    box_threshold = _default_thr if box_threshold is None else box_threshold

    ret, frame = PublishedFrameSource(rgb_path).read()
    if not ret or frame is None:
        print(f"[YOLO/비전 계층] No fresh Astra RGB frame at {rgb_path} - is astra_s_live.py running?")
        return None

    result = detect_and_segment(frame, text_prompt, box_threshold)
    if result is None:
        print(f"[YOLO/비전 계층] No detection for '{text_prompt}' (threshold={box_threshold})")
        return None
    det, mask, yaw_deg = result

    homography = _load_homography()
    if homography is None:
        print(f"[YOLO/비전 계층] No homography at {config.HOMOGRAPHY_PATH} - run calibrate_camera.py first")
        return None
    xy = _xy_from_pixel(homography, det.cx, det.cy)
    if xy is None:
        print("[YOLO/비전 계층] Homography mapping degenerate for this pixel")
        return None

    depth_mm = _load_depth_mm(depth_path)
    height_m = estimate_height_m(det, frame.shape[:2], depth_mm) if depth_mm is not None else None
    if height_m is not None:
        z = config.TABLE_Z + height_m - config.DESCEND_MARGIN_M
    else:
        print("[YOLO/비전 계층] No usable depth height estimate - falling back to bare TABLE_Z "
              "(matches visual_servo_pick_place.py's own fallback)")
        z = config.TABLE_Z

    target_roll_deg = DEFAULT_ROLL_DEG if yaw_deg is None else yaw_deg + IMAGE_YAW_TO_WORLD_ROLL_OFFSET_DEG

    return GraspPlan(
        text_prompt=text_prompt, detection=det, xy=xy, height_m=height_m,
        target_xyz=(xy[0], xy[1], z), yaw_deg=yaw_deg, target_roll_deg=target_roll_deg,
    )


def list_objects(rgb_path: str = config.ASTRA_RGB_FRAME_PATH,
                  depth_path: str = config.ASTRA_DEPTH_MM_PATH) -> list[GraspPlan]:
    """[YOLO/비전 계층] Every object perception_zeroshot.detect_all_objects
    finds in the current Astra view, each turned into a full GraspPlan
    (same xy/height/yaw/IK-roll pipeline as plan_grasp) - lets a caller
    pick a grasp target WITHOUT already knowing a text label that matches
    it, per the user's 2026-08-28 ask ("내가 설정하지 않은 것들도 탐지되게
    해줘" - detect things I haven't configured too). See main()'s
    --list/--index flags for the CLI side of this.

    Sorted plausible-tabletop-object-first (a real height_m reading),
    largest area within each group - NOT filtered out on a missing/
    implausible height, only deprioritized, since that's a heuristic (see
    detect_all_objects's docstring on why the robot's own gripper/wrist-
    camera housing tend to fail this check - they sit much higher above
    the table than any real object here does - but it's not a certainty)."""
    ret, frame = PublishedFrameSource(rgb_path).read()
    if not ret or frame is None:
        print(f"[YOLO/비전 계층] No fresh Astra RGB frame at {rgb_path} - is astra_s_live.py running?")
        return []
    homography = _load_homography()
    if homography is None:
        print(f"[YOLO/비전 계층] No homography at {config.HOMOGRAPHY_PATH} - run calibrate_camera.py first")
        return []
    depth_mm = _load_depth_mm(depth_path)

    plans = []
    for det, mask, yaw_deg in detect_all_objects(frame):
        plan = build_grasp_plan(det, yaw_deg, frame.shape[:2], homography, depth_mm)
        if plan is not None:
            plans.append(plan)

    plans.sort(key=lambda p: (p.height_m is None, -p.detection.area))
    return plans


def build_grasp_plan(det: Detection, yaw_deg: float | None, frame_shape_hw: tuple[int, int],
                      homography: np.ndarray, depth_mm: np.ndarray | None,
                      text_prompt: str | None = None) -> GraspPlan | None:
    """[YOLO/비전 계층] Same xy/height/roll construction list_objects uses per
    detection, factored out so any caller that already has a Detection (a
    text-prompt search, an auto-discovered list, or - 2026-08-28 - a user
    CLICK on a displayed box, see zeroshot_click_pick.py) builds an
    identical GraspPlan. None if the homography mapping is degenerate for
    this pixel (should be rare - the homography itself is checked by the
    caller before this is reached)."""
    xy = _xy_from_pixel(homography, det.cx, det.cy)
    if xy is None:
        return None
    height_m = estimate_height_m(det, frame_shape_hw, depth_mm) if depth_mm is not None else None
    z = config.TABLE_Z + height_m - config.DESCEND_MARGIN_M if height_m is not None else config.TABLE_Z
    target_roll_deg = DEFAULT_ROLL_DEG if yaw_deg is None else yaw_deg + IMAGE_YAW_TO_WORLD_ROLL_OFFSET_DEG
    return GraspPlan(
        text_prompt=text_prompt or f"<auto bbox={det.bbox}>", detection=det, xy=xy, height_m=height_m,
        target_xyz=(xy[0], xy[1], z), yaw_deg=yaw_deg, target_roll_deg=target_roll_deg,
    )


def plan_place_xy(text_prompt: str, box_threshold: float = None,
                   rgb_path: str = config.ASTRA_RGB_FRAME_PATH) -> tuple[float, float] | None:
    """[YOLO/비전 계층] Like plan_grasp but for a PLACE target - only needs xy
    (not a height/yaw grasp pose; the already-grasped object's own
    orientation is kept as-is while it's carried, see
    execute_pick_and_place), so this is a separate, simpler function rather
    than overloading GraspPlan with fields that wouldn't apply to a place
    target."""
    from perception_zeroshot import BOX_THRESHOLD as _default_thr

    box_threshold = _default_thr if box_threshold is None else box_threshold

    ret, frame = PublishedFrameSource(rgb_path).read()
    if not ret or frame is None:
        print(f"[YOLO/비전 계층] No fresh Astra RGB frame at {rgb_path}")
        return None
    det = detect_zeroshot(frame, text_prompt, box_threshold)
    if det is None:
        print(f"[YOLO/비전 계층] No detection for place target '{text_prompt}' (threshold={box_threshold})")
        return None
    homography = _load_homography()
    if homography is None:
        print(f"[YOLO/비전 계층] No homography at {config.HOMOGRAPHY_PATH}")
        return None
    return _xy_from_pixel(homography, det.cx, det.cy)


def _solve_for_xyz(kin, current_joint_deg: np.ndarray, target_xyz, target_roll_deg: float,
                    label: str = "move", check: bool = True):
    """[IK/조작 계층] check=True (default) calls _require_converged - every
    LIVE call site is protected against a bad/unreachable IK solution
    automatically, without each caller having to remember to check err_m
    itself. Only the dry-run/inspection call sites pass check=False (they
    print the residual instead of moving, so a bad number is informative,
    not dangerous). Roll is held FIXED (not solved) per the user's
    2026-08-28 decision to keep this over the spec's literal full-
    orientation constraint - see module docstring."""
    joints5 = solve_fixed_roll_ik(kin, current_joint_deg, target_xyz, target_roll_deg)
    err_m = solved_position_error_m(kin, joints5, target_xyz)
    if check:
        _require_converged(err_m, label)
    return joints5, err_m


def solve_plan(plan: GraspPlan, current_joint_deg: np.ndarray, kin=None,
                label: str = "grasp", check: bool = True) -> tuple[np.ndarray, float]:
    """[IK/조작 계층] Returns (5-vector arm joint degrees, position residual
    in meters) for plan.target_xyz at plan.target_roll_deg via grasp_ik's
    fixed-roll solver. kin can be passed in to reuse an already-built
    RobotKinematics (real hardware use, one per session) or left None to
    build a fresh one (dry-run/one-off use)."""
    kin = build_kinematics() if kin is None else kin
    return _solve_for_xyz(kin, current_joint_deg, plan.target_xyz, plan.target_roll_deg, label=label, check=check)


def describe_plan(plan: GraspPlan, joints5: np.ndarray, ik_err_m: float) -> str:
    """[IK/조작 계층] [YOLO/비전 계층]의 GraspPlan 출력과 이를 풀어낸 IK 해를
    함께 사람이 읽을 수 있는 형태로 요약 - 두 계층의 경계에 있는 로그 함수."""
    lines = [
        f"prompt: '{plan.text_prompt}'",
        f"  detection px: cx={plan.detection.cx:.1f} cy={plan.detection.cy:.1f} bbox={plan.detection.bbox}",
        f"  xy (robot frame): ({plan.xy[0]:.4f}, {plan.xy[1]:.4f})",
        f"  height_m: {plan.height_m}",
        f"  target_xyz: ({plan.target_xyz[0]:.4f}, {plan.target_xyz[1]:.4f}, {plan.target_xyz[2]:.4f})",
        f"  yaw_deg (image): {plan.yaw_deg}  ->  target_roll_deg: {plan.target_roll_deg:.1f}",
        f"  solved arm joints (deg): {np.round(joints5, 2).tolist()}",
        f"  IK position residual: {ik_err_m * 1000:.2f}mm",
    ]
    return "\n".join(lines)


def _move_arm(arm: SOArm101, joints5: np.ndarray, gripper_pct: float,
               steps: int = 20, step_delay_s: float = 0.05, stall_check: bool = True) -> None:
    """[IK/조작 계층] Interpolated multi-step move to a precomputed
    (joints5, gripper_pct) target, with the same stall/CollisionDetected
    safety as
    SOArm101.move_to_xyz - reimplemented here (not calling move_to_xyz)
    because that method solves its own IK internally via kinematics.py's
    free-roll solve_ik, which would silently discard the fixed wrist_roll
    this whole module exists to honor.

    2026-08-28, added after a real live run: this function used to be a
    single raw arm.send_joint_deg() call for the whole delta. On that run's
    first real move (a ~90+ degree, multi-joint jump from wherever the arm
    was resting), lerobot's own per-call max_relative_target safety clamp
    (15deg, see kinematics.py's SOFollowerRobotConfig) silently truncated
    it to a pose barely related to the intended target - the arm ended up
    nowhere near the grasp target and the grasp predictably failed. Real
    root cause was this function bypassing move_to_xyz's interpolation, not
    the IK math (confirmed separately - grasp_ik's FK-based residual checks
    were accurate in that same run)."""
    current = arm.get_joint_deg()
    target = np.concatenate([joints5, [gripper_pct]])
    last_good = current.copy()
    stall_count = 0
    for i in range(1, steps + 1):
        interp = current + (target - current) * (i / steps)
        arm.send_joint_deg(interp)
        time.sleep(step_delay_s)
        if not stall_check or i % config.STALL_CHECK_EVERY != 0:
            continue
        actual = arm.get_joint_deg()
        lag = float(np.max(np.abs(actual[: len(config.ARM_JOINTS)] - interp[: len(config.ARM_JOINTS)])))
        if lag > config.STALL_THRESHOLD_DEG:
            stall_count += 1
        else:
            stall_count, last_good = 0, actual
        if stall_count >= config.STALL_CONSECUTIVE:
            arm.send_joint_deg(last_good)
            time.sleep(0.3)
            raise CollisionDetected(
                f"[IK/조작 계층] _move_arm aborted: joint lag {lag:.1f}deg exceeded {config.STALL_THRESHOLD_DEG}deg "
                f"for {config.STALL_CONSECUTIVE} consecutive checks - retreated to last known-good pose."
            )


def close_gripper_and_verify(arm: SOArm101) -> tuple[bool, float]:
    """[Feedback 계층] 파지 성공 여부를 검증 - 이 그리퍼엔 진짜 토크/힘 센서가
    없어서, lerobot의 정규화된 그리퍼 열림 퍼센트를 물리 센싱의 대용치로 사용
    (자세한 근거는 GRIPPER_EMPTY_CLOSED_PCT/GRASP_DETECT_MARGIN_PCT 참고).
    Closes the gripper fully, checks config.GRIPPER_EMPTY_CLOSED_PCT/
    GRASP_DETECT_MARGIN_PCT (same convention as red_cube_calib_pick.py's
    grasp check, just via lerobot's percent-open gripper norm instead of
    raw scservo ticks - this task package uses SOFollower, not raw
    scservo_sdk). On a confirmed grasp, backs off to a gentler hold target
    (GRIP_HOLD_MARGIN_PCT open from the measured close point) instead of
    continuing to command a full/extreme close - see that constant's
    docstring for why."""
    gripper_idx = config.ALL_JOINTS.index("gripper")
    closed = arm.get_joint_deg()
    closed[gripper_idx] = 0.0
    arm.send_joint_deg(closed)
    time.sleep(1.0)
    actual_pct = float(arm.get_joint_deg()[gripper_idx])
    grasped = actual_pct > (config.GRIPPER_EMPTY_CLOSED_PCT + config.GRASP_DETECT_MARGIN_PCT)
    if grasped:
        hi = config.JOINT_LIMITS_DEG["gripper"][1]
        hold_pct = min(actual_pct + GRIP_HOLD_MARGIN_PCT, hi)
        hold = arm.get_joint_deg()
        hold[gripper_idx] = hold_pct
        arm.send_joint_deg(hold)
        time.sleep(0.2)
    return grasped, actual_pct


def execute_grasp(plan: GraspPlan, live: bool = False) -> None:
    """[IK/조작 계층] Pick-only entry point (no place stage) - see
    execute_pick_and_place for the full pick-and-place task. live=False
    (default): solves IK against a fresh RobotKinematics with a neutral
    seed and just prints the plan - never opens the robot port. Internally
    calls into [YOLO/비전 계층] output (plan, already computed by the
    caller) and [Feedback 계층] (close_gripper_and_verify)."""
    if not live:
        kin = build_kinematics()
        joints5, err_m = solve_plan(plan, np.zeros(6), kin, check=False)
        print("[IK/조작 계층] [DRY RUN - no robot connection]")
        print(describe_plan(plan, joints5, err_m))
        return

    if not check_port_not_busy():
        print(f"[IK/조작 계층] {config.FOLLOWER_PORT} is already in use by another process - refusing to connect "
              "(this arm is shared - see orbbec-astra-s-lerobot.md)")
        return

    arm = SOArm101()
    arm.connect()
    try:
        home_joints = arm.get_joint_deg()
        current = home_joints
        _cap_roll_excursion(plan, current[config.ARM_JOINTS.index("wrist_roll")], label="pick")
        joints5, err_m = solve_plan(plan, current, arm.kin, label="pick")
        print(describe_plan(plan, joints5, err_m))

        hover_xyz = (plan.target_xyz[0], plan.target_xyz[1], plan.target_xyz[2] + HOVER_CLEARANCE_M)
        hover_joints, _ = _solve_for_xyz(arm.kin, current, hover_xyz, plan.target_roll_deg, label="pick hover")
        _move_arm(arm, hover_joints, current[5])
        _move_arm(arm, joints5, current[5])

        grasped, actual_pct = close_gripper_and_verify(arm)
        print(f"[Feedback 계층] gripper closed to {actual_pct:.1f}% -> grasped={grasped}")

        if grasped:
            lift_xyz = (plan.target_xyz[0], plan.target_xyz[1], plan.target_xyz[2] + config.LIFT_M)
            lift_joints, _ = _solve_for_xyz(arm.kin, arm.get_joint_deg(), lift_xyz, plan.target_roll_deg, label="lift")
            _move_arm(arm, lift_joints, float(arm.get_joint_deg()[config.ALL_JOINTS.index("gripper")]))
        else:
            open_action = arm.get_joint_deg()
            open_action[config.ALL_JOINTS.index("gripper")] = 90.0
            arm.send_joint_deg(open_action)
    finally:
        try:
            _safe_return_home(arm, home_joints)
        finally:
            arm.disconnect()


def _safe_return_home(arm: SOArm101, home_joints: np.ndarray) -> None:
    """[IK/조작 계층] Interpolated return to home_joints's xyz (not a single
    raw send_joint_deg jump) - matches this project's established
    RETURN_HOME_XYZ convention. Catches broadly (not just CollisionDetected)
    and only ever warns, never raises - callers rely on this to be a true
    best-effort step that can never prevent arm.disconnect() from running.

    2026-08-28: a real live run hit a raw ConnectionError here (a serial
    comms hiccup, "no status packet"), which - back when this only caught
    CollisionDetected - propagated straight out of this function and
    skipped the caller's arm.disconnect() entirely (it was a sibling
    statement in the same finally block, not a nested try/finally). On a
    robot shared with other people, silently leaving a connection/torque
    state open because of an unrelated comms blip is a real hazard - not
    hypothetical, this actually happened. Callers additionally wrap this in
    their own try/finally around arm.disconnect() as defense in depth, but
    this function itself should also never be the thing that skips it."""
    try:
        home_xyz = tuple(gripper_position(arm.kin, home_joints))
        arm.move_to_xyz_converge(home_xyz)
    except CollisionDetected as e:
        print(f"[IK/조작 계층] WARNING: return-to-home move hit CollisionDetected ({e}) - arm retreated to its "
              "last known-good pose but may not be exactly at the original home position.")
    except Exception as e:
        print(f"[IK/조작 계층] WARNING: return-to-home failed ({type(e).__name__}: {e}) - could not confirm the "
              "arm reached its home pose. This arm is shared - physically check its position/torque before "
              "the next use.")


def execute_pick_and_place(pick_prompt: str | None, place_prompt: str, live: bool = False,
                            box_threshold: float = None, pick_plan: GraspPlan | None = None) -> None:
    """[IK/조작 계층] Full task orchestration across all three active layers -
    [YOLO/비전 계층] (plan_grasp/plan_place_xy), [IK/조작 계층] (the actual
    moves), [Feedback 계층] (close_gripper_and_verify). Grasp whatever
    pick_prompt describes (or, if pick_plan is already given - e.g. from
    list_objects()'s auto-discovery, see main()'s --index - use that
    directly and skip plan_grasp/pick_prompt entirely), carry it to and
    release it at place_prompt's location. place_prompt only needs an xy
    (plan_place_xy) - z is config.TABLE_Z + config.BIN_DESCEND_M (this
    project's existing "approach height above the place target" constant,
    not a depth-measured height of the place target itself - a solid
    object there is a real collision risk if BIN_DESCEND_M ever changes
    without checking against the place target's actual height). The grasp
    orientation (target_roll_deg) from the pick is reused unchanged for the
    place move - no reason to re-roll the wrist while an object is already
    held, and doing so would risk disturbing the grip for no benefit."""
    if pick_plan is None:
        pick_plan = plan_grasp(pick_prompt, box_threshold=box_threshold)
        if pick_plan is None:
            print(f"[YOLO/비전 계층] Aborting: no grasp plan for '{pick_prompt}'")
            return
    place_xy = plan_place_xy(place_prompt, box_threshold=box_threshold)
    if place_xy is None:
        print(f"[YOLO/비전 계층] Aborting: no place location for '{place_prompt}'")
        return
    place_xyz = (place_xy[0], place_xy[1], config.TABLE_Z + config.BIN_DESCEND_M)
    place_hover_xyz = (place_xy[0], place_xy[1], place_xyz[2] + HOVER_CLEARANCE_M)

    if not live:
        kin = build_kinematics()
        joints5, err_m = solve_plan(pick_plan, np.zeros(6), kin, check=False)
        print("[IK/조작 계층] [DRY RUN - no robot connection]")
        print(describe_plan(pick_plan, joints5, err_m))
        place_joints, place_err = _solve_for_xyz(kin, np.zeros(6), place_xyz, pick_plan.target_roll_deg, check=False)
        print(f"[IK/조작 계층] place: xy={place_xy} target_xyz={place_xyz} roll={pick_plan.target_roll_deg:.1f} "
              f"joints={np.round(place_joints, 2).tolist()} residual={place_err * 1000:.2f}mm")
        return

    if not check_port_not_busy():
        print(f"[IK/조작 계층] {config.FOLLOWER_PORT} is already in use by another process - refusing to connect "
              "(this arm is shared - see orbbec-astra-s-lerobot.md)")
        return

    arm = SOArm101()
    arm.connect()
    try:
        home_joints = arm.get_joint_deg()
        kin = arm.kin

        # --- pick ---
        current = home_joints
        _cap_roll_excursion(pick_plan, current[config.ARM_JOINTS.index("wrist_roll")], label="pick")
        joints5, err_m = _solve_for_xyz(kin, current, pick_plan.target_xyz, pick_plan.target_roll_deg, label="pick")
        print(describe_plan(pick_plan, joints5, err_m))
        pick_hover_xyz = (pick_plan.target_xyz[0], pick_plan.target_xyz[1], pick_plan.target_xyz[2] + HOVER_CLEARANCE_M)
        pick_hover_joints, _ = _solve_for_xyz(kin, current, pick_hover_xyz, pick_plan.target_roll_deg, label="pick hover")
        _move_arm(arm, pick_hover_joints, current[5])
        _move_arm(arm, joints5, current[5])

        grasped, actual_pct = close_gripper_and_verify(arm)
        print(f"[Feedback 계층] gripper closed to {actual_pct:.1f}% -> grasped={grasped}")
        if not grasped:
            print("[Feedback 계층] Grasp failed - opening gripper, skipping place, returning home.")
            open_action = arm.get_joint_deg()
            open_action[config.ALL_JOINTS.index("gripper")] = 90.0
            arm.send_joint_deg(open_action)
            return
        hold_pct = float(arm.get_joint_deg()[config.ALL_JOINTS.index("gripper")])

        lift_xyz = (pick_plan.target_xyz[0], pick_plan.target_xyz[1], pick_plan.target_xyz[2] + config.LIFT_M)
        lift_joints, _ = _solve_for_xyz(kin, arm.get_joint_deg(), lift_xyz, pick_plan.target_roll_deg, label="lift")
        _move_arm(arm, lift_joints, hold_pct)

        # --- place (same wrist_roll as the pick - see docstring) ---
        place_hover_joints, _ = _solve_for_xyz(kin, arm.get_joint_deg(), place_hover_xyz, pick_plan.target_roll_deg, label="place hover")
        _move_arm(arm, place_hover_joints, hold_pct)

        place_joints, place_err = _solve_for_xyz(kin, arm.get_joint_deg(), place_xyz, pick_plan.target_roll_deg, label="place")
        print(f"[IK/조작 계층] place: xy={place_xy} target_xyz={place_xyz} residual={place_err * 1000:.2f}mm")
        _move_arm(arm, place_joints, hold_pct)

        open_action = arm.get_joint_deg()
        open_action[config.ALL_JOINTS.index("gripper")] = 90.0
        arm.send_joint_deg(open_action)
        time.sleep(0.5)

        retreat_joints, _ = _solve_for_xyz(kin, arm.get_joint_deg(), place_hover_xyz, pick_plan.target_roll_deg, label="retreat")
        _move_arm(arm, retreat_joints, 90.0)
        print("[Feedback 계층] Pick-and-place complete.")  # final task-completion state - reported back per the spec's own layer 5 definition
    finally:
        try:
            _safe_return_home(arm, home_joints)
        finally:
            arm.disconnect()


def _describe_auto_plan(i: int, plan: GraspPlan) -> str:
    """[YOLO/비전 계층] list_objects()가 찾은 물체 하나를 사람이 읽을 수 있는
    한 줄로 요약 - main()의 --list/--index 출력에 쓰인다."""
    flag = "" if plan.height_m is not None else "  [no plausible table height - might be robot hardware, not a real object]"
    return (f"[YOLO/비전 계층] [{i}] xy=({plan.xy[0]:.3f}, {plan.xy[1]:.3f}) height_m={plan.height_m} "
            f"bbox={plan.detection.bbox} yaw_deg={plan.yaw_deg}{flag}")


def main() -> None:
    """CLI entrypoint - spans all three active layers ([YOLO/비전 계층] via
    plan_grasp/list_objects, [IK/조작 계층] via execute_grasp/
    execute_pick_and_place, [Feedback 계층] inside those), so this function
    itself carries no single tag - see the module docstring's layer legend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=None,
                         help="what to grasp, e.g. 'red cube', 'black cup' - omit if using --index")
    parser.add_argument("--list", action="store_true",
                         help="list every object currently visible (no text prompt needed - see "
                              "perception_zeroshot.detect_all_objects) and exit, without moving anything")
    parser.add_argument("--index", type=int, default=None,
                         help="grasp the Nth auto-detected object from --list's numbering, instead of a text "
                              "prompt - for objects you haven't named/configured a prompt for")
    parser.add_argument("--place", default=None, help="if given, also carry the grasped object here and release, "
                                                        "e.g. --place 'black bin'")
    parser.add_argument("--live", action="store_true", help="actually connect to and move the real robot")
    parser.add_argument("--box-threshold", type=float, default=None)
    args = parser.parse_args()

    if args.list:
        plans = list_objects()
        if not plans:
            print("[YOLO/비전 계층] No objects found (or no fresh Astra frame/homography - see message above).")
        for i, p in enumerate(plans):
            print(_describe_auto_plan(i, p))
        return

    if args.index is not None:
        plans = list_objects()
        if not (0 <= args.index < len(plans)):
            print(f"[YOLO/비전 계층] --index {args.index} out of range (0..{len(plans) - 1}); run --list first")
            raise SystemExit(1)
        pick_plan = plans[args.index]
        print(_describe_auto_plan(args.index, pick_plan))
        if args.place:
            execute_pick_and_place(None, args.place, live=args.live, box_threshold=args.box_threshold,
                                    pick_plan=pick_plan)
        else:
            execute_grasp(pick_plan, live=args.live)
        return

    if args.prompt is None:
        parser.error("give a text prompt, or use --list / --index")

    if args.place:
        execute_pick_and_place(args.prompt, args.place, live=args.live, box_threshold=args.box_threshold)
        return

    plan = plan_grasp(args.prompt, box_threshold=args.box_threshold)
    if plan is None:
        raise SystemExit(1)
    execute_grasp(plan, live=args.live)


if __name__ == "__main__":
    main()
