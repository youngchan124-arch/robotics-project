"""Standalone dry-run validation for zeroshot_pick.py's plan_grasp() +
solve_plan() - feeds the same real saved workspace photo
test_perception_zeroshot.py uses in as rgb_path/depth_path (a synthetic but
plausible depth array, since no real Astra depth capture exists for that
saved screenshot), so the FULL pipeline (detect -> segment -> yaw -> real
homography.json -> depth-delta height -> fixed-roll grasp IK) runs end to
end against real detection + real calibration data, without a live camera
or the robot. Never calls execute_grasp(..., live=True).

Run: `uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin/test_zeroshot_pick.py`
from ~/lerobot.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config
from zeroshot_pick import describe_plan, execute_grasp, plan_grasp, solve_plan

SCREENSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "cube_detector (left: annotated, right: raw mask)_screenshot_25.08.2026.png"
)
TMP_RGB_PATH = "/tmp/test_zeroshot_pick_rgb.png"
TMP_DEPTH_PATH = "/tmp/test_zeroshot_pick_depth.npy"


def _make_fixture() -> None:
    """Writes the real saved photo + a synthetic-but-plausible depth array
    to throwaway paths (never the shared production ASTRA_RGB_FRAME_PATH/
    ASTRA_DEPTH_MM_PATH - same test-isolation reasoning as perception.py's
    estimate_cube_height_m tests, see orbbec-astra-s-lerobot.md's
    2026-08-26 note on why that race was a real bug once)."""
    full = cv2.imread(str(SCREENSHOT_PATH))
    frame = full[:, :625]
    cv2.imwrite(TMP_RGB_PATH, frame)

    h, w = frame.shape[:2]
    depth = np.full((h // 2, w // 2), 480, dtype=np.uint16)  # table ~480mm, plausible for this desk setup
    dy0, dy1, dx0, dx1 = 85, 145, 108, 165  # ~ the black cube's bbox (219,171,109,116), halved
    depth[dy0:dy1, dx0:dx1] = 445  # ~35mm cube height, plausible for a real small cube
    np.save(TMP_DEPTH_PATH, depth)


def run() -> bool:
    # Warm up the client BEFORE writing the fixture (cheap for the current
    # Gemini backend - just reads the API key - but harmless to keep for
    # whatever backend is active).
    import perception_zeroshot

    perception_zeroshot._lazy_load()

    _make_fixture()

    # 2026-08-28: since the Gemini backend switch, the SLOW part moved from
    # model loading (before the fixture existed) to the per-call inference
    # itself (~8-9s, AFTER the fixture is written) - by the time plan_grasp
    # gets to reading TMP_DEPTH_PATH, real wall-clock time has passed and
    # FRAME_STALE_TIMEOUT_S (5s, tuned for a REAL continuously-republishing
    # camera - not a stale problem in production, only in this test's
    # static fixture) would false-fail this test on pure API latency, not
    # a real bug (found by actually hitting it once). Temporarily widened
    # just for this test's own duration, restored after.
    original_timeout = config.FRAME_STALE_TIMEOUT_S
    config.FRAME_STALE_TIMEOUT_S = 60.0
    try:
        plan = plan_grasp("black cube", rgb_path=TMP_RGB_PATH, depth_path=TMP_DEPTH_PATH)
    finally:
        config.FRAME_STALE_TIMEOUT_S = original_timeout
    if plan is None:
        print("FAIL: plan_grasp returned None")
        return False

    joints5, err_m = solve_plan(plan, np.zeros(6))
    print(describe_plan(plan, joints5, err_m))

    checks = {
        "xy inside real calibrated table region (x 0.15-0.35, y -0.12-0.12)":
            0.15 <= plan.target_xyz[0] <= 0.35 and -0.12 <= plan.target_xyz[1] <= 0.12,
        "height_m estimated and plausible (5-60mm)":
            plan.height_m is not None and 0.005 <= plan.height_m <= 0.06,
        "target z above bare TABLE_Z (a real height reading was actually used)":
            plan.target_xyz[2] > config.TABLE_Z,
        "IK residual small (<5mm)": err_m < 0.005,
        "wrist_roll exactly matches target_roll_deg (fixed-roll IK honored the yaw)":
            abs(joints5[config.ARM_JOINTS.index("wrist_roll")] - plan.target_roll_deg) < 1e-6,
    }
    all_ok = True
    for name, ok in checks.items():
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
        all_ok &= ok

    print("\n--- execute_grasp(dry_run) end-to-end smoke test (must not touch hardware) ---")
    execute_grasp(plan, live=False)

    print("PASS" if all_ok else "FAIL")
    return all_ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
