"""Standalone validation for perception_zeroshot.py against a REAL saved
workspace frame - no camera, no robot, same "prove it works before ever
touching hardware" convention as the sibling sim_dry_run.py files, except
here a synthetic image would defeat the point (the whole thing being tested
is whether a real vision-language model actually finds real objects), so
this uses an actual saved photo instead: the existing
`cube_detector (left: annotated, right: raw mask)_screenshot_25.08.2026.png`
in the parent vision_pick_place/ dir (a real Astra-view frame with a red
cube, a black cube, cables, and other workspace clutter - the annotated
overlay is cropped away, keeping just the left raw-camera half).

Run: `uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin/test_perception_zeroshot.py`
from ~/lerobot (needs the main venv's torch/transformers/cv2). Downloads
~2 small HF models on first run if not already cached.

Saves an annotated viz to /tmp/perception_zeroshot_test.png for manual
inspection - same "actually look at the saved frame" habit that caught the
GRASP_TARGET_PX offset and wrist-cam overexposure bugs earlier in this
project (see orbbec-astra-s-lerobot.md).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from perception_zeroshot import detect_and_segment

SCREENSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "cube_detector (left: annotated, right: raw mask)_screenshot_25.08.2026.png"
)
VIZ_OUT_PATH = "/tmp/perception_zeroshot_test.png"

# The black cube's known real extent in this specific saved frame (read off
# the annotated screenshot by eye) - a loose sanity bound, not a tight
# pixel-perfect check, since the point is "did it find roughly the right
# object", not exact reproducibility.
EXPECTED_CENTER_PX = (270.0, 230.0)
EXPECTED_CENTER_TOL_PX = 40.0


def load_test_frame() -> np.ndarray:
    full = cv2.imread(str(SCREENSHOT_PATH))
    if full is None:
        raise FileNotFoundError(f"Test screenshot not found: {SCREENSHOT_PATH}")
    return full[:, :625]  # drop the right-side mask-visualization panel


def run() -> bool:
    frame = load_test_frame()
    result = detect_and_segment(frame, "black cube")
    if result is None:
        print("FAIL: no detection for 'black cube' at all")
        return False
    det, mask, yaw_deg = result

    dx = abs(det.cx - EXPECTED_CENTER_PX[0])
    dy = abs(det.cy - EXPECTED_CENTER_PX[1])
    center_ok = dx <= EXPECTED_CENTER_TOL_PX and dy <= EXPECTED_CENTER_TOL_PX
    mask_area = int(mask.sum())
    mask_ok = mask_area > 500  # a real cube-sized mask, not a sliver

    print(f"detection: cx={det.cx:.1f} cy={det.cy:.1f} bbox={det.bbox} area={det.area:.0f}")
    print(f"  center within {EXPECTED_CENTER_TOL_PX}px of expected {EXPECTED_CENTER_PX}: {center_ok} (dx={dx:.1f} dy={dy:.1f})")
    print(f"mask: {mask_area}px  ok={mask_ok}")
    print(f"grasp yaw: {yaw_deg}")

    vis = frame.copy()
    vis[mask] = (vis[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    x, y, w, h = det.bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.circle(vis, (int(det.cx), int(det.cy)), 4, (0, 0, 255), -1)
    if yaw_deg is not None:
        L = 60
        rad = np.radians(yaw_deg)
        p1 = (int(det.cx - L * np.cos(rad)), int(det.cy - L * np.sin(rad)))
        p2 = (int(det.cx + L * np.cos(rad)), int(det.cy + L * np.sin(rad)))
        cv2.line(vis, p1, p2, (0, 165, 255), 2)
    cv2.imwrite(VIZ_OUT_PATH, vis)
    print(f"viz saved to {VIZ_OUT_PATH}")

    passed = center_ok and mask_ok
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
