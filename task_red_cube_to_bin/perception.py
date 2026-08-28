"""Object detection and 3D-ish coordinate estimation.

Per the spec's §4: HSV color segmentation is the primary method (no GPU/
training data needed, low latency, very stable for a single solid-color
object) - the interface (`Detection`, `detect_red_cube`, `detect_black_bin`,
each taking a raw BGR frame and returning a `Detection | None`) is deliberately
narrow so a YOLO-based implementation could be swapped in later without
touching any caller, if HSV's real failure mode (lighting sensitivity, or a
same-colored object in frame) ever proves worse in practice than it has been
today. It was tried today (YOLO-World, zero-shot, no training data available)
and performed worse than this HSV+shape approach on the actual objects in
this workspace - see this task's design notes.

Coordinate estimation is NOT a full pixel+depth -> camera-frame 3D point ->
base-frame transform via a proper 6-DOF T_cam_to_base extrinsic - that would
need a real hand-eye calibration (ArUco/checkerboard + cv2.calibrateHandEye),
which doesn't exist yet. What's used instead, and what's actually validated
against real hardware:
  - xy: a 2D homography (Astra RGB pixel -> robot-base xy on the table
    plane), from calibrate_camera.py's touch-point calibration
    (homography.json). Only valid for objects sitting on the table plane -
    which is exactly this task.
  - z: not from Astra depth's absolute reading at all (no camera-to-base
    transform to turn that into a base-frame coordinate) - instead a HEIGHT
    DELTA (table depth reading minus cube depth reading, both from Astra),
    which needs no extrinsics to be useful, added on top of the
    independently-measured TABLE_Z.
This is a coarse guess only - FINE_SERVO (see task_state_machine.py) is what
actually gets the gripper precisely onto the object, using the wrist camera
in closed loop, not trusting this estimate's absolute accuracy.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class Detection:
    cx: float  # pixel x of the object's centroid
    cy: float  # pixel y of the object's centroid
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class PublishedFrameSource:
    """Reads whatever camera_hub.py/astra_s_live.py last published to `path`
    instead of opening the camera device itself (see this module's docstring
    on why). Same cv2.VideoCapture-shaped isOpened()/read()/release() so it
    can substitute for one."""

    def __init__(self, path: str, stale_timeout_s: float = config.FRAME_STALE_TIMEOUT_S):
        self.path = path
        self.stale_timeout_s = stale_timeout_s

    def _fresh(self) -> bool:
        return os.path.exists(self.path) and (time.time() - os.path.getmtime(self.path)) < self.stale_timeout_s

    def isOpened(self) -> bool:
        return self._fresh()

    def read(self):
        if not self._fresh():
            return False, None
        frame = cv2.imread(self.path)
        return (frame is not None), frame

    def release(self) -> None:
        pass


def is_frame_corrupted(bgr_frame: np.ndarray) -> bool:
    """Flags USB frame-tearing (a real, validated issue on the cheap wrist
    UVC camera specifically): (a) a noisy multicolor band - several rows in
    a row with an abnormally large jump from the row above, and (b) a solid
    anomalous color block - a tall run of near-identical rows (an all-
    zero/garbage USB transfer decodes to a flat, often greenish block) whose
    color sits far from the rest of the frame's average."""
    row_means = bgr_frame.mean(axis=1)
    diffs = np.abs(np.diff(row_means, axis=0)).sum(axis=1)
    noisy_band = int((diffs > 25).sum()) >= 4 or bool((diffs > 100).any())

    flat = diffs < 3
    max_run = run = best_start = 0
    for i, f in enumerate(flat):
        if f:
            run += 1
            if run > max_run:
                max_run, best_start = run, i - run + 1
        else:
            run = 0
    h = bgr_frame.shape[0]
    block_color_far = False
    if max_run >= h * 0.12:
        block_mean = row_means[best_start : best_start + max_run].mean(axis=0)
        block_color_far = float(np.abs(block_mean - row_means.mean(axis=0)).sum()) > 60
    return bool(noisy_band or block_color_far)


def _best_candidate(contours, min_area, max_area, min_solidity, aspect_range):
    """Largest contour passing an area range + shape filter (solidity,
    aspect ratio) - not just the largest same-colored blob, so a hand/arm/
    cable in frame can't out-vote the real object. See config.py's
    MIN_CUBE_SOLIDITY comment for how these thresholds were tuned against a
    real miss."""
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or (area / hull_area) < min_solidity:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if h == 0 or not (aspect_range[0] <= (w / h) <= aspect_range[1]):
            continue
        if area > best_area:
            best, best_area = c, area
    return best, best_area


def _detect(bgr_frame, lower_ranges_upper, min_area, max_area_frac, min_solidity, aspect_range, kernel_size):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in lower_ranges_upper:
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c, area = _best_candidate(contours, min_area, config.FRAME_AREA_HINT * max_area_frac, min_solidity, aspect_range)
    if c is None:
        return None
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    bbox = cv2.boundingRect(c)
    return Detection(cx=m["m10"] / m["m00"], cy=m["m01"] / m["m00"], area=area, bbox=bbox)


def detect_red_cube(bgr_frame: np.ndarray) -> Detection | None:
    return _detect(
        bgr_frame,
        [(config.LOWER_RED_1, config.UPPER_RED_1), (config.LOWER_RED_2, config.UPPER_RED_2)],
        config.MIN_CUBE_CONTOUR_AREA, config.MAX_CUBE_AREA_FRAC, config.MIN_CUBE_SOLIDITY, config.CUBE_ASPECT_RANGE,
        kernel_size=5,
    )


def detect_black_bin(bgr_frame: np.ndarray) -> Detection | None:
    return _detect(
        bgr_frame,
        [(config.LOWER_BLACK, config.UPPER_BLACK)],
        config.MIN_BIN_CONTOUR_AREA, config.MAX_BIN_AREA_FRAC, config.MIN_BIN_SOLIDITY, config.BIN_ASPECT_RANGE,
        kernel_size=7,
    )


def _load_homography() -> np.ndarray | None:
    if not config.HOMOGRAPHY_PATH.exists():
        return None
    with open(config.HOMOGRAPHY_PATH) as f:
        data = json.load(f)
    return np.array(data["homography"], dtype=float)


_HOMOGRAPHY = _load_homography()


def estimate_xy_from_astra(
    detect_fn, rgb_path: str = config.ASTRA_RGB_FRAME_PATH, homography: np.ndarray | None = _HOMOGRAPHY
) -> tuple[float, float] | None:
    """Coarse robot-base (x, y) for whatever detect_fn finds in the Astra
    RGB view, via the table-plane homography - see this module's docstring
    for why this isn't a full 3D backprojection. None if there's no
    homography yet, Astra isn't running/in view, or nothing detected."""
    if homography is None:
        return None
    ret, color = PublishedFrameSource(rgb_path).read()
    if not ret or color is None:
        return None
    det = detect_fn(color)
    if det is None:
        return None
    mapped = homography @ np.array([det.cx, det.cy, 1.0])
    if abs(mapped[2]) < 1e-9:
        return None
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def estimate_cube_height_m(
    rgb_path: str = config.ASTRA_RGB_FRAME_PATH, depth_path: str = config.ASTRA_DEPTH_MM_PATH
) -> float | None:
    """Astra-depth height DELTA for the detected cube above the table -
    table depth (median over the whole frame - the table dominates the
    view) minus cube depth (median within the detection's bbox, scaled into
    depth's lower native resolution). No camera-to-base extrinsic needed for
    a delta - see this module's docstring. None if unavailable/implausible;
    every caller must have a TABLE_Z fallback for exactly that reason."""
    ret, color = PublishedFrameSource(rgb_path).read()
    if not ret or color is None:
        return None
    det = detect_red_cube(color)
    if det is None:
        return None
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= config.FRAME_STALE_TIMEOUT_S:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None

    sx, sy = depth_mm.shape[1] / color.shape[1], depth_mm.shape[0] / color.shape[0]
    bx, by, bw, bh = det.bbox
    x0, y0 = max(0, int(bx * sx)), max(0, int(by * sy))
    x1, y1 = min(depth_mm.shape[1], int((bx + bw) * sx)), min(depth_mm.shape[0], int((by + bh) * sy))
    if x1 <= x0 or y1 <= y0:
        return None

    cube_valid = depth_mm[y0:y1, x0:x1]
    cube_valid = cube_valid[cube_valid > 0]
    table_valid = depth_mm[depth_mm > 0]
    if cube_valid.size < 5 or table_valid.size < 100:
        return None

    height_m = (float(np.median(table_valid)) - float(np.median(cube_valid))) / 1000.0
    if not (config.CUBE_HEIGHT_MIN_M <= height_m <= config.CUBE_HEIGHT_MAX_M):
        return None
    return height_m
