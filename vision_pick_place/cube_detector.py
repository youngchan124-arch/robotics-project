"""Red cube detection via HSV color thresholding.

Red wraps around the HSV hue circle (0 and 180 are both "red" in OpenCV's 0-179
hue range), so two ranges are thresholded and combined rather than one.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# V(alue)/S(aturation) lower bounds widened from an earlier (70, 110) after sampling
# actual miss frames directly: under the Astra S's current lighting/exposure, large
# parts of the cube regularly read V=19-90 and S as low as 9-50 - a genuinely dim,
# partly-desaturated red on this camera, not occasional noise. The old V>=70 alone
# was throwing out a large fraction of real cube pixels. If detection is still
# unreliable, run this file directly (python cube_detector.py) to see the live mask
# overlay and adjust these against real frames rather than guessing again.
LOWER_RED_1 = (0, 60, 25)
UPPER_RED_1 = (10, 255, 255)
LOWER_RED_2 = (170, 60, 25)
UPPER_RED_2 = (180, 255, 255)

MIN_CONTOUR_AREA = 200  # px^2 - filters out small red noise/specular highlights

# 2026-08-26: color thresholding alone can't tell a cube from a hand/arm/
# cable that happens to fall in the same hue range (a hand under reddish
# lighting, or the maroon robot part already documented as a real false
# positive earlier this session) - if that blob is bigger than the actual
# cube's, "pick the largest contour" picks the wrong thing. Shape filtering
# doesn't need color to be perfect: a cube's silhouette is a solid, convex,
# roughly-square blob from any rotation; a hand's silhouette has fingers
# (concave, low solidity) and a forearm is long and thin (extreme aspect
# ratio) - neither looks like a cube regardless of color match. Candidates
# are now filtered on solidity + aspect ratio + a max-area cap (a blob
# covering most of the frame is far more likely to be an intruding hand/arm
# than the cube itself) before picking the largest survivor, instead of
# trusting color + size alone.
FRAME_AREA_HINT = 640 * 480  # both cameras run at this resolution as of 2026-08-26
MAX_CONTOUR_AREA_FRAC = 0.5  # a blob bigger than half the frame is not "the cube"
# 2026-08-26, same day as the filter was added: it immediately broke a real
# detection - a genuine cube frame measured solidity=0.837 (the cube's
# printed diamond/texture pattern + dim lighting notches the mask a little,
# not concave "fingers"), just under the original 0.85, so detect_red_cube
# returned None on a perfectly good frame. A synthetic hand-shaped blob
# measures ~0.41 - there's a wide gap between real texture noise (~0.84) and
# an actual concave hand silhouette (~0.4), so this only needed to be well
# below the noisy-cube reading, not right above it.
MIN_SOLIDITY = 0.65  # contour_area / convex_hull_area - a hand's fingers drag this down
ASPECT_RATIO_RANGE = (0.4, 2.5)  # w/h of the bounding box - rejects long thin shapes (an arm)


def _passes_shape_filter(contour, min_solidity: float, aspect_range: tuple[float, float]) -> bool:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = cv2.contourArea(contour) / hull_area
    if solidity < min_solidity:
        return False
    _, _, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False
    lo, hi = aspect_range
    return lo <= (w / h) <= hi


def _best_candidate(contours, min_area: float, max_area: float, min_solidity: float, aspect_range: tuple):
    """Largest contour that passes the area cap and shape filter, or None if
    nothing qualifies - used instead of a bare max(contours, key=area) so a
    big non-cube/non-bin blob (hand, arm, cable) can't win just by being the
    single largest red/black region in frame."""
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        if not _passes_shape_filter(c, min_solidity, aspect_range):
            continue
        if area > best_area:
            best, best_area = c, area
    return best, best_area


def is_frame_corrupted(bgr_frame: np.ndarray) -> bool:
    """Flags the two USB frame-tearing patterns actually observed on the wrist
    camera (a cheap generic UVC webcam, separate from the Orbbec units, that
    turned out to be the flakiest link here): (a) a noisy multicolor band -
    many rows in a row with an abnormally large jump from the row above, and
    (b) a solid anomalous color block - a tall run of near-identical rows
    (an all-zero/garbage USB transfer decodes to a flat, often greenish,
    block) whose color sits far from the rest of the frame's average. Real
    photographic content varies gradually row to row and doesn't have a big
    flat patch that differs sharply from its surroundings, so either pattern
    is a reasonable proxy for "this frame is garbage, don't trust it" without
    hardcoding the specific corruption color."""
    row_means = bgr_frame.mean(axis=1)  # (H, 3)
    diffs = np.abs(np.diff(row_means, axis=0)).sum(axis=1)  # (H-1,)

    noisy_band = int((diffs > 25).sum()) >= 4 or bool((diffs > 100).any())

    flat = diffs < 3
    max_run = 0
    run = 0
    best_start = 0
    for i, f in enumerate(flat):
        if f:
            run += 1
            if run > max_run:
                max_run = run
                best_start = i - run + 1
        else:
            run = 0
    h = bgr_frame.shape[0]
    block_color_far = False
    if max_run >= h * 0.12:
        block_mean = row_means[best_start : best_start + max_run].mean(axis=0)
        overall_mean = row_means.mean(axis=0)
        block_color_far = float(np.abs(block_mean - overall_mean).sum()) > 60

    return bool(noisy_band or block_color_far)


@dataclass
class Detection:
    cx: float  # pixel x of the cube's centroid
    cy: float  # pixel y of the cube's centroid
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


def detect_red_cube(bgr_frame: np.ndarray) -> Detection | None:
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up small speckles / fill small gaps before finding contours.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest, area = _best_candidate(
        contours, MIN_CONTOUR_AREA, FRAME_AREA_HINT * MAX_CONTOUR_AREA_FRAC, MIN_SOLIDITY, ASPECT_RATIO_RANGE
    )
    if largest is None:
        return None

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    bbox = cv2.boundingRect(largest)
    return Detection(cx=cx, cy=cy, area=area, bbox=bbox)


# Black bin: low value (dark), any hue, low-to-moderate saturation. Provisional
# thresholds - not sampled from real miss frames yet the way the cube's were.
# If bin detection is unreliable, run this file with --target bin (see __main__)
# and look at the saved /tmp/bin_miss_*.png snapshots the same way the cube's
# thresholds were tuned, rather than guessing again blind.
LOWER_BLACK = (0, 0, 0)
UPPER_BLACK = (180, 90, 70)
MIN_BIN_CONTOUR_AREA = 800  # bigger than the cube - the bin is a bigger object

# Same reasoning as the cube's shape filter above: a dark sleeve/glove, a
# shadow, a black cable (a real one is visible draped across this exact
# workspace - see astra_s_live.py session notes), or dark robot parts can all
# fall in this same low-value color range. Solidity is a bit more lenient
# than the cube's (a bin can have a visible rim/handle denting its
# silhouette a little) but a cable in particular has near-zero solidity and
# an extreme aspect ratio, so this alone rules most of that out.
# Lowered alongside MIN_SOLIDITY above for the same reason (a real cube frame
# measured 0.837, well under the cube's original 0.85 - the bin has no real-
# frame measurement yet, so its threshold gets the same safety margin down
# to a value still far above an actual hand's ~0.41 rather than guessing a
# tight number close to the untested nominal case).
MAX_BIN_CONTOUR_AREA_FRAC = 0.7  # the bin can legitimately fill more of the frame than the cube
MIN_BIN_SOLIDITY = 0.6
BIN_ASPECT_RATIO_RANGE = (0.3, 3.0)


def detect_black_bin(bgr_frame: np.ndarray) -> Detection | None:
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest, area = _best_candidate(
        contours,
        MIN_BIN_CONTOUR_AREA,
        FRAME_AREA_HINT * MAX_BIN_CONTOUR_AREA_FRAC,
        MIN_BIN_SOLIDITY,
        BIN_ASPECT_RATIO_RANGE,
    )
    if largest is None:
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    bbox = cv2.boundingRect(largest)
    return Detection(cx=cx, cy=cy, area=area, bbox=bbox)


def detect_black_bin_mask(bgr_frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_red_cube_mask(bgr_frame: np.ndarray) -> np.ndarray:
    """Same thresholding as detect_red_cube, but returns the cleaned binary mask
    directly - for diagnosing *why* detection drops a frame (color range too
    tight, lighting, camera artifact) rather than just seeing it drop."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


class SmoothedDetector:
    """Holds onto the last detection for a few frames when a single frame
    momentarily loses the cube (camera frame noise/tearing, brief occlusion),
    instead of flickering None every time detect_red_cube misses one frame.
    Only holds *position* - a truly gone cube still reads as gone once the
    hold expires, it doesn't paper over a real disappearance forever."""

    def __init__(self, hold_frames: int = 5):
        self.hold_frames = hold_frames
        self._last: Detection | None = None
        self._missing_streak = 0

    def update(self, bgr_frame: np.ndarray) -> Detection | None:
        det = detect_red_cube(bgr_frame)
        if det is not None:
            self._last = det
            self._missing_streak = 0
            return det
        self._missing_streak += 1
        if self._last is not None and self._missing_streak <= self.hold_frames:
            return self._last
        self._last = None
        return None


def draw_detection(bgr_frame: np.ndarray, det: Detection | None) -> np.ndarray:
    out = bgr_frame.copy()
    if det is not None:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (int(det.cx), int(det.cy)), 5, (0, 255, 0), -1)
        cv2.putText(
            out, f"({det.cx:.0f},{det.cy:.0f}) area={det.area:.0f}",
            (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
    return out


class _OpenCVCameraAdapter:
    """Same isOpened()/read()/release() shape as ThreadedOrbbecColorCamera, backed by
    a plain cv2.VideoCapture - lets the rest of this script's loop stay identical
    when testing detect_red_cube against a different, non-Astra camera."""

    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()

    def release(self) -> None:
        self.cap.release()


if __name__ == "__main__":
    # Standalone live check:
    #   python cube_detector.py             -> Astra S
    #   python cube_detector.py --source 4  -> plain UVC webcam at /dev/video4
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="astra",
        help="'astra' (default) or a /dev/videoN index to test a different camera model",
    )
    args = parser.parse_args()

    if args.source == "astra":
        from orbbec_color_camera import ThreadedOrbbecColorCamera

        cam = ThreadedOrbbecColorCamera()
        cam_desc = "Astra S"
    else:
        cam = _OpenCVCameraAdapter(int(args.source))
        cam_desc = f"/dev/video{args.source}"

    if not cam.isOpened():
        print(f"[cube_detector] {cam_desc}를 열 수 없습니다.")
        raise SystemExit(1)

    print("빨간 큐브 검출 미리보기 (컬러 | 마스크). raw=이번 프레임만, smoothed=최근 5개 '실제' 프레임 유지. 'q'로 종료.")
    # hold_frames counts *distinct camera frames*, not display-loop iterations - our
    # display loop was running at ~150-180Hz while the camera only delivers new
    # frames much slower, so most reads were re-processing the same buffer. That
    # alone was harmless, but it meant a single real miss got counted 10-15x against
    # the smoother's hold budget, exhausting it almost immediately - the actual bug
    # behind frames "flickering" despite smoothing. Gating everything below on
    # is_new_frame fixes both that and the wasted CPU.
    smoother = SmoothedDetector(hold_frames=5)
    last_frame_id: int | None = None
    annotated = None
    mask_bgr = None
    new_frame_count = 0
    raw_hit_count = 0
    miss_streak = 0
    max_miss_streak = 0
    snapshot_count = 0
    t_report = time.time()
    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_id = hash(frame.tobytes()[::997])  # sparse sample, cheap hash
            is_new_frame = frame_id != last_frame_id
            last_frame_id = frame_id

            if is_new_frame:
                raw_det = detect_red_cube(frame)
                smoothed_det = smoother.update(frame)
                mask = detect_red_cube_mask(frame)

                new_frame_count += 1
                if raw_det is not None:
                    raw_hit_count += 1
                    miss_streak = 0
                else:
                    miss_streak += 1
                    max_miss_streak = max(max_miss_streak, miss_streak)
                    if miss_streak == 3 and snapshot_count < 6:
                        # A run of 3+ real consecutive misses is long enough to be
                        # more than single-frame noise - save what the camera and
                        # mask actually looked like during it, to see *why* rather
                        # than keep guessing at thresholds blind.
                        snapshot_count += 1
                        cv2.imwrite(f"/tmp/cube_miss_{snapshot_count}_frame.png", frame)
                        cv2.imwrite(f"/tmp/cube_miss_{snapshot_count}_mask.png", mask)
                        print(f"[diag] saved miss snapshot {snapshot_count} (streak={miss_streak})")

                if time.time() - t_report >= 1.0:
                    print(
                        f"[diag] {new_frame_count} new frames/s from camera, raw hit-rate "
                        f"{raw_hit_count}/{new_frame_count} "
                        f"({100 * raw_hit_count / max(new_frame_count, 1):.0f}%), "
                        f"longest miss streak this window={max_miss_streak} frames "
                        f"(~{max_miss_streak * 33}ms)"
                    )
                    new_frame_count = 0
                    raw_hit_count = 0
                    max_miss_streak = 0
                    t_report = time.time()

                annotated = draw_detection(frame, smoothed_det)
                label = "RAW: MISS" if raw_det is None else "RAW: hit"
                cv2.putText(
                    annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if raw_det is None else (0, 255, 0), 2,
                )
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                h = annotated.shape[0]
                mask_bgr = cv2.resize(mask_bgr, (int(mask_bgr.shape[1] * h / mask_bgr.shape[0]), h))
            # else: redisplay the last computed overlay - no new data to process yet.

            if annotated is not None:
                cv2.imshow("cube_detector (left: annotated, right: raw mask)", cv2.hconcat([annotated, mask_bgr]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
