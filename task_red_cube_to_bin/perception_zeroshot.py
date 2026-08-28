"""Zero-shot, open-vocabulary object detection + grasp-pose estimation -
replaces perception.py's fixed HSV red-cube/black-bin detectors, per the
2026-08-28 pivot to "Vision -> 3D Grasp Pose -> IK/Motion Planning" (no
teleop demos, no per-object training/HSV tuning - see this session's own
~/.claude memory, orbbec-astra-s-lerobot.md, for the full history of why
HSV+fixed-YOLO was the prior approach). Kept as a SEPARATE module rather
than editing perception.py in place - perception.py's HSV path is real-
hardware-validated and still used by red_cube_calib_pick.py; this is the
new, not-yet-real-hardware-tested path, per the project's own test-before-
hardware convention (see module-level tests in test_perception_zeroshot.py).

Deliberately reuses perception.py's `Detection` dataclass shape (cx, cy,
area, bbox) - detect_zeroshot(text_prompt) below is built to be a drop-in
detect_fn for perception.py's estimate_xy_from_astra/estimate_cube_height_m
(both already real-hardware-validated), so the existing homography-xy +
depth-height-delta coarse-estimate pipeline is reused as-is rather than
rebuilt, per the user's explicit 2026-08-28 choice to validate the new
perception step against the OLD coordinate pipeline first, before ever
doing a proper hand-eye calibration for a full 3D backprojection.

Grasp orientation: for this 5-DOF SO-101 arm, IK is position-only
(config.IK_ORIENTATION_WEIGHT=0.0 - see kinematics.py's docstring: any
nonzero weight blew position error up to 16-253mm across the workspace) -
wrist_roll floats and is NOT settable via solve_ik's target pose.
estimate_grasp_yaw_deg() below is therefore consumed by grasp_ik.py's
fixed-roll solver as an explicit target, not by kinematics.py's solve_ik.

BACKEND HISTORY (2026-08-28):
  1. First version: Grounding DINO + SAM + CLIP - three separate models,
     ~1.8-2s per full detect+label pass.
  2. Replaced with YOLOE (yoloe-26l-seg.pt, Ultralytics' yolo26-based
     model) - one local model, ~0.017s/call, but the red cube's confidence
     stayed weak (~0.05) and occasionally lost to a competing "black cube"
     guess, and robot-hardware exclusion needed a hand-maintained
     ROBOT_PART_LABELS set.
  3. SAME DAY, replaced again with Gemini (Google's gemini-3.6-flash cloud
     vision-language model) per the user's explicit "이걸로 완전히 교체"
     (fully replace with this), after: (a) discovering the viewer's
     "frozen" complaint was actually a real bug (astra_s_live.py's USB
     stream had genuinely stalled after 3+ hours - fixed with `usbreset`,
     unrelated to this module), then (b) the user asking to try LLM-driven
     detection, first manually (Claude looking at a saved frame directly)
     then via a real API. Side-by-side on the same live frame: Gemini never
     confused the red cube with black (YOLOE's top guess was sometimes
     "black cube" at higher confidence than "red cube"), and correctly
     excluded the robot's own gripper/wrist hardware from a plain prompt
     instruction alone - no hand-maintained exclusion list needed as the
     primary mechanism (ROBOT_PART_LABELS is kept only as a cheap secondary
     filter, see label_all_objects).
     Real, load-bearing tradeoffs of this switch:
       - Latency: ~8-9s per call (cloud round-trip) vs YOLOE's ~0.017s -
         over 400x slower. A live viewer polling every ~1.5s is no longer
         realistic; zeroshot_viewer.py's refresh interval was raised
         accordingly (see that file). This backend fits a one-shot
         "detect once, then act" pattern, not continuous polling.
       - No segmentation masks: tested asking Gemini for a base64 mask per
         object - the request didn't even finish in 60s (probably the
         extra output tokens for image data), not usable. What this module
         calls a "mask" for a Gemini detection is just the axis-aligned
         rectangle of its bounding box, not a true per-pixel segmentation -
         estimate_grasp_yaw_deg still runs on it (PCA of a filled rectangle
         reduces to whichever side is longer), which is an approximation:
         fine for near-square objects (yaw comes back None anyway, see
         YAW_EVAL_RATIO_MIN), weaker for genuinely irregular/concave shapes
         where the true grasp axis isn't the same as the bbox's long side.
       - Non-deterministic: repeat calls on the identical frame can return
         a slightly different object list (temperature/model variance) -
         unlike YOLOE, which was exactly reproducible call to call.
       - Needs network + an API key (~/.gemini_api_key or GEMINI_API_KEY
         env var) - a real external dependency this project didn't have
         before (everything else runs fully local/offline).
     Given all of that, this is a genuine trade of speed+determinism for
     accuracy+simplicity - worth remembering if detection needs to run
     inside a tight closed loop again (e.g. a return to wrist-cam visual
     servoing) rather than the current open-loop "one detect, then move"
     design.
"""

from __future__ import annotations

import json
import os
import re

import cv2
import numpy as np

from perception import Detection

# 2026-08-28: switched from gemini-3.6-flash to gemini-flash-lite-latest
# mid-session after hitting the free-tier daily quota (20 requests/day) on
# 3.6-flash - this lite model has its OWN separate quota bucket, so it
# unblocked real-hardware testing immediately rather than waiting out an
# unknown daily reset window. Side-by-side on the same live frame: same
# box accuracy, actually faster (~3.5s vs ~8-9s) - no accuracy downside
# observed, may need to switch back or split load across both if THIS
# quota also runs out.
_GEMINI_MODEL_ID = "gemini-flash-lite-latest"
_GEMINI_API_KEY_FILE = os.path.expanduser("~/.gemini_api_key")

# 2026-08-28: Gemini has no fixed confidence scale to calibrate against the
# way YOLOE's raw objectness score did - it self-reports a 0-1 "confidence"
# per the prompt below, which real testing showed lands high (0.88-0.95)
# for genuine detections. Kept low/permissive since the model's own prompt-
# level exclusion of non-objects/robot-hardware is doing most of the real
# filtering work here, not this threshold.
BOX_THRESHOLD = 0.3
MIN_DISPLAY_SCORE = 0.3
YAW_EVAL_RATIO_MIN = 1.15  # see estimate_grasp_yaw_deg's docstring

# Kept only as an optional hint appended to the detect-everything prompt
# (Gemini doesn't need a closed candidate list the way YOLOE/CLIP did - it's
# genuinely open-vocabulary) and as the exclude-label vocabulary check below
# still uses lowercase multi-word phrases in this same style.
DEFAULT_LABELS = [
    "red cube", "black cube", "blue cube", "green cube", "wooden block",
    "black bin", "trash bin", "small box", "cardboard box",
    "blue glove", "rubber glove", "blue clamp", "orange clip", "plastic clip",
    "electrical cable", "power strip", "battery", "screw", "screwdriver",
    "circuit board", "connector", "bracket", "tool",
    "cup", "bottle", "phone", "pen", "tape", "sponge",
]
CLASSIFY_CROP_PAD_PX = 6  # kept only for _correct_flicker_confusion's crop padding, see below

_client = None


def _lazy_load() -> None:
    """Creates the Gemini client on first real use, not at import time -
    same name/role as the old YOLOE loader (test_zeroshot_pick.py calls
    this directly to warm up before writing its test fixture) even though
    "loading" now means "read the API key and open an HTTP client", not
    loading model weights."""
    global _client
    if _client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key and os.path.exists(_GEMINI_API_KEY_FILE):
            api_key = open(_GEMINI_API_KEY_FILE).read().strip()
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found - set GEMINI_API_KEY or save one to " + _GEMINI_API_KEY_FILE
            )
        _client = genai.Client(api_key=api_key)


_DETECT_PROMPT_TMPL = (
    'Detect {target} in this image. Ignore the robot arm/gripper hardware itself '
    "- only report real objects sitting on the table.\n"
    "For each detected object output a JSON array of objects with:\n"
    '- "label": short lowercase name of the object\n'
    '- "box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000\n'
    '- "confidence": your confidence 0.0-1.0 that this is a correct, real detection\n'
    "Return ONLY the JSON array, no markdown fences, no other text."
)


def _call_gemini(bgr_frame: np.ndarray, target_desc: str) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Runs one Gemini detection call, returns (label, confidence,
    bbox_xywh) tuples in pixel coordinates. Never raises on a malformed/
    empty model response - returns [] instead, same "don't crash the whole
    pipeline on one bad frame" convention every other function in this
    module already follows."""
    _lazy_load()
    from google.genai import types

    ok, buf = cv2.imencode(".png", bgr_frame)
    if not ok:
        return []
    prompt = _DETECT_PROMPT_TMPL.format(target=target_desc)
    resp = _client.models.generate_content(
        model=_GEMINI_MODEL_ID,
        contents=[types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png"), prompt],
    )
    text = (resp.text or "").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        items = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []

    h, w = bgr_frame.shape[:2]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = item.get("box_2d")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x0, y0 = max(0, int(xmin / 1000 * w)), max(0, int(ymin / 1000 * h))
        x1, y1 = min(w, int(xmax / 1000 * w)), min(h, int(ymax / 1000 * h))
        if x1 <= x0 or y1 <= y0:
            continue
        label = str(item.get("label", "object")).strip().lower()
        try:
            score = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            score = 1.0
        out.append((label, score, (x0, y0, x1 - x0, y1 - y0)))
    out.sort(key=lambda t: -t[1])
    return out


def _bbox_mask(frame_shape_hw: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    """See module docstring's "no segmentation masks" tradeoff - a filled
    rectangle standing in for a true per-pixel mask."""
    h, w = frame_shape_hw
    x, y, bw, bh = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[y : y + bh, x : x + bw] = True
    return mask


def estimate_grasp_yaw_deg(mask: np.ndarray) -> float | None:
    """PCA major-axis angle (image-plane degrees, atan2 convention) of the
    mask - the direction an elongated object's long axis runs, so the
    gripper should close ACROSS this (i.e. approach with jaws perpendicular
    to it). None if the mask is too small (<30px), or near-symmetric
    (major/minor eigenvalue ratio < YAW_EVAL_RATIO_MIN - e.g. a cube or a
    round object, where no particular yaw is better than another, so
    forcing one just adds risk for no benefit). Since 2026-08-28's Gemini
    switch, mask is usually just a bbox rectangle (see _bbox_mask) - this
    function doesn't know or care, it works on whatever boolean array it's
    given."""
    ys, xs = np.where(mask)
    if xs.size < 30:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    cov = np.cov((pts - pts.mean(axis=0)).T)
    evals, evecs = np.linalg.eigh(cov)
    if evals[0] <= 0 or evals[1] / evals[0] < YAW_EVAL_RATIO_MIN:
        return None
    major = evecs[:, np.argmax(evals)]
    return float(np.degrees(np.arctan2(major[1], major[0])))


def estimate_height_m(
    det: Detection,
    frame_shape_hw: tuple[int, int],
    depth_mm: np.ndarray,
    height_min_m: float = 0.005,
    height_max_m: float = 0.15,
) -> float | None:
    """Depth-delta height of det above the table, generalized from
    perception.py's estimate_cube_height_m (same median-over-bbox-vs-
    median-over-frame math, no camera-to-base extrinsic needed - see that
    function's docstring) but taking an already-computed Detection + already
    -loaded depth array directly. Backend-agnostic (pure depth math, only
    reads det.bbox) - unaffected by which detector produced det."""
    fh, fw = frame_shape_hw
    sx, sy = depth_mm.shape[1] / fw, depth_mm.shape[0] / fh
    bx, by, bw, bh = det.bbox
    x0, y0 = max(0, int(bx * sx)), max(0, int(by * sy))
    x1, y1 = min(depth_mm.shape[1], int((bx + bw) * sx)), min(depth_mm.shape[0], int((by + bh) * sy))
    if x1 <= x0 or y1 <= y0:
        return None

    obj_valid = depth_mm[y0:y1, x0:x1]
    obj_valid = obj_valid[obj_valid > 0]
    table_valid = depth_mm[depth_mm > 0]
    if obj_valid.size < 5 or table_valid.size < 100:
        return None

    height_m = (float(np.median(table_valid)) - float(np.median(obj_valid))) / 1000.0
    if not (height_min_m <= height_m <= height_max_m):
        return None
    return height_m


def detect_zeroshot(bgr_frame: np.ndarray, text_prompt: str, box_threshold: float = BOX_THRESHOLD) -> Detection | None:
    """Best-scoring Gemini box for text_prompt, in the same Detection shape
    perception.py's HSV detectors return - a drop-in detect_fn for
    estimate_xy_from_astra/estimate_cube_height_m-style callers. None if
    nothing scored above box_threshold. ~8-9s latency - see module
    docstring."""
    results = _call_gemini(bgr_frame, text_prompt.strip().lower())
    results = [r for r in results if r[1] >= box_threshold]
    if not results:
        return None
    _label, _score, (x, y, w, h) = results[0]
    return Detection(cx=x + w / 2.0, cy=y + h / 2.0, area=float(w * h), bbox=(x, y, w, h))


def detect_and_segment(bgr_frame: np.ndarray, text_prompt: str, box_threshold: float = BOX_THRESHOLD):
    """detect_zeroshot + a bbox-rectangle "mask" (see _bbox_mask) + grasp
    yaw, in one Gemini call. Returns (Detection, mask, yaw_deg_or_None) or
    None if nothing was detected."""
    results = _call_gemini(bgr_frame, text_prompt.strip().lower())
    results = [r for r in results if r[1] >= box_threshold]
    if not results:
        return None
    _label, _score, bbox = results[0]
    mask = _bbox_mask(bgr_frame.shape[:2], bbox)
    x, y, w, h = bbox
    det = Detection(cx=x + w / 2.0, cy=y + h / 2.0, area=float(w * h), bbox=bbox)
    yaw_deg = estimate_grasp_yaw_deg(mask)
    return det, mask, yaw_deg


def detect_all_objects(bgr_frame: np.ndarray, candidate_labels: list[str] = DEFAULT_LABELS,
                        min_score: float = MIN_DISPLAY_SCORE,
                        **_kwargs) -> list[tuple[Detection, np.ndarray, float | None]]:
    """"I don't need to name it in advance" detection: every object Gemini
    finds on the table (candidate_labels is accepted only for call-site
    compatibility with the old YOLOE signature - Gemini doesn't need a
    closed candidate list, it's asked for "all distinct physical objects"
    regardless), in the SAME (Detection, mask, yaw_deg) shape
    detect_and_segment returns for a single named object. Largest-area
    first. ~8-9s latency - see module docstring."""
    results = _call_gemini(bgr_frame, "all distinct physical objects on the table")
    out = []
    for _label, score, bbox in results:
        if score < min_score:
            continue
        mask = _bbox_mask(bgr_frame.shape[:2], bbox)
        x, y, w, h = bbox
        det = Detection(cx=x + w / 2.0, cy=y + h / 2.0, area=float(w * h), bbox=bbox)
        yaw_deg = estimate_grasp_yaw_deg(mask)
        out.append((det, mask, yaw_deg))
    out.sort(key=lambda item: -item[0].area)
    return out


# 2026-08-28: kept as a cheap SECONDARY filter, not the primary mechanism
# any more - Gemini's own prompt ("ignore the robot arm/gripper hardware
# itself") already excludes robot parts correctly in real testing, unlike
# the YOLOE era where this exact-label-match set was the only exclusion
# mechanism. Real free-form Gemini labels won't always match one of these
# four phrases exactly, so this only catches the cases where it happens to
# phrase it the same way - a real gap, not treated as a certainty.
ROBOT_PART_LABELS = {"robot gripper", "robot arm part", "camera", "connector"}

# 2026-08-28: the user reported the red cube sometimes gets classified as
# "black cube" ("플리커 때문에" - because of flicker) - this was against the
# CLIP-based backend, and real testing since (both YOLOE and Gemini) has
# NOT reproduced the confusion - Gemini in particular never once returned
# "black" for the red cube across several live test calls. Kept anyway as
# a harmless, cheap safety net (only fires on the one confirmed historical
# failure pattern - a "black ..." label - every other label passes through
# unchanged) since it doesn't care which model produced the label, just the
# crop's own HSV Hue/Saturation.
RED_HUE_RANGES = ((0, 15), (165, 180))  # OpenCV's 0-179 hue scale; generous band around perception.py's own red range
DARK_LABEL_CONFUSABLE = {"black cube", "black bin", "small box", "black box"}
MIN_SATURATION_FOR_COLOR = 15  # real black cube crop measured median S~0-4; real red cube stayed >=31 to 20% brightness


def _median_hue_saturation(bgr_frame: np.ndarray, bbox: tuple[int, int, int, int],
                            pad: int = CLASSIFY_CROP_PAD_PX) -> tuple[float, float]:
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr_frame.shape[1], x + w + pad), min(bgr_frame.shape[0], y + h + pad)
    hsv = cv2.cvtColor(bgr_frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h_ch, s_ch, _v_ch = cv2.split(hsv)
    return float(np.median(h_ch)), float(np.median(s_ch))


def _correct_flicker_confusion(bgr_frame: np.ndarray, bbox: tuple[int, int, int, int],
                                label: str, score: float) -> tuple[str, float]:
    """See RED_HUE_RANGES/MIN_SATURATION_FOR_COLOR's comment above - only
    touches the one historical failure mode (a dark-toned red object read
    as a "black ..." label); every other label passes through unchanged."""
    if label not in DARK_LABEL_CONFUSABLE:
        return label, score
    hue, sat = _median_hue_saturation(bgr_frame, bbox)
    in_red_band = any(lo <= hue <= hi for lo, hi in RED_HUE_RANGES)
    if sat >= MIN_SATURATION_FOR_COLOR and in_red_band:
        return "red cube", score
    return label, score


def label_all_objects(bgr_frame: np.ndarray, candidate_labels: list[str] = DEFAULT_LABELS,
                       exclude_labels: set[str] = ROBOT_PART_LABELS, min_score: float = MIN_DISPLAY_SCORE,
                       **_kwargs) -> list[tuple[Detection, np.ndarray, float | None, str, float]]:
    """Every object Gemini finds on the table, WITH its name+confidence
    (unlike detect_all_objects, which discards the label) - the "bounding
    box + name" pipeline behind zeroshot_pick.py's --list and the live
    viewer (zeroshot_viewer.py). Drops any candidate whose label is in
    exclude_labels or scores below min_score. Applies
    _correct_flicker_confusion first, so a corrected "red cube" is judged
    by its OWN min_score, not silently dropped for having originally been a
    low-confidence "black cube" guess. ~8-9s latency - see module
    docstring; zeroshot_viewer.py's refresh interval was raised to match."""
    results = _call_gemini(bgr_frame, "all distinct physical objects on the table")
    out = []
    for label, score, bbox in results:
        label, score = _correct_flicker_confusion(bgr_frame, bbox, label, score)
        if label in exclude_labels or score < min_score:
            continue
        mask = _bbox_mask(bgr_frame.shape[:2], bbox)
        x, y, w, h = bbox
        det = Detection(cx=x + w / 2.0, cy=y + h / 2.0, area=float(w * h), bbox=bbox)
        yaw_deg = estimate_grasp_yaw_deg(mask)
        out.append((det, mask, yaw_deg, label, score))
    return out


def draw_labeled_boxes(bgr_frame: np.ndarray,
                        labeled: list[tuple[Detection, np.ndarray, float | None, str, float]]) -> np.ndarray:
    """Annotates a COPY of bgr_frame with each entry's bbox + "label score"
    text - shared by zeroshot_pick.py's --list-viz (if added) and
    zeroshot_viewer.py, so the two never draw this differently."""
    vis = bgr_frame.copy()
    for det, _mask, _yaw, label, score in labeled:
        x, y, w, h = det.bbox
        color = (0, 255, 0) if score >= 0.6 else (0, 165, 255)  # 2026-08-28: Gemini's own confidence scale (0.88-0.95 typical for real hits)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 4, y)
        cv2.rectangle(vis, (x, ty - th - 4), (x + tw + 4, ty), color, -1)
        cv2.putText(vis, text, (x + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis
