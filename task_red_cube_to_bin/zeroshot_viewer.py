"""Live viewer: every object perception_zeroshot.label_all_objects finds in
the current Astra view, boxed and named, continuously - the visual form of
the user's 2026-08-28 ask ("카메라로 탐지하는거 다 바운딩 박스로 이름까지
붙여보자" - box AND name everything the camera detects). Read-only: never
opens the robot port, never touches astra_s_live.py's camera device (reads
its published frame like every other consumer in this project).

Detection+segmentation+per-object naming is slow (SAM automatic mode +
Grounding DINO-tiny + a CLIP classification per object - about 1.8s total,
not real-time), so this follows the same pattern red_cube_calib_pick.py's
2026-08-27 fix established for exactly this problem (see
orbbec-astra-s-lerobot.md): a BACKGROUND thread does the slow labeling work
on its own cadence while the main thread keeps calling imshow/waitKey every
loop tick.

2026-08-28: originally the background thread rendered a full annotated
IMAGE and the main thread just redisplayed that same static image for the
whole ~1.5-3s between labeling passes - technically "live" (each new frame
was real), but the user correctly called this choppy: the picture only
actually changed once every couple seconds, not smoothly. Fixed by
decoupling what updates at what rate: the background thread now only
computes DATA (the labeled-object list), and the main thread re-reads the
raw Astra frame and redraws the boxes on it fresh EVERY tick (~30ms,
whatever astra_s_live.py's own publish rate allows) - the video background
is now genuinely smooth/live, while the box positions+names themselves
still only refresh every ~1.5-3s (acceptable for a mostly-static tabletop
scene, and the actual expensive part).

Also atomically publishes each rendered frame to FULL_VIEW_PATH (same
atomic_write convention as astra_s_live.py/camera_hub.py, throttled - see
FILE_PUBLISH_INTERVAL_S) so the window's exact content can be inspected
without a GUI too - e.g. via Claude's own Read tool on a saved PNG, the
same "actually look at the saved frame" habit that caught real bugs
earlier in this project.

Run standalone via `uv run python3 zeroshot_viewer.py` from ~/lerobot's own
venv - checked live this session: unlike the older astra_s_live.py/
camera_hub.py note about needing ~/lerobot_song_venv for GUI opencv,
~/lerobot's own cv2 build (4.14.0) already has Qt5 highgui support, and
that venv is the one with transformers/torch/GPU - no venv split needed
here (lerobot_song_venv doesn't have transformers installed at all, tried
first, would need a separate install to use it instead). 'q'/ESC to quit.

2026-08-28: this window now also shows a colorized depth panel alongside
RGB+labels - astra_s_live.py's own "Astra S - RGB | Depth" window is being
run headless now (ASTRA_LIVE_HEADLESS=1, see that file) per the user's
"two camera windows is one too many" feedback, and depth was otherwise only
visible in that now-hidden window. Colorization (clip to [DEPTH_MIN_MM,
DEPTH_MAX_MM], normalize, cv2.COLORMAP_JET) matches astra_s_live.py's own
exactly (read from its source, not imported - that module also pulls in
classes that open the physical camera device at construction, which this
read-only viewer has no business touching) so the depth view looks the
same as before, just in this window instead of a separate one.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

import config
from perception import PublishedFrameSource
from perception_zeroshot import DEFAULT_LABELS, draw_labeled_boxes, label_all_objects

FULL_VIEW_PATH = "/tmp/vsp_zeroshot_viewer_full.png"  # RGB+labels+depth, exactly what the window shows each tick
# Matches astra_s_live.py's own DEPTH_MIN_MM/DEPTH_MAX_MM (350-800mm, tuned
# there for this table's close-range setup) - see this module's docstring.
DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800
# 2026-08-28, history: was 3.0 when a full pass took ~10-15s+ (unbatched
# CLIP calls, SAM's default 32x32 point grid) - lowered to 1.5 once the
# YOLOE backend made a pass take ~1.8s. SAME DAY, raised again to 12.0
# after perception_zeroshot.py switched to Gemini (cloud API, ~8-9s/call,
# per the user's "이걸로 완전히 교체") - polling every 1.5s would just queue
# up overlapping slow requests for no benefit. This backend fits a "detect
# once, act on it" pattern much better than continuous live polling - see
# perception_zeroshot.py's own module docstring for the full tradeoff.
REFRESH_INTERVAL_S = 12.0


def atomic_write(path: str, frame) -> None:
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    cv2.imwrite(tmp, frame)
    os.replace(tmp, path)


def load_depth_vis(depth_path: str = config.ASTRA_DEPTH_MM_PATH,
                    stale_timeout_s: float = config.FRAME_STALE_TIMEOUT_S):
    """Colorized depth panel from the raw mm-depth astra_s_live.py
    publishes - see module docstring for why this exists and why the
    colorization is duplicated rather than imported. None (not raised) if
    the file is missing/stale/unreadable - same convention as every other
    published-frame reader in this project."""
    if not os.path.exists(depth_path) or (time.time() - os.path.getmtime(depth_path)) >= stale_timeout_s:
        return None
    try:
        depth_mm = np.load(depth_path)
    except (OSError, ValueError):
        return None
    clipped = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    norm = ((clipped - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def combine_with_depth(display_bgr):
    """Appends the depth panel to the right of display_bgr, resized to the
    same height - returns display_bgr unchanged if no depth is available
    right now (never blocks the RGB+labels view on depth being present)."""
    depth_vis = load_depth_vis()
    if depth_vis is None:
        return display_bgr
    h = display_bgr.shape[0]
    d = cv2.resize(depth_vis, (max(1, int(depth_vis.shape[1] * h / depth_vis.shape[0])), h))
    cv2.putText(d, "DEPTH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.hconcat([display_bgr, d])


class LabelWorker(threading.Thread):
    """Owns the slow detect+label work on its own thread - see module
    docstring for why (mirrors red_cube_calib_pick.py's render-during-move
    fix for the exact same "long op freezes the GUI" problem). Stores only
    the labeled-object DATA (not a rendered image) - see module docstring's
    2026-08-28 note on why: the main thread draws it fresh onto whatever
    the current live frame is, every tick, instead of freezing on a stale
    captured image for the whole interval between passes."""

    def __init__(self, rgb_path: str = config.ASTRA_RGB_FRAME_PATH, interval_s: float = REFRESH_INTERVAL_S):
        super().__init__(daemon=True)
        self.rgb_path = rgb_path
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._latest_labeled = None
        self._latest_ts = 0.0
        self._stop = threading.Event()

    def latest(self):
        with self._lock:
            return self._latest_labeled, self._latest_ts

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        source = PublishedFrameSource(self.rgb_path)
        while not self._stop.is_set():
            ret, frame = source.read()
            if ret and frame is not None:
                try:
                    labeled = label_all_objects(frame)
                    with self._lock:
                        self._latest_labeled, self._latest_ts = labeled, time.time()
                except Exception as e:  # a single bad frame/inference hiccup shouldn't kill the whole viewer
                    print(f"[zeroshot_viewer] labeling pass failed ({type(e).__name__}: {e}) - retrying next cycle")
            self._stop.wait(self.interval_s)


# Writing a PNG to disk every ~30ms main-loop tick would be pure waste (the
# file only exists for inspection - via Claude's Read tool or otherwise -
# not for anything the viewer itself depends on) - throttled to roughly
# match REFRESH_INTERVAL_S instead, independent of the (now much faster,
# smooth) imshow rate.
FILE_PUBLISH_INTERVAL_S = 1.0


def _draw_liveness_indicator(img, frame_count: int) -> None:
    """2026-08-28: the user reported the window still LOOKED frozen even
    after the smoothness fix above - verified it genuinely wasn't (md5 of
    the published frame changed every second, confirmed live) - the real
    issue is that a mostly-static tabletop scene simply has no motion for a
    human eye to use as a "this is live" cue, so a truly live feed and a
    paused one look the same at a glance. Fix: an unmissable, unambiguous
    liveness cue that has nothing to do with the scene content - a wall-
    clock timestamp with sub-second resolution (visibly ticks even if
    nobody can perceive per-pixel sensor noise) plus a small dot that
    visibly moves in a fixed cycle. Drawn every main-loop tick regardless
    of frame/label state, bottom-left so it never collides with the top-
    left object-count text."""
    h = img.shape[0]
    ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}"
    cv2.putText(img, f"LIVE {ts}  frame#{frame_count}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cx = 30 + int(15 * np.sin(frame_count * 0.15))  # visibly oscillates left-right every ~40 ticks
    cv2.circle(img, (cx, h - 35), 6, (0, 255, 0), -1)


def main() -> None:
    print(f"[zeroshot_viewer] candidate labels: {DEFAULT_LABELS}")
    print("[zeroshot_viewer] starting background label worker (Gemini backend - each pass takes ~8-9s, not just the first)...")
    worker = LabelWorker()
    worker.start()
    source = PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    last_publish = 0.0
    frame_count = 0
    try:
        while True:
            ret, frame = source.read()
            labeled, label_ts = worker.latest()
            if not ret or frame is None:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            if labeled is None:
                display = frame.copy()
                cv2.putText(display, "labeling... (Gemini API call, ~8-9s each)", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                display = draw_labeled_boxes(frame, labeled)
                age_s = time.time() - label_ts
                cv2.putText(display, f"{len(labeled)} objects - boxes {age_s:.1f}s old", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame_count += 1
            _draw_liveness_indicator(display, frame_count)

            full = combine_with_depth(display)
            cv2.imshow("Zeroshot object viewer (box + name)", full)
            now = time.time()
            if now - last_publish >= FILE_PUBLISH_INTERVAL_S:
                atomic_write(FULL_VIEW_PATH, full)  # lets this window's exact content be checked without a GUI
                last_publish = now

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        worker.stop()
        worker.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
