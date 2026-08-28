"""Click-to-select pick-and-place - per the user's 2026-08-28 explicit
request ("클릭해서 빨간큐브를 집고 검은 쓰레기통에 넣어보자": click to pick
up the red cube and put it in the black bin), matching this project's
established click-driven pattern (red_cube_calib_pick.py's do_pick/do_place,
2026-08-26/27) but built on today's zero-shot Gemini + fixed-roll-IK
pipeline instead of a taught pixel-to-joint affine matrix.

Shows the live Astra RGB view with Gemini-labeled boxes (same
perception_zeroshot.label_all_objects + draw_labeled_boxes zeroshot_viewer.py
uses, background-polled the same way - see that file's LabelWorker for why).
Left-click anywhere inside a currently-displayed box selects it as the PICK
target and immediately runs a live pick-and-place to PLACE_PROMPT (default
"black bin"). 'q'/ESC to quit without picking.

NOT a fix for the accuracy gap found in the previous real run (reached the
target safely - the 2026-08-28 MAX_ROLL_EXCURSION_DEG cap worked - but
missed the physical grasp). Clicking only changes HOW the pick target is
chosen (visually, instead of a typed text prompt or --index number) - the
underlying homography+depth-delta coordinate pipeline computing WHERE to
move is identical to plan_grasp/list_objects, same known accuracy limits.

Run: `uv run python3 zeroshot_click_pick.py` from this directory (needs the
main ~/lerobot venv - transformers/torch/GPU aren't needed for Gemini calls
themselves, but google-genai + this project's own modules are).
"""

from __future__ import annotations

import threading
import time

import cv2

import config
from perception import PublishedFrameSource
from perception_zeroshot import DEFAULT_LABELS, draw_labeled_boxes, label_all_objects
from zeroshot_viewer import atomic_write, combine_with_depth  # reuse the same depth-panel rendering, see that file
from zeroshot_pick import (
    GraspPlan,
    _load_depth_mm,
    _load_homography,
    build_grasp_plan,
    check_port_not_busy,
    execute_pick_and_place,
)

PLACE_PROMPT = "black bin"
REFRESH_INTERVAL_S = 12.0  # matches zeroshot_viewer.py's own Gemini-latency-driven interval, see that file
FULL_VIEW_PATH = "/tmp/vsp_click_pick_full.png"  # exactly what the window shows, for checking without a GUI
FILE_PUBLISH_INTERVAL_S = 1.0


class LabelWorker(threading.Thread):
    """Identical role to zeroshot_viewer.py's LabelWorker (background
    Gemini polling so the displayed video stays smooth - see that file's
    module docstring for the "looked frozen" history this avoids) -
    reimplemented here rather than imported so this script has no
    dependency on zeroshot_viewer.py's own main()/window lifecycle."""

    def __init__(self, rgb_path: str = config.ASTRA_RGB_FRAME_PATH, interval_s: float = REFRESH_INTERVAL_S):
        super().__init__(daemon=True)
        self.rgb_path = rgb_path
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._latest_labeled = None
        self._latest_ts = 0.0
        self._stop = threading.Event()
        self._paused = threading.Event()

    def latest(self):
        with self._lock:
            return self._latest_labeled, self._latest_ts

    def pause(self) -> None:
        self._paused.set()  # stop starting new Gemini calls while a pick-and-place is executing

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        source = PublishedFrameSource(self.rgb_path)
        while not self._stop.is_set():
            if not self._paused.is_set():
                ret, frame = source.read()
                if ret and frame is not None:
                    try:
                        labeled = label_all_objects(frame)
                        with self._lock:
                            self._latest_labeled, self._latest_ts = labeled, time.time()
                    except Exception as e:
                        print(f"[zeroshot_click_pick] labeling pass failed ({type(e).__name__}: {e}) - retrying next cycle")
            self._stop.wait(self.interval_s)


class ClickState:
    """Shared between the mouse callback (fires on the Qt/X11 event thread)
    and the main loop - just the one pixel coordinate, guarded by a lock
    since both sides touch it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.pt: tuple[int, int] | None = None

    def set(self, x: int, y: int) -> None:
        with self._lock:
            self.pt = (x, y)

    def take(self) -> tuple[int, int] | None:
        with self._lock:
            pt, self.pt = self.pt, None
            return pt


def _bbox_contains(bbox: tuple[int, int, int, int], x: int, y: int) -> bool:
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def _find_clicked_plan(labeled, frame_shape_hw, homography, depth_mm, x: int, y: int) -> GraspPlan | None:
    """Smallest-area box containing the click wins (if boxes overlap, the
    more specific/smaller one is almost always the intended target)."""
    candidates = [(det, mask, yaw) for det, mask, yaw, _label, _score in labeled if _bbox_contains(det.bbox, x, y)]
    if not candidates:
        return None
    det, _mask, yaw_deg = min(candidates, key=lambda c: c[0].area)
    return build_grasp_plan(det, yaw_deg, frame_shape_hw, homography, depth_mm, text_prompt="<clicked>")


def main() -> None:
    if not check_port_not_busy():
        print(f"[zeroshot_click_pick] {config.FOLLOWER_PORT} is already in use by another process - refusing to "
              "start (this arm is shared - see orbbec-astra-s-lerobot.md)")
        return

    print(f"[zeroshot_click_pick] candidate labels for naming (display only): {DEFAULT_LABELS}")
    print(f"[zeroshot_click_pick] click a box to pick it up and place it at '{PLACE_PROMPT}'. 'q'/ESC to quit.")

    click_state = ClickState()

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_state.set(x, y)

    window = "Click to pick (left-click a box)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    worker = LabelWorker()
    worker.start()
    source = PublishedFrameSource(config.ASTRA_RGB_FRAME_PATH)
    last_publish = 0.0
    try:
        while True:
            ret, frame = source.read()
            labeled, label_ts = worker.latest()
            if not ret or frame is None:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            display = frame.copy() if labeled is None else draw_labeled_boxes(frame, labeled)
            age_s = 0.0 if label_ts == 0.0 else time.time() - label_ts
            status = "labeling... (Gemini API call, ~8-9s)" if labeled is None else f"{len(labeled)} objects - boxes {age_s:.1f}s old"
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            full = combine_with_depth(display)
            cv2.imshow(window, full)
            now = time.time()
            if now - last_publish >= FILE_PUBLISH_INTERVAL_S:
                atomic_write(FULL_VIEW_PATH, full)
                last_publish = now

            clicked = click_state.take()
            # combine_with_depth() appends a depth panel to the right, so the
            # displayed window is wider than frame itself - a click landing in
            # that panel (x >= frame width) isn't a real RGB pixel, ignore it.
            if clicked is not None and clicked[0] >= frame.shape[1]:
                print(f"[zeroshot_click_pick] click at {clicked} was in the DEPTH panel, not the RGB view - ignored.")
                clicked = None
            if clicked is not None and labeled:
                homography = _load_homography()
                depth_mm = _load_depth_mm()
                plan = None
                if homography is not None:
                    plan = _find_clicked_plan(labeled, frame.shape[:2], homography, depth_mm, *clicked)
                if plan is None:
                    print(f"[zeroshot_click_pick] click at {clicked} didn't land inside any currently-shown box - ignored.")
                else:
                    print(f"[zeroshot_click_pick] picked target at {clicked}: {plan.detection.bbox} -> "
                          f"executing pick-and-place to '{PLACE_PROMPT}'")
                    worker.pause()
                    try:
                        execute_pick_and_place(None, PLACE_PROMPT, live=True, pick_plan=plan)
                    except Exception as e:
                        print(f"[zeroshot_click_pick] pick-and-place raised {type(e).__name__}: {e}")
                    finally:
                        worker.resume()

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        worker.stop()
        worker.join(timeout=2)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
