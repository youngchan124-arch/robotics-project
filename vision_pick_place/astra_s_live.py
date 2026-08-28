"""Live Astra S RGB + Depth viewer (replaces depth_live_astra_s.py's depth-only
window with both streams from the SAME OpenNI2 device handle).

Astra S gives color and depth off one physical device - the earlier
depth_live_astra_s.py opened only the depth_stream, and this reuses that same
single-device-handle idea for both, via orbbec_color_camera.py's
ThreadedOrbbecRGBDCamera (already written for exactly this - one
Device.open_any() driving both streams on its own thread, with depth-to-color
image registration turned on so the two line up spatially).

Only ONE process may hold the Astra S device open at a time (same USB-device-
sharing constraint as the RGB/wrist cameras - see camera_hub.py's docstring),
so this script is the sole owner of it, same pattern: run standalone, one
window with RGB | DEPTH side by side. Also publishes the RGB frame to
/tmp/vsp_astra_rgb.png (atomic write, same convention as camera_hub.py) in
case a future detection pass wants a second RGB view - not read by the current
wrist-cam-only servoing pipeline.

Run standalone in ~/lerobot_song_venv (needs GUI opencv for imshow).

2026-08-28: added ASTRA_LIVE_HEADLESS env var (opt-in, default off - this
script's own window/behavior is UNCHANGED unless set) per the user's
request to stop this window from popping up once zeroshot_viewer.py (which
reads this script's published RGB+depth files) started showing its own
window with the depth panel folded in - two windows for the same camera
was the actual complaint, not this script's window specifically. When set
to "1", skips namedWindow/imshow/waitKey entirely; frame capture and the
ASTRA_RGB_FRAME_PATH/ASTRA_DEPTH_MM_PATH publishing this script exists for
in the first place are untouched. Stop with Ctrl-C in this mode (no window
to press 'q'/ESC in).
"""

import os
import time

import cv2
import numpy as np

from camera_utils import ASTRA_DEPTH_MM_PATH, ASTRA_RGB_FRAME_PATH
from cube_detector import detect_black_bin, detect_red_cube, draw_detection
from orbbec_color_camera import ThreadedOrbbecRGBDCamera

# Astra S is close-range on this table setup (see orbbec_color_camera.py /
# preview_all.py precedent) - 350-800mm covers the workspace without the
# far background washing out the color range.
DEPTH_MIN_MM = 350
DEPTH_MAX_MM = 800


def atomic_write(path: str, frame) -> None:
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    cv2.imwrite(tmp, frame)
    os.replace(tmp, path)


def atomic_write_npy(path: str, array) -> None:
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def main():
    headless = os.environ.get("ASTRA_LIVE_HEADLESS") == "1"

    cam = ThreadedOrbbecRGBDCamera(depth_min_mm=DEPTH_MIN_MM, depth_max_mm=DEPTH_MAX_MM)
    if not cam.isOpened():
        print("[astra_s_live] Astra S를 열 수 없습니다.")
        cam.release()
        return

    if not headless:
        cv2.namedWindow("Astra S - RGB | Depth", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Astra S - RGB | Depth", 1280, 480)
        print("[astra_s_live] 실행 중 - 'q' 또는 ESC로 종료.")
    else:
        print("[astra_s_live] 실행 중 (headless, ASTRA_LIVE_HEADLESS=1) - Ctrl-C로 종료.")
    frame_count = 0
    try:
        while True:
            ret, color, depth_vis = cam.read()
            if not ret or color is None:
                time.sleep(0.01)
                continue

            atomic_write(ASTRA_RGB_FRAME_PATH, color)  # publish the raw frame - annotation is display-only
            depth_mm = cam.read_raw_depth_mm()
            if depth_mm is not None:
                atomic_write_npy(ASTRA_DEPTH_MM_PATH, depth_mm)

            # Same detection the wrist cam already runs, applied here too so
            # the operator can visually confirm the cube/bin are recognized
            # from this angle as well - the wrist cam is still the only one
            # actually driving the servo control, this is purely a monitor
            # overlay. depth_vis is registered to color's frame (image
            # registration is on in ThreadedOrbbecRGBDCamera) but at a lower
            # native resolution (see that class's docstring on why depth
            # runs at QVGA), so the same detection's pixel coords are scaled
            # down before drawing on it rather than re-detected on depth.
            frame_count += 1
            if headless:
                time.sleep(0.01)  # publishing above already happened - just pace the loop, no window to update
                continue

            det_cube = detect_red_cube(color)
            det_bin = detect_black_bin(color)

            color_annot = color.copy()
            if det_cube is not None:
                color_annot = draw_detection(color_annot, det_cube)
            if det_bin is not None:
                bx, by, bw, bh = det_bin.bbox
                cv2.rectangle(color_annot, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)

            depth_annot = depth_vis.copy()
            sx = depth_vis.shape[1] / color.shape[1]
            sy = depth_vis.shape[0] / color.shape[0]
            if det_cube is not None:
                cv2.circle(depth_annot, (int(det_cube.cx * sx), int(det_cube.cy * sy)), 4, (0, 255, 0), 2)
            if det_bin is not None:
                bx, by, bw, bh = det_bin.bbox
                cv2.rectangle(
                    depth_annot,
                    (int(bx * sx), int(by * sy)),
                    (int((bx + bw) * sx), int((by + bh) * sy)),
                    (255, 0, 255), 2,
                )

            h = min(color_annot.shape[0], depth_annot.shape[0])
            c = cv2.resize(color_annot, (int(color_annot.shape[1] * h / color_annot.shape[0]), h))
            d = cv2.resize(depth_annot, (int(depth_annot.shape[1] * h / depth_annot.shape[0]), h))
            cv2.putText(c, "ASTRA RGB", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(d, "ASTRA DEPTH", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            combined = cv2.hconcat([c, d])
            cv2.imshow("Astra S - RGB | Depth", combined)

            key = cv2.waitKey(15) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cam.release()
        if not headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
