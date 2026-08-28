"""Live RGB + wrist camera viewer, run standalone (needs GUI opencv - use
~/lerobot_song_venv, NOT ~/lerobot's own headless-opencv venv).

Two jobs at once:
  1. Shows a combined RGB | WRIST window with the same red-cube / black-bin
     detection overlay the control script uses, so you can watch what it sees.
  2. Publishes each frame to /tmp/vsp_rgb.png and /tmp/vsp_wrist.png (atomic
     write via temp-file + rename) so other scripts (visual servoing,
     calibration) - which run in the OTHER venv, the one with feetech-servo-
     sdk but headless opencv - can read the latest frame without opening the
     camera device itself. Two processes can't both hold a UVC device open
     for streaming, so this hub is the only thing that actually calls
     cv2.VideoCapture; other scripts just cv2.imread() whatever this last
     wrote (imread/imwrite don't need a GUI backend, only imshow does).

Each camera runs on its own thread (CameraWorker), reading/filtering/
publishing at whatever pace that device actually delivers. They used to share
one thread, reading RGB then wrist every loop - a slow or momentarily-blocked
read on either device delayed the other's publish and display update too, on
top of the two devices already competing for USB bandwidth. Separate threads
mean one camera's hiccup no longer throttles the other; the main thread just
polls each worker's latest published frame for the combined display.

Depth isn't published here - run depth_live (already built, standalone, its
own window) alongside this in a separate terminal for the third screen.
"""

import os
import subprocess
import threading
import time

import cv2

from camera_utils import find_camera_index
from cube_detector import detect_black_bin, detect_red_cube, draw_detection, is_frame_corrupted

# Resolved by USB product name, not a hardcoded /dev/videoN - the wrist camera
# has already re-enumerated under a new index once this session after a real
# USB disconnect/reconnect under load, which would silently break a hardcoded
# number. Falls back to the last-known index if name lookup fails for some
# reason (v4l2-ctl missing, etc).
RGB_INDEX = find_camera_index("USB Camera")
WRIST_INDEX = find_camera_index("USB 2.0 PC Cam")

# 2026-08-26: the wrist cam was badly overexposed (background blown to solid
# white, real content unrecoverable - confirmed by inspecting saved frames)
# under this table's lighting. This camera exposes NO exposure_auto/
# exposure_absolute control at all (checked via `v4l2-ctl --list-ctrls` -
# only brightness/contrast/saturation/hue/gamma/white_balance_temperature/
# sharpness/backlight_compensation exist), so true manual exposure isn't
# possible - what fixed it live was dropping `brightness` to its minimum
# (113 default -> 0) and `gamma` to its minimum (2 default -> 1); contrast/
# saturation changes made it worse or did nothing and were left at default.
# This is specific to this exact camera model/lighting setup - re-tune by
# saving a frame and inspecting it if the lighting changes.
WRIST_V4L2_CTRLS = {"brightness": 0, "gamma": 1}
# No silent fallback to a hardcoded index: a stale guess can collide with
# whatever the OTHER camera re-enumerated to (this happened for real - both
# resolved to the same /dev/video4 after the wrist camera dropped off USB
# entirely and RGB happened to land on the index wrist used to have).
# Better to clearly report "not found" than open the wrong device.


def atomic_write(path: str, frame) -> None:
    # cv2.imwrite picks its encoder from the file extension, so the temp name
    # has to keep ".png" (a bare "path + .tmp" has extension ".tmp", which
    # has no registered writer and throws).
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    cv2.imwrite(tmp, frame)
    os.replace(tmp, path)


class CameraWorker(threading.Thread):
    """Owns one cv2.VideoCapture on its own thread: reads continuously,
    drops corrupted frames (see is_frame_corrupted), publishes good ones to
    `out_path`, and keeps the latest annotated (detection-overlay) frame in
    `self.vis` under a lock for the main thread's display loop to pick up -
    independent of how fast/slow this particular device is running."""

    def __init__(self, index: int, out_path: str, label: str, annotate_fn, v4l2_ctrls: dict | None = None):
        super().__init__(daemon=True)
        self.index = index
        self.out_path = out_path
        self.label = label
        self.annotate_fn = annotate_fn
        self.cap = cv2.VideoCapture(index)
        if v4l2_ctrls:
            # cv2's CAP_PROP_BRIGHTNESS/etc don't reliably map onto this
            # camera's actual UVC controls through the v4l2 backend - v4l2-ctl
            # directly against the device node is what was verified live to
            # actually change the output (see WRIST_V4L2_CTRLS below).
            ctrl_str = ",".join(f"{k}={v}" for k, v in v4l2_ctrls.items())
            subprocess.run(["v4l2-ctl", "-d", f"/dev/video{index}", f"--set-ctrl={ctrl_str}"], capture_output=True)
        self._lock = threading.Lock()
        self.vis = None
        self.total = 0
        self.corrupt = 0
        self.stop_flag = False

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def run(self) -> None:
        while not self.stop_flag:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            self.total += 1
            if is_frame_corrupted(frame):
                self.corrupt += 1
                continue
            atomic_write(self.out_path, frame)
            vis = self.annotate_fn(frame)
            cv2.putText(vis, self.label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            with self._lock:
                self.vis = vis

    def latest(self):
        with self._lock:
            return None if self.vis is None else self.vis.copy()

    def corruption_pct(self) -> float:
        return 100 * self.corrupt / self.total if self.total else 0.0

    def stop(self) -> None:
        self.stop_flag = True


def annotate_rgb(frame):
    det = detect_red_cube(frame)
    return draw_detection(frame.copy(), det) if det is not None else frame.copy()


def annotate_wrist(frame):
    vis = frame.copy()
    det_c = detect_red_cube(frame)
    det_b = detect_black_bin(frame)
    if det_c is not None:
        vis = draw_detection(vis, det_c)
    if det_b is not None:
        x, y, w, h = det_b.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return vis


def main():
    print(f"[camera_hub] RGB=/dev/video{RGB_INDEX} WRIST=/dev/video{WRIST_INDEX}")

    rgb_worker = CameraWorker(RGB_INDEX, "/tmp/vsp_rgb.png", "RGB (cube)", annotate_rgb) if RGB_INDEX is not None else None
    wrist_worker = (
        CameraWorker(WRIST_INDEX, "/tmp/vsp_wrist.png", "WRIST (cube+bin)", annotate_wrist, v4l2_ctrls=WRIST_V4L2_CTRLS)
        if WRIST_INDEX is not None
        else None
    )

    if rgb_worker is None:
        print("[camera_hub] RGB 카메라를 찾을 수 없습니다 (v4l2-ctl에 없음).")
    elif not rgb_worker.isOpened():
        print(f"[camera_hub] RGB(/dev/video{RGB_INDEX})를 열 수 없습니다.")
    if wrist_worker is None:
        print("[camera_hub] 손목캠을 찾을 수 없습니다 - USB 버스에서 완전히 빠진 상태입니다 (lsusb에도 없음). "
              "케이블/포트를 확인해주세요. RGB만으로 계속 진행합니다.")
    elif not wrist_worker.isOpened():
        print(f"[camera_hub] 손목캠(/dev/video{WRIST_INDEX})을 열 수 없습니다.")

    workers = [w for w in (rgb_worker, wrist_worker) if w is not None and w.isOpened()]
    if not workers:
        print("[camera_hub] 사용 가능한 카메라가 없습니다.")
        return

    for w in workers:
        w.start()

    print("[camera_hub] 실행 중 (스레드 분리) - 'q' 또는 ESC로 종료. 프레임을 /tmp/vsp_*.png 로 publish 합니다.")
    frame_count = 0
    try:
        while True:
            frames = [w.latest() for w in workers]
            panels = [f for f in frames if f is not None]
            if panels:
                h = min(p.shape[0] for p in panels)
                resized = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
                combined = cv2.hconcat(resized)
                cv2.imshow("RGB | WRIST", combined)

            frame_count += 1
            if frame_count % 150 == 0:
                stats = " ".join(f"{w.label}={w.corruption_pct():.0f}%" for w in workers)
                print(f"[camera_hub] 손상율: {stats}")

            key = cv2.waitKey(15) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=2)
            w.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
