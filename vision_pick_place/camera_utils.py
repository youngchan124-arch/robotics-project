"""Shared helper: resolve a camera's /dev/videoN index by matching its USB
product name instead of a hardcoded number. The wrist camera in particular has
re-enumerated under a new index at least once this session (a real USB
disconnect/reconnect under load, not a script bug - matches the general USB
flakiness already seen with these devices), so hardcoding "4" or "6" silently
breaks the next time that happens. Re-resolving by name at each script's
startup survives that.
"""

import os
import re
import subprocess
import time

import cv2
import numpy as np

# Where camera_hub.py (the sole process allowed to hold the RGB/wrist UVC
# devices open for streaming) publishes its latest frames. Any other script
# that wants a frame reads from here via PublishedFrameSource instead of
# opening the device itself - two processes can't both hold a UVC device open
# for streaming, and several scripts here (visual servoing, calibration) all
# want a look at the same live feed camera_hub.py is already showing.
RGB_FRAME_PATH = "/tmp/vsp_rgb.png"  # Astra Pro Plus RGB, via camera_hub.py - not connected as of 2026-08-26
WRIST_FRAME_PATH = "/tmp/vsp_wrist.png"
ASTRA_RGB_FRAME_PATH = "/tmp/vsp_astra_rgb.png"  # Astra S RGB, via astra_s_live.py - in use as of 2026-08-26
ASTRA_DEPTH_MM_PATH = "/tmp/vsp_astra_depth_mm.npy"  # raw uint16 mm depth, registered to ASTRA_RGB_FRAME_PATH's pixels
FRAME_STALE_TIMEOUT_S = 5.0


class PublishedFrameSource:
    """Drop-in replacement for cv2.VideoCapture's .read()/.isOpened(), backed
    by whatever camera_hub.py last wrote to `path` instead of a live device
    handle."""

    def __init__(self, path: str, stale_timeout_s: float = FRAME_STALE_TIMEOUT_S):
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
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        pass


class PublishedDepthSource:
    """Reads the raw mm-depth array astra_s_live.py publishes to
    ASTRA_DEPTH_MM_PATH (registered to ASTRA_RGB_FRAME_PATH's pixel grid, just
    at a lower native resolution - see astra_s_live.py's docstring on why).
    Same staleness convention as PublishedFrameSource. Returns None (never
    raises) whenever the data isn't there or isn't fresh - every caller must
    already have a "no depth info" fallback, since the Astra viewer might not
    be running, the cube might be out of Astra's view, etc."""

    def __init__(self, path: str = ASTRA_DEPTH_MM_PATH, stale_timeout_s: float = FRAME_STALE_TIMEOUT_S):
        self.path = path
        self.stale_timeout_s = stale_timeout_s

    def read(self) -> np.ndarray | None:
        if not os.path.exists(self.path):
            return None
        if (time.time() - os.path.getmtime(self.path)) >= self.stale_timeout_s:
            return None
        try:
            return np.load(self.path)
        except (OSError, ValueError):
            return None  # caught mid-write despite the atomic-rename convention, or a truncated read


def find_camera_index(name_substring: str) -> int | None:
    """Returns the first /dev/videoN for the device whose v4l2-ctl name
    contains name_substring, or None if not found/v4l2-ctl unavailable."""
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    blocks = out.split("\n\n")
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        header = lines[0]
        if name_substring.lower() not in header.lower():
            continue
        for line in lines[1:]:
            m = re.search(r"/dev/video(\d+)", line)
            if m:
                return int(m.group(1))
    return None
