"""Auto-labeled YOLO training data collection: moves the wrist camera (eye-
in-hand) to many viewpoints around the cube's current location, and at each
stop, auto-labels whatever frame comes back using the already-validated HSV
detector (perception.detect_red_cube) instead of hand-labeling anything.

Why this instead of hand-labeling: zero-shot YOLO-World couldn't find the
red cube at all (tested extensively - larger model, many prompts, upscaling,
near-zero confidence threshold; it DID find a black cube fine, just not this
specific small, texture-patterned red one) - not usable as-is. But an HSV
detector *already works* on this exact object under real conditions (that's
this whole project's working baseline) - it's a fine teacher/weak-labeler
even though it isn't itself the most robust detector, and real-image
diversity (viewpoint, lighting, background) from actually moving the arm is
what a single photo could never give a fine-tuned model.

Saves standard YOLO-format image/label pairs (dataset.yaml + images/train,
labels/train) under this directory's yolo_dataset/, ready for
ultralytics' own training CLI/API - see train_yolo.py.

Frames where HSV itself finds nothing are skipped (can't auto-label without
a detection) - this is a real limitation (the resulting dataset only
contains cases the teacher already succeeds on), but still gives a much more
diverse, real, generalizable set than one frame ever could.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import config
import perception
from kinematics import CollisionDetected, SOArm101

OUT_DIR = Path(__file__).resolve().parent / "yolo_dataset"
IMAGES_DIR = OUT_DIR / "images" / "train"
LABELS_DIR = OUT_DIR / "labels" / "train"

# Grid of small xy offsets (meters) + a couple of heights, around wherever
# the cube is actually found - denser than SEARCH_OFFSETS since the goal
# here is viewpoint diversity for training data, not just finding the target
# once.
OFFSET_GRID = [(dx, dy) for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02)]  # 3x3 - kept
# smaller than a denser grid deliberately: real hardware time is the actual cost here,
# not compute, and this still gives real viewpoint diversity across 27 total stops
HEIGHTS_ABOVE_TABLE = [0.05, 0.08, 0.11]
SETTLE_S = 0.3
FRAMES_PER_STOP = 3  # a few reads per stop - cheap, and detection/exposure can vary frame to frame


def save_yolo_label(image_path: Path, label_path: Path, frame: np.ndarray, det: perception.Detection) -> None:
    cv2.imwrite(str(image_path), frame)
    h, w = frame.shape[:2]
    bx, by, bw, bh = det.bbox
    cx_n, cy_n = (bx + bw / 2) / w, (by + bh / 2) / h
    bw_n, bh_n = bw / w, bh / h
    label_path.write_text(f"0 {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset.yaml").write_text(
        f"path: {OUT_DIR}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['red_cube']\n"
    )

    cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not cap.isOpened():
        print("[collect] 손목캠 프레임이 없습니다 - camera_hub.py가 실행 중인지 확인하세요.")
        return

    arm = SOArm101(port=config.FOLLOWER_PORT)
    arm.connect()
    print("[collect] 연결 성공. 현재 관절각:", arm.get_joint_deg())

    saved = skipped = 0
    try:
        # Find the cube once via the same coarse approach the real task uses,
        # then collect around wherever that actually landed.
        astra_xy = perception.estimate_xy_from_astra(perception.detect_red_cube)
        base_xy = astra_xy if astra_xy is not None else (config.SEARCH_HOVER_XYZ[0], config.SEARCH_HOVER_XYZ[1])
        print(f"[collect] 기준 위치: {base_xy}")

        stops = [(base_xy[0] + dx, base_xy[1] + dy, config.TABLE_Z + h)
                 for h in HEIGHTS_ABOVE_TABLE for dx, dy in OFFSET_GRID]
        print(f"[collect] 총 {len(stops)}개 위치 예정")

        for i, xyz in enumerate(stops):
            try:
                arm.move_to_xyz_converge(xyz, tolerance_m=0.015, max_iters=10)
            except CollisionDetected as e:
                print(f"[collect] {i}: 충돌 감지, 이 지점 건너뜀 ({e})")
                continue
            time.sleep(SETTLE_S)

            for _ in range(FRAMES_PER_STOP):
                ret, frame = cap.read()
                if not ret or frame is None or perception.is_frame_corrupted(frame):
                    time.sleep(0.05)
                    continue
                det = perception.detect_red_cube(frame)
                if det is None:
                    skipped += 1
                    time.sleep(0.05)
                    continue
                stem = f"{saved:05d}"
                save_yolo_label(IMAGES_DIR / f"{stem}.png", LABELS_DIR / f"{stem}.txt", frame, det)
                saved += 1
                time.sleep(0.05)

            if i % 10 == 0:
                print(f"[collect] {i}/{len(stops)} 지점 완료 - 저장 {saved}, 건너뜀 {skipped}")

    finally:
        print(f"[collect] 완료: 저장 {saved}장, 검출 실패로 건너뜀 {skipped}장")
        print("[collect] 홈으로 복귀합니다.")
        try:
            arm.move_to_xyz_converge(config.SEARCH_HOVER_XYZ, tolerance_m=0.015, max_iters=20)
        except CollisionDetected as e:
            print(f"[collect] 복귀 중 충돌 감지: {e}")
        cap.release()
        arm.disconnect()


if __name__ == "__main__":
    main()
