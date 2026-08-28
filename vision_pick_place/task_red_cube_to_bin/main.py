"""Entry point. Connects to the arm, reads its live joint positions as
home_pose (never hardcoded - see task_state_machine.run), checks the wrist
camera feed is available, and runs the fixed sequence.

Camera devices are NOT opened here - camera_hub.py (wrist) and
astra_s_live.py (Astra RGB+depth), the sibling scripts in vision_pick_place/,
must already be running and publishing frames (see perception.py's
PublishedFrameSource) before this starts; two processes can't both hold a
UVC/OpenNI2 device open for streaming.

Run: uv run python3 custom_scripts/vision_pick_place/task_red_cube_to_bin/main.py
(from ~/lerobot - needs the main venv for lerobot/feetech-servo-sdk, same as
every other robot-control script in this project; the camera scripts above
run separately in ~/lerobot_song_venv for GUI opencv.)
"""

from __future__ import annotations

import sys
import time

import config
import perception
from kinematics import SOArm101


def main() -> bool:
    cap = perception.PublishedFrameSource(config.WRIST_FRAME_PATH)
    if not cap.isOpened():
        print(
            "[main] 손목캠 프레임이 없습니다. 먼저 camera_hub.py를 ~/lerobot_song_venv로 "
            "실행해주세요:\n  source ~/lerobot_song_venv/bin/activate && "
            "python custom_scripts/vision_pick_place/camera_hub.py"
        )
        return False

    arm = SOArm101(port=config.FOLLOWER_PORT)
    arm.connect()
    print("[main] 연결 성공. 현재 관절각:", arm.get_joint_deg())
    print("[main] 3초 후 시작합니다 (Ctrl+C로 중단 가능)...")
    time.sleep(3)

    import task_state_machine

    try:
        return task_state_machine.run(arm, cap)
    except KeyboardInterrupt:
        print("\n[main] 사용자가 중단했습니다.")
        return False
    finally:
        cap.release()
        arm.disconnect()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
