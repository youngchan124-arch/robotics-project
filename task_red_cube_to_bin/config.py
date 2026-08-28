"""Central config for the red-cube -> bin task: every tunable number in one
place, per the 2026-08-26 spec's requirement that nothing be hardcoded deep
inside a state/module. Every constant below with a date comment was found or
tuned against real hardware today, not guessed - see
~/.claude memory (orbbec-astra-s-lerobot.md) for the full incident-by-
incident history if a number here ever needs revisiting.
"""

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
VISION_DIR = TASK_DIR.parent  # ~/lerobot/custom_scripts/vision_pick_place - shared camera publish paths live here

# --- Robot connection -------------------------------------------------------
# 2026-08-26: the follower arm's OWN USB adapter board (serial 5B3D042173) is
# dead - confirmed at the raw-byte level (zero response at any baud rate).
# Running on the leader arm's board instead (serial 5B3D042390), which udev
# enumerates as bare /dev/ttyACM0, not /dev/so101_follower. Revert to
# "/dev/so101_follower" once the original board is fixed/replaced.
FOLLOWER_PORT = "/dev/ttyACM0"

JOINT_LIMITS_DEG = {
    "shoulder_pan": (-118.0, 118.0),
    "shoulder_lift": (-105.0, 105.0),
    "elbow_flex": (-98.0, 98.0),
    "wrist_flex": (-102.0, 102.0),
    "wrist_roll": (-179.0, 179.0),
    "gripper": (-8.0, 99.0),  # percent-open, not degrees - so_follower.py's RANGE_0_100 norm mode
}
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_JOINTS = ARM_JOINTS + ["gripper"]

MAX_RELATIVE_TARGET_DEG = 15.0  # lerobot's own per-send_action clamp
MAX_MOVE_DELTA_DEG = 40.0  # outright-refuse-the-whole-move cap, before any motion is sent

STALL_THRESHOLD_DEG = 10.0  # actual-vs-commanded lag that counts as "not really moving"
STALL_CHECK_EVERY = 3  # steps between stall checks - ordinary servo catch-up lag isn't a stall
STALL_CONSECUTIVE = 3  # consecutive stalled checks before treating it as a real block

URDF_PATH = VISION_DIR / "so101_urdf" / "so_arm101.urdf"
IK_TARGET_FRAME = "gripper_frame_link"

# 2026-08-26: measured directly - forcing ANY nonzero orientation_weight on
# this 5-DOF arm's IK blew position error up to 16-253mm across the
# workspace (tested at weights 0.05-1.0, from both a neutral and the arm's
# real current-pose starting guess - not a starting-guess artifact). 5
# joints can't independently satisfy position (3 DoF) + orientation (3 DoF)
# almost anywhere in this workspace. Position-only IK plus the wrist-cam
# closed-loop servo (which re-measures the camera-to-gripper relationship
# fresh every approach, see FINE_SERVO state) is what actually handles a
# floating gripper orientation, not a hard IK constraint.
IK_ORIENTATION_WEIGHT = 0.0
IK_ITERATIONS = 6  # placo's solver needs several passes fed back as the next guess to land within ~1mm

# --- Table / grasp geometry --------------------------------------------------
# 2026-08-26: was 0.045 all session on an unverified "earlier IK test" value.
# Root cause of nearly every real grasp failure today: the user physically
# drove the gripper down onto the table and read back the live xyz at actual
# contact - (0.122, -0.0003, -0.0003). z was ~0.000, not 0.045 - a 45mm error.
TABLE_Z = 0.003  # 3mm margin above the measured real contact point
CUBE_HEIGHT_MIN_M = 0.005  # below this, an Astra height reading is treated as noise
CUBE_HEIGHT_MAX_M = 0.06  # above this, treated as a bad reading (this cube is a few cm)
DESCEND_MARGIN_M = 0.005  # stop this far short of the Astra-estimated cube top - let
# contact detection (not the depth estimate) catch the last few mm

LIFT_M = 0.08
BIN_DESCEND_M = 0.05

# --- Poses -------------------------------------------------------------------
# Raised hover pose to SEARCH from (wide camera view, clearance) - not a
# resting position. Center of the rectangle used in calibrate_camera.py's
# touch-point calibration.
SEARCH_HOVER_XYZ = (0.23, 0.0, 0.13)

# 2026-08-26: NOT hardcoded here as a fallback of last resort - main.py reads
# the arm's actual joint positions live at session start and uses THAT as
# home_pose, per the spec's "read_joint_positions(), no hardcoding"
# requirement. This constant only exists as a sanity-check reference (the
# pose the user physically drove the arm to and confirmed as "초기위치" once
# today) in case a caller needs a value before ever connecting.
REFERENCE_IDLE_XYZ = (0.10259099, 0.00435801, -0.02739574)

# --- Camera / vision -----------------------------------------------------
FRAME_W, FRAME_H = 640, 480
IMG_CENTER = (FRAME_W / 2.0, FRAME_H / 2.0)

# 2026-08-26: measured from real saved wrist-cam frames (the gripper's own
# jaw tips are visible in-shot, eye-in-hand mount) - the wrist camera does
# NOT look straight down the gripper's grasp axis, so "cube centered in the
# image" (IMG_CENTER) was never the same thing as "cube between the jaws".
# Caveat: since IK_ORIENTATION_WEIGHT=0.0 lets wrist_roll float, this offset
# could rotate around IMG_CENTER if a run lands on a very different roll
# than it was measured from - re-measure from a fresh saved frame if grasps
# keep missing.
GRASP_TARGET_PX = (195.0, 275.0)

# Published-frame paths: camera_hub.py / astra_s_live.py (the sibling
# vision_pick_place/ scripts) are the sole owners of the actual camera
# devices and publish here via atomic write - this task reads those files
# rather than opening any camera device itself, since two processes can't
# both hold a UVC/OpenNI2 device open for streaming. Same paths, defined
# fresh here rather than imported, per this task's own config being self-
# contained.
WRIST_FRAME_PATH = "/tmp/vsp_wrist.png"
ASTRA_RGB_FRAME_PATH = "/tmp/vsp_astra_rgb.png"
ASTRA_DEPTH_MM_PATH = "/tmp/vsp_astra_depth_mm.npy"
FRAME_STALE_TIMEOUT_S = 5.0

HOMOGRAPHY_PATH = VISION_DIR / "homography.json"

# HSV thresholds - see perception.py for how these were tuned (sampled
# against real miss-frames, not guessed).
LOWER_RED_1 = (0, 60, 25)
UPPER_RED_1 = (10, 255, 255)
LOWER_RED_2 = (170, 60, 25)
UPPER_RED_2 = (180, 255, 255)
MIN_CUBE_CONTOUR_AREA = 200

LOWER_BLACK = (0, 0, 0)
UPPER_BLACK = (180, 90, 70)
MIN_BIN_CONTOUR_AREA = 800

# Shape filter (solidity + aspect ratio + max-area cap), added 2026-08-26 so
# a hand/arm/cable in frame can't out-vote the real target just by being the
# largest same-colored blob. MIN_SOLIDITY lowered from an initial 0.85 the
# same day after it wrongly rejected a real, texture-noisy cube frame
# (measured solidity 0.837, just under 0.85) - 0.65 keeps a wide margin
# below that reading while staying far above an actual hand's ~0.41.
FRAME_AREA_HINT = FRAME_W * FRAME_H
MAX_CUBE_AREA_FRAC = 0.5
MIN_CUBE_SOLIDITY = 0.65
CUBE_ASPECT_RANGE = (0.4, 2.5)
MAX_BIN_AREA_FRAC = 0.7
MIN_BIN_SOLIDITY = 0.6
BIN_ASPECT_RANGE = (0.3, 3.0)

# 2026-08-26: this specific wrist camera ("USB 2.0 PC Cam") exposes no
# exposure_auto/exposure_absolute control at all (checked live via
# `v4l2-ctl --list-ctrls`) - true manual exposure isn't possible. What fixed
# real, badly-overexposed frames (background blown to solid white) was
# dropping brightness/gamma to their minimums; contrast/saturation changes
# made it worse or did nothing.
WRIST_V4L2_CTRLS = {"brightness": 0, "gamma": 1}

# --- Search sweep (fallback if the Astra coarse guess misses) ---------------
SEARCH_OFFSETS = [
    (0.0, 0.0),
    (0.03, 0.0), (0.06, 0.0), (-0.03, 0.0), (-0.06, 0.0),
    (0.0, 0.04), (0.0, 0.08), (0.0, -0.04), (0.0, -0.08),
    (0.03, 0.04), (-0.03, 0.04), (0.03, -0.04), (-0.03, -0.04),
]

# --- FINE_SERVO tuning (all found/tuned against real hardware 2026-08-26) --
PIXEL_TOLERANCE = 22.0  # secondary sanity bound only - see PHYSICAL_TOLERANCE_M
PHYSICAL_TOLERANCE_M = 0.004  # the real convergence gate: remaining pixel error run
# through the estimated Jacobian's inverse, in meters - see task_state_machine.py's
# FINE_SERVO state for why pixel error alone isn't enough once the image Jacobian
# is anisotropic (a real reading had singular values 8152 vs 1233 px/m, ~6.6x).
CENTER_STABLE_FRAMES = 3
MAX_SERVO_ITERS = 90
PROBE_DELTA_M = 0.008
SERVO_GAIN = 0.5
MAX_STEP_M = 0.006  # was 0.02 - too large a step in the sensitive axis caused endless
# oscillation once PHYSICAL_TOLERANCE_M made convergence genuinely tight
CLOSE_ERR_PX = 30.0  # once error is already under this, shrink the step cap further
CLOSE_MAX_STEP_FRAC = 0.4  # to CLOSE_ERR_PX * CLOSE_MAX_STEP_FRAC... see below
BROYDEN_MIN_STEP_M = 0.0025  # below this step size, skip the Broyden update - a small
# step amplifies ordinary detection noise into a wrong correction to J
STALL_ITERS = 5  # non-improving iterations before a full Jacobian re-probe
DIVERGE_PX = 40.0  # a jump this far past the best-seen error triggers an immediate
# re-probe, without waiting for STALL_ITERS
MAX_REESTIMATES = 4

COARSE_STEP_M = 0.025
COARSE_MAX_ITERS = 16
COARSE_TARGET_PX = 140.0

# --- Grasp verification -----------------------------------------------------
# 2026-08-26: placeholders, NOT yet measured on this unit - run
# manual_grasp_calibration.py (empty-close vs cube-close gripper readings)
# before trusting these on hardware.
GRIPPER_EMPTY_CLOSED_PCT = 5.0
GRASP_DETECT_MARGIN_PCT = 8.0

# --- Retry -------------------------------------------------------------------
MAX_GRASP_ATTEMPTS = 3  # per the spec's "실패 시 재시도 로직: 최대 N회"
