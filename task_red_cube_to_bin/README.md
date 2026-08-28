# task_red_cube_to_bin

Zero-shot vision pick-and-place for a **SO-101** 5-DOF robot arm: no
imitation-learning demos, no per-object training. A camera frame goes in,
Gemini finds the object by name, a fixed-roll IK solver moves the arm
there, and a gripper-position check tells you whether it actually grasped
anything.

```
[camera] → [Gemini: "what's on the table?"] → [pixel → robot xy/z/yaw]
         → [fixed-roll 4-DOF IK] → [move + grasp + verify]
```

## Hardware

- **SO-101** follower arm (Feetech STS3215 servos), connected via USB
  serial (`config.FOLLOWER_PORT`, default `/dev/ttyACM0`)
- **Orbbec Astra S** RGB-D camera, mounted overhead looking down at the
  table (the vision input for detection + depth)
- A wrist camera on the arm (optional, monitoring only - not read by this
  package's own scripts, see `camera_hub.py` one level up)

## Why this design

- **Position-only IK.** This is a 5-DOF arm - trying to also hold a full
  3-DOF orientation target blew position error up to 16-253mm across the
  workspace (see `kinematics.py`'s docstring). `grasp_ik.py` instead holds
  **wrist_roll fixed** at a chosen yaw and solves the other 4 joints
  numerically (damped least squares) - measured <1mm position error across
  a 200-case stress test, regardless of the requested roll.
- **No segmentation model.** Detection comes from Gemini
  (`gemini-flash-lite-latest` by default), which returns bounding boxes,
  not masks - see `perception_zeroshot.py`'s module docstring for the full
  backend history (Grounding DINO+SAM+CLIP → YOLOE → Gemini) and why each
  was replaced. A bbox rectangle stands in for a "mask" wherever the code
  needs one (grasp-yaw estimation); good enough for roughly-cuboid objects,
  weaker for irregular/concave shapes.
- **Open-loop.** One detection pass computes a target pose, the arm moves
  there once, no wrist-cam visual servo refinement. Real-hardware accuracy
  on an arbitrary new object has not been fully characterized - the
  homography (table-plane pixel → robot xy) and depth-delta height
  estimate are only as good as `homography.json`'s calibration.

## Setup

```bash
# from ~/lerobot (this package expects the main lerobot venv/deps)
uv add google-genai   # if not already installed

# Gemini API key - either:
export GEMINI_API_KEY="your-key-here"
# or:
echo "your-key-here" > ~/.gemini_api_key && chmod 600 ~/.gemini_api_key
```

You also need `homography.json` (table-plane pixel→xy calibration) one
directory up, in `vision_pick_place/` - see `calibrate_camera.py` there.
Camera frames are read from published files (`config.ASTRA_RGB_FRAME_PATH`,
`config.ASTRA_DEPTH_MM_PATH`), written by `astra_s_live.py` (also one
directory up) - run that first, in headless mode if you don't want its own
window:

```bash
ASTRA_LIVE_HEADLESS=1 python3 ../astra_s_live.py &
```

## Usage

All commands run from this directory with `uv run python3 <script>` (or
plain `python3` if you're already in the right venv).

### Pick one named object

```bash
python3 zeroshot_pick.py "red cube"                              # dry run - prints the plan, moves nothing
python3 zeroshot_pick.py "red cube" --live                        # actually moves the real arm
python3 zeroshot_pick.py "red cube" --place "black bin" --live    # pick AND place
```

### Discover objects without naming them

```bash
python3 zeroshot_pick.py --list                 # lists every object currently visible, with index numbers
python3 zeroshot_pick.py --index 2 --live        # grasp the 3rd one from that list
python3 zeroshot_pick.py --index 2 --place "black bin" --live
```

### Click-to-pick (live GUI window)

```bash
python3 zeroshot_click_pick.py
```

Shows the Astra view with Gemini-detected boxes + a depth panel side by
side. Left-click any box to immediately pick it up and place it at
`PLACE_PROMPT` (default `"black bin"`, edit the constant at the top of the
file to change). `q`/ESC to quit without moving anything.

### Just watch what the camera sees (no robot)

```bash
python3 zeroshot_viewer.py       # GUI window: RGB + detection boxes | depth
python3 zeroshot_http_view.py    # same feed over HTTP (http://localhost:8899/) -
                                  # use this if the GUI window doesn't repaint live
                                  # (seen under some Wayland/XWayland setups)
```

### Run the test suite (no camera or robot needed)

```bash
python3 test_grasp_ik.py             # pure math: IK convergence across targets/rolls
python3 test_perception_zeroshot.py  # detection against a saved real photo
python3 test_zeroshot_pick.py        # full pipeline dry-run against a saved photo
```

## Module map

| File | What it does |
|---|---|
| `perception_zeroshot.py` | Calls Gemini for detection, plus a small HSV-based safety net (`_correct_flicker_confusion`) for a real red/black confusion bug found in an earlier backend |
| `grasp_ik.py` | Fixed-roll 4-DOF numerical IK (damped least squares), with multi-restart so a bad seed doesn't strand the solver at a joint limit |
| `zeroshot_pick.py` | Orchestration: perception → 3D pose → IK → move/grasp/verify. Also the `--list`/`--index`/CLI entry point. Includes `MAX_ROLL_EXCURSION_DEG`, a safety cap that refuses a large wrist-roll sweep and keeps the current roll instead (added after a real run tripped a collision-retreat trying to satisfy an unreliable yaw estimate on a near-symmetric object) |
| `zeroshot_viewer.py` | Live GUI: RGB + labeled boxes + depth, decoupled so the video stays smooth even though detection itself is slow (~3-9s/call, cloud API) |
| `zeroshot_click_pick.py` | Same view as the viewer, plus click-to-select-and-execute |
| `zeroshot_http_view.py` | Browser-based fallback view (same published frame, served over local HTTP) |
| `kinematics.py`, `config.py`, `perception.py`, `gripper.py` | Older, real-hardware-validated modules from a previous (HSV/click-calibrated) pipeline iteration - imported read-only by the files above, not modified by this package |
| `mujoco_sim/` | SO-101 MJCF model + meshes, from an earlier simulation-based iteration of this task |
| `sim_dry_run.py`, `task_state_machine.py`, `collect_training_data.py`, `main.py`, `gripper.py` | From that same earlier iteration; kept for reference |

## Known limitations

- **Gemini free-tier quota**: 20 requests/day per model on some tiers -
  `perception_zeroshot._GEMINI_MODEL_ID` can be swapped to a model with a
  separate quota bucket if you hit this (that's why the default is
  `gemini-flash-lite-latest`, not a larger model).
- **No true segmentation** - grasp-yaw estimates are a bbox-rectangle
  approximation, not a real per-pixel mask.
- **Open-loop accuracy is unverified for small objects** - the first real
  `--live` run of this pipeline reached its target pose safely but missed
  the physical grasp on a ~50mm cube. If you hit this, the likely fixes
  are (a) redo `homography.json`'s calibration with more/better-placed
  points, or (b) add a wrist-cam closed-loop refinement step before the
  final descend (this package deliberately doesn't have one - see "Why
  this design" above).
- This follower arm may be shared with other people in your setup - all
  live scripts check the serial port isn't already in use before
  connecting (`check_port_not_busy`) and always exit through a
  return-to-home + disconnect path, even on error.
