# task_red_cube_to_bin

**Zero-shot vision pick-and-place for a SO-101 5-DOF robot arm.**

Point a camera at a table, describe (or click on) an object in plain
language, and the arm picks it up and moves it somewhere else — with
**no teleoperation demos, no per-object training, no fine-tuning**. A
vision-language model looks at the camera frame and tells the code what's
there and where; classical robotics (inverse kinematics, depth geometry)
handles turning that into an actual, safe arm motion.

```
 camera frame
     │
     ▼
┌─────────────────────┐   "detect all objects, ignore the robot itself"
│  Gemini (cloud VLM)  │◄──────────────────────────────────────────────
└─────────┬────────────┘
          │ per object: name, confidence, bounding box (pixel space)
          ▼
┌─────────────────────────────┐
│ pixel → robot-frame (x, y)  │  table-plane homography (pre-calibrated)
│ + z from depth-delta        │  Astra S depth: table height − object height
│ + yaw from bbox aspect      │  PCA on the bbox rectangle (no true mask)
└─────────┬────────────────────┘
          ▼
┌─────────────────────────────┐
│ fixed-roll 4-DOF numerical  │  wrist_roll held at the chosen yaw,
│ IK (damped least squares)   │  other 4 joints solved to reach (x,y,z)
└─────────┬────────────────────┘
          ▼
┌─────────────────────────────┐
│ interpolated move + grasp   │  stall/collision detection, gripper-%
│ + gripper-position verify   │  grasp check, gentle post-grasp hold
└──────────────────────────────┘
```

## Why build it this way

This package is the result of a same-day pivot away from imitation
learning. The original plan for this robot (see the sibling
`task_red_cube_to_bin`-adjacent history in this repo's commit log) was
ACT/behavioral-cloning trained on ~50-100 teleoperated demonstrations —
expensive to collect, and brittle to any change in the object or its
position. This package instead follows a **"Vision → 3D Grasp Pose →
IK/Motion Planning"** design: no demonstration data at all, and it
generalizes to objects it has never seen, because the "what is this and
where is it" step is delegated to a general-purpose vision-language model
instead of a model trained on this specific task.

The trade-off is real and intentional: this is **open-loop** (one
detection pass → one move → done, no closed-loop visual servoing) and
depends on a **cloud API call** (seconds of latency, a daily quota, a
network dependency) rather than a fully local, millisecond-latency
pipeline. It targets simple pick-and-place of individual objects, not
high-throughput or closed-loop-precision tasks.

### The 5-DOF constraint that shapes everything downstream

The SO-101 arm has 5 joints. A full 6-DOF end-effector pose (3D position +
3D orientation) is therefore, in general, **not simultaneously
achievable** — asking the IK solver to satisfy both blew position error up
to 16–253mm across the workspace in real testing (`kinematics.py`'s own
solver, `IK_ORIENTATION_WEIGHT` docstring). Every design decision below
traces back to working around this:

- **Position-only IK, roll held fixed instead of solved.** `grasp_ik.py`
  never asks for a full orientation — it fixes `wrist_roll` at whatever
  yaw the vision step picked, and numerically solves the *other 4 joints*
  (damped least squares, with automatic multi-restart from several seeds
  if the first attempt gets stuck against a joint limit) to reach the
  target (x, y, z) at that fixed roll. Measured across a 200-case stress
  test (5 targets × multiple rolls × 40 randomized starting poses): **0
  failures to converge under 5mm, worst-case residual 4.09mm.**
- **A wrist-roll "excursion cap."** Even a *correct* target roll can be
  expensive to reach if the arm happens to be resting far from it — one
  real run tried to satisfy a 90° yaw estimate on a small, nearly-square
  cube (where the "correct" yaw is close to arbitrary) and had to sweep
  167.7° of wrist rotation to get there, tripping the stall/collision
  safety check before ever reaching the object. `MAX_ROLL_EXCURSION_DEG`
  (in `zeroshot_pick.py`) now checks this *before* moving: if reaching the
  requested roll would require more than 90° of rotation from the arm's
  *current* position, it keeps the current roll instead of forcing the
  sweep. Cheap objects like near-square cubes don't have a strongly
  "correct" grasp yaw anyway, so this loses little and avoids a real,
  reproduced failure mode.
- **No true segmentation.** Because there's no mask, the yaw estimate for
  a detected object is just PCA on its *bounding-box rectangle*, not its
  actual silhouette. This is a fine approximation for boxy objects and a
  weak one for irregular/concave shapes — see "Known limitations" below.

## Architecture in more detail

Mapping this project onto a standard household-robot layering (the
project it grew out of separates concerns into an LLM/planning layer, a
navigation layer, a vision layer, an IK/manipulation layer, and a feedback
layer — this package only implements the last three; there's no mobile
base and no natural-language task planner here, every script is invoked
directly):

| Layer | What lives here | Files |
|---|---|---|
| **Vision** | Turns a camera frame into named objects + pixel boxes, then into robot-frame (x, y, z) + a grasp yaw | `perception_zeroshot.py` |
| **IK / manipulation** | Turns a target pose into safe joint motion; owns every hardware safety check | `grasp_ik.py`, `zeroshot_pick.py` (the move/execute functions), `kinematics.py` (read-only, from an earlier iteration) |
| **Feedback** | Did the grasp actually work? Reports back a boolean + confidence | `zeroshot_pick.py`'s `close_gripper_and_verify` |

### Vision: `perception_zeroshot.py`

One function, `_call_gemini(frame, "all distinct physical objects on the
table")`, does all the real work: it sends the current camera frame to
Gemini with a plain-English instruction (no fixed candidate-object list —
Gemini decides what's there and what to call it) and gets back JSON:
object name, a 0–1 confidence, and a bounding box in normalized
0–1000 coordinates. Everything else in the file is either format
conversion or a narrow safety net:

- `_correct_flicker_confusion` — a real bug, found and fixed during
  development: a dark-toned red object was sometimes read as "black" by
  an earlier CLIP-based backend, especially under dim/flickering light.
  Verified fix: HSV hue/saturation of the actual pixels survive a
  brightness change far better than raw RGB does (a red-cube crop held
  median saturation ≥31 all the way down to 20% simulated brightness; a
  genuinely black crop stayed at median saturation ~0–4 even scaled *up*
  to 200%). If a detection is labeled "black ..." but its own pixels are
  clearly saturated and in the red hue band, it's relabeled "red cube".
  Kept as a cheap backend-agnostic safety net even though later backends
  (YOLOE, Gemini) haven't reproduced the original bug.
- `ROBOT_PART_LABELS` — a small exact-match exclude list
  (`robot gripper`, `robot arm part`, `camera`, `connector`) as a *backup*
  filter. The primary mechanism is just asking Gemini in the prompt to
  ignore the robot's own hardware, which worked correctly and reliably in
  real testing — this list only catches the rare case where a label
  happens to match one of these four exact phrases.
- `estimate_grasp_yaw_deg` — PCA on whatever mask-shaped array it's given.
  Since Gemini returns boxes, not masks, callers pass a filled rectangle
  (`_bbox_mask`); a near-square or round object naturally returns `None`
  (below `YAW_EVAL_RATIO_MIN`, its major/minor-axis ratio isn't decisive
  enough to commit to a yaw) rather than a noisy, arbitrary angle.
- `estimate_height_m` — no camera-to-robot 3D calibration exists (no full
  extrinsic/hand-eye calibration has been done for this rig), so height
  isn't computed as an absolute 3D coordinate. Instead it's a **depth
  delta**: median depth over the whole frame (≈ the table) minus median
  depth inside the object's bounding box, converted to meters. This needs
  no camera calibration beyond having a depth stream at all.

#### Backend history (why Gemini, not something local)

Three backends were tried, in order, each replacing the last:

1. **Grounding DINO (text-prompted detection) + SAM (segmentation) + CLIP
   (zero-shot naming)** — three separate local models, ~1.8–2s per full
   pass. Worked, but CLIP's classification for the red cube specifically
   was unreliable under this rig's lighting (the flicker-confusion bug
   above was found here).
2. **YOLOE** (`yoloe-26l-seg.pt`, Ultralytics' newest architecture at the
   time) — one local model, detection + segmentation + naming in one
   call, **~0.017s** warmed. Bounding boxes matched Grounding DINO's own
   to within 3px, and a specificity check (absent-object prompts like
   "banana"/"elephant" against a real frame with neither present) gave 0
   false detections down to a 0.01 confidence threshold. The real
   weakness: its confidence score for the red cube specifically stayed
   low (~0.05) and it occasionally ranked a competing "black cube" guess
   higher for the same region — smaller, subtler mistakes than the older
   backends, but a real accuracy ceiling for this exact object.
3. **Gemini** (currently `gemini-flash-lite-latest`) — a cloud
   vision-language model, ~3–9s per call depending on load and which
   model variant. In side-by-side testing on the same live frame, it
   never confused the red cube with black, and correctly excluded the
   robot's own gripper/wrist hardware from a plain prompt instruction
   alone, with no hand-maintained label list needed as the primary
   mechanism. The real cost is latency (400×+ slower than YOLOE) and an
   external dependency (network + API key + a daily request quota on the
   free tier) — `zeroshot_viewer.py`'s poll interval was raised from 1.5s
   to 12s accordingly, and this pipeline is built around "detect once,
   then act", not continuous closed-loop polling.

`perception_zeroshot.py`'s public functions
(`detect_zeroshot`/`detect_and_segment`/`detect_all_objects`/
`label_all_objects`) kept the *exact same signatures* across all three
backend swaps, so nothing downstream (`zeroshot_pick.py`,
`zeroshot_viewer.py`) needed to change when the backend did.

### IK / manipulation: `grasp_ik.py` + `zeroshot_pick.py`

- `grasp_ik.solve_fixed_roll_ik` — the numerical solver described above.
  `_solve_from_seed` does one damped-least-squares attempt (finite-
  difference Jacobian on the 4 free joints); the public function tries the
  caller's seed first, and if the residual exceeds `CONVERGENCE_TOL_M`
  (5mm), retries from four fixed, spread-out fallback seeds and keeps
  whichever attempt converged best. This exists because a single-seed
  solve can get stuck against a joint limit — reproduced systematically in
  a stress test (21/60 random realistic seeds failed under 5mm for one
  fixed target before this fix; 0/60 after, and 0/200 across a broader
  multi-target sweep).
- `zeroshot_pick.py`'s `_move_arm` — every actual robot motion goes
  through this, never a single raw `send_action` call. It interpolates
  over ~20 steps and checks actual-vs-commanded joint lag every few steps,
  raising `CollisionDetected` (and retreating to the last known-good pose)
  if the arm falls more than 10° behind for 3 consecutive checks. This
  specifically fixes a real bug found in testing: a raw single-call move
  got silently truncated by lerobot's own per-call safety clamp (15°/call)
  and the arm ended up nowhere near its intended target.
- `IK_SAFETY_TOL_M` — a second gate, independent of `grasp_ik`'s own
  convergence check: even a solved IK result is refused (no move sent) if
  its residual exceeds 8mm, so a genuinely unreachable target never
  silently sends the arm somewhere wrong.
- `_safe_return_home` — every execution path, success or failure, ends by
  interpolating back to wherever the arm was at the start of the run, and
  is wrapped so that even an unrelated error here (a real one hit once: a
  serial "no status packet" comms hiccup) can never prevent the final
  `disconnect()` from running. This matters because the arm in the
  original development setup was shared with other people — leaving the
  port open or the arm mid-air because of an unrelated exception would be
  a real problem for the next person.
- `check_port_not_busy` — refuses to connect at all if something else
  already has the serial port open, rather than fighting over it.

### Feedback: gripper-position grasp verification

This gripper has no force/torque sensor, so "did I actually grasp
something" is inferred from the gripper's own commanded-vs-actual closed
position: closing on empty air lands near a measured empty-closed
baseline (`GRIPPER_EMPTY_CLOSED_PCT`); closing on a real object stops
further open (wedged), by at least `GRASP_DETECT_MARGIN_PCT`. On a
confirmed grasp, the gripper backs off from a full/extreme close to a
gentler hold target (`GRIP_HOLD_MARGIN_PCT`) rather than staying pinned at
maximum force for the whole carry — found necessary after a real run where
staying at max close current the whole time tripped the servo's own
overload protection and dropped an already-successful grasp.

## Hardware

- **SO-101** follower arm (Feetech STS3215 servos), USB serial
  (`config.FOLLOWER_PORT`, default `/dev/ttyACM0`)
- **Orbbec Astra S** RGB-D camera, mounted overhead looking down at the
  table — the only camera this package's own scripts read from
- A wrist camera on the arm exists in this rig but is **not** read by any
  script here (monitoring only, via `camera_hub.py` one directory up)

## Setup

This repo mirrors the real directory layout: this file lives in
`vision_pick_place/task_red_cube_to_bin/`, and everything this package
reads at runtime but doesn't itself contain lives one directory up in
`vision_pick_place/` — the Astra S camera driver, the SO-101 URDF, and the
table-plane calibration this specific rig produced. See that directory's
own files (`astra_s_live.py`, `camera_hub.py`, `camera_utils.py`,
`cube_detector.py`, `orbbec_color_camera.py`, `openni2_redist/`,
`so101_urdf/`, `homography.json`) — nothing outside this repo is required
for those, but a few Python packages are:

```bash
# from ~/lerobot (or wherever your lerobot checkout lives - this package
# expects the main lerobot venv/deps: torch, opencv, numpy already present)
uv add google-genai   # Gemini SDK, if not already installed
uv pip install primesense   # ctypes wrapper the Astra S driver uses over openni2_redist/'s SDK

# Gemini API key - either:
export GEMINI_API_KEY="your-key-here"
# or, so it survives across shells/sessions:
echo "your-key-here" > ~/.gemini_api_key && chmod 600 ~/.gemini_api_key
```

`homography.json` here is **rig-specific** (pixel↔xy calibration tied to
exactly how this Astra S was physically mounted over this table) — if
your camera/table/robot-base geometry differs at all, redo it with
`calibrate_camera.py` (one directory up; not included in this repo, since
it additionally depends on `robot_control.py`, a raw-servo-register
control layer from an earlier pipeline iteration not otherwise needed
here) before trusting any of this package's coordinates.

Camera frames are read from published files
(`config.ASTRA_RGB_FRAME_PATH`, `config.ASTRA_DEPTH_MM_PATH`), written by
`astra_s_live.py` — **that script owns the actual camera device**; nothing
in this package opens it directly (only one process can hold a UVC/OpenNI2
device open at a time). Run it first:

```bash
cd ../                                               # into vision_pick_place/
ASTRA_LIVE_HEADLESS=1 python3 astra_s_live.py &       # headless: no GUI window of its own
cd task_red_cube_to_bin/
```

## Usage

Run from this directory with `uv run python3 <script>` (or plain
`python3` if you're already in the right venv). Every entry point defaults
to a **dry run** — it prints the full plan (detected object, computed
robot coordinates, solved joint angles, IK residual) without moving
anything; add `--live` to actually move the real arm.

### Pick one named object

```bash
python3 zeroshot_pick.py "red cube"
# prompt: 'red cube'
#   detection px: cx=390.0 cy=206.5 bbox=(364, 178, 52, 56)
#   xy (robot frame): (0.2198, 0.0364)
#   height_m: 0.041
#   target_xyz: (0.2198, 0.0364, 0.0390)
#   solved arm joints (deg): [-13.87, -31.33, 51.55, 47.27, 90.0]
#   IK position residual: 0.82mm

python3 zeroshot_pick.py "red cube" --live                        # actually move the real arm
python3 zeroshot_pick.py "red cube" --place "black bin" --live    # pick AND place
```

### Discover objects without naming them

Useful for objects you haven't described in advance — Gemini names
whatever it finds, you just pick which one by number:

```bash
python3 zeroshot_pick.py --list
# [0] xy=(0.220, 0.037) height_m=0.041 bbox=(364, 178, 52, 56) yaw_deg=90.0
# [1] xy=(0.223, 0.134) height_m=0.035 bbox=(220, 174, 106, 112) yaw_deg=None
# [2] xy=(0.069, -0.051) height_m=None bbox=(485, 423, 86, 55)  [no plausible table height - might be robot hardware]

python3 zeroshot_pick.py --index 0 --place "black bin" --live
```

### Click-to-pick (live GUI window)

```bash
python3 zeroshot_click_pick.py
```

Shows the Astra view with Gemini-detected boxes + a colorized depth panel
side by side, refreshed continuously (the video itself updates every
frame; only the detection boxes refresh on Gemini's own ~3–9s cadence —
decoupled specifically so the window never *looks* frozen while a
detection call is in flight). Left-click any box to immediately pick it up
and place it at `PLACE_PROMPT` (module-level constant, default
`"black bin"`). `q`/ESC quits without moving anything.

### Just watch what the camera sees (no robot)

```bash
python3 zeroshot_viewer.py       # GUI window: RGB + detection boxes | depth, plus a
                                  # visible "LIVE hh:mm:ss frame#N" ticker + moving dot,
                                  # so it's unmistakable whether the feed is actually live
python3 zeroshot_http_view.py    # same feed over HTTP (http://localhost:8899/) instead of
                                  # a native window - use this if the GUI window doesn't
                                  # visibly repaint (seen under some Wayland/XWayland setups)
```

### Run the test suite (no camera or robot needed)

```bash
python3 test_grasp_ik.py             # pure math: IK convergence across targets/rolls/seeds
python3 test_perception_zeroshot.py  # detection against a saved real photo
python3 test_zeroshot_pick.py        # full pipeline dry-run against a saved photo + real homography.json
```

All three pass without touching any hardware — they're the standard way
to sanity-check a code change before ever pointing it at the real arm.

## Module map

### `vision_pick_place/task_red_cube_to_bin/` (this project)

| File | Role |
|---|---|
| `perception_zeroshot.py` | Gemini-based detection/naming, depth-delta height, bbox-PCA yaw, the flicker-confusion safety net |
| `grasp_ik.py` | Fixed-roll 4-DOF numerical IK with multi-restart |
| `zeroshot_pick.py` | Orchestration + CLI (`--list`/`--index`/`--place`/`--live`), all the move/grasp/verify/safety logic, `MAX_ROLL_EXCURSION_DEG` |
| `zeroshot_viewer.py` | Live GUI viewer: RGB + boxes + depth, decoupled from detection latency |
| `zeroshot_click_pick.py` | Same view, plus click-to-select-and-execute |
| `zeroshot_http_view.py` | Browser-based fallback view over local HTTP |
| `kinematics.py`, `config.py`, `perception.py`, `gripper.py` | From an earlier (HSV-detection, click-calibrated) iteration of this task — imported **read-only** by everything above, never modified by this package, since they're already real-hardware-validated |
| `mujoco_sim/` | SO-101 MJCF model + collision meshes, from an earlier MuJoCo-simulation iteration |
| `sim_dry_run.py`, `task_state_machine.py`, `collect_training_data.py`, `main.py` | Also from that earlier iteration; kept for reference, not part of the active pipeline |

### `vision_pick_place/` (one directory up — runtime dependencies)

| File / folder | Role |
|---|---|
| `astra_s_live.py` | **Owns the Astra S camera device.** Publishes RGB + depth frames to disk (`/tmp/vsp_astra_rgb.png`, `/tmp/vsp_astra_depth_mm.npy`) that everything in `task_red_cube_to_bin/` reads instead of opening the camera itself. Must be running (see Setup) before any detection call will find a fresh frame. `ASTRA_LIVE_HEADLESS=1` skips its own GUI window. |
| `orbbec_color_camera.py` | The actual OpenNI2 device wrapper (`ThreadedOrbbecRGBDCamera`) `astra_s_live.py` and `camera_hub.py` use — one background thread reading color+depth off one device handle. |
| `openni2_redist/` | The Orbbec/OpenNI2 SDK's redistributable binaries (`libOpenNI2.so` + drivers) — the Astra S needs these specifically; the `primesense` pip package alone isn't sufficient. |
| `camera_utils.py` | Shared `PublishedFrameSource`/`PublishedDepthSource` (read-a-published-file-instead-of-the-device pattern) + `find_camera_index` (resolves a UVC camera's `/dev/videoN` by USB product name, since it can renumber across reconnects). |
| `cube_detector.py` | Fixed-HSV-threshold red-cube/black-bin detection — used for the monitor overlay in `astra_s_live.py`/`camera_hub.py`'s own windows only; **not** part of `task_red_cube_to_bin`'s actual (Gemini-based) detection path. |
| `camera_hub.py` | Wrist-camera viewer + publisher (`/tmp/vsp_wrist.png`) — optional, monitoring only, not read by any script in `task_red_cube_to_bin/`. |
| `so101_urdf/` | URDF + collision meshes `kinematics.py`'s `RobotKinematics` (placo-based IK) loads. |
| `homography.json` | Table-plane pixel→robot-xy calibration for **this specific physical rig** — see the rig-specific warning above. |

## Known limitations

- **Gemini free-tier quota.** Some model variants cap out around 20
  requests/day; `perception_zeroshot._GEMINI_MODEL_ID` can be pointed at a
  different model (a separate quota bucket) if you hit this — that's why
  the default is the smaller `gemini-flash-lite-latest` rather than a
  larger/newer model.
- **No true segmentation** — grasp-yaw estimates are a bounding-box
  approximation, not a real per-pixel mask; fine for boxy objects, weaker
  for irregular or concave ones.
- **Open-loop accuracy on small objects is not yet fully verified.** The
  first real `--live` run of this exact pipeline reached its target pose
  safely (no collision, the roll-excursion cap did its job) but missed
  the physical grasp on a ~50mm cube — the gripper closed to near its
  empty-closed baseline. Likely fixes if you hit this: redo
  `homography.json`'s calibration with more/better-distributed points, or
  add a wrist-cam closed-loop refinement pass before the final descend
  (deliberately not present here — see "Why build it this way" above).
- **This follower arm may be shared** in some setups — every live script
  checks the port isn't already in use before connecting and always exits
  through a return-to-home-then-disconnect path, but if you kill a script
  with `SIGKILL` instead of `Ctrl-C`/`SIGINT`, that cleanup is skipped.
