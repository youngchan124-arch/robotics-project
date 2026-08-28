# robotics-project

로봇 프로젝트 1

## Projects

- [`vision_pick_place/task_red_cube_to_bin/`](vision_pick_place/task_red_cube_to_bin/) -
  Zero-shot vision pick-and-place for a SO-101 robot arm (camera → Gemini
  object detection → fixed-roll IK → grasp). No imitation-learning demos
  required. `vision_pick_place/` also holds the camera-device and
  kinematics files this project reads at runtime (Astra S camera driver,
  URDF, table-plane calibration). See the project's own README for
  architecture, setup, and usage.
