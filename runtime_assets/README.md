# Runtime Assets

This directory contains only the small artifacts required to launch the final MuJoCo demonstration path after cloning the repository.

| File | Purpose |
| --- | --- |
| `clip_semantic_waypoint_core_v2_v1_20260721_110325.npz` | Saved lightweight semantic-waypoint policy used by the final RGB-feedback demonstration. |
| `top_rgb_core_v2_calibration_v1.json` | Top-camera pixel-to-table calibration for the `core_v2` workspace. |
| `final_closure_audit_v1.json` | Recorded aggregate evidence for the final MuJoCo closure audit. |

These files are small enough for Git and are necessary to reproduce the supplied final viewer command. Raw demonstrations, full experiment outputs, MP4 videos, and larger model checkpoints are deliberately excluded from the repository. They can be regenerated with the scripts in `scripts/`.
