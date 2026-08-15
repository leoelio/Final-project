# Final Project: WidowX MuJoCo Research Reproduction Bundle

This directory is the code-and-data review package for the MSc project on resource-efficient, vision-language-conditioned tabletop manipulation. It is a copy: the working repository remains unchanged.

## Included

- `widowx_env/`: MuJoCo environment, control, scripted expert and RGB grounding.
- `scripts/`: data collection, replay, training, closed-loop evaluation and viewer entry points.
- `research_platform/`: local experiment-management platform source.
- `assets/` and `external/wx250s_assets/`: tabletop MJCF and licensed WX250S meshes.
- `data/`: all local training data from the working project, including demonstrations, RGB frames, Core V2, contact, recovery, counterfactual and VLA-bridge exports.
- `runtime_assets/`: final calibration, compact policy model and closure audit.
- `outputs/`: primary machine-readable metrics and model checkpoints from BC, kNN, ACT-lite, diffusion-lite, action-head, CLIP, LoRA-proxy and preference experiments.
- `kaggle/`: GPU-probe and remote-training source plus packaged Kaggle datasets; no API key or credential is included.
- `MANIFEST.csv`: relative path, byte size and SHA-256 hash for every packaged file.

Generated prose reports, Word/PowerPoint files, posters, showcase videos, rendered figures, caches, virtual environments, credentials and duplicate evidence/remote-run packs are intentionally excluded. Primary training data, model checkpoints and quantitative JSON/CSV records are retained.

## Environment

Python 3.11 or 3.12 and an OpenGL-capable display are recommended. From this `Code` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first CLIP-based run may download `openai/clip-vit-base-patch32`. Core MuJoCo simulation does not require a GPU. MP4 export additionally requires a system `ffmpeg` executable on `PATH`; simulation, training and the research platform do not.

## Reproduce The Main Workflow

Check the environment:

```powershell
python .\scripts\smoke_test.py
```

Open the interactive tabletop viewer:

```powershell
python .\scripts\run_viewer.py --task place_blue_cube_blue_pad --complexity medium
```

Run the scripted feasibility controller visibly and slowly:

```powershell
python .\scripts\run_expert.py --task place_blue_cube_blue_pad --complexity medium --viewer --speed 0.10 --duration 60 --retries 2
```

Train the standard MLP behavioural-cloning baseline on the packaged 100-demonstration dataset:

```powershell
python .\scripts\train_mlp_bc.py --run-dir .\data\demos\place_blue_cube_blue_pad_medium_20260702_051752 --hidden-sizes 128,128 --batch-size 1024 --lr 0.001
```

Run the retained lightweight semantic-RGB controller in the MuJoCo viewer:

```powershell
python .\scripts\run_clip_semantic_rgb_feedback.py --model .\runtime_assets\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz --calibration .\runtime_assets\top_rgb_core_v2_calibration_v1.json --task move_leftmost_cube_to_bowl --complexity language --workspace-profile core_v2 --seed 4006 --feedback-attempts 1 --recovery-search table --viewer --duration 35 --speed 0.18 --arm-kp 105 --arm-force 70 --gripper-kp 550 --gripper-force 75 --friction 0.8
```

Start the local research platform:

```powershell
python .\research_platform\server.py --host 127.0.0.1 --port 8050
```

Then open `http://127.0.0.1:8050/`.

## Verified In This Bundle

- All formal source files are present; Python syntax validation passed.
- The repository verifier loaded MuJoCo 3.10.0 and produced a non-blank RGB render.
- Linear BC, MLP BC, action-chunk/ACT-lite and diffusion-lite completed smoke training and wrote loadable checkpoints from the packaged data.
- The retained CLIP semantic + RGB feedback controller completed seed `4006` with `1/1` task success and `0.0073 m` target error.
- The research platform health endpoint reported `healthy`, all four registered datasets available, and evidence integrity `5/5`.

PyTorch/Transformers and internet access for the initial CLIP weight download are required for VLM training. Kaggle GPU scripts are included for the remote GPU stage; the API credential is deliberately not packaged.

## Evidence Boundary

The retained controller is VLA-inspired rather than an end-to-end OpenVLA system: frozen CLIP resolves language intent, top-view RGB supplies object and target geometry, and a structured controller executes pick-and-place with at most one visual re-localisation retry. MuJoCo state is used for labels and scoring, not runtime target selection. The final independent replication recorded `278/288` strict successes (`96.5%`) across two seed-disjoint cohorts; method-specific records are under `outputs/evaluations/` and the authoritative closure record is `runtime_assets/final_closure_audit_v1.json`.
