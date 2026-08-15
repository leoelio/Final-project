# WidowX Research Console

This local platform adds a real-time execution, control and evidence layer without changing historical experiment data or the original HTML showcase.

该本地平台在不修改历史实验数据和原 HTML 展示页的前提下，增加实时仿真、指令控制、训练监控和双语展示层。

See `docs/research_platform_v3_guide_zh_en.md` for the bilingual architecture, TRACE governance model, evidence portfolio, paired protocol and interpretation boundary.

## What is live

- The simulation image is rendered from the current MuJoCo state, not from a recorded video.
- Bilingual commands map to the four validated tabletop tasks.
- The RGB-grounded route localises the source object from top-view RGB and the versioned plane calibration before structured execution.
- Training jobs call the repository's actual MLP BC, Action-Chunk BC and Diffusion Policy-lite scripts. Epoch losses are parsed from their real output.
- The benchmark lab runs both policies on the same task/seed pairs and reports per-policy 95% Wilson intervals plus paired disagreements.
- Every new simulation archives its initial top-view scene, final front-view state and offline RGB localisation error when applicable; the top view is the policy input only for RGB-grounded runs.
- The append-only experiment ledger records simulation, training and benchmark configuration, metrics, artifacts, parent-child relationships and reproduction commands.
- The experiment workspace compares new platform runs, previews visual evidence and exports UTF-8 CSV plus per-run JSON/Markdown reports.
- The TRACE governance workspace pre-registers immutable protocols, links real benchmarks and converts eight evidence gates into a reportability decision.
- The TRACE Evidence Portfolio maps thesis claims to exact metrics, source-file fingerprints, method lifecycles and prohibited overclaims. It separates replicated findings, bounded pilots, negative evidence and unsupported future work.
- The Evidence Release Gate freezes a self-contained thesis evidence bundle only after five gates pass. Every append-only release includes copied source records, the portfolio, experiment ledger, file hashes and bilingual manifests.
- The Low-resource Adaptation Studio combines a constrained Task Forge, real MuJoCo demonstration collection, three-seed holdout evaluation and four candidates: LoRA-style, Adapter, Micro Head and a zero-gradient Registry RGB skill adapter.
- Its resource gate self-calibrates from earlier runs of the same method/viewer mode, while `RESOURCE-PARETO-1.0` promotes a candidate only after minimum episode, success and target-error gates. Cross-method deltas require identical holdout seeds.
- `PAIR-OPT-1.0` gives two to four candidates one fingerprinted demonstration set and one identical held-out seed sequence. It executes candidates sequentially, retains failures, computes paired disagreements and exact McNemar statistics, and writes a parent optimization record with auditable candidate children.
- Fingerprint-matched demonstration reuse is optional. The platform shows an observed fresh-versus-cache measurement only when both runs use the same protocol, and labels the result as a local single-pair measurement.
- The latest completed paired result is restored from the append-only ledger after a server restart, so the decision board remains evidence-backed instead of depending on in-memory session state.
- The original `docs/integrated_research_showcase.html` is loaded unchanged in its own workspace.

## Start

From the repository root:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\run_research_platform.py" --host 127.0.0.1 --port 8050 --open-browser
```

Open `http://127.0.0.1:8050/` if the browser does not open automatically.

也可以在 VS Code 的“运行和调试”中选择 `Run Live Research Platform`。

## Verify / 验证

With the platform running:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\verify_research_platform.py"
```

To additionally exercise the live control path:

```powershell
& ".\.venv\Scripts\python.exe" ".\scripts\verify_research_platform.py" --exercise-controls
```

默认验证为只读。`--exercise-controls` 会短暂运行一个 MuJoCo 会话，并验证启动、暂停、继续和停止状态。

Use `--exercise-benchmark` to run one real same-seed policy pair and verify confidence metrics, visual evidence, reports and parent-child ledger records.

Use `--exercise-governance` to create one immutable TRACE protocol, launch its paired benchmark and verify the automatically generated decision verdict.

Use `--exercise-adaptation` to exercise the complete Registry RGB skill path on a registered task. It verifies demonstration collection, the compiled artifact, zero trainable parameters, self-calibrated process-tree memory, three held-out evaluations, Pareto promotion and the linked ledger record.

Use `--exercise-arena` to run a real LoRA-style versus Registry RGB comparison with one shared demonstration set, identical three-seed evaluation, paired statistics, candidate child records and a machine-readable optimization artifact.

The default verifier also audits the read-only portfolio API, five source-integrity gates, source downloads and the generated Markdown claim report.

Use `--exercise-release` to create and verify one immutable evidence release, including all seven bundled files and the manifest hash.

## Safety boundary

The browser cannot run arbitrary shell commands. Simulation tasks, native viewer commands, datasets and trainers are selected from fixed server-side allowlists. Task Forge accepts only the four declared cube sources, three declared targets and bounded instructions; registrations are persisted in `outputs/platform_research/adaptation_tasks.json`. Historical datasets and reports are read-only; new training checkpoints are written under `outputs/platform_training/`, new visual evidence under `outputs/platform_research/runs/`, new run records are appended to `outputs/platform_research/experiment_ledger.jsonl`, and immutable evidence bundles are created under `outputs/platform_research/releases/`.
