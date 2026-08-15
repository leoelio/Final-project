from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_experiment_command_index import viewer_command  # noqa: E402


QUICK_TARGETS = {
    "dashboard": ROOT / "docs" / "experiment_dashboard.html",
    "deck": ROOT / "docs" / "defense_deck.html",
    "gallery": ROOT / "docs" / "video_evidence_gallery.html",
    "storyboard": ROOT / "docs" / "video_presentation_storyboard.html",
    "playlist": ROOT / "docs" / "defense_video_playlist.html",
    "cue-sheet": ROOT / "docs" / "defense_video_cue_sheet.md",
    "comparison": ROOT / "docs" / "method_comparison_dashboard.html",
    "visual-index": ROOT / "docs" / "thesis_visual_evidence_index.html",
    "qa": ROOT / "docs" / "defense_qa_playbook.html",
    "lineage": ROOT / "docs" / "version_lineage_index.html",
    "package": ROOT / "docs" / "final_experiment_package.md",
    "handoff": ROOT / "docs" / "final_showcase_handoff.md",
    "narrative-script": ROOT / "docs" / "final_defense_narrative_script.md",
    "remaining-board": ROOT / "docs" / "remaining_experiment_execution_board.md",
    "matrix": ROOT / "docs" / "result_matrix.md",
    "trajectory-act-brief": ROOT / "docs" / "trajectory_act_conclusion_brief.md",
    "trajectory-act-slow": ROOT / "docs" / "trajectory_act_slow_viewer_guide.md",
    "manifest": ROOT / "docs" / "final_artifact_manifest.md",
    "evidence-pack": ROOT / "docs" / "defense_evidence_pack.md",
    "preference-gate": ROOT / "docs" / "preference_post_training_upgrade_gate.md",
    "preference-ablation": ROOT / "docs" / "preference_post_training_ablation_matrix.md",
    "launcher": ROOT / "docs" / "showcase_launcher_guide.md",
    "live-runbook": ROOT / "docs" / "defense_live_runbook.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open local experiment showcase artifacts by claim, stage, method, candidate, or quick target.",
    )
    parser.add_argument("--list", choices=["quick", "claims", "stages", "methods", "candidates"], help="List available targets.")
    parser.add_argument(
        "--target",
        help="Target to open, for example claim:C03, stage:2, method:torch_act_state_chunk_v1, candidate:grasp_gated_torch_act_state_chunk_v1_candidate, dashboard, deck, matrix, narrative-script, remaining-board, trajectory-act-brief, trajectory-act-slow, comparison, playlist, cue-sheet, visual-index, qa, lineage, preference-gate, preference-ablation, evidence-pack, live-runbook.",
    )
    parser.add_argument(
        "--action",
        choices=["open", "open-all", "viewer", "print"],
        default="open",
        help="open opens the primary artifact; open-all opens related evidence; viewer starts a MuJoCo viewer for method or candidate targets.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for method viewer targets.")
    parser.add_argument("--task", default="place_blue_cube_blue_pad", help="Task for method viewer targets.")
    parser.add_argument("--complexity", default="medium", help="Complexity for method viewer targets.")
    parser.add_argument("--viewer-speed", type=float, default=0.05, help="MuJoCo viewer playback speed. Smaller is slower.")
    parser.add_argument("--viewer-duration", type=float, default=60.0, help="Seconds to keep the viewer open after rollout.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without opening files or starting viewers.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_versions(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8-sig"))["methods"]


def split_items(value: str) -> list[str]:
    return [part.strip().strip("`") for part in value.replace("；", "\n").splitlines() if part.strip()]


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def print_action(message: str) -> None:
    print(message, flush=True)


def open_path(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    print_action(f"open: {path}")
    if dry_run:
        return
    suffix = path.suffix.lower()
    if suffix in {".md", ".csv", ".json", ".txt"}:
        subprocess.Popen(["notepad.exe", str(path)])
    else:
        os.startfile(path)  # type: ignore[attr-defined]


def launch_viewer(command: str, *, dry_run: bool) -> None:
    rendered = command.replace("\n", "; ")
    print_action(rendered)
    if dry_run:
        return
    ps_command = f"Start-Process powershell.exe -ArgumentList '-NoExit','-Command',{ps_quote(rendered)}"
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps_command], check=True)


def override_viewer_timing(command: str, *, speed: float, duration: float) -> str:
    command = re.sub(r"(--speed)\s+\S+", rf"\1 {speed:g}", command)
    return re.sub(r"(--duration)\s+\S+", rf"\1 {duration:g}", command)


def list_quick() -> None:
    for name, path in QUICK_TARGETS.items():
        print(f"{name}: {path}", flush=True)


def list_claims(rows: list[dict[str, str]]) -> None:
    for row in rows:
        print(f"{row['claim_id']}: {row['claim_type']} -> {row['primary_video']}", flush=True)


def list_stages(rows: list[dict[str, str]]) -> None:
    for row in rows:
        print(f"{row['阶段编号']}: {row['阶段名称']} -> {split_items(row['视频证据'])[0]}", flush=True)


def list_methods(methods: list[dict[str, str]]) -> None:
    for method in methods:
        print(f"{method['version']}: {method['method']} -> {method['clip']}", flush=True)


def list_candidates(rows: list[dict[str, str]]) -> None:
    for row in rows:
        print(f"{row['版本']}: {row['方法定位']} -> {row['视频文件']}", flush=True)


def handle_claim(target_id: str, action: str, dry_run: bool) -> None:
    rows = read_csv(ROOT / "docs" / "claim_video_playback_index.csv")
    by_id = {row["claim_id"].upper(): row for row in rows}
    claim = by_id.get(target_id.upper())
    if claim is None:
        raise KeyError(f"unknown claim target: {target_id}")
    if action == "print":
        print(json.dumps(claim, ensure_ascii=False, indent=2), flush=True)
        return
    open_path(resolve(claim["primary_video"]), dry_run=dry_run)
    if action == "open-all":
        for command in split_items(claim["helper_commands"]):
            path_text = command.split('"')[1] if '"' in command else command.replace("Start-Process", "").replace("notepad.exe", "").strip()
            open_path(Path(path_text), dry_run=dry_run)


def handle_stage(target_id: str, action: str, dry_run: bool) -> None:
    rows = read_csv(ROOT / "docs" / "stage_reproduction_runbook.csv")
    by_id = {row["阶段编号"]: row for row in rows}
    stage = by_id.get(target_id)
    if stage is None:
        raise KeyError(f"unknown stage target: {target_id}")
    if action == "print":
        print(json.dumps(stage, ensure_ascii=False, indent=2), flush=True)
        return
    for video in split_items(stage["视频证据"])[:1 if action == "open" else None]:
        open_path(resolve(video), dry_run=dry_run)
    if action == "open-all":
        for entry in split_items(stage["展示入口"]):
            open_path(resolve(entry), dry_run=dry_run)


def handle_method(target_id: str, action: str, args: argparse.Namespace) -> None:
    methods = read_versions(ROOT / "docs" / "experiment_versions.json")
    by_version = {method["version"]: method for method in methods}
    method = by_version.get(target_id)
    if method is None:
        raise KeyError(f"unknown method target: {target_id}")
    if action == "print":
        print(json.dumps(method, ensure_ascii=False, indent=2), flush=True)
        return
    if action == "viewer":
        launch_viewer(
            viewer_command(
                method,
                task=args.task,
                complexity=args.complexity,
                seed=args.seed,
                duration=args.viewer_duration,
                speed=args.viewer_speed,
            ),
            dry_run=args.dry_run,
        )
        return
    open_path(resolve(method["clip"]), dry_run=args.dry_run)


def handle_candidate(target_id: str, action: str, args: argparse.Namespace) -> None:
    rows = read_csv(ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    by_version = {row["版本"]: row for row in rows}
    candidate = by_version.get(target_id)
    if candidate is None:
        raise KeyError(f"unknown candidate target: {target_id}")
    if action == "print":
        print(json.dumps(candidate, ensure_ascii=False, indent=2), flush=True)
        return
    if action == "viewer":
        launch_viewer(
            override_viewer_timing(
                candidate["完整viewer命令"],
                speed=args.viewer_speed,
                duration=args.viewer_duration,
            ),
            dry_run=args.dry_run,
        )
        return
    open_path(resolve(candidate["视频文件"]), dry_run=args.dry_run)
    if action == "open-all":
        open_path(resolve(candidate["报告文件"]), dry_run=args.dry_run)
        open_path(resolve(candidate["元数据文件"]), dry_run=args.dry_run)


def handle_quick(target: str, action: str, dry_run: bool) -> None:
    path = QUICK_TARGETS.get(target)
    if path is None:
        raise KeyError(f"unknown quick target: {target}")
    if action == "print":
        print(path, flush=True)
        return
    open_path(path, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    if args.list:
        if args.list == "quick":
            list_quick()
        elif args.list == "claims":
            list_claims(read_csv(ROOT / "docs" / "claim_video_playback_index.csv"))
        elif args.list == "stages":
            list_stages(read_csv(ROOT / "docs" / "stage_reproduction_runbook.csv"))
        elif args.list == "methods":
            list_methods(read_versions(ROOT / "docs" / "experiment_versions.json"))
        else:
            list_candidates(read_csv(ROOT / "docs" / "candidate_diagnostic_video_index.csv"))
        return

    if not args.target:
        raise SystemExit("Use --list or --target. Example: --target claim:C03 --action open")

    if ":" in args.target:
        kind, target_id = args.target.split(":", 1)
        if kind == "claim":
            handle_claim(target_id, args.action, args.dry_run)
        elif kind == "stage":
            handle_stage(target_id, args.action, args.dry_run)
        elif kind == "method":
            handle_method(target_id, args.action, args)
        elif kind == "candidate":
            handle_candidate(target_id, args.action, args)
        else:
            raise KeyError(f"unknown target kind: {kind}")
    else:
        handle_quick(args.target, args.action, args.dry_run)


if __name__ == "__main__":
    main()
