from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "preference_post_training_upgrade_gate_v1"


CANDIDATES = [
    {
        "version": "preference_trajectory_post_training_v1_candidate",
        "method": "偏好加权 trajectory-kNN 后训练候选",
        "preference_source": "scripted demo episode/attempt outcome proxy",
        "preference_strategy": "trajectory-level target-distance reward with success/failure/out-of-table multipliers",
        "report": "docs/preference_trajectory_post_training_report.md",
        "csv": "docs/preference_trajectory_post_training_report.csv",
        "json": "outputs/evaluations/preference_trajectory_post_training_v1.json",
        "model": "outputs/preference_post_training/preference_trajectory_post_training_20260720_165005.npz",
        "video": "outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4",
        "status_note": "旧候选只记录 grasp_success 聚合，缺少批量 tcp_grasp_lift_success 和 strict_grasp_lift_success 字段。",
    },
    {
        "version": "preference_contact_aware_trajectory_post_training_v1_candidate",
        "method": "偏好加权 + 相对几何 trajectory-conditioned BC 后训练候选",
        "preference_source": "scripted demo episode/attempt outcome proxy",
        "preference_strategy": "trajectory-level target-distance reward with success/failure/out-of-table multipliers + relative geometry",
        "report": "docs/preference_contact_aware_trajectory_post_training_report.md",
        "csv": "docs/preference_contact_aware_trajectory_post_training_report.csv",
        "json": "outputs/evaluations/preference_contact_aware_trajectory_post_training_v1_candidate.json",
        "model": "outputs/preference_post_training/preference_contact_aware_trajectory_post_training_20260721_000449.npz",
        "video": "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4",
        "status_note": "train-range 目标距离和 TCP 抬升改善，但 held-out 抬升为 0，标准抓取和严格抓取仍为 0。",
    },
    {
        "version": "preference_ranked_trajectory_post_training_v1_candidate",
        "method": "episode 内排序偏好 + 相对几何 trajectory-kNN 后训练候选",
        "preference_source": "episode-level ranked attempt preference",
        "preference_strategy": "success/failure/out-of-table ranked attempt weights + relative geometry",
        "report": "docs/preference_ranked_trajectory_post_training_report.md",
        "csv": "docs/preference_ranked_trajectory_post_training_report.csv",
        "json": "outputs/evaluations/preference_ranked_trajectory_post_training_v1_candidate.json",
        "model": "outputs/preference_post_training/preference_ranked_trajectory_post_training_20260721_031024.npz",
        "video": "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "status_note": "快速 2+2 诊断显示 train-range 放置成功，但 held-out 仍弱，标准抓取和严格抓取仍为 0。",
    },
    {
        "version": "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "method": "episode 内排序偏好 + dense trajectory-kNN 后训练候选",
        "preference_source": "episode-level ranked attempt preference",
        "preference_strategy": "success/failure/out-of-table ranked attempt weights without relative geometry, sample-stride=8",
        "report": "docs/preference_trajectory_post_training_v1_ranked_objective_summary.md",
        "csv": "docs/preference_trajectory_post_training_v1_ranked_objective_report.csv",
        "json": "outputs/evaluations/preference_trajectory_post_training_v1_ranked_objective_candidate.json",
        "model": "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz",
        "video": "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4",
        "status_note": "sample-stride=8 的密集样本让 train-range 放置达到 4/5，但 held-out 仍为 0/5，固定 seed0 视频失败，标准抓取仍为 0。",
    },
    {
        "version": "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "method": "TCP 抬升排序偏好 + relative-geometry trajectory-kNN 后训练候选",
        "preference_source": "episode-level placed/tcp-lift attempt proxy",
        "preference_strategy": "placed/tcp-lift/out-of-table ranked attempt weights + relative geometry, sample-stride=16",
        "report": "docs/preference_trajectory_post_training_v1_tcp_lift_rank_report.md",
        "csv": "docs/preference_trajectory_post_training_v1_tcp_lift_rank_report.csv",
        "json": "outputs/evaluations/preference_trajectory_post_training_v1_tcp_lift_rank_candidate.json",
        "model": "outputs/preference_post_training/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_20260721_090438.npz",
        "video": "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.mp4",
        "status_note": "train-range 放置 5/5 且 TCP 抬升 5/5，但标准抓取和严格抓取仍为 0；held-out 只有 1/5 放置且 TCP 抬升为 0。",
    },
]


FIELDNAMES = [
    "版本",
    "当前定位",
    "偏好来源",
    "权重策略",
    "train_range 放置成功",
    "heldout 放置成功",
    "train_range TCP抬升",
    "heldout TCP抬升",
    "train_range 标准抓取",
    "heldout 标准抓取",
    "train_range 严格抓取",
    "heldout 严格抓取",
    "固定视频",
    "报告",
    "viewer命令",
    "是否允许升级formal",
    "不能升级原因",
    "论文边界",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese formal-upgrade gate for preference post-training candidates.")
    parser.add_argument("--candidate-index", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    parser.add_argument("--planned-registry", type=Path, default=ROOT / "docs" / "next_experiment_registry.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "preference_post_training_upgrade_gate.md")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "preference_post_training_upgrade_gate.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "是"}


def ratio(successes: int | None, episodes: int | None) -> str:
    if successes is None or episodes is None:
        return "未记录"
    return f"{successes}/{episodes}"


def aggregate_per_episode(rows: list[dict[str, str]], split: str) -> dict[str, int | None]:
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        return {"episodes": None, "successes": None, "tcp": None, "ever": None, "strict": None}
    return {
        "episodes": len(selected),
        "successes": sum(bool_value(row.get("success")) for row in selected),
        "tcp": sum(bool_value(row.get("tcp_grasp_lift_success")) for row in selected),
        "ever": sum(bool_value(row.get("ever_grasp_success")) or bool_value(row.get("grasp_success")) for row in selected),
        "strict": sum(bool_value(row.get("strict_grasp_lift_success")) for row in selected),
    }


def aggregate_report(path: Path, version: str, split: str) -> dict[str, int | None]:
    rows = read_csv(path)
    if rows and "successes" in rows[0] and "episodes" in rows[0]:
        selected = [row for row in rows if row.get("split") == split]
        if selected:
            row = selected[0]
            episodes = int(float(row["episodes"]))
            return {
                "episodes": episodes,
                "successes": int(float(row["successes"])),
                "tcp": int(float(row["tcp_grasp_lift_successes"])) if row.get("tcp_grasp_lift_successes") not in (None, "") else None,
                "ever": int(float(row.get("grasp_successes", "0"))),
                "strict": int(float(row["strict_grasp_lift_successes"])) if row.get("strict_grasp_lift_successes") not in (None, "") else None,
            }
    return aggregate_per_episode(rows, split)


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    key = "版本" if rows and "版本" in rows[0] else "version"
    return {row[key]: row for row in rows if row.get(key)}


def build_row(candidate: dict[str, str], candidate_index: dict[str, dict[str, str]]) -> dict[str, str]:
    csv_path = ROOT / candidate["csv"]
    train = aggregate_report(csv_path, candidate["version"], "train_range")
    heldout = aggregate_report(csv_path, candidate["version"], "heldout")
    indexed = candidate_index.get(candidate["version"], {})
    viewer = indexed.get("完整viewer命令", "")
    paper_boundary = indexed.get("论文边界", "不能写成在线 RL、真实人类偏好优化、OpenVLA/Robot VLA 后训练或真实 WidowX 结果。")

    missing_formal_fields = []
    if train["strict"] is None or heldout["strict"] is None:
        missing_formal_fields.append("批量 strict_grasp_lift_success 字段不完整")
    strict_total = (train["strict"] or 0) + (heldout["strict"] or 0)
    if strict_total == 0:
        missing_formal_fields.append("标准严格抓取成功为 0")
    if heldout["successes"] in (None, 0):
        missing_formal_fields.append("held-out 放置泛化不足")
    if not (ROOT / candidate["video"]).exists():
        missing_formal_fields.append("固定视频缺失")
    if not viewer:
        missing_formal_fields.append("慢速 viewer 命令缺失")

    reason = "；".join(dict.fromkeys([*missing_formal_fields, candidate["status_note"]]))
    return {
        "版本": candidate["version"],
        "当前定位": candidate["method"],
        "偏好来源": candidate["preference_source"],
        "权重策略": candidate["preference_strategy"],
        "train_range 放置成功": ratio(train["successes"], train["episodes"]),
        "heldout 放置成功": ratio(heldout["successes"], heldout["episodes"]),
        "train_range TCP抬升": ratio(train["tcp"], train["episodes"]) if train["tcp"] is not None else "未记录",
        "heldout TCP抬升": ratio(heldout["tcp"], heldout["episodes"]) if heldout["tcp"] is not None else "未记录",
        "train_range 标准抓取": ratio(train["ever"], train["episodes"]) if train["ever"] is not None else "未记录",
        "heldout 标准抓取": ratio(heldout["ever"], heldout["episodes"]) if heldout["ever"] is not None else "未记录",
        "train_range 严格抓取": ratio(train["strict"], train["episodes"]) if train["strict"] is not None else "未记录",
        "heldout 严格抓取": ratio(heldout["strict"], heldout["episodes"]) if heldout["strict"] is not None else "未记录",
        "固定视频": candidate["video"],
        "报告": candidate["report"],
        "viewer命令": viewer,
        "是否允许升级formal": "否",
        "不能升级原因": reason,
        "论文边界": paper_boundary,
    }


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str]], planned_row: dict[str, str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_formal_version": "preference_trajectory_post_training_v1",
        "candidate_count": len(rows),
        "formal_upgrade_allowed_now": False,
        "formal_upgrade_allowed_count": sum(row["是否允许升级formal"] == "是" for row in rows),
        "planned_registry_status": planned_row.get("status") if planned_row else "missing",
        "planned_success_gate": planned_row.get("success_gate") if planned_row else "",
        "rows": rows,
        "formal_gate_requirements": [
            "记录 preference 来源和权重策略",
            "同时保留成功与失败轨迹比例",
            "至少包含 train-range 与 held-out 闭环评测",
            "批量记录 success、grasp_success、object_z、tcp_grasp_lift_success 和 strict_grasp_lift_success",
            "固定视频与慢速 viewer 命令必须存在",
            "通过 verify_experiment_artifacts.py 后才可进入正式方法版本统计",
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, rows: list[dict[str, str]], planned_row: dict[str, str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preference Post-training 正式升级门禁",
        "",
        f"版本：`{VERSION}`",
        "",
        f"用途：把 `preference_trajectory_post_training_v1` 从 planned 升级为正式后训练方法之前必须满足的证据条件固化下来，并统一审计当前 {len(rows)} 个偏好后训练候选。该文件不新增策略结果，只说明哪些候选可以写、哪些不能升级 formal。",
        "",
        "## 1. 总体判断",
        "",
        "- 目标正式版本：`preference_trajectory_post_training_v1`。",
        f"- 注册表当前状态：`{planned_row.get('status', 'missing') if planned_row else 'missing'}`。",
        "- 当前允许升级 formal：`否`。",
        f"- 当前审计候选：`{len(rows)}` 个。",
        "- 主要原因：所有候选都没有形成标准严格抓取成功；部分候选 held-out 泛化弱，第一版候选还缺少批量 strict grasp 字段。",
        "",
        "## 2. formal 升级要求",
        "",
        "- 必须记录 preference 来源、权重策略、成功/失败轨迹比例，不能只保留成功片段。",
        "- 必须同时给出 train-range、held-out 和固定视频，而不是只看 seed0 成功样例。",
        "- 必须记录 `success`、`grasp_success`、`object_z`、`tcp_grasp_lift_success` 和 `strict_grasp_lift_success`。",
        "- 若 `success=True` 但 `grasp_success=False`，只能写成目标距离达标或局部抬升迹象，不能写成稳定抓取。",
        "- 升级正式方法前必须补 `docs/evaluation_summary.csv`、`docs/model_resource_summary.csv`、`docs/video_evidence_index.csv`、失败模式分类和慢速 viewer 命令，并通过总体验证。",
        "",
        "## 3. 候选审计表",
        "",
        md_row(FIELDNAMES),
        md_row(["---"] * len(FIELDNAMES)),
    ]
    for row in rows:
        lines.append(md_row([row[field] for field in FIELDNAMES]))

    lines.extend(
        [
            "",
            "## 4. 当前可写结论",
            "",
            "- 可以写：已经完成 trajectory-level preference weighting、relative-geometry preference、episode-ranked preference 和 ranked-objective dense sample 四类本地候选诊断，并保留失败视频、评测表和 viewer 命令。",
            "- 可以写：偏好权重可以改善训练范围的目标距离或 TCP 抬升迹象，但没有证明标准抓取或 held-out 稳定泛化。",
            "- 不能写：`preference_trajectory_post_training_v1` 已作为正式后训练方法完成。",
            "- 不能写：当前结果是在线 RL、真实 human preference optimization、OpenVLA/OFT 后训练或真实 WidowX 结果。",
            "",
            "## 5. 下一步",
            "",
            "1. 若继续做本地偏好后训练，先补带 `strict_grasp_lift_success` 的 5+5 或 10+10 批量评测，不再沿用缺字段旧口径。",
            "2. 把 preference reward 从目标距离扩展到接触保持、夹爪闭合、物体高度和放置阶段完成度。",
            "3. 若接入真实 robot VLA 表征，必须另起 `robot_vla_*` 版本，并通过远端结果回填门禁，不能复用这些本地候选名。",
            "",
            "## 6. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_preference_post_training_upgrade_gate.py"}"',
            "```",
            "",
            "## 7. 代表性慢速 viewer 命令",
            "",
            "```powershell",
            rows[-1]["viewer命令"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_index = by_version(read_csv(args.candidate_index))
    planned_rows = by_version(read_csv(args.planned_registry))
    planned_row = planned_rows.get("preference_trajectory_post_training_v1")
    rows = [build_row(candidate, candidate_index) for candidate in CANDIDATES]
    write_csv(args.output_csv, rows)
    write_json(args.output_json, rows, planned_row)
    write_md(args.output_md, rows, planned_row)
    print(f"preference_post_training_upgrade_gate_md: {args.output_md}", flush=True)
    print(f"preference_post_training_upgrade_gate_csv: {args.output_csv}", flush=True)
    print(f"preference_post_training_upgrade_gate_json: {args.output_json}", flush=True)
    print(f"preference_post_training_upgrade_gate_rows: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
