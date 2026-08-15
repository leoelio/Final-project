from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "place_blue_cube_blue_pad",
    "place_blue_cube_red_pad",
    "place_red_cube_red_pad",
    "move_leftmost_cube_to_bowl",
)
TASK_NAMES = {
    "place_blue_cube_blue_pad": "蓝色立方体 -> 蓝色盘",
    "place_blue_cube_red_pad": "蓝色立方体 -> 红色盘",
    "place_red_cube_red_pad": "红色立方体 -> 红色盘",
    "move_leftmost_cube_to_bowl": "最左立方体 -> 碗",
}


def read_rows(protocol: str, method: str, suffix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task in TASKS:
        path = ROOT / "docs" / f"{protocol}_{method}{suffix}_{task}.csv"
        with path.open(encoding="utf-8-sig", newline="") as file:
            rows.extend(csv.DictReader(file))
    return rows


def is_success(row: dict[str, str]) -> bool:
    return row["success"].lower() == "true"


def task_summary(standard: list[dict[str, str]], fusion: list[dict[str, str]]) -> dict[str, object]:
    standard_by_key = {(row["task"], row["seed"]): row for row in standard}
    fusion_by_key = {(row["task"], row["seed"]): row for row in fusion}
    if standard_by_key.keys() != fusion_by_key.keys():
        raise ValueError("paired evaluation keys do not match")
    pairs = [(standard_by_key[key], fusion_by_key[key]) for key in sorted(standard_by_key)]
    baseline_successes = sum(is_success(row) for row, _ in pairs)
    fusion_successes = sum(is_success(row) for _, row in pairs)
    improved = sum(not is_success(row) and is_success(other) for row, other in pairs)
    regressed = sum(is_success(row) and not is_success(other) for row, other in pairs)
    tied_success = sum(is_success(row) and is_success(other) for row, other in pairs)
    tied_failure = sum(not is_success(row) and not is_success(other) for row, other in pairs)
    distances = {
        "standard": sum(float(row["target_distance"]) for row, _ in pairs) / len(pairs),
        "fusion": sum(float(row["target_distance"]) for _, row in pairs) / len(pairs),
    }
    regrasp_attempts = sum(int(row.get("contact_regrasp_attempts", "0")) for _, row in pairs)
    fusion_sustained_transport_successes = sum(
        is_success(row) and row.get("transport_hold_confirmed", "False").lower() == "true"
        for _, row in pairs
    )
    return {
        "episodes": len(pairs),
        "standard_successes": baseline_successes,
        "fusion_successes": fusion_successes,
        "standard_success": f"{baseline_successes}/{len(pairs)}",
        "fusion_success": f"{fusion_successes}/{len(pairs)}",
        "delta_success_points": (fusion_successes - baseline_successes) / len(pairs) * 100,
        "standard_mean_target_distance": distances["standard"],
        "fusion_mean_target_distance": distances["fusion"],
        "improved": improved,
        "regressed": regressed,
        "tied_success": tied_success,
        "tied_failure": tied_failure,
        "regrasp_attempts": regrasp_attempts,
        "fusion_sustained_transport_successes": fusion_sustained_transport_successes,
    }


def exact_mcnemar_p_value(improved: int, regressed: int) -> float:
    disagreements = improved + regressed
    if disagreements == 0:
        return 1.0
    tail = sum(comb(disagreements, count) for count in range(min(improved, regressed) + 1)) / (2**disagreements)
    return min(1.0, 2.0 * tail)


def ratio(summary: dict[str, object], method: str) -> str:
    return str(summary[f"{method}_success"])


def write_csv(path: Path, stress_by_task: dict[str, dict[str, object]], stress_all: dict[str, object], nominal_all: dict[str, object]) -> None:
    fieldnames = [
        "protocol", "task", "episodes", "standard_success", "fusion_success", "delta_success_points",
        "standard_mean_target_distance_m", "fusion_mean_target_distance_m", "regrasp_attempts",
    ]
    rows: list[dict[str, object]] = []
    for task, summary in stress_by_task.items():
        rows.append({
            "protocol": "low_friction_multitask_40_paired",
            "task": task,
            "episodes": summary["episodes"],
            "standard_success": ratio(summary, "standard"),
            "fusion_success": ratio(summary, "fusion"),
            "delta_success_points": f"{float(summary['delta_success_points']):.1f}",
            "standard_mean_target_distance_m": f"{float(summary['standard_mean_target_distance']):.4f}",
            "fusion_mean_target_distance_m": f"{float(summary['fusion_mean_target_distance']):.4f}",
            "regrasp_attempts": summary["regrasp_attempts"],
        })
    for protocol, summary in (("low_friction_multitask_40_paired", stress_all), ("nominal_multitask_20_paired", nominal_all)):
        rows.append({
            "protocol": protocol,
            "task": "all_tasks",
            "episodes": summary["episodes"],
            "standard_success": ratio(summary, "standard"),
            "fusion_success": ratio(summary, "fusion"),
            "delta_success_points": f"{float(summary['delta_success_points']):.1f}",
            "standard_mean_target_distance_m": f"{float(summary['standard_mean_target_distance']):.4f}",
            "fusion_mean_target_distance_m": f"{float(summary['fusion_mean_target_distance']):.4f}",
            "regrasp_attempts": summary["regrasp_attempts"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    stress = payload["low_friction_40_paired"]
    nominal = payload["nominal_20_paired"]
    by_task = stress["by_task"]
    lines = [
        "# CLIP 语义 + 接触反馈融合执行器",
        "",
        "版本：`clip_semantic_contact_fusion_low_friction_multitask_v1`",
        "",
        "## 方法边界",
        "",
        "该方法复用冻结 `openai/clip-vit-base-patch32` 和原有 4 类意图 adapter；不新增训练参数，也不把 scripted waypoint 写成端到端 VLA。融合发生在 MuJoCo 执行层：语义模块选择对象/目标，低层以物体-TCP 相对距离确认运输阶段是否仍保持；若检测到掉落或未放置，最多从当前物理状态重规划一次更紧、更长的夹爪保持抓取。没有重置 simulator。",
        "融合配置从第一次抓取即采用更紧、更长的夹爪保持（`0.007 / 420`，标准执行器为 `0.015 / 260`），因此低摩擦收益不能单独归因于恢复分支；恢复分支的独立贡献尚未做消融。",
        "",
        "## 固定低摩擦压力协议",
        "",
        "- 40 条配对轨迹：4 个桌面搬运任务 x 10 seed（3100-3104 与 3200-3204）。",
        "- 所有对照复用相同初始状态、任务、`hard` 物体复杂度和冻结 CLIP adapter。",
        "- MuJoCo 参数：arm kp/force = 120/80，gripper kp/force = 600/90，sliding friction = 1.2。",
        "",
        "| 任务 | 标准语义执行器 | 接触融合 | 成功率变化 | 标准距离(m) | 融合距离(m) | 恢复次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        summary = by_task[task]
        lines.append(
            f"| {TASK_NAMES[task]} | {ratio(summary, 'standard')} | {ratio(summary, 'fusion')} | "
            f"{float(summary['delta_success_points']):+.1f} pp | {float(summary['standard_mean_target_distance']):.4f} | "
            f"{float(summary['fusion_mean_target_distance']):.4f} | {summary['regrasp_attempts']} |"
        )
    lines.extend([
        "",
        f"合计：标准 `{ratio(stress, 'standard')}`，融合 `{ratio(stress, 'fusion')}`，总体成功率变化 "
        f"`{float(stress['delta_success_points']):+.1f} pp`，全样本平均目标距离 "
        f"`{float(stress['standard_mean_target_distance']):.4f} m -> {float(stress['fusion_mean_target_distance']):.4f} m`。",
        "",
        f"配对结果：失败转成功 `{stress['improved']}`，成功转失败 `{stress['regressed']}`，共同成功 `{stress['tied_success']}`，共同失败 `{stress['tied_failure']}`；精确 McNemar 双侧 p=`{stress['exact_mcnemar_p_value']:.4f}`。因此该 40 条实验支持物理鲁棒性改善趋势，但样本量仍不足以宣称 p<0.05 的统计显著性。",
        f"融合专用的最终运输保持代理为 `{stress['fusion_sustained_transport_successes']}/{stress['episodes']}`：其中 3 条按原严格成功口径成功、但在融合器的最终运输保持检查中为 false，故它们不能作为“全程稳定持物”证据。",
        "",
        "## 常规域回归检查",
        "",
        "- 20 条配对轨迹：相同四任务 x seed 3300-3304。",
        "- MuJoCo 参数：arm kp/force = 150/100，gripper kp/force = 1200/200，sliding friction = 5.0。",
        f"- 标准与融合均为 `{ratio(nominal, 'standard')}` / `{ratio(nominal, 'fusion')}`；融合没有常规任务成功率退化。",
        "",
        "## 限制与下一步",
        "",
        "- 反馈信号是 MuJoCo 可得的物体-TCP 相对位姿代理，不是现实触觉传感器；真实 WidowX 上需要另做传感与安全验证。",
        "- 阈值、一次恢复上限与夹爪保持长度均为固定工程规则，尚未学习；不应写成 LoRA、RL 后训练或通用连续动作 VLA。",
        "- 低摩擦压力域只覆盖一种参数组合；需要扩大到扰动网格、更多物体形状和独立训练/测试划分。",
        "",
        "## 生成命令",
        "",
        "```powershell",
        f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_clip_semantic_contact_fusion_report.py"}"',
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stress_standard = read_rows("low_friction_multitask", "standard") + read_rows("low_friction_multitask", "standard", "_validation")
    stress_fusion = read_rows("low_friction_multitask", "contact_fusion") + read_rows("low_friction_multitask", "contact_fusion", "_validation")
    nominal_standard = read_rows("nominal_multitask", "standard")
    nominal_fusion = read_rows("nominal_multitask", "contact_fusion")
    stress_by_task = {
        task: task_summary(
            [row for row in stress_standard if row["task"] == task],
            [row for row in stress_fusion if row["task"] == task],
        )
        for task in TASKS
    }
    stress_all = task_summary(stress_standard, stress_fusion)
    stress_all["exact_mcnemar_p_value"] = exact_mcnemar_p_value(int(stress_all["improved"]), int(stress_all["regressed"]))
    nominal_all = task_summary(nominal_standard, nominal_fusion)
    payload = {
        "version": "clip_semantic_contact_fusion_low_friction_multitask_v1",
        "method_key": "clip_semantic_contact_fusion",
        "method_boundary": "Frozen CLIP intent adapter plus structured MuJoCo executor; not an end-to-end VLA or learned tactile policy.",
        "low_friction_40_paired": {"by_task": stress_by_task, **stress_all},
        "nominal_20_paired": nominal_all,
        "protocol": {
            "stress": {"seeds": "3100-3104,3200-3204", "complexity": "hard", "arm_kp_force": "120/80", "gripper_kp_force": "600/90", "friction": 1.2},
            "nominal": {"seeds": "3300-3304", "complexity": "hard", "arm_kp_force": "150/100", "gripper_kp_force": "1200/200", "friction": 5.0},
        },
        "source_files": {
            "stress_standard": "docs/low_friction_multitask_standard_*.csv and *_validation_*.csv",
            "stress_fusion": "docs/low_friction_multitask_contact_fusion_*.csv and *_validation_*.csv",
            "nominal": "docs/nominal_multitask_{standard,contact_fusion}_*.csv",
        },
    }
    write_csv(ROOT / "docs" / "clip_semantic_contact_fusion_report.csv", stress_by_task, stress_all, nominal_all)
    write_markdown(ROOT / "docs" / "clip_semantic_contact_fusion_report.md", payload)
    output_path = ROOT / "outputs" / "evaluations" / "clip_semantic_contact_fusion_v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"low_friction: standard={stress_all['standard_success']} fusion={stress_all['fusion_success']}")
    print(f"nominal: standard={nominal_all['standard_success']} fusion={nominal_all['fusion_success']}")
    print(f"report: {output_path}")


if __name__ == "__main__":
    main()
