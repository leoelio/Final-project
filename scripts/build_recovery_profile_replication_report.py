from __future__ import annotations

import json
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(rows: list[dict]) -> dict:
    standard = {(row["task"], row["seed"]): row for row in rows if row["profile"] == "standard"}
    deep = {(row["task"], row["seed"]): row for row in rows if row["profile"] == "deep_tight_slow"}
    if set(standard) != set(deep):
        raise ValueError("standard/deep paired rows differ")
    improved = sum(int(not standard[key]["success"] and deep[key]["success"]) for key in standard)
    regressed = sum(int(standard[key]["success"] and not deep[key]["success"]) for key in standard)
    discordant = improved + regressed
    lower = min(improved, regressed)
    p = 1.0 if not discordant else min(1.0, 2 * sum(comb(discordant, value) for value in range(lower + 1)) / (2**discordant))
    return {
        "episodes": len(standard),
        "standard_successes": sum(int(row["success"]) for row in standard.values()),
        "deep_successes": sum(int(row["success"]) for row in deep.values()),
        "improved": improved,
        "regressed": regressed,
        "discordant": discordant,
        "exact_two_sided_p": p,
    }


def main() -> None:
    v1 = load(ROOT / "outputs" / "evaluations" / "rgb_recovery_profile_heldout_v1.json")
    v2 = load(ROOT / "outputs" / "evaluations" / "rgb_recovery_profile_heldout_v2.json")
    preregistered = load(ROOT / "outputs" / "evaluations" / "rgb_recovery_profile_multidomain_preregistered_v1.json")
    budget = load(ROOT / "outputs" / "evaluations" / "recovery_budget_preregistered_v1.json")
    sweep = load(ROOT / "outputs" / "evaluations" / "recovery_profile_contact_sweep_v1.json")
    action_v1 = load(ROOT / "data" / "action_profile_bank" / "action_profile_bank_v1_summary.json")
    action_v2 = load(ROOT / "data" / "action_profile_bank" / "action_profile_bank_proprio_v2_summary.json")
    v1_result = paired(v1["rows"])
    v2_result = paired(v2["rows"])
    combined = paired(v1["rows"] + v2["rows"])
    preregistered_severe = preregistered["by_domain"]["severe_contact_shift"]
    preregistered_retry_failures = sum(
        row["profile"] == "standard" and row["recovery_reason"] == "source_not_visually_in_workspace"
        for row in preregistered["rows"]
    )
    action_counts = {name: 0 for name in ("both_failed", "both_success_prefer_standard", "standard_only_success", "deep_only_success")}
    for bank in (action_v1, action_v2):
        for split in ("train", "test"):
            for name, count in bank["split_outcome_counts"][split].items():
                action_counts[name] += count
    for domain in sweep["by_domain"].values():
        for name, count in domain["outcome_counts"].items():
            action_counts[name.replace("both_success", "both_success_prefer_standard")] += count

    lines = [
        "# RGB 恢复轨迹复制报告",
        "",
        "版本：`recovery_profile_replication_v2`",
        "",
        "## 协议",
        "",
        "对比两条固定的第二次恢复轨迹：`standard` 与 `deep_tight_slow`。首轮轨迹相同；只有当顶视 RGB 确认目标未完成且源物体可视觉重定位时才执行一次恢复。运行时不使用 MuJoCo 物体真值。",
        "",
        "环境初始化或首轮 RGB 无法定位源物体的 seed 会成对跳过，避免只给某个 profile 造成缺失。这一处理估计的是共享视觉前端可执行条件下的 profile 差异，跳过数量单独报告。",
        "",
        "## 两个独立闭环批次",
        "",
        "| 批次 | 有效配对 | 标准 | 深抓取 | 标准失败转深抓取成功 | 标准成功转深抓取失败 | 精确双侧 p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| seed 1300-1309 | {v1_result['episodes']} | {v1_result['standard_successes']}/{v1_result['episodes']} | {v1_result['deep_successes']}/{v1_result['episodes']} | {v1_result['improved']} | {v1_result['regressed']} | {v1_result['exact_two_sided_p']:.4f} |",
        f"| seed 2200-2214 | {v2_result['episodes']} | {v2_result['standard_successes']}/{v2_result['episodes']} | {v2_result['deep_successes']}/{v2_result['episodes']} | {v2_result['improved']} | {v2_result['regressed']} | {v2_result['exact_two_sided_p']:.4f} |",
        f"| 合并 | {combined['episodes']} | {combined['standard_successes']}/{combined['episodes']} | {combined['deep_successes']}/{combined['episodes']} | {combined['improved']} | {combined['regressed']} | {combined['exact_two_sided_p']:.4f} |",
        "",
        "合并后深抓取净增加 1 个成功，但只有 3 个不一致对，p=1.0。它是条件性改善趋势，而不是统计显著的全局替换依据；默认部署仍保持标准恢复。",
        "",
        "## 预注册跨域独立测试",
        "",
        f"预先固定 seed `2600-2623`、两项 RGB 可验证任务和三档接触域后，得到 {preregistered['episodes_per_profile']} 个有效配对（请求 144，对称跳过 {len(preregistered['skipped_resets'])} 个）。全域标准/深抓取同为 {preregistered['profile_successes']['standard']}/{preregistered['episodes_per_profile']}，改进 0、回退 0。主检验极端域为 {preregistered_severe['profile_successes']['standard']}/{preregistered_severe['episodes_per_profile']} 与 {preregistered_severe['profile_successes']['deep_tight_slow']}/{preregistered_severe['episodes_per_profile']}，p={preregistered_severe['paired']['exact_two_sided_p']:.4f}。",
        "",
        f"该批次只有 {preregistered['profile_retries']['standard']} 个 episode 实际进入第二次恢复，且两种轨迹都成功；另有 {preregistered_retry_failures} 个共享失败来自源物体离开可视觉重定位工作区，profile 根本未执行。故结果既没有支持深抓取，也不能把零差异误解为两条轨迹在所有失败状态下等价。",
        "",
        "预注册结论覆盖此前的“极端接触候选”表述：深抓取保留为离线反事实动作消融，不作为在线默认或条件化部署方案。",
        "",
        "## 恢复预算消融",
        "",
        f"在可分离最左方块的严重接触域中，另一个预注册配对集 seed `{budget['seed_range']}` 比较最多 1 次与最多 2 次标准 RGB 重试：均为 {budget['successes']['one_retry']}/{budget['episodes_per_budget']}，改进 {budget['paired']['improved']}、回退 {budget['paired']['regressed']}、p={budget['paired']['exact_two_sided_p']:.4f}。因此不增加默认重试预算。",
        "",
        "## 接触参数扫描",
        "",
        f"扫描 seed `{sweep['seed_range']}` 的三档接触条件，得到 {sweep['candidate_states']} 个 RGB 可恢复失败状态。极端域有 {sweep['by_domain']['severe_contact_shift']['candidate_states']} 个候选，标准/深抓取成功分别为 {sweep['by_domain']['severe_contact_shift']['standard_successes']}/{sweep['by_domain']['severe_contact_shift']['candidate_states']} 与 {sweep['by_domain']['severe_contact_shift']['deep_successes']}/{sweep['by_domain']['severe_contact_shift']['candidate_states']}；其中 `deep_only_success={sweep['by_domain']['severe_contact_shift']['outcome_counts']['deep_only_success']}`、`standard_only_success={sweep['by_domain']['severe_contact_shift']['outcome_counts']['standard_only_success']}`。轻度和低接触域很少产生可恢复失败，不能据此比较 profile。",
        "",
        "## 多候选反事实样本",
        "",
        f"两个动作银行和参数扫描合计包含 `deep_only_success={action_counts['deep_only_success']}`、`standard_only_success={action_counts['standard_only_success']}`、`both_success={action_counts['both_success_prefer_standard']}`、`both_failed={action_counts['both_failed']}`。这些是探索性数据，包含为选择器采集的训练段，不能替代上表中的独立闭环结果。",
        "",
        "## 成对视频",
        "",
        "- `videos/recovery_profile_v1/seed1308_standard_success.mp4` 与 `seed1308_deep_tight_slow_failure.mp4`：标准成功、深抓取回退。",
        "- `videos/recovery_profile_v2/seed2200_standard_failure.mp4` 与 `seed2200_deep_tight_slow_success.mp4`：标准失败、深抓取成功。",
        "",
        "## 当前决策",
        "",
        "保持最多一次标准 RGB 恢复作为默认方案；深抓取和第二次标准重试均仅保留为已否决的离线消融。下一阶段不再调固定 profile 或预算，而是针对严重接触域收集失败后状态与动作结果差异更丰富的数据，再重新评估是否需要本体感知选择器。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        'cd "C:\\Users\\Administrator\\Desktop\\final project-robot ic\\vla_robot_grasping"',
        '$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"',
        '& .\\.venv\\Scripts\\python.exe .\\scripts\\evaluate_rgb_recovery_profiles.py `',
        '  --model .\\outputs\\clip_semantic_waypoint\\clip_semantic_waypoint_core_v2_v1_20260721_110325.npz `',
        '  --seed 2600 --episodes 24 `',
        '  --domains mild_contact_shift,low_contact_shift,severe_contact_shift `',
        '  --tasks place_blue_cube_red_pad,move_leftmost_cube_to_bowl `',
        '  --output-json .\\outputs\\evaluations\\rgb_recovery_profile_multidomain_preregistered_v1.json `',
        '  --output-md .\\docs\\rgb_recovery_profile_multidomain_preregistered_v1.md',
        "```",
    ]
    output = ROOT / "docs" / "recovery_profile_replication_v1.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
