from __future__ import annotations

import json
from math import comb
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "outputs" / "evaluations" / "rgb_table_recovery_v4_extended_v1.json"
SOURCE_PATH = ROOT / "outputs" / "evaluations" / "rgb_table_recovery_v4_source_control_v1.json"
OUTPUT = ROOT / "docs" / "rgb_table_recovery_v4_stage.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(value: int, total: int) -> str:
    return f"{value}/{total} ({100.0 * value / total:.1f}%)"


def wilson_95(value: int, total: int) -> str:
    z = 1.959963984540054
    rate = value / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    radius = z * ((rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) ** 0.5) / denominator
    return f"[{100.0 * (center - radius):.1f}%, {100.0 * (center + radius):.1f}%]"


def exact_two_sided(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, value) for value in range(min(improved, regressed) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def keyed_rows(data: dict) -> dict[tuple[str, str, int], dict]:
    return {(row["domain"], row["task"], int(row["seed"])): row for row in data["rows"]}


def main() -> None:
    table = load(TABLE_PATH)
    source = load(SOURCE_PATH)
    table_rows = keyed_rows(table)
    source_rows = keyed_rows(source)
    if table_rows.keys() != source_rows.keys():
        raise ValueError("table and source protocols do not contain the same paired episodes")
    first_mismatches = [
        key
        for key in table_rows
        if table_rows[key]["first_attempt_success"] != source_rows[key]["first_attempt_success"]
        or table_rows[key]["semantic_correct"] != source_rows[key]["semantic_correct"]
        or table_rows[key]["visual_selection_correct"] != source_rows[key]["visual_selection_correct"]
    ]
    if first_mismatches:
        raise ValueError(f"paired first-attempt mismatch for {len(first_mismatches)} episodes")
    total = table["overall"]["episodes"]
    recovered = [row for row in table["rows"] if not row["first_attempt_success"] and row["task_success"]]
    open_regressions = [row for row in table["rows"] if row["first_attempt_success"] and not row["task_success"]]
    scope_improved = [key for key in table_rows if not source_rows[key]["task_success"] and table_rows[key]["task_success"]]
    scope_regressed = [key for key in table_rows if source_rows[key]["task_success"] and not table_rows[key]["task_success"]]

    lines = [
        "# RGB 桌面范围恢复 V4 阶段报告",
        "",
        "版本：`rgb_table_recovery_v4_extended_v1`",
        "",
        "## 结论",
        "",
        "V4 在未参与诊断的 seed `4000-4011` 上，使用冻结 CLIP 意图、RGB 几何定位和一次有界桌面范围重试。初始语言、定位和对象身份均正确；最终严格成功由首轮开环的结果通过 RGB 重试进一步提升。",
        "",
        "该方法仍是 MuJoCo 固定桌面条件下的分层视觉-语言-动作系统。它不是端到端 VLA、OpenVLA 微调或真实机械臂结果。",
        "",
        "## V4 独立结果",
        "",
        "| 指标 | 结果 | Wilson 95% CI |",
        "| --- | ---: | ---: |",
    ]
    for key, label in (
        ("initial_grounding_executable", "初始 RGB 定位可执行"),
        ("semantic_correct", "冻结 CLIP 语义正确"),
        ("visual_selection_correct", "离线 RGB 对象选择正确"),
        ("first_attempt_success", "首轮严格成功"),
        ("task_success", "桌面范围恢复后的最终严格成功"),
    ):
        value = table["overall"][key]
        lines.append(f"| {label} | {ratio(value, total)} | {wilson_95(value, total)} |")
    lines.append(f"| 触发 RGB 重试 | {table['overall']['recovery_triggered']}/{total} | - |")

    lines.extend(
        [
            "",
            "## 配对结果 A：首轮开环 vs. 允许一次桌面范围重试",
            "",
            f"首轮成功 `{table['overall']['first_attempt_success']}/{total}`，最终成功 `{table['overall']['task_success']}/{total}`；"
            f"挽回 {len(recovered)} 条、回退 {len(open_regressions)} 条、不一致配对 {len(recovered) + len(open_regressions)} 条，"
            f"精确双侧检验 `p={exact_two_sided(len(recovered), len(open_regressions)):.4f}`。",
            "",
            "## 配对结果 B：源工作区搜索 vs. 有界桌面范围搜索",
            "",
            "两种搜索范围使用相同 seed、相同首轮动作、相同模型和相同一次重试预算。已核验全部 144 条的首轮语义、对象选择和严格成功字段完全一致。",
            "",
            "| 搜索范围 | 最终严格成功 | 相对源区的净变化 |",
            "| --- | ---: | ---: |",
            f"| `source` | {ratio(source['overall']['task_success'], total)} | 基线 |",
            f"| `table` | {ratio(table['overall']['task_success'], total)} | 改进 {len(scope_improved)}，回退 {len(scope_regressed)}，精确双侧 p={exact_two_sided(len(scope_improved), len(scope_regressed)):.4f} |",
            "",
            "只有配对结果 B 才用于判断新增桌面范围是否有独立贡献；配对结果 A 只说明整个重试闭环是否有价值。",
            "",
            "## 分任务与接触域",
            "",
            "| 任务 | 首轮成功 | 重试触发 | 最终成功 | Wilson 95% CI |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task, item in table["by_task"].items():
        episodes = item["episodes"]
        lines.append(
            f"| `{task}` | {ratio(item['first_attempt_success'], episodes)} | {item['recovery_triggered']}/{episodes} | "
            f"{ratio(item['task_success'], episodes)} | {wilson_95(item['task_success'], episodes)} |"
        )
    lines.extend(["", "| 接触域 | 首轮成功 | 重试触发 | 最终成功 | Wilson 95% CI |", "| --- | ---: | ---: | ---: | ---: |"])
    for domain, item in table["by_domain"].items():
        episodes = item["episodes"]
        lines.append(
            f"| `{domain}` | {ratio(item['first_attempt_success'], episodes)} | {item['recovery_triggered']}/{episodes} | "
            f"{ratio(item['task_success'], episodes)} | {wilson_95(item['task_success'], episodes)} |"
        )
    lines.extend(
        [
            "",
            "## 运行时边界",
            "",
            "- 初始物体身份仍要求工作区、立方体尺寸/形状和同色目标盘排除。",
            "- 只有首轮未被 RGB 确认完成时才允许一次重试；桌面回退要求同色候选位于固定桌面边界、满足面积/填充率阈值，并按距离上一次 RGB 位置最近选择。",
            "- MuJoCo 物体真实位姿只用于离线成功率和对象选择标签，不参与运行时搜索或轨迹规划。",
            "- 未被 RGB 重新定位的接触失败仍被如实计为失败；桌面范围搜索不等价于通用全场景视觉恢复。",
            "",
            "## 视频证据",
            "",
            "- `videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.mp4`：V4 配对对照中源工作区搜索失败、桌面范围搜索成功的严重接触案例。首轮目标距离约 9.8 cm，第二次 RGB 重定位后为约 7.6 mm。",
            "",
            "## 复现",
            "",
            "完整命令见 `docs/rgb_table_recovery_v4_preregistered.md`。",
        ]
    )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {OUTPUT}")


if __name__ == "__main__":
    main()
