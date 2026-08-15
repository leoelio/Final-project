from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def counts(rows: list[dict]) -> dict:
    return {
        "episodes": len(rows),
        "initial_grounding": sum(row["initial_grounding_executable"] for row in rows),
        "visual_selection": sum(row["visual_selection_correct"] for row in rows),
        "first_success": sum(row["first_attempt_success"] for row in rows),
        "retry": sum(row["recovery_triggered"] for row in rows),
        "final_success": sum(row["task_success"] for row in rows),
    }


def ratio(item: dict, field: str) -> str:
    return f"{item[field]}/{item['episodes']}"


def main() -> None:
    source = ROOT / "outputs" / "evaluations" / "rgb_frontend_end_to_end_preregistered_v1.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data["rows"]
    overall = counts(rows)
    by_domain = {name: counts([row for row in rows if row["domain"] == name]) for name in data["domains"]}
    by_task = {name: counts([row for row in rows if row["task"] == name]) for name in data["tasks"]}
    recovered_successes = sum(row["recovery_triggered"] and row["task_success"] for row in rows)
    recovery_failures = sum(row["recovery_triggered"] and not row["task_success"] for row in rows)
    no_retry_failures = sum(not row["recovery_triggered"] and not row["task_success"] for row in rows)
    selection_errors = [row for row in rows if row["initial_grounding_executable"] and not row["visual_selection_correct"]]
    failures = [row for row in rows if not row["task_success"]]
    failure_reasons = Counter(str(row.get("recovery_reason", row.get("error", "unknown"))) for row in failures)

    lines = [
        "# 修订 RGB 前端端到端阶段报告",
        "",
        "版本：`rgb_frontend_end_to_end_preregistered_v1`",
        "",
        "## 固定协议",
        "",
        "使用 seed `3000-3023`，两项任务、三档接触域，共 144 个不重叠 episode。所有源定位异常均计为系统不可执行，不作跳过。运行时为冻结 CLIP 意图、修订 RGB 初始定位、结构化标准抓取/放置和最多一次 RGB 触发的标准重试。MuJoCo 真值只用于离线评分。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 初始 RGB 定位可执行 | {ratio(overall, 'initial_grounding')} |",
        f"| 视觉对象选择正确 | {ratio(overall, 'visual_selection')} |",
        f"| 严格首轮成功 | {ratio(overall, 'first_success')} |",
        f"| 触发一次 RGB 重试 | {ratio(overall, 'retry')} |",
        f"| 最终严格任务成功 | {ratio(overall, 'final_success')} |",
        "",
        f"22 个 episode 触发重试，其中 {recovered_successes} 个最终成功、{recovery_failures} 个仍失败；首轮不成功但未触发重试而最终失败的 episode 有 {no_retry_failures} 个。重试相对严格首轮净增加 {overall['final_success'] - overall['first_success']} 个成功，但本阶段没有与另一条恢复轨迹作配对比较。",
        "",
        "## 分域与分任务",
        "",
        "| 接触域 | 初始定位 | 首轮成功 | 触发重试 | 最终成功 |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| `{name}` | {ratio(item, 'initial_grounding')} | {ratio(item, 'first_success')} | {ratio(item, 'retry')} | {ratio(item, 'final_success')} |"
            for name, item in by_domain.items()
        ],
        "",
        "| 任务 | 初始定位 | 视觉选择 | 最终成功 |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| `{name}` | {ratio(item, 'initial_grounding')} | {ratio(item, 'visual_selection')} | {ratio(item, 'final_success')} |"
            for name, item in by_task.items()
        ],
        "",
        f"视觉选择错误为 {len(selection_errors)} 个，均来自空间任务 seed `{', '.join(str(row['seed']) for row in selection_errors)}` 在不同接触域的重复场景。最终失败的终止原因计数为 `{dict(failure_reasons)}`。",
        "",
        "## 可写结论",
        "",
        "工作区裁剪修订消除了本测试中的初始定位不可执行：144/144。蓝方块到红盘在三档接触域的组合下为 71/72；空间最左方块到碗为 60/72，且包含全部视觉选择错误。因而下一阶段应优先加强空间关系对象选择和失败后物体可重定位，而不是继续调固定深抓取 profile。",
        "",
        "不能写为“端到端 VLA 已解决接触控制”或“修订相对旧系统提升了端到端成功率”：这里没有在同一新 seed 上执行旧前端的配对对照。定位层的 before/after 证据仅限 `rgb_frontend_stage_v1.md`。",
        "",
        "## 关联证据",
        "",
        "- 初始定位审计：`docs/rgb_frontend_stage_v1.md`。",
        "- 预注册协议：`docs/rgb_frontend_end_to_end_preregistered_v1.md`。",
        "- viewer 回归视频：`videos/rgb_frontend_v1/seed2809_workspaceclip_success.mp4`。",
        "- 恢复 profile 阴性结果：`docs/recovery_profile_replication_v1.md`。",
    ]
    output = ROOT / "docs" / "rgb_frontend_end_to_end_stage_v1.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_path: {output}")


if __name__ == "__main__":
    main()
