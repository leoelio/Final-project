from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


FIELDNAMES = [
    "版本",
    "方法",
    "阶段",
    "视频类型",
    "任务",
    "seed",
    "结果",
    "失败模式",
    "证据用途",
    "目标距离",
    "末端到物体距离",
    "物体高度",
    "抓取标志",
    "视频文件",
    "元数据文件",
    "论文可写",
    "论文红线",
]


MODE_NOTES = {
    "成功样例": "可写作固定任务、固定 seed 或训练分布内的一次可视化成功证据，但仍需以批量成功率为准。",
    "数据回放/可复现": "可写作示范轨迹可被重新加载和回放，用于证明数据采集链路可复现；不能作为策略学习成功率。",
    "未形成有效抓取/未抬升": "可写作闭环接触阶段失败：策略没有稳定完成靠近、夹紧和抬升，说明离线动作误差低不等于闭环操作成功。",
    "目标选择/放置偏差": "可写作策略产生了明显目标偏差，物体没有被放到目标区域，适合支撑放置精度和任务条件化不足的分析。",
    "语言/空间泛化失败": "可写作普通 state/action 或轻量代理模型缺少可靠语言与空间泛化能力，不能把一次语言失败归因于完整 VLA 失效。",
    "扰动域接触鲁棒性不足": "可写作 MuJoCo 扰动域下的接触鲁棒性下降，不能写成 Isaac 或真实机器人验证。",
    "闭环漂移/阶段切换失败": "可写作策略在长时序闭环中误差累积，阶段切换或动作块重规划没有稳定保持任务进度。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese failure-mode taxonomy from fixed rollout video evidence.")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.csv")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.md")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_success(row: dict[str, str]) -> bool:
    return row.get("结果", "").strip().lower() == "success=true"


def classify(row: dict[str, str]) -> str:
    version = row.get("版本", "")
    result = row.get("结果", "")
    evidence = row.get("证据用途", "")
    if version.startswith("replay_") or result.startswith("replay") or "可复现" in evidence:
        return "数据回放/可复现"
    if is_success(row):
        return "成功样例"

    stage = row.get("阶段", "")
    video_type = row.get("视频类型", "")
    task = row.get("任务", "")
    complexity = row.get("复杂度", "")
    if version.startswith("domain_randomization_"):
        return "扰动域接触鲁棒性不足"
    if video_type == "语言/空间泛化片段" or complexity == "language" or "language" in task or "language" in stage:
        return "语言/空间泛化失败"

    target_distance = as_float(row.get("目标距离", ""))
    eef_distance = as_float(row.get("末端到物体距离", ""))
    object_height = as_float(row.get("物体高度", ""))
    grasped = row.get("抓取标志", "").strip().lower() == "true"

    if target_distance is not None and target_distance > 0.18 and eef_distance is not None and eef_distance < 0.24:
        return "目标选择/放置偏差"
    if not grasped and object_height is not None and object_height <= 0.035:
        return "未形成有效抓取/未抬升"
    if eef_distance is not None and eef_distance > 0.30:
        return "闭环漂移/阶段切换失败"
    return "目标选择/放置偏差"


def build_rows(video_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in video_rows:
        mode = classify(row)
        red_line = row.get("论文红线", "")
        if mode == "扰动域接触鲁棒性不足" and "不能写成真实机器人验证" not in red_line:
            red_line = f"{red_line}；不能写成真实机器人验证"

        rows.append(
            {
                "版本": row.get("版本", ""),
                "方法": row.get("方法", ""),
                "阶段": row.get("阶段", ""),
                "视频类型": row.get("视频类型", ""),
                "任务": row.get("任务", ""),
                "seed": row.get("seed", ""),
                "结果": row.get("结果", ""),
                "失败模式": mode,
                "证据用途": row.get("证据用途", ""),
                "目标距离": row.get("目标距离", ""),
                "末端到物体距离": row.get("末端到物体距离", ""),
                "物体高度": row.get("物体高度", ""),
                "抓取标志": row.get("抓取标志", ""),
                "视频文件": row.get("视频文件", ""),
                "元数据文件": row.get("元数据文件", ""),
                "论文可写": MODE_NOTES[mode],
                "论文红线": red_line,
            }
        )
    return rows


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode_counts = Counter(row["失败模式"] for row in rows)
    stage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        stage_counts[row["阶段"]][row["失败模式"]] += 1

    trajectory_rows = [
        row
        for row in rows
        if row["版本"]
        in {
            "act_lite_chunk_bc_v1",
            "trajectory_conditioned_chunk_bc_v2",
            "trajectory_knn_chunk_bc_v1",
            "torch_act_state_chunk_v1",
            "torch_act_state_chunk_cuda_v1",
            "phase_conditioned_torch_act_v1",
            "torch_act_cvae_state_chunk_v1",
            "visual_feature_act_lite_v1",
            "visual_act_cnn_cvae_v1",
        }
    ]

    lines = [
        "# 失败模式分类记录",
        "",
        "版本：`failure_mode_taxonomy_v1`",
        "",
        "用途：把固定 rollout 视频索引中的成功/失败样例整理为论文可解释的失败模式。该表服务于 trajectory-conditioned BC / ACT、普通 BC、轻量 action-head、语言泛化和 MuJoCo domain randomization 代理评测的结果解释。",
        "",
        "数据来源：`docs/video_evidence_index.csv`；可视化入口：`docs/video_evidence_gallery.html`。",
        "",
        "重要边界：本表只解释 MuJoCo 仿真视频证据；domain randomization 是 MuJoCo 代理评测，不能写成真实机器人验证，也不能写成 Isaac 高保真验证。",
        "",
        "## 1. 失败模式总览",
        "",
        md_row(["失败模式", "样例数", "论文写法"]),
        md_row(["---", "---:", "---"]),
    ]
    for mode, count in mode_counts.most_common():
        lines.append(md_row([mode, str(count), MODE_NOTES[mode]]))

    lines.extend(
        [
            "",
            "## 2. trajectory-conditioned BC / ACT 重点记录",
            "",
            "这一组方法的核心观察是：动作块和 Transformer/CVAE 结构能让动作更平滑，但在当前小规模示范数据和 state/轻量视觉代理条件下，闭环接触、夹紧、抬升仍不稳定。`trajectory_knn_chunk_bc_v1` 在训练范围成功，说明轨迹记忆能复现局部示范；但它在留出范围失败，不能写成策略泛化。",
            "",
            md_row(["版本", "阶段", "结果", "失败模式", "目标距离", "物体高度", "论文红线"]),
            md_row(["---", "---", "---", "---", "---:", "---:", "---"]),
        ]
    )
    for row in trajectory_rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["阶段"],
                    row["结果"],
                    row["失败模式"],
                    row["目标距离"],
                    row["物体高度"],
                    row["论文红线"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 3. 阶段分布",
            "",
            md_row(["阶段", "失败模式", "样例数"]),
            md_row(["---", "---", "---:"]),
        ]
    )
    for stage in sorted(stage_counts):
        for mode, count in stage_counts[stage].most_common():
            lines.append(md_row([stage, mode, str(count)]))

    lines.extend(
        [
            "",
            "## 4. 明细表",
            "",
            md_row(["版本", "视频类型", "结果", "失败模式", "证据用途", "视频文件", "论文红线"]),
            md_row(["---", "---", "---", "---", "---", "---", "---"]),
        ]
    )
    for row in rows:
        lines.append(
            md_row(
                [
                    f"`{row['版本']}`",
                    row["视频类型"],
                    row["结果"],
                    row["失败模式"],
                    row["证据用途"],
                    f"`{row['视频文件']}`",
                    row["论文红线"],
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 5. 论文写作红线",
            "",
            "- 成功样例只能作为视频证据，量化结论必须引用批量评测表。",
            "- `ACT-lite`、`State ACT`、`ACT-CVAE-lite` 和 `Visual ACT-CNN-CVAE-lite` 都是本地轻量 baseline，不能写成官方完整 ACT 复现。",
            "- `Frozen CLIP Action Head-lite` 只能写成通用 VLM 表征代理，不能写成 OpenVLA/RT-2 后训练。",
            "- MuJoCo domain randomization 代理评测不能写成真实机器人验证，也不能写成 Isaac 高保真验证。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows(read_csv(args.video_evidence))
    write_csv(args.output_csv, rows)
    write_md(args.output_md, rows)
    print(f"rows: {len(rows)}", flush=True)
    print(f"output_csv: {args.output_csv}", flush=True)
    print(f"output_md: {args.output_md}", flush=True)


if __name__ == "__main__":
    main()
