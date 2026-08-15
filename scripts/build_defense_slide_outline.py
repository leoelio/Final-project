from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SLIDES = [
    {
        "id": "01",
        "title": "研究问题与总体路线",
        "message": "在有限算力、有限示范数据和有限真实机械臂时间条件下，先用 MuJoCo 建立可复现实验闭环，再比较普通 BC、ACT/Diffusion、action-head、VLM/PEFT proxy。",
        "versions": ["expert_scripted_v1", "structured_waypoint_policy_v1"],
        "figure": "",
        "video": "outputs/presentation_clips/00_defense_video_reel.mp4",
        "script": "开场先说明本阶段不是从零训练大 VLA，而是建立轻量后训练和对照评测链路。播放总览 reel 的前 10 秒即可，让评委先看到任务和仿真环境。",
    },
    {
        "id": "02",
        "title": "任务环境与数据链路",
        "message": "WidowX 桌面任务、示范采集、轨迹回放和固定视频导出已经可复现。",
        "versions": ["expert_scripted_v1", "replay_demo_v1", "structured_waypoint_policy_v1"],
        "figure": "",
        "video": "outputs/presentation_clips/01_task_data_oracle.mp4",
        "script": "说明主任务是把蓝色方块放到蓝色盘子，环境包含干扰物体。强调 replay 不是策略，而是数据可复现性证明。",
    },
    {
        "id": "03",
        "title": "评测协议与量化指标",
        "message": "所有方法统一记录版本名、artifact、train-range、held-out、language、参数量、训练时间、显存和固定 rollout 视频；同时用 strict_grasp_success_audit_v1 区分目标距离达标和真正抓取/抬升。",
        "versions": ["linear_bc_v1", "knn_bc_v1", "mlp_bc_v1"],
        "figure": "outputs/figures/main_task_success.svg",
        "video": "",
        "script": "先讲评测协议，再解释为什么只看离线 MSE 不够，必须看 MuJoCo 闭环成功率和视频证据。这里主动补充严格抓取口径：当前原始放置成功有 {STRICT_LOOSE}，但 strict grasp 成功是 {STRICT_SUCCESS}，所以不能写成稳定抓取成功。",
    },
    {
        "id": "04",
        "title": "普通 BC baseline",
        "message": "Linear/MLP 单步回归闭环不稳定；kNN 在训练范围成功但 held-out 下降，说明轨迹记忆不等于泛化。",
        "versions": ["linear_bc_v1", "knn_bc_v1", "mlp_bc_v1"],
        "figure": "outputs/figures/main_task_success.svg",
        "video": "outputs/presentation_clips/02_basic_bc_baselines.mp4",
        "script": "播放普通 BC 短片，指出 linear/MLP 的失败和 kNN 的训练范围记忆现象。这里是后面 ACT/VLA proxy 的基础对照。",
    },
    {
        "id": "05",
        "title": "Trajectory / ACT / Diffusion baseline",
        "message": "动作块、历史窗口、Transformer、CVAE 和 diffusion 更接近序列策略，但 state-only 小数据版本仍不能稳定完成接触、抬升和放置。",
        "versions": [
            "trajectory_conditioned_chunk_bc_v2",
            "trajectory_knn_chunk_bc_v1",
            "torch_act_state_chunk_cuda_v1",
            "phase_conditioned_torch_act_v1",
            "torch_act_cvae_state_chunk_v1",
            "torch_diffusion_policy_state_chunk_v1",
            "visual_feature_act_lite_v1",
        ],
        "figure": "outputs/figures/resource_vs_success.svg",
        "video": "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
        "script": "强调这些是 ACT/Diffusion 的轻量或 state-only baseline，不是完整视觉 ACT 或完整视觉 Diffusion Policy。",
    },
    {
        "id": "06",
        "title": "Action-head 与 VLM/PEFT proxy",
        "message": "object/vision/language action-head、reward-weighted BC、phase-conditioned、Adapter/LoRA-style 和 CLIP proxy 已建立，但闭环成功率仍不足。",
        "versions": [
            "object_language_action_head_lite_v1",
            "phase_conditioned_action_head_lite_v1",
            "reward_weighted_action_head_lite_v1",
            "adapter_action_head_lite_v1",
            "lora_action_head_lite_v1",
            "clip_action_head_lite_v1",
            "vision_language_action_head_lite_v1",
            "multi_task_object_action_head_lite_v1",
        ],
        "figure": "outputs/figures/resource_vs_success.svg",
        "video": "outputs/presentation_clips/04_action_head_peft_proxy.mp4",
        "script": "这里重点讲资源和结构，而不是夸大效果。Adapter/LoRA-style 参数量很小，但目前没有带来稳定闭环成功。",
    },
    {
        "id": "07",
        "title": "语言/空间泛化测试",
        "message": "leftmost-to-bowl 任务中，规则和结构化策略可完成，当前 learned/action-head proxy 仍为 0/5。",
        "versions": [
            "expert_scripted_v1",
            "structured_waypoint_policy_v1",
            "object_language_action_head_lite_v1",
            "clip_action_head_lite_v1",
            "multi_task_object_action_head_lite_v1",
        ],
        "figure": "outputs/figures/language_success.svg",
        "video": "outputs/presentation_clips/05_language_generalization.mp4",
        "script": "这一页回答 VLA 是否更会理解语言。当前结论是否定的：评测链路建立了，但 proxy 还没有学出空间语言泛化。",
    },
    {
        "id": "08",
        "title": "MuJoCo Domain Randomization 代理评测",
        "message": "低摩擦、弱夹爪和执行器扰动下，结构化策略仍稳定，trajectory-kNN 出现鲁棒性下降，Visual ACT-CNN-CVAE-lite 仍失败。",
        "versions": ["structured_waypoint_policy_v1", "trajectory_knn_chunk_bc_v1", "visual_act_cnn_cvae_v1"],
        "figure": "",
        "video": "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "script": "这一页只能写成 MuJoCo 层面的动力学扰动代理评测。说明 Isaac 和真实 WidowX 的交接门禁已经建立，但本机尚未完成 Isaac/Isaac Sim 高保真 domain randomization，也没有真实 WidowX 迁移结果。",
    },
    {
        "id": "09",
        "title": "数据效率对比",
        "message": "kNN/trajectory-kNN 随数据增加在训练范围改善，但 held-out 仍弱；object-language action head 在当前预算下没有稳定成功。",
        "versions": ["knn_bc_v1", "trajectory_knn_chunk_bc_v1", "object_language_action_head_lite_v1"],
        "figure": "outputs/figures/data_efficiency.svg",
        "video": "",
        "script": "解释 10、25、50、92 条示范预算。重点是不要把训练范围记忆写成省数据泛化。",
    },
    {
        "id": "10",
        "title": "算力与参数效率",
        "message": "Adapter/LoRA-style proxy 的可训练参数约 2.1k，显著低于普通 action-head；Torch ACT/Diffusion CUDA 路线可记录训练时间和显存。",
        "versions": [
            "adapter_action_head_lite_v1",
            "lora_action_head_lite_v1",
            "object_language_action_head_lite_v1",
            "torch_act_state_chunk_cuda_v1",
            "torch_diffusion_policy_state_chunk_v1",
            "clip_action_head_lite_v1",
        ],
        "figure": "outputs/figures/resource_vs_success.svg",
        "video": "",
        "script": "这一页回答是否省算力：参数确实少，但成功率没有同步提升。因此表述应是资源效率链路已建立，而不是方法已胜出。",
    },
    {
        "id": "11",
        "title": "视频证据与演示入口",
        "message": "每个方法都有固定 mp4/json 元数据，另有全量宫格、语言宫格、domain randomization 阶段短片、阶段总览 reel、OpenVLA 数据桥接、外部验证 handoff 和 readiness audit 入口，便于论文和答辩展示。",
        "versions": [],
        "figure": "",
        "video": "outputs/presentation_clips/00_defense_video_reel.mp4",
        "links": [
            {
                "title": "OpenVLA bridge 样本浏览页",
                "path": "docs/openvla_bridge_gallery.html",
                "note": "openvla_dataset_bridge_v1：72 条 image + instruction + state + action 样本。",
            },
            {
                "title": "OpenVLA 数据桥接报告",
                "path": "docs/openvla_dataset_bridge_report.md",
                "note": "明确这是后续真实 VLA 后训练的数据准备，不参与当前策略成功率比较。",
            },
            {
                "title": "OpenVLA 本地可行性检查",
                "path": "docs/openvla_feasibility_report.md",
                "note": "openvla_feasibility_check_v1：记录本机不适合直接训练真实 OpenVLA/机器人 VLA LoRA。",
            },
            {
                "title": "Robot VLA action-head 交接门禁",
                "path": "docs/robot_vla_action_head_handoff.md",
                "note": "robot_vla_action_head_handoff_v1：定义真实 robot VLA action-head 远端运行的输入、输出和入包门禁。",
            },
            {
                "title": "Robot VLA 远端运行包",
                "path": "docs/robot_vla_remote_run_pack.md",
                "note": "robot_vla_remote_run_pack_v1：打包 bridge 数据、远端命令模板和结果回填 schema。",
            },
            {
                "title": "Robot VLA 远端结果回填门禁",
                "path": "docs/robot_vla_remote_result_intake.md",
                "note": "robot_vla_remote_result_intake_v1：检查远端模型、feature cache、评测 JSON、视频和报告能否进入正式方法包。",
            },
            {
                "title": "Isaac domain randomization 交接门禁",
                "path": "docs/isaac_domain_randomization_handoff.md",
                "note": "isaac_domain_randomization_handoff_v1：固定 Isaac 复现实验字段和回填文件；不是 Isaac 结果。",
            },
            {
                "title": "真实 WidowX 验证交接门禁",
                "path": "docs/real_widowx_validation_handoff.md",
                "note": "real_widowx_validation_handoff_v1：固定真实机械臂安全门禁和 50 条 trial 模板；不是真实 trial 结果。",
            },
            {
                "title": "严格抓取成功口径审计",
                "path": "docs/strict_grasp_success_audit.md",
                "note": "strict_grasp_success_audit_v1：区分原始放置 success、grasp_success 和 object_z；当前严格抓取成功为 {STRICT_SUCCESS}。",
            },
            {
                "title": "外部依赖阶段 readiness audit",
                "path": "docs/external_dependency_readiness_audit.md",
                "note": "external_dependency_readiness_audit_v1：统一说明真实 robot VLA、Isaac 和真实 WidowX planned 版本的阻塞条件、回填工件和论文边界；不是策略成功率结果。",
            },
        ],
        "script": "现场可播放 00_defense_video_reel.mp4；如评委追问某类方法，切到 01-06 阶段短片或 outputs/videos 下的单方法视频；如追问 success=True 但是否真的抓起来，打开 strict grasp audit；如追问真实 VLA、Isaac 或真实机械臂，打开 bridge、remote pack、external readiness audit、Isaac handoff 和真实 WidowX handoff，并说明当前只是数据桥接、运行/回填门禁、readiness 审计和 trial 模板。",
    },
    {
        "id": "12",
        "title": "阶段性结论",
        "message": "当前结果支持继续做机器人预训练 VLA 表征 + action head / Adapter / LoRA，而不是把本地 CLIP 或 RGB proxy 写成真实 VLA。",
        "versions": ["structured_waypoint_policy_v1", "clip_action_head_lite_v1", "adapter_action_head_lite_v1", "lora_action_head_lite_v1"],
        "figure": "outputs/figures/main_task_success.svg",
        "video": "",
        "script": "明确贡献：环境、数据、对照、评测、视频证据链已经建立；当前 proxy 效果不足反而给下一阶段 VLA 后训练提供基线。",
    },
    {
        "id": "13",
        "title": "表述边界与后续工作",
        "message": "不能宣称已完成真实 OpenVLA/RT-2/机器人 VLA 后训练、完整视觉 ACT、完整视觉 Diffusion Policy、Isaac 或真实机械臂验证；当前只完成 bridge、可行性审计、Robot VLA 远端运行/回填门禁、外部依赖 readiness audit、Isaac handoff 和真实 WidowX trial 模板。",
        "versions": ["torch_act_cvae_state_chunk_v1", "torch_diffusion_policy_state_chunk_v1", "clip_action_head_lite_v1"],
        "figure": "",
        "video": "",
        "script": "最后主动说明边界，给出下一步：基于 docs/robot_vla_remote_run_pack.md 在远端接入机器人预训练 VLA 表征，并用 docs/external_dependency_readiness_audit.md 检查 planned 版本是否具备回填条件，再按 docs/isaac_domain_randomization_handoff.md 和 docs/real_widowx_validation_handoff.md 推进 Isaac 与真实 WidowX 小规模验证。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a slide-by-slide Chinese defense outline from current experiment artifacts.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resources", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--strict-grasp-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "defense_slide_outline.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def by_version(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["version"]: row for row in rows if row.get("version")}


def language_alias(version: str) -> str:
    if version == "expert_scripted_v1":
        return "expert_scripted_language_v1"
    return version


def value(row: dict | None, key: str, default: str = "未记录") -> str:
    if not row:
        return default
    item = row.get(key, "")
    return item if item not in ("", None) else default


def number(value_text: str) -> str:
    if value_text in ("", "未记录", None):
        return "未记录"
    try:
        return f"{int(float(value_text)):,}"
    except ValueError:
        return str(value_text)


def md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(md_escape(item) for item in values) + " |"


def strict_counts(args: argparse.Namespace) -> tuple[str, str]:
    summary = read_json(args.strict_grasp_json).get("summary", {})
    episodes = summary.get("episodes", "?")
    loose = f"{summary.get('loose_successes', '?')}/{episodes}"
    strict = f"{summary.get('strict_grasp_successes', '?')}/{episodes}"
    return loose, strict


def method_table(slide: dict, methods: dict[str, dict], summary: dict[str, dict], language: dict[str, dict], resources: dict[str, dict]) -> list[str]:
    if not slide["versions"]:
        return []
    lines = [
        md_row(["版本", "方法", "train", "held-out", "language", "参数"]),
        md_row(["---", "---", "---:", "---:", "---:", "---:"]),
    ]
    for version in slide["versions"]:
        method = methods.get(version)
        if not method:
            continue
        main = summary.get(version, method)
        lang = language.get(language_alias(version), {})
        res = resources.get(version, {})
        lines.append(
            md_row(
                [
                    f"`{version}`",
                    method["method"],
                    value(main, "train_range_success", method.get("train_range_success", "")),
                    value(main, "heldout_success", method.get("heldout_success", "")),
                    value(lang, "success", "未评测"),
                    number(value(res, "trainable_params", "0")),
                ]
            )
        )
    return lines


def ensure_references_exist(slides: list[dict]) -> None:
    missing = []
    for slide in slides:
        for key in ("figure", "video"):
            path = slide.get(key)
            if path and not (ROOT / path).exists():
                missing.append(path)
        for link in slide.get("links", []):
            path = link.get("path")
            if path and not (ROOT / path).exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError("\n".join(missing))


def write_doc(args: argparse.Namespace) -> None:
    versions = read_json(args.versions)
    methods = {method["version"]: method for method in versions["methods"]}
    summary = by_version(read_csv(args.summary))
    language = by_version(read_csv(args.language_summary))
    resources = by_version(read_csv(args.resources))
    loose, strict = strict_counts(args)
    ensure_references_exist(SLIDES)

    lines = [
        "# 答辩幻灯片大纲",
        "",
        "版本：`defense_slide_outline_v1`",
        "",
        "用途：把当前实验结果整理为 PPT 页级讲解脚本。每一页都明确核心结论、引用版本、量化结果、图表或视频 cue，便于后续制作毕业答辩幻灯片。",
        "",
        "建议总时长：8-12 分钟。若时间紧，播放 `outputs/presentation_clips/00_defense_video_reel.mp4` 的前 40-60 秒；若时间充足，再展开第 4-8 页的阶段短片。",
        "",
        "## 幻灯片总览",
        "",
        md_row(["页码", "标题", "核心结论", "图表", "视频"]),
        md_row(["---:", "---", "---", "---", "---"]),
    ]
    for slide in SLIDES:
        lines.append(
            md_row(
                [
                    slide["id"],
                    slide["title"],
                    slide["message"],
                    f"`{slide['figure']}`" if slide["figure"] else "无",
                    f"`{slide['video']}`" if slide["video"] else "无",
                ]
            )
        )

    lines.extend(["", "## 页级讲解脚本", ""])
    for slide in SLIDES:
        lines.extend(
            [
                f"### Slide {slide['id']}：{slide['title']}",
                "",
                f"核心结论：{slide['message']}",
                "",
                f"讲稿提示：{slide['script']}",
                "",
            ]
        )
        if slide["figure"]:
            lines.extend(["图表 cue：", "", f"- `{slide['figure']}`", ""])
        if slide["video"]:
            lines.extend(["视频 cue：", "", f"- `{slide['video']}`", ""])
        if slide.get("links"):
            lines.extend(["资料入口：", ""])
            for link in slide["links"]:
                lines.append(f"- `{link['path']}`：{link['title']}。{link['note']}")
            lines.append("")
        table = method_table(slide, methods, summary, language, resources)
        if table:
            lines.extend(["引用版本与指标：", "", *table, ""])

    lines.extend(
        [
            "## 讲解红线",
            "",
            "- `structured_waypoint_policy_v1` 是显式状态/阶段控制，不是 learned VLA。",
            "- `torch_act_state_chunk_v1`、`torch_act_state_chunk_cuda_v1`、`phase_conditioned_torch_act_v1`、`torch_act_cvae_state_chunk_v1` 是 state-only ACT-style/ACT-CVAE-lite baseline，不能写成完整视觉 ACT。",
            "- `torch_diffusion_policy_state_chunk_v1` 是 state-only diffusion action-chunk baseline，不能写成完整视觉 Diffusion Policy。",
            "- `clip_action_head_lite_v1` 使用通用 CLIP，不是机器人 VLA 或 OpenVLA。",
            "- `adapter_action_head_lite_v1` 和 `lora_action_head_lite_v1` 是本地 PEFT proxy，不能写成 pretrained VLA LoRA/Adapter。",
            "- `openvla_dataset_bridge_v1`、`openvla_feasibility_check_v1`、`robot_vla_action_head_handoff_v1`、`robot_vla_remote_run_pack_v1` 和 `robot_vla_remote_result_intake_v1` 是数据桥接、本地可行性检查、远端运行包和回填门禁，不能写成真实 OpenVLA LoRA 或 `robot_vla_action_head_lite_v1` 已完成。",
            "- `external_dependency_readiness_audit_v1` 是外部依赖阶段门禁审计，不是策略成功率结果；当前所有 planned 外部版本都不能直接进入正式方法统计。",
            "- `isaac_domain_randomization_handoff_v1` 和 `real_widowx_validation_handoff_v1` 是交接门禁和 trial 模板，不能写成 Isaac 或真实 WidowX 运行结果。",
            "- `strict_grasp_success_audit_v1` 是评测口径审计，不能把原始放置 success 写成稳定抓取成功；需要同时报告 `grasp_success` 和 `object_z`。",
            "",
            "## 关联材料",
            "",
            "- 论文结果章节草稿：`docs/thesis_results_chapter_draft.md`",
            "- OpenVLA 数据桥接浏览页：`docs/openvla_bridge_gallery.html`",
            "- OpenVLA 数据桥接报告：`docs/openvla_dataset_bridge_report.md`",
            "- OpenVLA 本地可行性检查：`docs/openvla_feasibility_report.md`",
            "- Robot VLA action-head 交接门禁：`docs/robot_vla_action_head_handoff.md`",
            "- Robot VLA 远端运行包：`docs/robot_vla_remote_run_pack.md`",
            "- Robot VLA 远端结果回填门禁：`docs/robot_vla_remote_result_intake.md`",
            "- 外部依赖阶段 readiness audit：`docs/external_dependency_readiness_audit.md`",
            "- Isaac domain randomization 交接门禁：`docs/isaac_domain_randomization_handoff.md`",
            "- 真实 WidowX 验证交接门禁：`docs/real_widowx_validation_handoff.md`",
            "- 严格抓取成功口径审计：`docs/strict_grasp_success_audit.md`",
            "- 方法阶段审计表：`docs/method_stage_audit.md`",
            "- 答辩视频片段包：`docs/presentation_video_pack.md`",
            "- 总交付入口：`docs/final_experiment_package.md`",
            "",
            "## 重新生成命令",
            "",
            "```powershell",
            f'& "{ROOT / ".venv" / "Scripts" / "python.exe"}" "{ROOT / "scripts" / "build_defense_slide_outline.py"}"',
            "```",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).replace("{STRICT_LOOSE}", loose).replace("{STRICT_SUCCESS}", strict)
    args.output.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_doc(args)
    print(f"defense_slide_outline: {args.output}", flush=True)
    print(f"slides: {len(SLIDES)}", flush=True)


if __name__ == "__main__":
    main()
