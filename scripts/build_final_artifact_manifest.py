from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


CORE_ARTIFACTS = [
    ("总交付入口", "docs/final_experiment_package.md"),
    ("最终闭环审计", "docs/final_closure_audit_v1.md"),
    ("最终闭环审计 JSON", "outputs/evaluations/final_closure_audit_v1.json"),
    ("V4 独立复核", "docs/v4_independent_replication_v1.md"),
    ("接触监测器独立闭环", "docs/contact_phase_monitor_heldout_v1_analysis.md"),
    ("提前重抓反事实审计", "docs/counterfactual_intervention_pilot_v1_audit.md"),
    ("MuJoCo-only 研究范围决议", "docs/mujoco_only_scope.md"),
    ("最终展示与交付 Handoff 索引", "docs/final_showcase_handoff.md"),
    ("最终展示与交付 Handoff CSV", "docs/final_showcase_handoff.csv"),
    ("最终展示与交付 Handoff 脚本", "scripts/build_final_showcase_handoff.py"),
    ("最终答辩讲解脚本", "docs/final_defense_narrative_script.md"),
    ("最终答辩讲解脚本 CSV", "docs/final_defense_narrative_script.csv"),
    ("最终答辩讲解脚本生成脚本", "scripts/build_final_defense_narrative_script.py"),
    ("剩余实验执行看板", "docs/remaining_experiment_execution_board.md"),
    ("剩余实验执行看板 CSV", "docs/remaining_experiment_execution_board.csv"),
    ("剩余实验执行看板脚本", "scripts/build_remaining_experiment_execution_board.py"),
    ("可复现命令索引", "docs/reproducible_command_index.md"),
    ("实验版本登记", "docs/experiment_versions.json"),
    ("最终方法版本索引", "docs/final_method_version_index.md"),
    ("最终方法版本索引 CSV", "docs/final_method_version_index.csv"),
    ("主任务评测表", "docs/evaluation_summary.csv"),
    ("语言泛化评测表", "docs/language_generalization_summary.csv"),
    ("模型资源表", "docs/model_resource_summary.csv"),
    ("数据效率表", "docs/data_efficiency_summary.csv"),
    ("Domain randomization 代理报告", "docs/domain_randomization_summary.md"),
    ("Domain randomization 逐集表", "docs/domain_randomization_summary.csv"),
    ("Domain randomization JSON", "outputs/evaluations/domain_randomization_eval_v1.json"),
    ("Isaac domain randomization 交接门禁报告", "docs/isaac_domain_randomization_handoff.md"),
    ("Isaac domain randomization 交接门禁 CSV", "docs/isaac_domain_randomization_handoff.csv"),
    ("Isaac domain randomization 交接门禁 JSON", "outputs/evaluations/isaac_domain_randomization_handoff_v1.json"),
    ("Isaac domain randomization 交接门禁脚本", "scripts/build_isaac_domain_randomization_handoff.py"),
    ("真实 WidowX 验证交接门禁报告", "docs/real_widowx_validation_handoff.md"),
    ("真实 WidowX 验证交接门禁 CSV", "docs/real_widowx_validation_handoff.csv"),
    ("真实 WidowX 验证交接门禁 JSON", "outputs/evaluations/real_widowx_validation_handoff_v1.json"),
    ("真实 WidowX 验证 trial 模板", "outputs/real_robot/real_widowx_validation_v1_trial_template.csv"),
    ("真实 WidowX 验证交接门禁脚本", "scripts/build_real_widowx_validation_handoff.py"),
    ("阶段结果矩阵", "docs/result_matrix.md"),
    ("方法评测比较看板", "docs/method_comparison_dashboard.md"),
    ("方法评测比较看板 CSV", "docs/method_comparison_dashboard.csv"),
    ("方法评测比较看板 HTML", "docs/method_comparison_dashboard.html"),
    ("方法评测比较看板脚本", "scripts/build_method_comparison_dashboard.py"),
    ("核心多任务对比矩阵", "docs/core_task_comparison_matrix.md"),
    ("核心多任务对比矩阵 CSV", "docs/core_task_comparison_matrix.csv"),
    ("核心多任务对比矩阵 JSON", "outputs/evaluations/core_task_comparison_matrix_v1.json"),
    ("核心多任务对比矩阵脚本", "scripts/build_core_task_comparison_matrix.py"),
    ("Core V2 留出集对比矩阵", "docs/core_v2_holdout_comparison_matrix.md"),
    ("Core V2 留出集对比矩阵 CSV", "docs/core_v2_holdout_comparison_matrix.csv"),
    ("Core V2 留出集对比矩阵 JSON", "outputs/evaluations/core_v2_holdout_comparison_matrix_v1.json"),
    ("Core V2 留出集对比矩阵脚本", "scripts/build_core_v2_comparison_matrix.py"),
    ("Core V2 预训练 VLM 动作头报告", "docs/core_v2_pretrained_vlm_action_head_report.md"),
    ("Core V2 预训练 VLM 动作头 CSV", "docs/core_v2_pretrained_vlm_action_head_report.csv"),
    ("Core V2 预训练 VLM 动作头 JSON", "outputs/evaluations/core_v2_pretrained_vlm_action_head_v1.json"),
    ("Core V2 预训练 VLM 动作头脚本", "scripts/build_core_v2_pretrained_vlm_report.py"),
    ("Core V2 预训练 CLIP 动作头模型", "outputs/clip_action_head/clip_core_v2_multitask_v1_20260721_104743.npz"),
    ("Core V2 预训练 CLIP 主任务视频", "outputs/videos/clip_core_v2_multitask_v1_seed0.mp4"),
    ("Core V2 预训练 CLIP 空间任务视频", "outputs/videos/clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4"),
    ("Core V2 CLIP 语义-结构化执行报告", "docs/core_v2_clip_semantic_waypoint_report.md"),
    ("Core V2 CLIP 语义-结构化执行 CSV", "docs/core_v2_clip_semantic_waypoint_report.csv"),
    ("Core V2 CLIP 语义-结构化执行 JSON", "outputs/evaluations/core_v2_clip_semantic_waypoint_v1.json"),
    ("Core V2 CLIP 语义-结构化执行报告脚本", "scripts/build_core_v2_clip_semantic_waypoint_report.py"),
    ("Core V2 CLIP 语义 adapter 训练脚本", "scripts/train_clip_semantic_waypoint.py"),
    ("Core V2 CLIP 语义执行脚本", "scripts/run_clip_semantic_waypoint.py"),
    ("Core V2 CLIP 语义评测脚本", "scripts/evaluate_clip_semantic_waypoint.py"),
    ("Core V2 CLIP 语义 adapter 模型", "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_v1_20260721_110325.npz"),
    ("Core V2 CLIP 语义空间任务视频", "outputs/videos/clip_semantic_waypoint_core_v2_v1_leftmost_cube_seed420.mp4"),
    ("Kaggle 冻结 CLIP 语义适配器补充实验报告", "docs/kaggle_clip_semantic_adapter_core_v2_v1_report.md"),
    ("Kaggle 冻结 CLIP 语义适配器逐任务 CSV", "docs/kaggle_clip_semantic_adapter_core_v2_v1.csv"),
    ("Kaggle 冻结 CLIP 语义适配器汇总 JSON", "outputs/evaluations/kaggle_clip_semantic_adapter_core_v2_v1.json"),
    ("Kaggle 冻结 CLIP 语义适配器报告构建脚本", "scripts/build_kaggle_clip_semantic_adapter_report.py"),
    ("Kaggle 冻结 CLIP 语义适配器远程训练脚本", "kaggle/kernels/widowx_mujoco_clip_semantic_adapter_v1/train_clip_semantic_adapter.py"),
    ("Kaggle 冻结 CLIP 语义适配器 Kernel 元数据", "kaggle/kernels/widowx_mujoco_clip_semantic_adapter_v1/kernel-metadata.json"),
    ("Kaggle 冻结 CLIP 语义适配器模型", "outputs/clip_semantic_waypoint/kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3.npz"),
    ("Kaggle 冻结 CLIP 语义适配器远程指标", "outputs/kaggle_remote/kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3/kaggle_clip_semantic_adapter_core_v2_v1_metrics.json"),
    ("Kaggle 冻结 CLIP 语义适配器闭环视频", "outputs/videos/kaggle_clip_semantic_adapter_core_v2_v1_hard_leftmost_seed1900.mp4"),
    ("Kaggle 冻结 CLIP 语义适配器视频元数据", "outputs/videos/kaggle_clip_semantic_adapter_core_v2_v1_hard_leftmost_seed1900.json"),
    ("Kaggle 冻结 CLIP 语义适配器 OOD 报告", "docs/kaggle_clip_semantic_adapter_core_v2_ood_v1_report.md"),
    ("Kaggle 冻结 CLIP 语义适配器 OOD CSV", "docs/kaggle_clip_semantic_adapter_core_v2_ood_v1.csv"),
    ("Kaggle 冻结 CLIP 语义适配器 OOD JSON", "outputs/evaluations/kaggle_clip_semantic_adapter_core_v2_ood_v1.json"),
    ("Kaggle 冻结 CLIP 语义适配器 OOD 语义误判视频", "outputs/videos/kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.mp4"),
    ("Kaggle 冻结 CLIP 语义适配器 OOD 语义误判视频元数据", "outputs/videos/kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.json"),
    ("冻结 CLIP 线性头与 Kaggle 适配器同协议对照报告", "docs/frozen_clip_semantic_adapter_same_protocol_comparison.md"),
    ("冻结 CLIP 线性头与 Kaggle 适配器同协议对照 CSV", "docs/frozen_clip_semantic_adapter_same_protocol_comparison.csv"),
    ("冻结 CLIP 线性头与 Kaggle 适配器同协议对照 JSON", "outputs/evaluations/frozen_clip_semantic_adapter_same_protocol_comparison_v1.json"),
    ("冻结 CLIP 线性头与 Kaggle 适配器同协议对照构建脚本", "scripts/build_frozen_clip_semantic_adapter_comparison.py"),
    ("Core V2 CLIP 语义数据效率报告", "docs/core_v2_clip_semantic_data_efficiency.md"),
    ("Core V2 CLIP 语义数据效率 CSV", "docs/core_v2_clip_semantic_data_efficiency.csv"),
    ("Core V2 CLIP 语义数据效率 JSON", "outputs/evaluations/core_v2_clip_semantic_data_efficiency_v1.json"),
    ("Core V2 CLIP 语义数据效率评测脚本", "scripts/evaluate_clip_semantic_waypoint_data_efficiency.py"),
    ("Core V2 CLIP 语义数据效率报告脚本", "scripts/build_core_v2_clip_semantic_data_efficiency_report.py"),
    ("Core V2 CLIP 语义 5 条预算模型", "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_5eps_v1_20260721_111415.npz"),
    ("Core V2 CLIP 语义 10 条预算模型", "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_10eps_v1_20260721_111439.npz"),
    ("Core V2 CLIP 语义 OOD 泛化报告", "docs/core_v2_clip_semantic_ood_generalization.md"),
    ("Core V2 CLIP 语义 OOD 泛化 CSV", "docs/core_v2_clip_semantic_ood_generalization.csv"),
    ("Core V2 CLIP 语义 OOD 泛化 JSON", "outputs/evaluations/core_v2_clip_semantic_ood_generalization_v1.json"),
    ("Core V2 CLIP 语义 OOD 泛化脚本", "scripts/evaluate_clip_semantic_ood_generalization.py"),
    ("Core V2 CLIP 语义 OOD 全物体成功视频", "outputs/videos/clip_semantic_ood_hard_leftmost_cube_seed1300.mp4"),
    ("Core V2 CLIP 语义 OOD 语言失败视频", "outputs/videos/clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4"),
    ("Core V2 代表性视频证据说明", "docs/core_v2_video_evidence.md"),
    ("Core V2 专家成功视频", "outputs/videos/core_v2_expert_leftmost_cube_seed420.mp4"),
    ("Core V2 action-head 失败视频", "outputs/videos/core_v2_object_action_head_leftmost_cube_seed420.mp4"),
    ("Core V2 高效训练子集脚本", "scripts/create_demo_subset.py"),
    ("核心任务蓝方块到蓝盘 CSV", "docs/core_task_blue_cube_blue_pad.csv"),
    ("核心任务蓝方块到蓝盘 JSON", "outputs/evaluations/core_task_blue_cube_blue_pad.json"),
    ("核心任务蓝方块到红盘 CSV", "docs/core_task_blue_cube_red_pad.csv"),
    ("核心任务蓝方块到红盘 JSON", "outputs/evaluations/core_task_blue_cube_red_pad.json"),
    ("核心任务红方块到红盘 CSV", "docs/core_task_red_cube_red_pad.csv"),
    ("核心任务红方块到红盘 JSON", "outputs/evaluations/core_task_red_cube_red_pad.json"),
    ("核心任务最左物体到碗 CSV", "docs/core_task_leftmost_to_bowl.csv"),
    ("核心任务最左物体到碗 JSON", "outputs/evaluations/core_task_leftmost_to_bowl.json"),
    ("论文图表与视频证据索引", "docs/thesis_visual_evidence_index.md"),
    ("论文图表与视频证据索引 CSV", "docs/thesis_visual_evidence_index.csv"),
    ("论文图表与视频证据索引 HTML", "docs/thesis_visual_evidence_index.html"),
    ("论文图表与视频证据索引脚本", "scripts/build_thesis_visual_evidence_index.py"),
    ("答辩追问 Q&A Playbook", "docs/defense_qa_playbook.md"),
    ("答辩追问 Q&A Playbook CSV", "docs/defense_qa_playbook.csv"),
    ("答辩追问 Q&A Playbook HTML", "docs/defense_qa_playbook.html"),
    ("答辩追问 Q&A Playbook 脚本", "scripts/build_defense_qa_playbook.py"),
    ("实验版本谱系索引", "docs/version_lineage_index.md"),
    ("实验版本谱系索引 CSV", "docs/version_lineage_index.csv"),
    ("实验版本谱系索引 HTML", "docs/version_lineage_index.html"),
    ("实验版本谱系索引脚本", "scripts/build_version_lineage_index.py"),
    ("方法阶段审计", "docs/method_stage_audit.md"),
    ("方法证据门禁", "docs/method_evidence_gate.md"),
    ("方法证据门禁 CSV", "docs/method_evidence_gate.csv"),
    ("版本命名与入包门禁规范", "docs/version_naming_and_gate_spec.md"),
    ("版本命名与入包门禁规范 CSV", "docs/version_naming_and_gate_spec.csv"),
    ("版本命名与入包门禁规范 JSON", "outputs/evaluations/version_naming_and_gate_spec_v1.json"),
    ("版本命名与入包门禁规范脚本", "scripts/build_version_naming_and_gate_spec.py"),
    ("阶段对比报告", "docs/stage_comparison_report.md"),
    ("任务/数据/普通 BC 阶段报告", "docs/task_bc_stage_report.md"),
    ("任务/数据/普通 BC 阶段 CSV", "docs/task_bc_stage_report.csv"),
    ("Trajectory/ACT 阶段报告", "docs/trajectory_act_stage_report.md"),
    ("Trajectory/ACT 阶段 CSV", "docs/trajectory_act_stage_report.csv"),
    ("Trajectory/ACT 中文实验台账", "docs/trajectory_act_experiment_record.md"),
    ("Trajectory/ACT 中文实验台账 CSV", "docs/trajectory_act_experiment_record.csv"),
    ("Trajectory/ACT 中文实验台账脚本", "scripts/build_trajectory_act_experiment_record.py"),
    ("Trajectory/ACT 失败诊断矩阵", "docs/trajectory_act_failure_diagnosis.md"),
    ("Trajectory/ACT 失败诊断 CSV", "docs/trajectory_act_failure_diagnosis.csv"),
    ("Trajectory/ACT 论文结论摘要", "docs/trajectory_act_conclusion_brief.md"),
    ("Trajectory/ACT 论文结论摘要 CSV", "docs/trajectory_act_conclusion_brief.csv"),
    ("Trajectory/ACT 论文结论摘要脚本", "scripts/build_trajectory_act_conclusion_brief.py"),
    ("Trajectory/ACT 超慢可视化指南", "docs/trajectory_act_slow_viewer_guide.md"),
    ("Trajectory/ACT 超慢可视化指南 CSV", "docs/trajectory_act_slow_viewer_guide.csv"),
    ("Trajectory/ACT 超慢可视化指南脚本", "scripts/build_trajectory_act_slow_viewer_guide.py"),
    ("Trajectory phase template BC 候选模型", "outputs/trajectory_phase_template_bc/trajectory_phase_template_bc_20260720_160007.npz"),
    ("Trajectory phase template BC 候选报告", "docs/trajectory_phase_template_bc_report.md"),
    ("Trajectory phase template BC 候选 CSV", "docs/trajectory_phase_template_bc_report.csv"),
    ("Trajectory phase template BC 候选 JSON", "outputs/evaluations/trajectory_phase_template_bc_v1.json"),
    ("Trajectory phase template BC 候选视频", "outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4"),
    ("Trajectory-prior Residual BC 候选模型", "outputs/trajectory_prior_residual_bc/trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz"),
    ("Trajectory-prior Residual BC 候选报告", "docs/trajectory_prior_residual_bc_report.md"),
    ("Trajectory-prior Residual BC 候选 CSV", "docs/trajectory_prior_residual_bc_report.csv"),
    ("Trajectory-prior Residual BC 候选 JSON", "outputs/evaluations/trajectory_prior_residual_bc_v1_candidate.json"),
    ("Trajectory-prior Residual BC 候选视频", "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4"),
    ("Trajectory-prior Residual BC 候选视频元数据", "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.json"),
    ("Trajectory-prior Residual BC 公共脚本", "scripts/trajectory_prior_residual_common.py"),
    ("Trajectory-prior Residual BC 训练脚本", "scripts/train_trajectory_prior_residual_bc.py"),
    ("Trajectory-prior Residual BC 运行脚本", "scripts/run_trajectory_prior_residual_policy.py"),
    ("Trajectory-prior Residual BC 评测脚本", "scripts/evaluate_trajectory_prior_residual_bc.py"),
    ("Timing-aware Trajectory-prior Residual BC 候选模型", "outputs/timing_aware_trajectory_prior_residual_bc/timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz"),
    ("Timing-aware Trajectory-prior Residual BC 候选报告", "docs/timing_aware_trajectory_prior_residual_bc_report.md"),
    ("Timing-aware Trajectory-prior Residual BC 候选 CSV", "docs/timing_aware_trajectory_prior_residual_bc_report.csv"),
    ("Timing-aware Trajectory-prior Residual BC 候选 JSON", "outputs/evaluations/timing_aware_trajectory_prior_residual_bc_v1_candidate.json"),
    ("Timing-aware Trajectory-prior Residual BC 候选视频", "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4"),
    ("Timing-aware Trajectory-prior Residual BC 候选视频元数据", "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.json"),
    ("Timing-aware Trajectory-prior Residual BC 公共脚本", "scripts/timing_aware_trajectory_prior_residual_common.py"),
    ("Timing-aware Trajectory-prior Residual BC 训练脚本", "scripts/train_timing_aware_trajectory_prior_residual_bc.py"),
    ("Timing-aware Trajectory-prior Residual BC 运行脚本", "scripts/run_timing_aware_trajectory_prior_residual_policy.py"),
    ("Timing-aware Trajectory-prior Residual BC 评测脚本", "scripts/evaluate_timing_aware_trajectory_prior_residual_bc.py"),
    ("Grasp-gated trajectory/ACT 候选诊断报告", "docs/grasp_gated_trajectory_act_report.md"),
    ("Grasp-gated trajectory/ACT 候选诊断 CSV", "docs/grasp_gated_trajectory_act_report.csv"),
    ("Grasp-gated trajectory/ACT 候选诊断 JSON", "outputs/evaluations/grasp_gated_trajectory_act_v1_candidate.json"),
    ("Grasp-gated trajectory/ACT 候选诊断脚本", "scripts/evaluate_grasp_gated_trajectory_act.py"),
    ("Grasp-gated trajectory chunk BC 候选视频", "outputs/videos/grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4"),
    ("Grasp-gated Torch ACT 候选视频", "outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4"),
    ("Phase-weighted Torch ACT 候选模型", "outputs/torch_act/phase_weighted_torch_act_v1_candidate_20260720_225108.pt"),
    ("Phase-weighted Torch ACT 候选诊断报告", "docs/phase_weighted_torch_act_report.md"),
    ("Phase-weighted Torch ACT 候选诊断 CSV", "docs/phase_weighted_torch_act_report.csv"),
    ("Phase-weighted Torch ACT 候选诊断 JSON", "outputs/evaluations/phase_weighted_torch_act_v1_candidate.json"),
    ("Phase-weighted Torch ACT 候选诊断脚本", "scripts/evaluate_phase_weighted_torch_act.py"),
    ("Phase-weighted Torch ACT 候选视频", "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.mp4"),
    ("Contact/Phase-gated Torch ACT 候选模型", "outputs/torch_act/contact_phase_gated_torch_act_v1_candidate_20260721_003304.pt"),
    ("Contact/Phase-gated Torch ACT 候选报告", "docs/contact_phase_gated_torch_act_report.md"),
    ("Contact/Phase-gated Torch ACT 候选 CSV", "docs/contact_phase_gated_torch_act_report.csv"),
    ("Contact/Phase-gated Torch ACT 候选 JSON", "outputs/evaluations/contact_phase_gated_torch_act_v1_candidate.json"),
    ("Contact/Phase-gated Torch ACT 候选视频", "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.mp4"),
    ("Contact/Phase-gated Torch ACT 候选视频元数据", "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.json"),
    ("Contact/Phase-gated Torch ACT 候选评测脚本", "scripts/evaluate_contact_phase_gated_torch_act.py"),
    ("Contact-aware Phase-gated Torch ACT 候选模型", "outputs/torch_act/contact_aware_phase_gated_torch_act_v1_candidate_20260721_004944.pt"),
    ("Contact-aware Phase-gated Torch ACT 候选报告", "docs/contact_aware_phase_gated_torch_act_report.md"),
    ("Contact-aware Phase-gated Torch ACT 候选 CSV", "docs/contact_aware_phase_gated_torch_act_report.csv"),
    ("Contact-aware Phase-gated Torch ACT 候选 JSON", "outputs/evaluations/contact_aware_phase_gated_torch_act_v1_candidate.json"),
    ("Contact-aware Phase-gated Torch ACT 候选视频", "outputs/videos/contact_aware_phase_gated_torch_act_v1_candidate_seed0.mp4"),
    ("Contact-aware Phase-gated Torch ACT 候选视频元数据", "outputs/videos/contact_aware_phase_gated_torch_act_v1_candidate_seed0.json"),
    ("Contact-aware Phase-gated Torch ACT 候选评测脚本", "scripts/evaluate_contact_aware_phase_gated_torch_act.py"),
    ("Grasp/Lift 子策略上界诊断报告", "docs/grasp_lift_subpolicy_probe_report.md"),
    ("Grasp/Lift 子策略上界诊断 CSV", "docs/grasp_lift_subpolicy_probe_report.csv"),
    ("Grasp/Lift 子策略上界诊断 JSON", "outputs/evaluations/grasp_lift_subpolicy_probe_v1_candidate.json"),
    ("Grasp/Lift 子策略上界诊断脚本", "scripts/evaluate_grasp_lift_subpolicy_probe.py"),
    ("Grasp/Lift 子策略上界诊断视频", "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4"),
    ("Contact-stage subpolicy 候选报告", "docs/contact_stage_subpolicy_report.md"),
    ("Contact-stage subpolicy 候选 CSV", "docs/contact_stage_subpolicy_report.csv"),
    ("Contact-stage subpolicy 候选 JSON", "outputs/evaluations/contact_stage_subpolicy_v1_candidate.json"),
    ("Contact-stage subpolicy 候选运行脚本", "scripts/run_contact_stage_subpolicy.py"),
    ("Contact-stage subpolicy 候选评测脚本", "scripts/evaluate_contact_stage_subpolicy.py"),
    ("Contact-stage subpolicy 候选视频", "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4"),
    ("Contact-stage subpolicy 候选视频元数据", "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.json"),
    ("Contact-stage 示范数据 metadata", "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/metadata.jsonl"),
    ("Contact-stage 示范数据 summary", "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/summary.json"),
    ("Contact-stage Demo Torch ACT 候选模型", "outputs/torch_act/contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt"),
    ("Contact-stage Demo Torch ACT 候选报告", "docs/contact_stage_demo_torch_act_report.md"),
    ("Contact-stage Demo Torch ACT 候选 CSV", "docs/contact_stage_demo_torch_act_report.csv"),
    ("Contact-stage Demo Torch ACT 候选 JSON", "outputs/evaluations/contact_stage_demo_torch_act_v1_candidate.json"),
    ("Contact-stage Demo Torch ACT 候选视频", "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4"),
    ("Contact-stage Demo Torch ACT 候选视频元数据", "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.json"),
    ("Contact-stage 示范采集脚本", "scripts/collect_contact_stage_demos.py"),
    ("Contact-stage Demo Torch ACT 评测脚本", "scripts/evaluate_contact_stage_demo_torch_act.py"),
    ("Contact-stage Phase Action-Head 候选模型", "outputs/phase_action_head/contact_stage_phase_action_head_v1_candidate_20260721_020941.npz"),
    ("Contact-stage Phase Action-Head 候选报告", "docs/contact_stage_phase_action_head_report.md"),
    ("Contact-stage Phase Action-Head 候选 CSV", "docs/contact_stage_phase_action_head_report.csv"),
    ("Contact-stage Phase Action-Head 候选 JSON", "outputs/evaluations/contact_stage_phase_action_head_v1_candidate.json"),
    ("Contact-stage Phase Action-Head 候选视频", "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4"),
    ("Contact-stage Phase Action-Head 候选视频元数据", "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.json"),
    ("Contact-stage Phase Action-Head 评测脚本", "scripts/evaluate_contact_stage_phase_action_head.py"),
    ("Contact-hold Weighted Torch ACT 候选模型", "outputs/torch_act/contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt"),
    ("Contact-hold Weighted Torch ACT 候选报告", "docs/contact_hold_weighted_torch_act_report.md"),
    ("Contact-hold Weighted Torch ACT 候选 CSV", "docs/contact_hold_weighted_torch_act_report.csv"),
    ("Contact-hold Weighted Torch ACT 候选 JSON", "outputs/evaluations/contact_hold_weighted_torch_act_v1_candidate.json"),
    ("Contact-hold Weighted Torch ACT 候选视频", "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4"),
    ("Contact-hold Weighted Torch ACT 候选视频元数据", "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.json"),
    ("Contact-hold Weighted Torch ACT 评测脚本", "scripts/evaluate_contact_hold_weighted_torch_act.py"),
    ("Gripper timing/contact probe 候选报告", "docs/gripper_timing_contact_probe_report.md"),
    ("Gripper timing/contact probe 候选 CSV", "docs/gripper_timing_contact_probe_report.csv"),
    ("Gripper timing/contact probe 候选 JSON", "outputs/evaluations/gripper_timing_contact_probe_v1_candidate.json"),
    ("Gripper timing/contact probe 候选视频", "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.mp4"),
    ("Gripper timing/contact probe 候选视频元数据", "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.json"),
    ("Gripper timing/contact probe 运行脚本", "scripts/run_gripper_timing_probe.py"),
    ("Gripper timing/contact probe 评测脚本", "scripts/evaluate_gripper_timing_probe.py"),
    ("Grasp-gated trajectory-kNN 候选报告", "docs/grasp_gated_trajectory_knn_report.md"),
    ("Grasp-gated trajectory-kNN 候选 CSV", "docs/grasp_gated_trajectory_knn_report.csv"),
    ("Grasp-gated trajectory-kNN 候选 JSON", "outputs/evaluations/grasp_gated_trajectory_knn_v1.json"),
    ("Grasp-gated trajectory-kNN 候选视频", "outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4"),
    ("Contact-aware trajectory-kNN 候选模型", "outputs/trajectory_knn_bc/contact_aware_trajectory_knn_20260720_233445.npz"),
    ("Contact-aware trajectory-kNN 候选报告", "docs/contact_aware_trajectory_knn_report.md"),
    ("Contact-aware trajectory-kNN 候选 CSV", "docs/contact_aware_trajectory_knn_report.csv"),
    ("Contact-aware trajectory-kNN 候选 JSON", "outputs/evaluations/contact_aware_trajectory_knn_v1_candidate.json"),
    ("Contact-aware trajectory-kNN 候选视频", "outputs/videos/contact_aware_trajectory_knn_v1_candidate_seed0.mp4"),
    ("Contact-aware trajectory-kNN 候选视频元数据", "outputs/videos/contact_aware_trajectory_knn_v1_candidate_seed0.json"),
    ("Contact-aware trajectory-kNN 候选评测脚本", "scripts/evaluate_contact_aware_trajectory_knn.py"),
    ("Preference trajectory post-training 候选模型", "outputs/preference_post_training/preference_trajectory_post_training_20260720_165005.npz"),
    ("Preference trajectory post-training 候选报告", "docs/preference_trajectory_post_training_report.md"),
    ("Preference trajectory post-training 候选 CSV", "docs/preference_trajectory_post_training_report.csv"),
    ("Preference trajectory post-training 候选 JSON", "outputs/evaluations/preference_trajectory_post_training_v1.json"),
    ("Preference trajectory post-training 候选视频", "outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4"),
    ("Ranked-objective preference trajectory post-training 候选总结", "docs/preference_trajectory_post_training_v1_ranked_objective_summary.md"),
    ("Ranked-objective preference trajectory post-training 候选报告", "docs/preference_trajectory_post_training_v1_ranked_objective_report.md"),
    ("Ranked-objective preference trajectory post-training 候选 CSV", "docs/preference_trajectory_post_training_v1_ranked_objective_report.csv"),
    ("Ranked-objective preference trajectory post-training 候选 JSON", "outputs/evaluations/preference_trajectory_post_training_v1_ranked_objective_candidate.json"),
    ("Ranked-objective preference trajectory post-training 候选模型", "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz"),
    ("Ranked-objective preference trajectory post-training 候选视频", "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4"),
    ("Ranked-objective preference trajectory post-training 候选视频元数据", "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.json"),
    ("Ranked-fast preference trajectory post-training 候选总结", "docs/preference_trajectory_post_training_v1_ranked_fast_summary.md"),
    ("Ranked-fast preference trajectory post-training 候选报告", "docs/preference_trajectory_post_training_v1_ranked_fast_report.md"),
    ("Ranked-fast preference trajectory post-training 候选 CSV", "docs/preference_trajectory_post_training_v1_ranked_fast_report.csv"),
    ("Ranked-fast preference trajectory post-training 候选 JSON", "outputs/evaluations/preference_trajectory_post_training_v1_ranked_fast_candidate.json"),
    ("Ranked-fast preference trajectory post-training 候选模型", "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz"),
    ("Ranked-fast preference trajectory post-training 候选视频", "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4"),
    ("Ranked-fast preference trajectory post-training 候选视频元数据", "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.json"),
    ("TCP-lift preference trajectory post-training 候选报告", "docs/preference_trajectory_post_training_v1_tcp_lift_rank_report.md"),
    ("TCP-lift preference trajectory post-training 候选 CSV", "docs/preference_trajectory_post_training_v1_tcp_lift_rank_report.csv"),
    ("TCP-lift preference trajectory post-training 候选 JSON", "outputs/evaluations/preference_trajectory_post_training_v1_tcp_lift_rank_candidate.json"),
    ("TCP-lift preference trajectory post-training 候选模型", "outputs/preference_post_training/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_20260721_090438.npz"),
    ("TCP-lift preference trajectory post-training 候选视频", "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.mp4"),
    ("TCP-lift preference trajectory post-training 候选视频元数据", "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.json"),
    ("Preference trajectory post-training 训练脚本", "scripts/train_preference_trajectory_post_training.py"),
    ("Preference trajectory post-training 运行脚本", "scripts/run_preference_trajectory_post_training_policy.py"),
    ("Preference trajectory post-training 评测脚本", "scripts/evaluate_preference_trajectory_post_training.py"),
    ("Preference post-training 正式升级门禁", "docs/preference_post_training_upgrade_gate.md"),
    ("Preference post-training 正式升级门禁 CSV", "docs/preference_post_training_upgrade_gate.csv"),
    ("Preference post-training 正式升级门禁 JSON", "outputs/evaluations/preference_post_training_upgrade_gate_v1.json"),
    ("Preference post-training 正式升级门禁脚本", "scripts/build_preference_post_training_upgrade_gate.py"),
    ("Preference 后训练消融矩阵", "docs/preference_post_training_ablation_matrix.md"),
    ("Preference 后训练消融矩阵 CSV", "docs/preference_post_training_ablation_matrix.csv"),
    ("Preference 后训练消融矩阵脚本", "scripts/build_preference_post_training_ablation_matrix.py"),
    ("Preference + contact-aware trajectory post-training 候选模型", "outputs/preference_post_training/preference_contact_aware_trajectory_post_training_20260721_000449.npz"),
    ("Preference + contact-aware trajectory post-training 候选报告", "docs/preference_contact_aware_trajectory_post_training_report.md"),
    ("Preference + contact-aware trajectory post-training 候选 CSV", "docs/preference_contact_aware_trajectory_post_training_report.csv"),
    ("Preference + contact-aware trajectory post-training 候选 JSON", "outputs/evaluations/preference_contact_aware_trajectory_post_training_v1_candidate.json"),
    ("Preference + contact-aware trajectory post-training 候选视频", "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4"),
    ("Preference + contact-aware trajectory post-training 候选视频元数据", "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.json"),
    ("Preference + contact-aware trajectory post-training 评测脚本", "scripts/evaluate_preference_contact_aware_trajectory_post_training.py"),
    ("Ranked preference trajectory post-training 候选模型", "outputs/preference_post_training/preference_ranked_trajectory_post_training_20260721_031024.npz"),
    ("Ranked preference trajectory post-training 候选报告", "docs/preference_ranked_trajectory_post_training_report.md"),
    ("Ranked preference trajectory post-training 候选 CSV", "docs/preference_ranked_trajectory_post_training_report.csv"),
    ("Ranked preference trajectory post-training 候选 JSON", "outputs/evaluations/preference_ranked_trajectory_post_training_v1_candidate.json"),
    ("Ranked preference trajectory post-training 候选视频", "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4"),
    ("Ranked preference trajectory post-training 候选视频元数据", "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.json"),
    ("Ranked preference trajectory post-training 评测脚本", "scripts/evaluate_preference_ranked_trajectory_post_training.py"),
    ("候选方法诊断视频索引", "docs/candidate_diagnostic_video_index.md"),
    ("候选方法诊断视频 CSV", "docs/candidate_diagnostic_video_index.csv"),
    ("候选方法诊断视频索引脚本", "scripts/build_candidate_diagnostic_video_index.py"),
    ("Grasp-gated trajectory-kNN 运行脚本", "scripts/run_grasp_gated_trajectory_knn_policy.py"),
    ("Grasp-gated trajectory-kNN 评测脚本", "scripts/evaluate_grasp_gated_trajectory_knn.py"),
    ("控制限幅扫表", "docs/control_safety_sweep.md"),
    ("控制限幅扫表 CSV", "docs/control_safety_sweep.csv"),
    ("控制限幅扫表 JSON", "outputs/evaluations/control_safety_sweep_v1.json"),
    ("Action-head/PEFT 阶段报告", "docs/action_head_stage_report.md"),
    ("Action-head/PEFT 阶段 CSV", "docs/action_head_stage_report.csv"),
    ("Action-head 控制限幅扫表", "docs/action_head_control_safety_sweep.md"),
    ("Action-head 控制限幅扫表 CSV", "docs/action_head_control_safety_sweep.csv"),
    ("Action-head 控制限幅扫表 JSON", "outputs/evaluations/action_head_control_safety_sweep_v1.json"),
    ("严格抓取成功口径审计", "docs/strict_grasp_success_audit.md"),
    ("严格抓取成功口径审计 CSV", "docs/strict_grasp_success_audit.csv"),
    ("严格抓取成功口径审计 JSON", "outputs/evaluations/strict_grasp_success_audit_v1.json"),
    ("严格抓取成功口径审计脚本", "scripts/build_strict_grasp_success_audit.py"),
    ("阶段证据总表", "docs/stage_evidence_index.md"),
    ("阶段证据总表 CSV", "docs/stage_evidence_index.csv"),
    ("阶段展示总索引", "docs/stage_showcase_index.md"),
    ("阶段展示总索引 HTML", "docs/stage_showcase_index.html"),
    ("阶段复现实验手册", "docs/stage_reproduction_runbook.md"),
    ("阶段复现实验手册 CSV", "docs/stage_reproduction_runbook.csv"),
    ("研究问题证据映射", "docs/research_evidence_map.md"),
    ("研究问题展示选择表", "docs/research_question_showcase_plan.md"),
    ("研究问题展示选择表 CSV", "docs/research_question_showcase_plan.csv"),
    ("Claim 证据追踪矩阵", "docs/claim_evidence_traceability.md"),
    ("Claim 证据追踪矩阵 CSV", "docs/claim_evidence_traceability.csv"),
    ("Claim 视频播放清单", "docs/claim_video_playback_index.md"),
    ("Claim 视频播放清单 CSV", "docs/claim_video_playback_index.csv"),
    ("总目标完成度审计", "docs/goal_completion_audit.md"),
    ("视频证据索引", "docs/video_evidence_index.md"),
    ("视频质量审计", "docs/video_quality_audit.md"),
    ("视频质量审计 CSV", "docs/video_quality_audit.csv"),
    ("视频证据浏览页", "docs/video_evidence_gallery.html"),
    ("视频展示讲稿与时间线", "docs/video_presentation_storyboard.md"),
    ("视频展示讲稿与时间线 HTML", "docs/video_presentation_storyboard.html"),
    ("答辩视频播放清单", "docs/defense_video_playlist.md"),
    ("答辩视频播放清单 CSV", "docs/defense_video_playlist.csv"),
    ("答辩视频播放清单 HTML", "docs/defense_video_playlist.html"),
    ("答辩视频播放清单脚本", "scripts/build_defense_video_playlist.py"),
    ("答辩视频 Cue Sheet", "docs/defense_video_cue_sheet.md"),
    ("答辩视频 Cue Sheet CSV", "docs/defense_video_cue_sheet.csv"),
    ("答辩视频 Cue Sheet 脚本", "scripts/build_defense_video_cue_sheet.py"),
    ("失败模式分类记录", "docs/failure_mode_taxonomy.md"),
    ("失败模式分类 CSV", "docs/failure_mode_taxonomy.csv"),
    ("论文结果章节草稿", "docs/thesis_results_chapter_draft.md"),
    ("论文附录结果表", "docs/thesis_appendix_tables.md"),
    ("论文方法对比 CSV", "docs/thesis_method_comparison_table.csv"),
    ("论文 domain randomization CSV", "docs/thesis_domain_randomization_table.csv"),
    ("OpenVLA 数据桥接报告", "docs/openvla_dataset_bridge_report.md"),
    ("OpenVLA 数据桥接 JSON", "outputs/evaluations/openvla_dataset_bridge_v1.json"),
    ("OpenVLA 数据桥接 manifest", "data/vla_bridge/openvla_dataset_bridge_v1/manifest.json"),
    ("OpenVLA 数据桥接样本 JSONL", "data/vla_bridge/openvla_dataset_bridge_v1/samples.jsonl"),
    ("OpenVLA 数据桥接预览图", "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png"),
    ("OpenVLA 数据桥接浏览页", "docs/openvla_bridge_gallery.html"),
    ("WidowX MuJoCo RLDS 源数据报告", "docs/widowx_mujoco_rlds_source_v1_report.md"),
    ("WidowX MuJoCo RLDS 源数据 JSON", "outputs/evaluations/widowx_mujoco_rlds_source_v1.json"),
    ("WidowX MuJoCo RLDS 源数据验证报告", "docs/widowx_mujoco_rlds_source_validation_v1.md"),
    ("WidowX MuJoCo RLDS 源数据验证 JSON", "outputs/evaluations/widowx_mujoco_rlds_source_validation_v1.json"),
    ("WidowX MuJoCo RLDS 源数据导出脚本", "scripts/export_openvla_rlds_source.py"),
    ("WidowX MuJoCo RLDS 源数据验证脚本", "scripts/validate_openvla_rlds_source.py"),
    ("WidowX MuJoCo TFDS builder", "scripts/remote_openvla/widowx_mujoco_pick_place_dataset_builder.py"),
    ("OpenVLA 本地可行性检查报告", "docs/openvla_feasibility_report.md"),
    ("OpenVLA 本地可行性检查 JSON", "outputs/evaluations/openvla_feasibility_check_v1.json"),
    ("Robot VLA action-head 交接门禁报告", "docs/robot_vla_action_head_handoff.md"),
    ("Robot VLA action-head 交接门禁 JSON", "outputs/evaluations/robot_vla_action_head_handoff_v1.json"),
    ("Robot VLA 远端运行包报告", "docs/robot_vla_remote_run_pack.md"),
    ("Robot VLA 远端运行包 JSON", "outputs/evaluations/robot_vla_remote_run_pack_v1.json"),
    ("Robot VLA 远端运行包 ZIP", "outputs/robot_vla_remote_run_pack/robot_vla_remote_run_pack_v1.zip"),
    ("Robot VLA 远端运行包脚本", "scripts/build_robot_vla_remote_run_pack.py"),
    ("Robot VLA 远端结果回填门禁报告", "docs/robot_vla_remote_result_intake.md"),
    ("Robot VLA 远端结果回填门禁 CSV", "docs/robot_vla_remote_result_intake.csv"),
    ("Robot VLA 远端结果回填门禁 JSON", "outputs/evaluations/robot_vla_remote_result_intake_v1.json"),
    ("Robot VLA 远端结果回填门禁脚本", "scripts/build_robot_vla_remote_result_intake.py"),
    ("下一阶段实验注册表", "docs/next_experiment_registry.md"),
    ("下一阶段实验注册表 CSV", "docs/next_experiment_registry.csv"),
    ("外部依赖阶段 Readiness Audit", "docs/external_dependency_readiness_audit.md"),
    ("外部依赖阶段 Readiness Audit CSV", "docs/external_dependency_readiness_audit.csv"),
    ("外部依赖阶段 Readiness Audit JSON", "outputs/evaluations/external_dependency_readiness_audit_v1.json"),
    ("外部依赖阶段 Readiness Audit 脚本", "scripts/build_external_dependency_readiness_audit.py"),
    ("答辩现场展示 Runbook", "docs/defense_live_runbook.md"),
    ("答辩现场展示 Runbook CSV", "docs/defense_live_runbook.csv"),
    ("答辩现场展示 Runbook 脚本", "scripts/build_defense_live_runbook.py"),
    ("答辩故事板", "docs/defense_storyboard.md"),
    ("答辩幻灯片大纲", "docs/defense_slide_outline.md"),
    ("本地 dashboard", "docs/experiment_dashboard.html"),
    ("本地展示启动器", "scripts/showcase_launcher.py"),
    ("本地展示启动器说明", "docs/showcase_launcher_guide.md"),
    ("答辩证据包说明", "docs/defense_evidence_pack.md"),
    ("答辩证据包 JSON", "outputs/evaluations/defense_evidence_pack_v1.json"),
    ("答辩证据包 ZIP", "outputs/defense_evidence_pack/defense_evidence_pack_v1.zip"),
    ("答辩证据包脚本", "scripts/build_defense_evidence_pack.py"),
    ("OpenVLA bridge gallery", "docs/openvla_bridge_gallery.html"),
    ("本地答辩 deck", "docs/defense_deck.html"),
    ("答辩视频包说明", "docs/presentation_video_pack.md"),
    ("视频宫格说明", "docs/video_showcase.md"),
    ("下一阶段实施计划", "docs/next_phase_implementation.md"),
]


DISPLAY_ARTIFACTS = [
    ("Ranked-objective preference trajectory post-training 候选诊断视频", "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4"),
    ("Ranked-fast preference trajectory post-training 候选诊断视频", "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4"),
    ("TCP-lift preference trajectory post-training 候选诊断视频", "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.mp4"),
    ("答辩总览视频", "outputs/presentation_clips/00_defense_video_reel.mp4"),
    ("阶段 1 视频", "outputs/presentation_clips/01_task_data_oracle.mp4"),
    ("阶段 2 视频", "outputs/presentation_clips/02_basic_bc_baselines.mp4"),
    ("阶段 3 视频", "outputs/presentation_clips/03_trajectory_act_diffusion.mp4"),
    ("阶段 4 视频", "outputs/presentation_clips/04_action_head_peft_proxy.mp4"),
    ("阶段 5 视频", "outputs/presentation_clips/05_language_generalization.mp4"),
    ("阶段 6 视频", "outputs/presentation_clips/06_domain_randomization_proxy.mp4"),
    ("阶段 7 候选诊断视频", "outputs/presentation_clips/07_candidate_diagnostics.mp4"),
    ("答辩视频 Cue Sheet", "docs/defense_video_cue_sheet.md"),
    ("最终答辩讲解脚本", "docs/final_defense_narrative_script.md"),
    ("剩余实验执行看板", "docs/remaining_experiment_execution_board.md"),
    ("Trajectory/ACT 论文结论摘要", "docs/trajectory_act_conclusion_brief.md"),
    ("Trajectory/ACT 超慢可视化指南", "docs/trajectory_act_slow_viewer_guide.md"),
    ("Preference 后训练消融矩阵", "docs/preference_post_training_ablation_matrix.md"),
    ("核心方法宫格", "outputs/showcase/core_methods_grid.mp4"),
    ("全部方法宫格", "outputs/showcase/all_registered_methods_grid.mp4"),
    ("语言泛化宫格", "outputs/showcase/language_generalization_grid.mp4"),
    ("低摩擦结构化对照视频", "outputs/videos/domain_randomization_structured_low_friction_seed0.mp4"),
    ("低摩擦 trajectory-kNN 对照视频", "outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4"),
    ("低摩擦 Visual ACT-CNN-CVAE 对照视频", "outputs/videos/domain_randomization_visual_act_cnn_cvae_low_friction_seed0.mp4"),
    ("Trajectory phase template BC 候选诊断视频", "outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4"),
    ("Trajectory-prior Residual BC 候选诊断视频", "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4"),
    ("Timing-aware Trajectory-prior Residual BC 候选诊断视频", "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4"),
    ("Grasp-gated trajectory chunk BC 候选诊断视频", "outputs/videos/grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4"),
    ("Grasp-gated Torch ACT 候选诊断视频", "outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4"),
    ("Grasp-gated trajectory-kNN 候选诊断视频", "outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4"),
    ("Contact-aware trajectory-kNN 候选诊断视频", "outputs/videos/contact_aware_trajectory_knn_v1_candidate_seed0.mp4"),
    ("Preference trajectory post-training 候选诊断视频", "outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4"),
    ("Preference + contact-aware trajectory post-training 候选诊断视频", "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4"),
    ("Ranked preference trajectory post-training 候选诊断视频", "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4"),
    ("Phase-weighted Torch ACT 候选诊断视频", "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.mp4"),
    ("Contact/Phase-gated Torch ACT 候选诊断视频", "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.mp4"),
    ("Contact-aware Phase-gated Torch ACT 候选诊断视频", "outputs/videos/contact_aware_phase_gated_torch_act_v1_candidate_seed0.mp4"),
    ("Grasp/Lift 子策略上界诊断视频", "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4"),
    ("Contact-stage subpolicy 候选诊断视频", "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4"),
    ("Contact-stage Demo Torch ACT 候选诊断视频", "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4"),
    ("Contact-stage Phase Action-Head 候选诊断视频", "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4"),
    ("Contact-hold Weighted Torch ACT 候选诊断视频", "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4"),
    ("Gripper timing/contact probe 候选诊断视频", "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.mp4"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a final manifest of registered experiment artifacts.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--core-task-comparison", type=Path, default=ROOT / "docs" / "core_task_comparison_matrix.csv")
    parser.add_argument("--core-v2-comparison", type=Path, default=ROOT / "docs" / "core_v2_holdout_comparison_matrix.csv")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--data-efficiency-summary", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--domain-randomization", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--stage-comparison", type=Path, default=ROOT / "docs" / "stage_comparison_report.csv")
    parser.add_argument("--method-evidence-gate", type=Path, default=ROOT / "docs" / "method_evidence_gate.csv")
    parser.add_argument("--final-method-index", type=Path, default=ROOT / "docs" / "final_method_version_index.csv")
    parser.add_argument("--method-comparison-dashboard", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.csv")
    parser.add_argument("--thesis-visual-evidence", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.csv")
    parser.add_argument("--defense-qa-playbook", type=Path, default=ROOT / "docs" / "defense_qa_playbook.csv")
    parser.add_argument("--version-lineage", type=Path, default=ROOT / "docs" / "version_lineage_index.csv")
    parser.add_argument("--task-bc-stage", type=Path, default=ROOT / "docs" / "task_bc_stage_report.csv")
    parser.add_argument("--trajectory-act-stage", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--trajectory-act-record", type=Path, default=ROOT / "docs" / "trajectory_act_experiment_record.csv")
    parser.add_argument("--trajectory-act-diagnosis", type=Path, default=ROOT / "docs" / "trajectory_act_failure_diagnosis.csv")
    parser.add_argument("--trajectory-act-conclusion", type=Path, default=ROOT / "docs" / "trajectory_act_conclusion_brief.csv")
    parser.add_argument("--trajectory-act-slow-viewer", type=Path, default=ROOT / "docs" / "trajectory_act_slow_viewer_guide.csv")
    parser.add_argument("--preference-post-training-ablation", type=Path, default=ROOT / "docs" / "preference_post_training_ablation_matrix.csv")
    parser.add_argument("--preference-ranked-objective", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_objective_report.csv")
    parser.add_argument("--preference-ranked-fast", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_fast_report.csv")
    parser.add_argument("--timing-aware-trajectory-prior-residual", type=Path, default=ROOT / "docs" / "timing_aware_trajectory_prior_residual_bc_report.csv")
    parser.add_argument("--gripper-timing-contact-probe", type=Path, default=ROOT / "docs" / "gripper_timing_contact_probe_report.csv")
    parser.add_argument("--control-safety-sweep", type=Path, default=ROOT / "docs" / "control_safety_sweep.csv")
    parser.add_argument("--action-head-stage", type=Path, default=ROOT / "docs" / "action_head_stage_report.csv")
    parser.add_argument("--action-head-control-safety-sweep", type=Path, default=ROOT / "docs" / "action_head_control_safety_sweep.csv")
    parser.add_argument("--strict-grasp-audit", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.csv")
    parser.add_argument("--strict-grasp-audit-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--stage-evidence-index", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--research-evidence", type=Path, default=ROOT / "docs" / "research_evidence_map.csv")
    parser.add_argument("--research-showcase", type=Path, default=ROOT / "docs" / "research_question_showcase_plan.csv")
    parser.add_argument("--claim-evidence", type=Path, default=ROOT / "docs" / "claim_evidence_traceability.csv")
    parser.add_argument("--claim-video-playback", type=Path, default=ROOT / "docs" / "claim_video_playback_index.csv")
    parser.add_argument("--goal-completion", type=Path, default=ROOT / "docs" / "goal_completion_audit.csv")
    parser.add_argument("--external-dependency-readiness", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.csv")
    parser.add_argument("--external-dependency-readiness-json", type=Path, default=ROOT / "outputs" / "evaluations" / "external_dependency_readiness_audit_v1.json")
    parser.add_argument("--defense-live-runbook", type=Path, default=ROOT / "docs" / "defense_live_runbook.csv")
    parser.add_argument("--defense-video-playlist", type=Path, default=ROOT / "docs" / "defense_video_playlist.csv")
    parser.add_argument("--defense-video-cue-sheet", type=Path, default=ROOT / "docs" / "defense_video_cue_sheet.csv")
    parser.add_argument("--final-defense-narrative", type=Path, default=ROOT / "docs" / "final_defense_narrative_script.csv")
    parser.add_argument("--remaining-experiment-board", type=Path, default=ROOT / "docs" / "remaining_experiment_execution_board.csv")
    parser.add_argument("--video-evidence", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--video-quality-audit", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--failure-mode-taxonomy", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.csv")
    parser.add_argument("--presentation-manifest", type=Path, default=ROOT / "outputs" / "presentation_clips" / "presentation_video_pack_manifest.json")
    parser.add_argument("--defense-evidence-pack-json", type=Path, default=ROOT / "outputs" / "evaluations" / "defense_evidence_pack_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "final_artifact_manifest.md")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def artifact_entry(role: str, path_text: str) -> dict[str, object]:
    path = ROOT / path_text
    return {
        "role": role,
        "path": path_text.replace("\\", "/"),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def method_entries(methods: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "version": method["version"],
            "stage": method["stage"],
            "method": method["method"],
            "artifact": method["artifact"],
            "clip": method["clip"],
            "train_range_success": method["train_range_success"],
            "heldout_success": method["heldout_success"],
        }
        for method in methods
    ]


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    versions = read_json(args.versions)
    methods = versions["methods"]
    language_rows = read_csv(args.language_summary)
    resource_rows = read_csv(args.resource_summary)
    data_efficiency_rows = read_csv(args.data_efficiency_summary)
    domain_randomization_rows = read_csv(args.domain_randomization)
    stage_rows = read_csv(args.stage_comparison)
    method_evidence_rows = read_csv(args.method_evidence_gate)
    final_method_rows = read_csv(args.final_method_index)
    method_comparison_rows = read_csv(args.method_comparison_dashboard)
    core_task_comparison_rows = read_csv(args.core_task_comparison)
    core_v2_comparison_rows = read_csv(args.core_v2_comparison)
    thesis_visual_evidence_rows = read_csv(args.thesis_visual_evidence)
    defense_qa_rows = read_csv(args.defense_qa_playbook)
    version_lineage_rows = read_csv(args.version_lineage)
    task_bc_rows = read_csv(args.task_bc_stage)
    trajectory_act_rows = read_csv(args.trajectory_act_stage)
    trajectory_act_record_rows = read_csv(args.trajectory_act_record)
    trajectory_act_diagnosis_rows = read_csv(args.trajectory_act_diagnosis)
    trajectory_act_conclusion_rows = read_csv(args.trajectory_act_conclusion)
    trajectory_act_slow_viewer_rows = read_csv(args.trajectory_act_slow_viewer)
    preference_ablation_rows = read_csv(args.preference_post_training_ablation)
    preference_ranked_objective_rows = read_csv(args.preference_ranked_objective)
    preference_ranked_fast_rows = read_csv(args.preference_ranked_fast)
    timing_aware_trajectory_rows = read_csv(args.timing_aware_trajectory_prior_residual)
    gripper_timing_rows = read_csv(args.gripper_timing_contact_probe)
    control_safety_rows = read_csv(args.control_safety_sweep)
    action_head_rows = read_csv(args.action_head_stage)
    action_head_control_safety_rows = read_csv(args.action_head_control_safety_sweep)
    strict_grasp_rows = read_csv(args.strict_grasp_audit)
    strict_grasp = read_json(args.strict_grasp_audit_json)
    strict_grasp_summary = strict_grasp.get("summary", {})
    stage_evidence_rows = read_csv(args.stage_evidence_index)
    research_rows = read_csv(args.research_evidence)
    research_showcase_rows = read_csv(args.research_showcase)
    claim_evidence_rows = read_csv(args.claim_evidence)
    claim_video_playback_rows = read_csv(args.claim_video_playback)
    goal_rows = read_csv(args.goal_completion)
    external_dependency_rows = read_csv(args.external_dependency_readiness)
    external_dependency_readiness = read_json(args.external_dependency_readiness_json)
    defense_live_rows = read_csv(args.defense_live_runbook)
    defense_video_playlist_rows = read_csv(args.defense_video_playlist)
    defense_video_cue_sheet_rows = read_csv(args.defense_video_cue_sheet)
    final_defense_narrative_rows = read_csv(args.final_defense_narrative)
    remaining_experiment_board_rows = read_csv(args.remaining_experiment_board)
    video_rows = read_csv(args.video_evidence)
    video_quality_rows = read_csv(args.video_quality_audit)
    failure_mode_rows = read_csv(args.failure_mode_taxonomy)
    presentation = read_json(args.presentation_manifest)
    defense_evidence_pack = read_json(args.defense_evidence_pack_json)
    openvla_bridge = read_json(ROOT / "outputs" / "evaluations" / "openvla_dataset_bridge_v1.json")
    rlds_source = read_json(ROOT / "outputs" / "evaluations" / "widowx_mujoco_rlds_source_v1.json")
    rlds_source_validation = read_json(ROOT / "outputs" / "evaluations" / "widowx_mujoco_rlds_source_validation_v1.json")
    openvla_feasibility = read_json(ROOT / "outputs" / "evaluations" / "openvla_feasibility_check_v1.json")
    robot_vla_handoff = read_json(ROOT / "outputs" / "evaluations" / "robot_vla_action_head_handoff_v1.json")
    robot_vla_remote_pack = read_json(ROOT / "outputs" / "evaluations" / "robot_vla_remote_run_pack_v1.json")
    robot_vla_remote_intake = read_json(ROOT / "outputs" / "evaluations" / "robot_vla_remote_result_intake_v1.json")
    preference_upgrade_gate = read_json(ROOT / "outputs" / "evaluations" / "preference_post_training_upgrade_gate_v1.json")
    isaac_handoff = read_json(ROOT / "outputs" / "evaluations" / "isaac_domain_randomization_handoff_v1.json")
    real_widowx_handoff = read_json(ROOT / "outputs" / "evaluations" / "real_widowx_validation_handoff_v1.json")

    core = [artifact_entry(role, path) for role, path in CORE_ARTIFACTS]
    display = [artifact_entry(role, path) for role, path in DISPLAY_ARTIFACTS]
    registered_video_entries = [artifact_entry(f"单方法视频：{method['version']}", method["clip"]) for method in methods]

    return {
        "version": "final_artifact_manifest_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "按照计划进行实验，保留各部分版本名称，并支持不同方法和阶段的说明、评测比较与仿真视频片段展示。",
        "task": versions.get("task", ""),
        "complexity": versions.get("complexity", ""),
        "counts": {
            "methods": len(methods),
            "language_rows": len(language_rows),
            "resource_rows": len(resource_rows),
            "data_efficiency_rows": len(data_efficiency_rows),
            "domain_randomization_rows": len(domain_randomization_rows),
            "isaac_handoff_checked": 1 if isaac_handoff.get("version") == "isaac_domain_randomization_handoff_v1" else 0,
            "isaac_handoff_rows": len(isaac_handoff.get("rows", [])),
            "isaac_handoff_required_files": len(isaac_handoff.get("required_return_files", [])),
            "real_widowx_handoff_checked": 1 if real_widowx_handoff.get("version") == "real_widowx_validation_handoff_v1" else 0,
            "real_widowx_handoff_rows": len(real_widowx_handoff.get("rows", [])),
            "real_widowx_trial_template_rows": int(real_widowx_handoff.get("trial_template_rows", 0)),
            "real_widowx_required_files": len(real_widowx_handoff.get("required_return_files", [])),
            "stage_groups": len(stage_rows),
            "method_evidence_rows": len(method_evidence_rows),
            "final_method_index_rows": len(final_method_rows),
            "method_comparison_rows": len(method_comparison_rows),
            "core_task_comparison_rows": len(core_task_comparison_rows),
            "core_v2_holdout_comparison_rows": len(core_v2_comparison_rows),
            "thesis_visual_evidence_rows": len(thesis_visual_evidence_rows),
            "defense_qa_rows": len(defense_qa_rows),
            "version_lineage_rows": len(version_lineage_rows),
            "task_bc_stage_rows": len(task_bc_rows),
            "trajectory_act_stage_rows": len(trajectory_act_rows),
            "trajectory_act_record_rows": len(trajectory_act_record_rows),
            "trajectory_act_diagnosis_rows": len(trajectory_act_diagnosis_rows),
            "trajectory_act_conclusion_rows": len(trajectory_act_conclusion_rows),
            "trajectory_act_slow_viewer_rows": len(trajectory_act_slow_viewer_rows),
            "preference_post_training_ablation_rows": len(preference_ablation_rows),
            "preference_ranked_objective_rows": len(preference_ranked_objective_rows),
            "preference_ranked_fast_rows": len(preference_ranked_fast_rows),
            "timing_aware_trajectory_prior_residual_rows": len(timing_aware_trajectory_rows),
            "gripper_timing_contact_probe_rows": len(gripper_timing_rows),
            "control_safety_sweep_rows": len(control_safety_rows),
            "action_head_stage_rows": len(action_head_rows),
            "action_head_control_safety_sweep_rows": len(action_head_control_safety_rows),
            "strict_grasp_audit_rows": len(strict_grasp_rows),
            "strict_grasp_audit_loose_successes": int(strict_grasp_summary.get("loose_successes", 0)),
            "strict_grasp_audit_strict_successes": int(strict_grasp_summary.get("strict_grasp_successes", 0)),
            "strict_grasp_audit_loose_without_grasp_rows": int(strict_grasp_summary.get("loose_without_grasp_rows", 0)),
            "stage_evidence_rows": len(stage_evidence_rows),
            "research_questions": len(research_rows),
            "research_showcase_rows": len(research_showcase_rows),
            "claim_evidence_rows": len(claim_evidence_rows),
            "claim_video_playback_rows": len(claim_video_playback_rows),
            "goal_completion_rows": len(goal_rows),
            "external_dependency_readiness_checked": 1 if external_dependency_readiness.get("version") == "external_dependency_readiness_audit_v1" else 0,
            "external_dependency_readiness_rows": len(external_dependency_rows),
            "external_dependency_waiting_remote_result": int(external_dependency_readiness.get("readiness_counts", {}).get("waiting_remote_result", 0)),
            "external_dependency_waiting_isaac_runtime": int(external_dependency_readiness.get("readiness_counts", {}).get("waiting_isaac_runtime", 0)),
            "external_dependency_waiting_real_robot_trials": int(external_dependency_readiness.get("readiness_counts", {}).get("waiting_real_robot_trials", 0)),
            "defense_live_runbook_rows": len(defense_live_rows),
            "defense_video_playlist_rows": len(defense_video_playlist_rows),
            "defense_video_cue_sheet_rows": len(defense_video_cue_sheet_rows),
            "final_defense_narrative_rows": len(final_defense_narrative_rows),
            "remaining_experiment_board_rows": len(remaining_experiment_board_rows),
            "video_evidence_rows": len(video_rows),
            "video_quality_rows": len(video_quality_rows),
            "failure_mode_rows": len(failure_mode_rows),
            "registered_method_videos": len(registered_video_entries),
            "presentation_pack_items": 1 + len(presentation.get("stages", [])),
            "display_artifacts": len(display),
            "defense_evidence_pack_checked": 1 if defense_evidence_pack.get("version") == "defense_evidence_pack_v1" else 0,
            "defense_evidence_pack_files": int(defense_evidence_pack.get("file_count", 0)),
            "defense_evidence_pack_archive_bytes": int(defense_evidence_pack.get("archive_size_bytes", 0)),
            "openvla_bridge_samples": int(openvla_bridge.get("samples_exported", 0)),
            "openvla_bridge_gallery": 1 if (ROOT / "docs" / "openvla_bridge_gallery.html").exists() else 0,
            "rlds_source_checked": 1 if rlds_source.get("version") == "widowx_mujoco_rlds_source_v1" else 0,
            "rlds_source_episodes": int(rlds_source.get("episodes_exported", 0)),
            "rlds_source_steps": int(rlds_source.get("steps_exported", 0)),
            "rlds_source_validation_checked": 1 if rlds_source_validation.get("version") == "widowx_mujoco_rlds_source_validation_v1" else 0,
            "openvla_feasibility_checked": 1 if openvla_feasibility.get("version") == "openvla_feasibility_check_v1" else 0,
            "robot_vla_handoff_checked": 1 if robot_vla_handoff.get("version") == "robot_vla_action_head_handoff_v1" else 0,
            "robot_vla_remote_pack_checked": 1 if robot_vla_remote_pack.get("version") == "robot_vla_remote_run_pack_v1" else 0,
            "robot_vla_remote_pack_files": int(robot_vla_remote_pack.get("packaged_file_count", 0)),
            "robot_vla_remote_intake_checked": 1 if robot_vla_remote_intake.get("version") == "robot_vla_remote_result_intake_v1" else 0,
            "robot_vla_remote_intake_returned_files": int(robot_vla_remote_intake.get("returned_files_present", 0)),
            "preference_post_training_upgrade_gate_rows": int(preference_upgrade_gate.get("candidate_count", 0)),
            "preference_post_training_formal_upgrade_allowed": int(preference_upgrade_gate.get("formal_upgrade_allowed_count", 0)),
        },
        "methods": method_entries(methods),
        "core_artifacts": core,
        "display_artifacts": display,
        "registered_method_videos": registered_video_entries,
        "rebuild_commands": [
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_domain_randomization.py"}" --episodes 2 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_isaac_domain_randomization_handoff.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_real_widowx_validation_handoff.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "summarize_experiments.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "summarize_model_resources.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_method_stage_audit.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_method_evidence_gate.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_final_method_version_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_method_comparison_dashboard.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_core_task_comparison_matrix.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_core_v2_comparison_matrix.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_thesis_visual_evidence_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_qa_playbook.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_version_lineage_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_stage_comparison_report.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_task_bc_stage_report.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_stage_report.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_experiment_record.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_failure_diagnosis.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_conclusion_brief.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_trajectory_act_slow_viewer_guide.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_final_defense_narrative_script.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_remaining_experiment_execution_board.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_trajectory_phase_template_bc.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --bins 128 --sample-stride 4 --ridge 0.001 --min-bin-samples 64 --feature-mode planned',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_trajectory_phase_template_bc.py"}" --model "{ROOT / "outputs" / "trajectory_phase_template_bc" / "trajectory_phase_template_bc_20260720_160007.npz"}" --episodes 5 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_trajectory_prior_residual_bc.py"}" --run-dir "{ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1"}" --sample-stride 4 --ridge 0.001 --model-prefix trajectory_prior_residual_bc_v1_candidate',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_trajectory_prior_residual_bc.py"}" --model "{ROOT / "outputs" / "trajectory_prior_residual_bc" / "trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz"}" --episodes 5 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method trajectory_prior_residual_bc --version trajectory_prior_residual_bc_v1_candidate --model "{ROOT / "outputs" / "trajectory_prior_residual_bc" / "trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz"}" --task place_blue_cube_blue_pad --complexity medium --seed 0 --steps 2840 --camera top_rgb --fps 24 --frame-stride 12 --width 640 --height 480 --gripper-kp 900 --gripper-force 180 --friction 3.0 --action-alpha 1.0 --max-arm-delta 0.02 --max-gripper-delta 0.0008 --log-every 0',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_timing_aware_trajectory_prior_residual_bc.py"}" --run-dir "{ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1"}" --sample-stride 4 --ridge 0.001 --model-prefix timing_aware_trajectory_prior_residual_bc_v1_candidate',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_timing_aware_trajectory_prior_residual_bc.py"}" --model "{ROOT / "outputs" / "timing_aware_trajectory_prior_residual_bc" / "timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz"}" --steps 3500 --residual-scale 0.02',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method timing_aware_trajectory_prior_residual_bc --version timing_aware_trajectory_prior_residual_bc_v1_candidate --model "{ROOT / "outputs" / "timing_aware_trajectory_prior_residual_bc" / "timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz"}" --task place_blue_cube_blue_pad --complexity medium --seed 0 --steps 3500 --camera top_rgb --fps 24 --frame-stride 12 --width 640 --height 480 --gripper-kp 900 --gripper-force 180 --friction 3.0 --residual-scale 0.02 --action-alpha 1.0 --max-arm-delta 0.02 --max-gripper-delta 0.0008 --log-every 0',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_grasp_gated_trajectory_act.py"}" --episodes 5',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method grasp_gated_trajectory_chunk_bc --task place_blue_cube_blue_pad --complexity medium --seed 0 --steps 2840 --camera top_rgb --fps 24 --frame-stride 12 --width 640 --height 480 --gripper-kp 900 --gripper-force 180 --friction 3.0 --log-every 0',
            f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"; & "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method grasp_gated_torch_act --task place_blue_cube_blue_pad --complexity medium --seed 0 --steps 2840 --camera top_rgb --fps 24 --frame-stride 12 --width 640 --height 480 --gripper-kp 900 --gripper-force 180 --friction 3.0 --log-every 0',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_torch_act.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --horizon 8 --history 8 --sample-stride 16 --d-model 64 --nhead 4 --encoder-layers 2 --decoder-layers 2 --dim-feedforward 128 --epochs 8 --batch-size 256 --lr 0.0003 --gripper-loss-weight 8 --phase-one-hot --phase-loss-weights "grasp:5,lift:5,transfer:2,place_release:3" --model-prefix contact_phase_gated_torch_act_v1_candidate',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_contact_phase_gated_torch_act.py"}" --model "{ROOT / "outputs" / "torch_act" / "contact_phase_gated_torch_act_v1_candidate_20260721_003304.pt"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_grasp_gated_trajectory_knn.py"}" --model "{ROOT / "outputs" / "trajectory_knn_bc" / "trajectory_knn_chunk_bc_20260720_053423.npz"}" --episodes 5 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_preference_trajectory_post_training.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --horizon 8 --history 8 --sample-stride 16 --no-augment-relative',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_preference_trajectory_post_training.py"}" --model "{ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_20260720_165005.npz"}" --episodes 5 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_preference_trajectory_post_training.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --model-prefix preference_trajectory_post_training_v1_ranked_objective_candidate --version preference_trajectory_post_training_v1_ranked_objective_candidate --horizon 8 --history 8 --sample-stride 8 --preference-mode episode_rank --rank-decay 0.40 --success-multiplier 3.0 --failed-attempt-multiplier 0.25 --failed-episode-multiplier 0.15 --out-of-table-multiplier 0.10 --min-preference 0.02 --no-augment-relative',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_preference_trajectory_post_training.py"}" --model "{ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz"}" --version preference_trajectory_post_training_v1_ranked_objective_candidate --episodes 5 --steps 2840 --output-json "{ROOT / "outputs" / "evaluations" / "preference_trajectory_post_training_v1_ranked_objective_candidate.json"}" --output-csv "{ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_objective_report.csv"}" --output-md "{ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_objective_report.md"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method preference_trajectory_post_training --version preference_trajectory_post_training_v1_ranked_objective_candidate --model "{ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz"}" --task place_blue_cube_blue_pad --complexity medium --seed 0 --steps 2840 --camera top_rgb --fps 24 --frame-stride 12 --width 640 --height 480 --arm-kp 150 --arm-force 100 --gripper-kp 800 --gripper-force 140 --friction 3.0 --log-every 0',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_preference_trajectory_post_training.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --model-prefix preference_trajectory_post_training_v1_ranked_fast_candidate --version preference_trajectory_post_training_v1_ranked_fast_candidate --horizon 8 --history 8 --sample-stride 32 --preference-mode episode_rank --rank-decay 0.40 --success-multiplier 3.0 --failed-attempt-multiplier 0.25 --failed-episode-multiplier 0.15 --out-of-table-multiplier 0.10 --min-preference 0.02 --no-augment-relative',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_preference_trajectory_post_training.py"}" --model "{ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz"}" --version preference_trajectory_post_training_v1_ranked_fast_candidate --episodes 5 --steps 2840 --output-json "{ROOT / "outputs" / "evaluations" / "preference_trajectory_post_training_v1_ranked_fast_candidate.json"}" --output-csv "{ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_fast_report.csv"}" --output-md "{ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_fast_report.md"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_video.py"}" --method preference_trajectory_post_training --version preference_trajectory_post_training_v1_ranked_fast_candidate --model "{ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz"}" --task place_blue_cube_blue_pad --complexity medium --seed 0 --steps 2840 --camera top_rgb --fps 24 --frame-stride 12 --width 640 --height 480 --arm-kp 150 --arm-force 100 --gripper-kp 800 --gripper-force 140 --friction 3.0 --log-every 0',
            f'& "{PYTHON}" "{ROOT / "scripts" / "train_preference_trajectory_post_training.py"}" --run-dir "{ROOT / "data" / "demos" / "place_blue_cube_blue_pad_medium_20260702_051752"}" --horizon 8 --history 8 --sample-stride 16 --augment-relative --model-prefix preference_contact_aware_trajectory_post_training --version preference_contact_aware_trajectory_post_training_v1_candidate',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_preference_contact_aware_trajectory_post_training.py"}" --model "{ROOT / "outputs" / "preference_post_training" / "preference_contact_aware_trajectory_post_training_20260721_000449.npz"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_preference_post_training_upgrade_gate.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_preference_post_training_ablation_matrix.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_gripper_timing_probe.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_candidate_diagnostic_video_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_control_safety_sweep.py"}" --episodes 2 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_action_head_stage_report.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "evaluate_action_head_control_safety_sweep.py"}" --episodes 2 --steps 2840',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_strict_grasp_success_audit.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_stage_evidence_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_stage_showcase_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_stage_reproduction_runbook.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_video_presentation_storyboard.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_video_evidence_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_video_quality_audit.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_video_evidence_gallery.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_failure_mode_taxonomy.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_research_evidence_map.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_research_question_showcase_plan.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_claim_evidence_traceability.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_claim_video_playback_index.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_goal_completion_audit.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_final_closure_audit.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_live_runbook.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_video_playlist.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_video_cue_sheet.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_deck_html.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_thesis_appendix_tables.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_openvla_dataset_bridge.py"}" --episodes 6 --steps-per-episode 12 --image-size 128',
            f'& "{PYTHON}" "{ROOT / "scripts" / "export_openvla_rlds_source.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "validate_openvla_rlds_source.py"}"',
            f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"; & "{PYTHON}" "{ROOT / "scripts" / "check_openvla_feasibility.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_openvla_bridge_gallery.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_robot_vla_action_head_handoff.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_robot_vla_remote_run_pack.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_robot_vla_remote_result_intake.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_next_experiment_registry.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_external_dependency_readiness_audit.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "showcase_launcher.py"}" --list quick',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_evidence_pack.py"}"',
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_final_artifact_manifest.py"}"',
        ],
        "verification_command": f'$env:VLA_TORCH_PACKAGE_DIR="D:\\vla_torch_cuda_pkgs"; & "{PYTHON}" "{ROOT / "scripts" / "verify_experiment_artifacts.py"}"',
    }


def missing_artifacts(manifest: dict[str, object]) -> list[str]:
    missing: list[str] = []
    for section in ("core_artifacts", "display_artifacts", "registered_method_videos"):
        for item in manifest[section]:  # type: ignore[index]
            if not item["exists"]:  # type: ignore[index]
                missing.append(str(item["path"]))  # type: ignore[index]
    return missing


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_md(path: Path, manifest: dict[str, object]) -> None:
    counts = manifest["counts"]  # type: ignore[assignment]
    assert isinstance(counts, dict)
    missing = missing_artifacts(manifest)
    lines = [
        "# 最终实验 Artifact Manifest",
        "",
        "版本：`final_artifact_manifest_v1`",
        "",
        "用途：把当前实验登记、评测结果、阶段说明、视频证据、答辩展示入口和验证命令集中成一个总清单。后续新增方法时，先更新各评测表和视频，再重建本 manifest。",
        "",
        "最终展示与交付 Handoff 索引包含 `final_showcase_handoff_v1`，用于把版本名称、方法/阶段说明、评测比较、仿真视频片段、viewer 可视化和外部依赖入口压缩成一页式最短路径。",
        "",
        "最终答辩讲解脚本包含 `final_defense_narrative_script_v1`，用于把研究问题、阶段方法、量化指标、视频证据和论文边界整理成一条中文答辩讲解顺序。",
        "",
        "剩余实验执行看板包含 `remaining_experiment_execution_board_v1`，用于把还不能写成正式结果的 planned/readiness 实验整理为优先级、执行环境、回填工件、升级门槛和论文红线。",
        "",
        "方法级证据包含 `method_evidence_gate_v1`，用于逐个检查正式方法版本是否具备 artifact、评测、资源记录、固定视频、视频审计、viewer 命令和论文红线。",
        "",
        "版本命名与入包门禁规范包含 `version_naming_and_gate_spec_v1`，用于固定正式方法、候选诊断、前置门禁、planned 外部版本、视频命名和 planned→formal 升级规则。",
        "",
        "最终方法版本索引包含 `final_method_version_index_v1`，用于把 25 个正式方法的最终版说明、固定视频和慢速 viewer 命令集中到一个中文入口。",
        "",
        "论文图表与视频证据索引包含 `thesis_visual_evidence_index_v1`，用于把论文图、附录表、答辩 HTML、阶段视频和候选诊断视频映射到中文图注、支撑结论和论文红线。",
        "",
        "答辩追问 Q&A Playbook 包含 `defense_qa_playbook_v1`，用于把常见追问映射到推荐回答、首选证据、视频/图表和必须坚持的论文边界。",
        "",
        "实验版本谱系索引包含 `version_lineage_index_v1`，用于把数据集、正式方法、候选诊断、前置门禁和计划版本区分到同一张谱系表。",
        "",
        "`widowx_mujoco_rlds_source_v1` 包含 79 条 Core V2 成功 episode 和 2,528 个 joint-delta source step，`widowx_mujoco_rlds_source_validation_v1` 已验证图像、8D state/action 和终止标记；它们仍不是已注册 RLDS 或 OpenVLA 训练结果。",
        "",
        "Robot VLA 远端运行包包含 `robot_vla_remote_run_pack_v1`，用于把已验证 RLDS source、TFDS builder、handoff 契约、远端命令模板、结果回填 schema 和 zip 归档固定下来；它不是策略结果。",
        "",
        "Robot VLA 远端结果回填门禁包含 `robot_vla_remote_result_intake_v1`，用于检查远端返回的模型、feature cache、评测 JSON、视频和报告是否足够把 planned 方法登记为正式版本；它不是策略结果。",
        "",
        "Isaac domain randomization 交接门禁包含 `isaac_domain_randomization_handoff_v1`，用于把 MuJoCo 扰动域、桌面场景契约、Isaac 回填指标、必须输出文件和论文红线固定下来；它不是 Isaac 运行结果。",
        "",
        "真实 WidowX 验证交接门禁包含 `real_widowx_validation_handoff_v1`，用于把真实机械臂安全门禁、trial 字段、20-50 次记录模板、视频回填和论文红线固定下来；它不是真实机械臂运行结果。",
        "",
        "外部依赖阶段 Readiness Audit 包含 `external_dependency_readiness_audit_v1`，用于统一检查真实 robot VLA、Isaac 和真实 WidowX planned 版本当前是等待远端结果、等待 Isaac runtime、等待真实 trial，还是只能作为前置门禁；它不是策略成功率结果。",
        "",
        f"阶段复现入口包含 `stage_reproduction_runbook_v1`，用于按 {counts.get('stage_evidence_rows', 0)} 个阶段查找代表 viewer 命令、量化证据、视频证据、重建命令和论文红线。",
        "",
        "Trajectory/ACT 诊断入口包含 `trajectory_act_failure_diagnosis_v1`，用于基于固定视频元数据拆分接触、夹紧、抬升、泛化和动作速度问题。",
        "",
        "Trajectory/ACT 中文实验台账包含 `trajectory_act_experiment_record_v1`，用于把 9 个 trajectory-conditioned BC / ACT 版本的中文记录、最终模型、主任务/语言 viewer 命令、训练命令和论文红线固定下来。",
        "",
        "Trajectory/ACT 论文结论摘要包含 `trajectory_act_conclusion_brief_v1`，用于把 trajectory-conditioned BC / ACT-style baseline 的版本名称、评测结果、视频证据、可写结论和论文红线压缩成论文/答辩可引用入口。",
        "",
        "Trajectory/ACT 超慢可视化指南包含 `trajectory_act_slow_viewer_guide_v1`，用于把 trajectory-conditioned BC / ACT 阶段的标准慢速和超慢学习 viewer 命令集中登记；该指南不新增量化评测结果。",
        "",
        "Trajectory phase template BC 候选实验包含 `trajectory_phase_template_bc_v1_candidate`，用于记录显式 phase 模板仍不能稳定解决抓取/抬升的负例；它不是正式可靠 ACT baseline。",
        "",
        "Trajectory-prior Residual BC 候选实验包含 `trajectory_prior_residual_bc_v1_candidate`，用于记录分阶段轨迹先验 + residual BC action head 可以显著提高放置和 TCP 抬升，但它不是纯 BC、完整官方 ACT 或 VLA 后训练。",
        "",
        "Timing-aware Trajectory-prior Residual BC 候选实验包含 `timing_aware_trajectory_prior_residual_bc_v1_candidate`，用于记录把强闭合、闭合保持和抬升保持并入轨迹先验后，在 residual-scale=0.02 下达到 train-range 5/5、held-out 4/5，但严格抓取仍为 0/10；它不是完整官方 ACT、OpenVLA/OFT 或真实 WidowX 结果。",
        "",
        "Grasp-gated trajectory/ACT 候选诊断包含 `grasp_gated_trajectory_act_v1_candidate`，用于记录保守夹爪门控、慢速动作和更强夹爪力仍不能把 trajectory-conditioned BC 或 Torch ACT 转化为稳定抓取策略。",
        "",
        "Phase-weighted Torch ACT 候选诊断包含 `phase_weighted_torch_act_v1_candidate`，用于记录 grasp/lift/place_release 阶段 loss 加权降低离线误差但没有转化为闭环抓取和抬升成功；它不是正式方法。",
        "",
        "Grasp/Lift 子策略上界诊断包含 `grasp_lift_subpolicy_probe_v1_candidate`，用于记录 scripted expert / IK 控制上界可以完成放置和 TCP 抬升，但标准严格抓取口径仍需单独报告；它不是 learned baseline。",
        "",
        "Grasp-gated trajectory-kNN 候选实验包含 `grasp_gated_trajectory_knn_v1_candidate`，用于记录单独夹爪门控不能把 trajectory-kNN 转化为真实抓取策略的负例；它不是正式可靠 ACT baseline。",
        "",
        "Preference post-training 正式升级门禁包含 `preference_post_training_upgrade_gate_v1`，用于审计三个偏好后训练候选是否足以把 planned 的 `preference_trajectory_post_training_v1` 升级为正式方法；当前结论是不能升级 formal。",
        "",
        "Preference 后训练消融矩阵包含 `preference_post_training_ablation_matrix_v1`，用于把三个 preference post-training 候选的偏好来源、权重策略、TCP 抬升、严格抓取和下一轮 objective 设计方向统一到中文矩阵。",
        "",
        "Gripper timing/contact probe 包含 `gripper_timing_contact_probe_v1_candidate`，用于把夹爪闭合时序、闭合后保持、TCP 抬升和标准严格抓取口径差异整理成 ACT/trajectory 后续 objective 设计依据；它不是 learned BC、ACT、Diffusion Policy 或 VLA 方法。",
        "",
        "控制限幅扫表包含 `control_safety_sweep_v1`，用于量化检查更慢动作平滑和更小增量是否能改善 trajectory/ACT 代表方法。",
        "",
        "Action-head 控制限幅扫表包含 `action_head_control_safety_sweep_v1`，用于量化检查更慢动作平滑和更小增量是否能改善 action-head/PEFT proxy。",
        "",
        "严格抓取成功口径审计包含 `strict_grasp_success_audit_v1`，用于同时报告原始放置成功、`grasp_success` 和 `object_z`，避免把推到目标区域附近误写成稳定抓取放置。",
        "",
        "研究问题展示入口包含 `research_question_showcase_plan_v1`，用于按研究问题选择图表、视频片段、辅助入口、建议讲稿和论文红线。",
        "",
        "论文 claim 追踪入口包含 `claim_evidence_traceability_v1`，用于把可写结论逐条绑定到量化证据、视频证据和论文红线。",
        "",
        "Claim 视频播放入口包含 `claim_video_playback_index_v1`，用于把每条 claim 绑定到首选播放命令、辅助入口、讲稿提示和论文红线。",
        "",
        "答辩现场展示 Runbook 包含 `defense_live_runbook_v1`，用于把开场前检查、推荐播放顺序、现场追问 viewer 和必须坚持的论文红线整理为一页式操作清单。",
        "",
        "答辩视频 Cue Sheet 包含 `defense_video_cue_sheet_v1`，用于把播放顺序、建议时间点、打开命令、备用 viewer 命令、讲解提示、证据引用和论文红线整理成现场可执行清单。",
        "",
        "本地展示启动器包含 `showcase_launcher_v1`，用于按 quick、claim、stage、method 和 candidate 打开视频、文档或 MuJoCo viewer。",
        "",
        "答辩证据包包含 `defense_evidence_pack_v1`，用于把当前 MuJoCo 实验包的中文文档、HTML 入口、CSV/JSON 量化表、论文图表和仿真视频集中到可复制目录与 zip；它是证据归档，不代表真实 OpenVLA、Isaac 或真实 WidowX 已完成。",
        "",
        "视频展示证据额外包含 `video_quality_audit_v1`，用于确认 `docs/video_evidence_index.csv` 中登记的视频可播放、元数据存在、时长和分辨率满足展示审计门槛；它不是成功率评测。",
        "",
        "## 1. 覆盖统计",
        "",
        md_row(["项目", "数量"]),
        md_row(["---", "---:"]),
    ]
    for key, value in counts.items():
        lines.append(md_row([key, str(value)]))

    lines.extend(
        [
            "",
            "## 2. 核心交付物",
            "",
            md_row(["角色", "路径", "大小"]),
            md_row(["---", "---", "---:"]),
        ]
    )
    for item in manifest["core_artifacts"]:  # type: ignore[index]
        lines.append(md_row([item["role"], f"`{item['path']}`", str(item["size_bytes"])]))

    lines.extend(
        [
            "",
            "## 3. 展示入口",
            "",
            md_row(["角色", "路径", "大小"]),
            md_row(["---", "---", "---:"]),
        ]
    )
    for item in manifest["display_artifacts"]:  # type: ignore[index]
        lines.append(md_row([item["role"], f"`{item['path']}`", str(item["size_bytes"])]))

    lines.extend(
        [
            "",
            "## 4. 方法版本",
            "",
            md_row(["版本", "阶段", "方法", "Train", "Held-out", "固定视频"]),
            md_row(["---", "---", "---", "---:", "---:", "---"]),
        ]
    )
    for method in manifest["methods"]:  # type: ignore[index]
        lines.append(
            md_row(
                [
                    f"`{method['version']}`",
                    method["stage"],
                    method["method"],
                    method["train_range_success"],
                    method["heldout_success"],
                    f"`{method['clip']}`",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## 5. 重建命令",
            "",
            "```powershell",
            *manifest["rebuild_commands"],  # type: ignore[arg-type]
            "```",
            "",
            "## 6. 总体验证命令",
            "",
            "```powershell",
            str(manifest["verification_command"]),
            "```",
            "",
            "## 7. 缺失项",
            "",
        ]
    )
    if missing:
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("当前 manifest 中登记的核心交付物、展示入口和单方法视频均存在。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(args.output_md, manifest)
    print(f"final_artifact_manifest_json: {args.output_json}", flush=True)
    print(f"final_artifact_manifest_md: {args.output_md}", flush=True)
    print(f"methods: {manifest['counts']['methods']}", flush=True)  # type: ignore[index]
    print(f"method_evidence_rows: {manifest['counts']['method_evidence_rows']}", flush=True)  # type: ignore[index]
    print(f"method_comparison_rows: {manifest['counts']['method_comparison_rows']}", flush=True)  # type: ignore[index]
    print(f"thesis_visual_evidence_rows: {manifest['counts']['thesis_visual_evidence_rows']}", flush=True)  # type: ignore[index]
    print(f"defense_qa_rows: {manifest['counts']['defense_qa_rows']}", flush=True)  # type: ignore[index]
    print(f"version_lineage_rows: {manifest['counts']['version_lineage_rows']}", flush=True)  # type: ignore[index]
    print(f"research_showcase_rows: {manifest['counts']['research_showcase_rows']}", flush=True)  # type: ignore[index]
    print(f"claim_evidence_rows: {manifest['counts']['claim_evidence_rows']}", flush=True)  # type: ignore[index]
    print(f"claim_video_playback_rows: {manifest['counts']['claim_video_playback_rows']}", flush=True)  # type: ignore[index]
    print(f"defense_video_playlist_rows: {manifest['counts']['defense_video_playlist_rows']}", flush=True)  # type: ignore[index]
    print(f"defense_video_cue_sheet_rows: {manifest['counts']['defense_video_cue_sheet_rows']}", flush=True)  # type: ignore[index]
    print(f"final_defense_narrative_rows: {manifest['counts']['final_defense_narrative_rows']}", flush=True)  # type: ignore[index]
    print(f"remaining_experiment_board_rows: {manifest['counts']['remaining_experiment_board_rows']}", flush=True)  # type: ignore[index]
    print(f"trajectory_act_conclusion_rows: {manifest['counts']['trajectory_act_conclusion_rows']}", flush=True)  # type: ignore[index]
    print(f"trajectory_act_slow_viewer_rows: {manifest['counts']['trajectory_act_slow_viewer_rows']}", flush=True)  # type: ignore[index]
    print(f"preference_post_training_ablation_rows: {manifest['counts']['preference_post_training_ablation_rows']}", flush=True)  # type: ignore[index]
    print(f"preference_ranked_objective_rows: {manifest['counts']['preference_ranked_objective_rows']}", flush=True)  # type: ignore[index]
    print(f"preference_ranked_fast_rows: {manifest['counts']['preference_ranked_fast_rows']}", flush=True)  # type: ignore[index]
    print(f"timing_aware_trajectory_prior_residual_rows: {manifest['counts']['timing_aware_trajectory_prior_residual_rows']}", flush=True)  # type: ignore[index]
    print(f"gripper_timing_contact_probe_rows: {manifest['counts']['gripper_timing_contact_probe_rows']}", flush=True)  # type: ignore[index]
    print(f"control_safety_sweep_rows: {manifest['counts']['control_safety_sweep_rows']}", flush=True)  # type: ignore[index]
    print(f"action_head_control_safety_sweep_rows: {manifest['counts']['action_head_control_safety_sweep_rows']}", flush=True)  # type: ignore[index]
    print(f"strict_grasp_audit_rows: {manifest['counts']['strict_grasp_audit_rows']}", flush=True)  # type: ignore[index]
    print(f"video_evidence_rows: {manifest['counts']['video_evidence_rows']}", flush=True)  # type: ignore[index]
    print(f"video_quality_rows: {manifest['counts']['video_quality_rows']}", flush=True)  # type: ignore[index]
    print(f"external_dependency_readiness_rows: {manifest['counts']['external_dependency_readiness_rows']}", flush=True)  # type: ignore[index]
    print(f"defense_evidence_pack_files: {manifest['counts']['defense_evidence_pack_files']}", flush=True)  # type: ignore[index]


if __name__ == "__main__":
    main()
