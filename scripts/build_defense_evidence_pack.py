from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VERSION = "defense_evidence_pack_v1"


CORE_DOCS = [
    "README.md",
    "requirements.txt",
    "docs/final_experiment_package.md",
    "docs/final_closure_audit_v1.md",
    "docs/v4_independent_replication_v1.md",
    "docs/contact_phase_monitor_heldout_v1_analysis.md",
    "docs/counterfactual_intervention_pilot_v1_audit.md",
    "docs/mujoco_only_scope.md",
    "docs/final_showcase_handoff.md",
    "docs/final_showcase_handoff.csv",
    "docs/final_defense_narrative_script.md",
    "docs/final_defense_narrative_script.csv",
    "docs/remaining_experiment_execution_board.md",
    "docs/remaining_experiment_execution_board.csv",
    "docs/experiment_dashboard.html",
    "docs/defense_deck.html",
    "docs/final_artifact_manifest.md",
    "docs/final_artifact_manifest.json",
    "docs/reproducible_command_index.md",
    "docs/showcase_launcher_guide.md",
    "docs/experiment_versions.json",
    "docs/evaluation_summary.csv",
    "docs/language_generalization_summary.csv",
    "docs/model_resource_summary.csv",
    "docs/data_efficiency_summary.csv",
    "docs/domain_randomization_summary.md",
    "docs/domain_randomization_summary.csv",
    "docs/result_matrix.md",
    "docs/final_method_version_index.md",
    "docs/final_method_version_index.csv",
    "docs/method_comparison_dashboard.html",
    "docs/method_comparison_dashboard.md",
    "docs/method_comparison_dashboard.csv",
    "docs/core_task_comparison_matrix.md",
    "docs/core_task_comparison_matrix.csv",
    "outputs/evaluations/core_task_comparison_matrix_v1.json",
    "scripts/build_core_task_comparison_matrix.py",
    "docs/core_v2_holdout_comparison_matrix.md",
    "docs/core_v2_holdout_comparison_matrix.csv",
    "outputs/evaluations/core_v2_holdout_comparison_matrix_v1.json",
    "scripts/build_core_v2_comparison_matrix.py",
    "docs/core_v2_pretrained_vlm_action_head_report.md",
    "docs/core_v2_pretrained_vlm_action_head_report.csv",
    "outputs/evaluations/core_v2_pretrained_vlm_action_head_v1.json",
    "scripts/build_core_v2_pretrained_vlm_report.py",
    "docs/core_v2_clip_semantic_waypoint_report.md",
    "docs/core_v2_clip_semantic_waypoint_report.csv",
    "outputs/evaluations/core_v2_clip_semantic_waypoint_v1.json",
    "scripts/build_core_v2_clip_semantic_waypoint_report.py",
    "docs/kaggle_clip_semantic_adapter_core_v2_v1_report.md",
    "docs/kaggle_clip_semantic_adapter_core_v2_v1.csv",
    "outputs/evaluations/kaggle_clip_semantic_adapter_core_v2_v1.json",
    "scripts/build_kaggle_clip_semantic_adapter_report.py",
    "kaggle/kernels/widowx_mujoco_clip_semantic_adapter_v1/train_clip_semantic_adapter.py",
    "kaggle/kernels/widowx_mujoco_clip_semantic_adapter_v1/kernel-metadata.json",
    "outputs/clip_semantic_waypoint/kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3.npz",
    "outputs/kaggle_remote/kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3/kaggle_clip_semantic_adapter_core_v2_v1_metrics.json",
    "outputs/kaggle_remote/kaggle_clip_semantic_adapter_core_v2_v1_kernel_v3/kaggle_clip_semantic_adapter_core_v2_v1_predictions.csv",
    "docs/kaggle_clip_semantic_adapter_core_v2_ood_v1_report.md",
    "docs/kaggle_clip_semantic_adapter_core_v2_ood_v1.csv",
    "outputs/evaluations/kaggle_clip_semantic_adapter_core_v2_ood_v1.json",
    "docs/frozen_clip_semantic_adapter_same_protocol_comparison.md",
    "docs/frozen_clip_semantic_adapter_same_protocol_comparison.csv",
    "outputs/evaluations/frozen_clip_semantic_adapter_same_protocol_comparison_v1.json",
    "scripts/build_frozen_clip_semantic_adapter_comparison.py",
    "docs/core_v2_clip_semantic_data_efficiency.md",
    "docs/core_v2_clip_semantic_data_efficiency.csv",
    "outputs/evaluations/core_v2_clip_semantic_data_efficiency_v1.json",
    "scripts/evaluate_clip_semantic_waypoint_data_efficiency.py",
    "scripts/build_core_v2_clip_semantic_data_efficiency_report.py",
    "docs/core_v2_clip_semantic_ood_generalization.md",
    "docs/core_v2_clip_semantic_ood_generalization.csv",
    "outputs/evaluations/core_v2_clip_semantic_ood_generalization_v1.json",
    "scripts/evaluate_clip_semantic_ood_generalization.py",
    "docs/core_v2_video_evidence.md",
    "scripts/create_demo_subset.py",
    "docs/core_v2_holdout_blue_cube_blue_pad.csv",
    "outputs/evaluations/core_v2_holdout_blue_cube_blue_pad.json",
    "docs/core_v2_holdout_blue_cube_red_pad.csv",
    "outputs/evaluations/core_v2_holdout_blue_cube_red_pad.json",
    "docs/core_v2_holdout_red_cube_red_pad.csv",
    "outputs/evaluations/core_v2_holdout_red_cube_red_pad.json",
    "docs/core_v2_holdout_leftmost_cube_to_bowl.csv",
    "outputs/evaluations/core_v2_holdout_leftmost_cube_to_bowl.json",
    "docs/core_v2_prior_holdout_blue_cube_blue_pad.csv",
    "outputs/evaluations/core_v2_prior_holdout_blue_cube_blue_pad.json",
    "docs/core_v2_prior_holdout_blue_cube_red_pad.csv",
    "outputs/evaluations/core_v2_prior_holdout_blue_cube_red_pad.json",
    "docs/core_v2_prior_holdout_red_cube_red_pad.csv",
    "outputs/evaluations/core_v2_prior_holdout_red_cube_red_pad.json",
    "docs/core_v2_prior_holdout_leftmost_cube_to_bowl.csv",
    "outputs/evaluations/core_v2_prior_holdout_leftmost_cube_to_bowl.json",
    "docs/core_task_blue_cube_blue_pad.csv",
    "outputs/evaluations/core_task_blue_cube_blue_pad.json",
    "docs/core_task_blue_cube_red_pad.csv",
    "outputs/evaluations/core_task_blue_cube_red_pad.json",
    "docs/core_task_red_cube_red_pad.csv",
    "outputs/evaluations/core_task_red_cube_red_pad.json",
    "docs/core_task_leftmost_to_bowl.csv",
    "outputs/evaluations/core_task_leftmost_to_bowl.json",
    "docs/method_evidence_gate.md",
    "docs/method_evidence_gate.csv",
    "docs/version_naming_and_gate_spec.md",
    "docs/version_naming_and_gate_spec.csv",
    "docs/method_stage_audit.md",
    "docs/method_stage_audit.csv",
    "docs/stage_comparison_report.md",
    "docs/stage_comparison_report.csv",
    "docs/stage_showcase_index.html",
    "docs/stage_showcase_index.md",
    "docs/stage_evidence_index.md",
    "docs/stage_evidence_index.csv",
    "docs/stage_reproduction_runbook.md",
    "docs/stage_reproduction_runbook.csv",
    "docs/task_bc_stage_report.md",
    "docs/task_bc_stage_report.csv",
    "docs/trajectory_act_stage_report.md",
    "docs/trajectory_act_stage_report.csv",
    "docs/trajectory_act_experiment_record.md",
    "docs/trajectory_act_experiment_record.csv",
    "docs/trajectory_act_failure_diagnosis.md",
    "docs/trajectory_act_failure_diagnosis.csv",
    "docs/trajectory_act_conclusion_brief.md",
    "docs/trajectory_act_conclusion_brief.csv",
    "docs/trajectory_act_slow_viewer_guide.md",
    "docs/trajectory_act_slow_viewer_guide.csv",
    "scripts/build_final_defense_narrative_script.py",
    "scripts/build_remaining_experiment_execution_board.py",
    "scripts/build_trajectory_act_slow_viewer_guide.py",
    "docs/trajectory_prior_residual_bc_report.md",
    "docs/trajectory_prior_residual_bc_report.csv",
    "docs/phase_weighted_torch_act_report.md",
    "docs/phase_weighted_torch_act_report.csv",
    "docs/grasp_lift_subpolicy_probe_report.md",
    "docs/grasp_lift_subpolicy_probe_report.csv",
    "docs/contact_stage_subpolicy_report.md",
    "docs/contact_stage_subpolicy_report.csv",
    "docs/contact_stage_demo_torch_act_report.md",
    "docs/contact_stage_demo_torch_act_report.csv",
    "docs/contact_stage_phase_action_head_report.md",
    "docs/contact_stage_phase_action_head_report.csv",
    "docs/contact_hold_weighted_torch_act_report.md",
    "docs/contact_hold_weighted_torch_act_report.csv",
    "docs/gripper_timing_contact_probe_report.md",
    "docs/gripper_timing_contact_probe_report.csv",
    "docs/contact_aware_trajectory_knn_report.md",
    "docs/contact_aware_trajectory_knn_report.csv",
    "docs/preference_contact_aware_trajectory_post_training_report.md",
    "docs/preference_contact_aware_trajectory_post_training_report.csv",
    "docs/preference_ranked_trajectory_post_training_report.md",
    "docs/preference_ranked_trajectory_post_training_report.csv",
    "docs/preference_trajectory_post_training_v1_ranked_objective_summary.md",
    "docs/preference_trajectory_post_training_v1_ranked_objective_report.md",
    "docs/preference_trajectory_post_training_v1_ranked_objective_report.csv",
    "docs/preference_trajectory_post_training_v1_ranked_fast_summary.md",
    "docs/preference_trajectory_post_training_v1_ranked_fast_report.md",
    "docs/preference_trajectory_post_training_v1_ranked_fast_report.csv",
    "docs/preference_post_training_upgrade_gate.md",
    "docs/preference_post_training_upgrade_gate.csv",
    "docs/preference_post_training_ablation_matrix.md",
    "docs/preference_post_training_ablation_matrix.csv",
    "docs/contact_phase_gated_torch_act_report.md",
    "docs/contact_phase_gated_torch_act_report.csv",
    "docs/contact_aware_phase_gated_torch_act_report.md",
    "docs/contact_aware_phase_gated_torch_act_report.csv",
    "docs/action_head_stage_report.md",
    "docs/action_head_stage_report.csv",
    "docs/strict_grasp_success_audit.md",
    "docs/strict_grasp_success_audit.csv",
    "docs/video_evidence_gallery.html",
    "docs/video_evidence_index.md",
    "docs/video_evidence_index.csv",
    "docs/video_quality_audit.md",
    "docs/video_quality_audit.csv",
    "docs/failure_mode_taxonomy.md",
    "docs/failure_mode_taxonomy.csv",
    "docs/presentation_video_pack.md",
    "docs/video_presentation_storyboard.html",
    "docs/video_presentation_storyboard.md",
    "docs/defense_video_playlist.html",
    "docs/defense_video_playlist.md",
    "docs/defense_video_playlist.csv",
    "docs/defense_video_cue_sheet.md",
    "docs/defense_video_cue_sheet.csv",
    "docs/defense_live_runbook.md",
    "docs/defense_live_runbook.csv",
    "docs/defense_qa_playbook.html",
    "docs/defense_qa_playbook.md",
    "docs/defense_qa_playbook.csv",
    "docs/thesis_visual_evidence_index.html",
    "docs/thesis_visual_evidence_index.md",
    "docs/thesis_visual_evidence_index.csv",
    "docs/version_lineage_index.html",
    "docs/version_lineage_index.md",
    "docs/version_lineage_index.csv",
    "docs/claim_evidence_traceability.md",
    "docs/claim_evidence_traceability.csv",
    "docs/claim_video_playback_index.md",
    "docs/claim_video_playback_index.csv",
    "docs/research_evidence_map.md",
    "docs/research_evidence_map.csv",
    "docs/research_question_showcase_plan.md",
    "docs/research_question_showcase_plan.csv",
    "docs/goal_completion_audit.md",
    "docs/goal_completion_audit.csv",
    "docs/thesis_results_chapter_draft.md",
    "docs/thesis_appendix_tables.md",
    "docs/thesis_method_comparison_table.csv",
    "docs/thesis_domain_randomization_table.csv",
    "docs/defense_slide_outline.md",
    "docs/defense_storyboard.md",
    "docs/next_phase_implementation.md",
    "docs/next_experiment_registry.md",
    "docs/next_experiment_registry.csv",
    "docs/external_dependency_readiness_audit.md",
    "docs/external_dependency_readiness_audit.csv",
    "docs/openvla_dataset_bridge_report.md",
    "docs/openvla_bridge_gallery.html",
    "docs/widowx_mujoco_rlds_source_v1_report.md",
    "docs/widowx_mujoco_rlds_source_validation_v1.md",
    "docs/openvla_feasibility_report.md",
    "docs/robot_vla_action_head_handoff.md",
    "docs/robot_vla_remote_run_pack.md",
    "docs/robot_vla_remote_result_intake.md",
    "docs/robot_vla_remote_result_intake.csv",
    "docs/isaac_domain_randomization_handoff.md",
    "docs/isaac_domain_randomization_handoff.csv",
    "docs/real_widowx_validation_handoff.md",
    "docs/real_widowx_validation_handoff.csv",
    "docs/video_showcase.md",
    "scripts/build_final_showcase_handoff.py",
    "scripts/build_version_naming_and_gate_spec.py",
    "scripts/trajectory_prior_residual_common.py",
    "scripts/train_trajectory_prior_residual_bc.py",
    "scripts/run_trajectory_prior_residual_policy.py",
    "scripts/evaluate_trajectory_prior_residual_bc.py",
    "docs/timing_aware_trajectory_prior_residual_bc_report.md",
    "docs/timing_aware_trajectory_prior_residual_bc_report.csv",
    "scripts/timing_aware_trajectory_prior_residual_common.py",
    "scripts/train_timing_aware_trajectory_prior_residual_bc.py",
    "scripts/run_timing_aware_trajectory_prior_residual_policy.py",
    "scripts/evaluate_timing_aware_trajectory_prior_residual_bc.py",
]

EXTRA_FILES = [
    "outputs/evaluations/final_closure_audit_v1.json",
    "outputs/evaluations/v4_independent_replication_v1.json",
    "outputs/evaluations/contact_phase_monitor_heldout_v1.json",
    "outputs/evaluations/counterfactual_intervention_pilot_v1.json",
    "videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.mp4",
    "videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.json",
    "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_v1_20260721_110325.npz",
    "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_5eps_v1_20260721_111415.npz",
    "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_10eps_v1_20260721_111439.npz",
    "outputs/clip_action_head/clip_core_v2_multitask_v1_20260721_104743.npz",
    "data/vla_bridge/openvla_dataset_bridge_v1/manifest.json",
    "data/vla_bridge/openvla_dataset_bridge_v1/samples.jsonl",
    "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png",
    "outputs/evaluations/data_efficiency_v2.json",
    "outputs/evaluations/domain_randomization_eval_v1.json",
    "outputs/evaluations/strict_grasp_success_audit_v1.json",
    "outputs/evaluations/openvla_dataset_bridge_v1.json",
    "outputs/evaluations/widowx_mujoco_rlds_source_v1.json",
    "outputs/evaluations/widowx_mujoco_rlds_source_validation_v1.json",
    "outputs/evaluations/openvla_feasibility_check_v1.json",
    "outputs/evaluations/robot_vla_action_head_handoff_v1.json",
    "outputs/evaluations/robot_vla_remote_run_pack_v1.json",
    "outputs/evaluations/robot_vla_remote_result_intake_v1.json",
    "outputs/evaluations/isaac_domain_randomization_handoff_v1.json",
    "outputs/evaluations/real_widowx_validation_handoff_v1.json",
    "outputs/evaluations/external_dependency_readiness_audit_v1.json",
    "outputs/evaluations/version_naming_and_gate_spec_v1.json",
    "outputs/evaluations/trajectory_prior_residual_bc_v1_candidate.json",
    "outputs/trajectory_prior_residual_bc/trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz",
    "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
    "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.json",
    "outputs/evaluations/timing_aware_trajectory_prior_residual_bc_v1_candidate.json",
    "outputs/timing_aware_trajectory_prior_residual_bc/timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz",
    "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
    "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.json",
    "outputs/evaluations/phase_weighted_torch_act_v1_candidate.json",
    "outputs/torch_act/phase_weighted_torch_act_v1_candidate_20260720_225108.pt",
    "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.mp4",
    "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.json",
    "scripts/evaluate_phase_weighted_torch_act.py",
    "outputs/evaluations/grasp_lift_subpolicy_probe_v1_candidate.json",
    "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
    "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.json",
    "scripts/evaluate_grasp_lift_subpolicy_probe.py",
    "outputs/evaluations/contact_stage_subpolicy_v1_candidate.json",
    "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4",
    "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.json",
    "scripts/run_contact_stage_subpolicy.py",
    "scripts/evaluate_contact_stage_subpolicy.py",
    "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/metadata.jsonl",
    "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/summary.json",
    "outputs/evaluations/contact_stage_demo_torch_act_v1_candidate.json",
    "outputs/torch_act/contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt",
    "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
    "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.json",
    "scripts/collect_contact_stage_demos.py",
    "scripts/evaluate_contact_stage_demo_torch_act.py",
    "outputs/evaluations/contact_stage_phase_action_head_v1_candidate.json",
    "outputs/phase_action_head/contact_stage_phase_action_head_v1_candidate_20260721_020941.npz",
    "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4",
    "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.json",
    "scripts/evaluate_contact_stage_phase_action_head.py",
    "outputs/evaluations/contact_hold_weighted_torch_act_v1_candidate.json",
    "outputs/torch_act/contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt",
    "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
    "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.json",
    "scripts/evaluate_contact_hold_weighted_torch_act.py",
    "outputs/evaluations/gripper_timing_contact_probe_v1_candidate.json",
    "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.mp4",
    "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.json",
    "scripts/run_gripper_timing_probe.py",
    "scripts/evaluate_gripper_timing_probe.py",
    "outputs/evaluations/contact_aware_trajectory_knn_v1_candidate.json",
    "outputs/trajectory_knn_bc/contact_aware_trajectory_knn_20260720_233445.npz",
    "outputs/videos/contact_aware_trajectory_knn_v1_candidate_seed0.mp4",
    "outputs/videos/contact_aware_trajectory_knn_v1_candidate_seed0.json",
    "scripts/evaluate_contact_aware_trajectory_knn.py",
    "outputs/evaluations/preference_contact_aware_trajectory_post_training_v1_candidate.json",
    "outputs/preference_post_training/preference_contact_aware_trajectory_post_training_20260721_000449.npz",
    "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4",
    "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.json",
    "scripts/evaluate_preference_contact_aware_trajectory_post_training.py",
    "outputs/evaluations/preference_ranked_trajectory_post_training_v1_candidate.json",
    "outputs/evaluations/preference_post_training_upgrade_gate_v1.json",
    "outputs/preference_post_training/preference_ranked_trajectory_post_training_20260721_031024.npz",
    "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
    "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.json",
    "scripts/evaluate_preference_ranked_trajectory_post_training.py",
    "outputs/evaluations/preference_trajectory_post_training_v1_ranked_objective_candidate.json",
    "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz",
    "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4",
    "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.json",
    "outputs/evaluations/preference_trajectory_post_training_v1_ranked_fast_candidate.json",
    "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz",
    "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4",
    "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.json",
    "scripts/build_preference_post_training_upgrade_gate.py",
    "scripts/build_preference_post_training_ablation_matrix.py",
    "outputs/evaluations/contact_phase_gated_torch_act_v1_candidate.json",
    "outputs/torch_act/contact_phase_gated_torch_act_v1_candidate_20260721_003304.pt",
    "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.mp4",
    "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.json",
    "scripts/evaluate_contact_phase_gated_torch_act.py",
    "outputs/evaluations/contact_aware_phase_gated_torch_act_v1_candidate.json",
    "outputs/torch_act/contact_aware_phase_gated_torch_act_v1_candidate_20260721_004944.pt",
    "outputs/videos/contact_aware_phase_gated_torch_act_v1_candidate_seed0.mp4",
    "outputs/videos/contact_aware_phase_gated_torch_act_v1_candidate_seed0.json",
    "scripts/evaluate_contact_aware_phase_gated_torch_act.py",
    "outputs/robot_vla_remote_run_pack/robot_vla_remote_run_pack_v1.zip",
    "scripts/export_openvla_rlds_source.py",
    "scripts/validate_openvla_rlds_source.py",
    "scripts/remote_openvla/widowx_mujoco_pick_place_dataset_builder.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable defense evidence pack with docs, videos, and figures.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "defense_evidence_pack" / VERSION)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / f"{VERSION}.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "defense_evidence_pack.md")
    return parser.parse_args()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_reset_dir(path: Path) -> None:
    root = (ROOT / "outputs" / "defense_evidence_pack").resolve()
    resolved = path.resolve()
    if resolved.parent != root or resolved.name != VERSION:
        raise RuntimeError(f"refuse to clear unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def add_file(paths: list[tuple[str, Path]], role: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    paths.append((role, path))


def collect_sources() -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for item in CORE_DOCS:
        add_file(sources, "核心中文文档/索引", ROOT / item)
    for item in EXTRA_FILES:
        add_file(sources, "外部阶段前置证据", ROOT / item)
    for path in sorted((ROOT / "scripts").glob("*.py")):
        add_file(sources, "复现脚本", path)
    for path in sorted((ROOT / "outputs" / "figures").glob("*.svg")):
        add_file(sources, "论文图表", path)
    for pattern in ("*.mp4", "*.json"):
        for path in sorted((ROOT / "outputs" / "videos").glob(pattern)):
            add_file(sources, "全量单方法视频证据", path)
    for path in sorted((ROOT / "outputs" / "presentation_clips").glob("*.mp4")):
        add_file(sources, "答辩阶段短片", path)
    add_file(sources, "答辩阶段短片", ROOT / "outputs" / "presentation_clips" / "presentation_video_pack_manifest.json")
    for path in sorted((ROOT / "outputs" / "showcase").glob("*.mp4")):
        add_file(sources, "宫格展示视频", path)

    unique: dict[str, tuple[str, Path]] = {}
    for role, path in sources:
        unique[path.resolve().as_posix()] = (role, path)
    return list(unique.values())


def copy_sources(sources: list[tuple[str, Path]], output_dir: Path) -> list[dict[str, object]]:
    entries = []
    for role, source in sources:
        relative = source.relative_to(ROOT)
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append(
            {
                "role": role,
                "source": relative.as_posix(),
                "packaged_path": target.relative_to(output_dir).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    return entries


def count_roles(entries: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        role = str(entry["role"])
        counts[role] = counts.get(role, 0) + 1
    return counts


def write_start_here(output_dir: Path, entries: list[dict[str, object]]) -> Path:
    path = output_dir / "START_HERE.md"
    lines = [
        "# 答辩证据包",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 MuJoCo 实验包的核心中文文档、HTML 展示入口、CSV/JSON 量化表、论文图表和仿真视频集中到一个可复制目录。它只组织已有证据，不新增实验结果。",
        "",
        "## 推荐打开顺序",
        "",
        "1. `docs/final_closure_audit_v1.md`：最终采用方案、独立复核、被拒绝候选和边界。",
        "2. `docs/final_showcase_handoff.md`：按原始目标查找版本、阶段、评测和视频入口。",
        "3. `docs/final_defense_narrative_script.md`：按研究问题和阶段组织中文答辩讲解顺序。",
        "4. `docs/experiment_dashboard.html`：总览 dashboard。",
        "5. `docs/defense_deck.html`：本地答辩 deck。",
        "6. `outputs/presentation_clips/00_defense_video_reel.mp4`：阶段总览视频。",
        "7. `docs/method_comparison_dashboard.html`：方法横向比较。",
        "8. `docs/version_lineage_index.html`：版本谱系。",
        "9. `docs/defense_qa_playbook.html`：追问回答和证据入口。",
        "",
        "## 证据边界",
        "",
        "- 视频只作为定性展示，成功率和资源结论仍以 CSV/JSON 为准。",
        "- 当前包是 MuJoCo 实验包；真实 OpenVLA、Isaac 和真实 WidowX 仍是后续阶段。",
        "- `completed_prerequisite` 和 handoff/remote pack 不是策略成功率结果。",
        "- `trajectory_prior_residual_bc_v1_candidate` 是分阶段轨迹先验 + residual BC 诊断候选，不是纯 BC、完整官方 ACT 或 VLA 后训练。",
        "- `timing_aware_trajectory_prior_residual_bc_v1_candidate` 是时序感知 trajectory-conditioned / ACT-like residual BC 候选；residual-scale=0.02 时 train-range 5/5、held-out 4/5，但 strict grasp 仍为 0，不能写成完整官方 ACT 或真实 VLA 后训练。",
        "",
        "## 包内计数",
        "",
    ]
    for role, count in sorted(count_roles(entries).items()):
        lines.append(f"- {role}: {count}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_archive(output_dir: Path) -> Path:
    archive = output_dir.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir).as_posix())
    return archive


def build_pack(args: argparse.Namespace) -> dict[str, object]:
    safe_reset_dir(args.output_dir)
    sources = collect_sources()
    entries = copy_sources(sources, args.output_dir)
    start_here = write_start_here(args.output_dir, entries)
    entries.append(
        {
            "role": "证据包入口",
            "source": "generated",
            "packaged_path": start_here.relative_to(args.output_dir).as_posix(),
            "size_bytes": start_here.stat().st_size,
            "sha256": sha256(start_here),
        }
    )
    pack_manifest = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "集中交付当前 MuJoCo 实验包的版本、阶段、评测、图表和仿真视频证据。",
        "output_dir": rel(args.output_dir),
        "file_count": len(entries),
        "role_counts": count_roles(entries),
        "entries": entries,
        "open_commands": {
            "final_closure": f'Start-Process notepad.exe "{args.output_dir / "docs" / "final_closure_audit_v1.md"}"',
            "dashboard": f'Start-Process "{args.output_dir / "docs" / "experiment_dashboard.html"}"',
            "handoff": f'Start-Process notepad.exe "{args.output_dir / "docs" / "final_showcase_handoff.md"}"',
            "narrative": f'Start-Process notepad.exe "{args.output_dir / "docs" / "final_defense_narrative_script.md"}"',
            "remaining_board": f'Start-Process notepad.exe "{args.output_dir / "docs" / "remaining_experiment_execution_board.md"}"',
            "deck": f'Start-Process "{args.output_dir / "docs" / "defense_deck.html"}"',
            "reel": f'Start-Process "{args.output_dir / "outputs" / "presentation_clips" / "00_defense_video_reel.mp4"}"',
            "lineage": f'Start-Process "{args.output_dir / "docs" / "version_lineage_index.html"}"',
        },
        "paper_boundary": "当前证据包只证明 MuJoCo 实验包的版本、评测和视频证据完整；真实 OpenVLA、Isaac 和真实 WidowX 仍必须单独运行和登记。",
    }
    pack_manifest_path = args.output_dir / "PACK_MANIFEST.json"
    pack_manifest_path.write_text(json.dumps(pack_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    entries.append(
        {
            "role": "证据包入口",
            "source": "generated",
            "packaged_path": pack_manifest_path.relative_to(args.output_dir).as_posix(),
            "size_bytes": pack_manifest_path.stat().st_size,
            "sha256": sha256(pack_manifest_path),
        }
    )
    pack_manifest["file_count"] = len(entries)
    pack_manifest["role_counts"] = count_roles(entries)
    pack_manifest["entries"] = entries
    pack_manifest_path.write_text(json.dumps(pack_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    archive = make_archive(args.output_dir)
    pack_manifest["archive_path"] = rel(archive)
    pack_manifest["archive_size_bytes"] = archive.stat().st_size
    pack_manifest["archive_sha256"] = sha256(archive)
    return pack_manifest


def md_row(values: list[str]) -> str:
    return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"


def write_report(path: Path, pack: dict[str, object]) -> None:
    role_counts = pack["role_counts"]
    assert isinstance(role_counts, dict)
    lines = [
        "# 答辩证据包",
        "",
        f"版本：`{VERSION}`",
        "",
        "用途：把当前 MuJoCo 实验包的核心中文文档、HTML 展示入口、CSV/JSON 量化表、论文图表和仿真视频集中到一个可复制目录和 zip。该包只组织已有证据，不新增实验结果。",
        "",
        "## 1. 输出位置",
        "",
        "```text",
        str(pack["output_dir"]),
        str(pack["archive_path"]),
        "```",
        "",
        "包内入口：",
        "",
        "- `START_HERE.md`：答辩前优先打开的说明页。",
        "- `PACK_MANIFEST.json`：包内文件清单和哈希记录。",
        "- `docs/final_closure_audit_v1.md`：最终采用 V4、独立复核、接触候选拒绝和论文边界。",
        "- `docs/final_showcase_handoff.md`：按原始目标查找版本、阶段、评测、视频和 viewer 入口。",
        "- `docs/final_defense_narrative_script.md`：`final_defense_narrative_script_v1`，按研究问题、阶段方法、指标和视频证据组织的中文答辩讲解脚本。",
        "- `docs/remaining_experiment_execution_board.md`：`remaining_experiment_execution_board_v1`，剩余 planned/readiness 实验的执行顺序、回填工件、升级门槛和论文红线。",
        "- `docs/trajectory_act_conclusion_brief.md`：`trajectory_act_conclusion_brief_v1`，trajectory-conditioned BC / ACT-style baseline 的论文/答辩结论摘要。",
        "- `docs/trajectory_act_slow_viewer_guide.md`：`trajectory_act_slow_viewer_guide_v1`，trajectory-conditioned BC / ACT 阶段的标准慢速与超慢学习 viewer 完整命令。",
        "- `docs/preference_post_training_ablation_matrix.md`：`preference_post_training_ablation_matrix_v1`，三个 preference post-training 候选的中文消融矩阵和下一轮 objective 设计方向。",
        "- `docs/widowx_mujoco_rlds_source_v1_report.md` 与 `docs/widowx_mujoco_rlds_source_validation_v1.md`：79 条 Core V2 成功 episode、2,528 个 joint-delta source step 及其完整性验证；尚未注册为 RLDS。",
        "",
        "## 2. 覆盖统计",
        "",
        md_row(["类别", "文件数"]),
        md_row(["---", "---:"]),
    ]
    for role, count in sorted(role_counts.items()):
        lines.append(md_row([role, count]))
    lines.extend(
        [
            "",
            "## 3. 推荐打开顺序",
            "",
            "```powershell",
            str(pack["open_commands"]["final_closure"]),  # type: ignore[index]
            str(pack["open_commands"]["handoff"]),  # type: ignore[index]
            str(pack["open_commands"]["narrative"]),  # type: ignore[index]
            str(pack["open_commands"]["remaining_board"]),  # type: ignore[index]
            str(pack["open_commands"]["dashboard"]),  # type: ignore[index]
            str(pack["open_commands"]["deck"]),  # type: ignore[index]
            str(pack["open_commands"]["reel"]),  # type: ignore[index]
            str(pack["open_commands"]["lineage"]),  # type: ignore[index]
            "```",
            "",
            "## 4. 重建命令",
            "",
            "```powershell",
            f'& "{PYTHON}" "{ROOT / "scripts" / "build_defense_evidence_pack.py"}"',
            "```",
            "",
            "## 5. 使用边界",
            "",
            "- 该包是当前 MuJoCo 实验包的可交付证据包，不代表真实 OpenVLA、Isaac 或真实 WidowX 已完成。",
            "- `trajectory_prior_residual_bc_v1_candidate` 已入包；它记录分阶段轨迹先验 + residual BC action head 明显改善放置和 TCP 抬升，但不是纯 BC、完整官方 ACT 或 VLA 后训练成功结果。",
            "- `timing_aware_trajectory_prior_residual_bc_v1_candidate` 已入包；它记录时序感知 trajectory-conditioned / ACT-like residual BC 在低残差幅度下提高放置稳定性，但 strict grasp 仍为 0，不能写成完整官方 ACT 或真实 VLA 后训练成功结果。",
            "- `contact_stage_subpolicy_v1_candidate` 已入包；它是 scripted contact-stage 上界诊断，不是 learned BC/ACT/VLA 成功结果。",
            "- `contact_stage_demo_torch_act_v1_candidate` 已入包；它是用 contact-stage 成功示范训练 ACT 的负例诊断，不是完整官方 ACT 成功结果。",
            "- `contact_stage_phase_action_head_v1_candidate` 已入包；它是用 contact-stage 成功示范训练轻量 phase action-head 的负例诊断，不是真实 VLA 后训练成功结果。",
            "- `contact_hold_weighted_torch_act_v1_candidate` 已入包；它记录 contact-hold loss 加权带来一次 TCP 抬升迹象，但仍不是稳定抓取、完整官方 ACT 或真实 VLA 后训练成功结果。",
            "- `preference_ranked_trajectory_post_training_v1_candidate` 已入包；它记录 episode-ranked preference 带来目标距离成功和局部 TCP 抬升迹象，但 strict grasp 仍为 0，不能写成在线 RL 或真实偏好优化成功。",
            "- `preference_trajectory_post_training_v1_ranked_objective_candidate` 已入包；它记录 sample-stride=8 ranked objective 候选，train-range 4/5、held-out 0/5、标准抓取 0/10，固定 seed0 视频仍失败，不能升级为正式 preference post-training 方法。",
            "- `preference_trajectory_post_training_v1_ranked_fast_candidate` 已入包；它记录 ranked preference 的快速候选，train-range 2/5、held-out 0/5、标准抓取 0/10，不能升级为正式 preference post-training 方法。",
            "- 视频证据只支持定性展示，成功率、目标距离、资源消耗和语言泛化仍以 CSV/JSON 为准。",
            "- 后续新增真实 VLA/Isaac/真实机械臂结果后，必须重新运行本脚本并更新 manifest。",
            "",
            f"生成时间：{pack['generated_at']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pack = build_pack(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_md, pack)
    print(f"defense_evidence_pack_dir: {args.output_dir}", flush=True)
    print(f"defense_evidence_pack_archive: {pack['archive_path']}", flush=True)
    print(f"defense_evidence_pack_json: {args.output_json}", flush=True)
    print(f"defense_evidence_pack_md: {args.output_md}", flush=True)
    print(f"defense_evidence_pack_files: {pack['file_count']}", flush=True)


if __name__ == "__main__":
    main()
