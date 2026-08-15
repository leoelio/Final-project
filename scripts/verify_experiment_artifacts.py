from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


MOJIBAKE_MARKERS = (
    "锛",
    "銆",
    "鈥",
    "闃舵",
    "瑙嗛",
    "璇勬",
    "妯″",
    "鐗堟",
    "鍙",
    "涓€",
    "\ufffd",
)


EXTRA_VIDEO_STEMS = (
    "object_language_action_head_lite_v1_seed1_success_example",
    "expert_scripted_language_v1_seed200",
    "structured_waypoint_policy_v1_language_seed200",
    "object_language_action_head_lite_v1_language_seed200",
    "reward_weighted_action_head_lite_v1_language_seed200",
    "phase_conditioned_action_head_lite_v1_language_seed200",
    "torch_act_cvae_state_chunk_v1_language_seed200",
    "torch_act_state_chunk_cuda_v1_language_seed200",
    "torch_diffusion_policy_state_chunk_v1_language_seed200",
    "visual_feature_act_lite_v1_language_seed200",
    "adapter_action_head_lite_v1_language_seed200",
    "lora_action_head_lite_v1_language_seed200",
    "vision_language_action_head_lite_v1_language_seed200",
    "clip_action_head_lite_v1_language_seed200",
    "multi_task_object_action_head_lite_v1_language_seed400",
    "domain_randomization_structured_low_friction_seed0",
    "domain_randomization_trajectory_knn_low_friction_seed0",
    "domain_randomization_visual_act_cnn_cvae_low_friction_seed0",
    "trajectory_phase_template_bc_v1_candidate_seed1",
    "trajectory_prior_residual_bc_v1_candidate_seed0",
    "timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0",
    "grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0",
    "grasp_gated_torch_act_state_chunk_v1_candidate_seed0",
    "grasp_gated_trajectory_knn_v1_candidate_seed0",
    "preference_trajectory_post_training_v1_candidate_seed0",
    "preference_trajectory_post_training_v1_ranked_objective_candidate_seed0",
    "preference_trajectory_post_training_v1_ranked_fast_candidate_seed0",
    "preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0",
    "grasp_lift_subpolicy_probe_v1_candidate_seed0",
    "contact_aware_trajectory_knn_v1_candidate_seed0",
    "preference_contact_aware_trajectory_post_training_v1_candidate_seed0",
    "contact_phase_gated_torch_act_v1_candidate_seed0",
    "contact_aware_phase_gated_torch_act_v1_candidate_seed0",
    "contact_stage_subpolicy_v1_candidate_seed0",
    "contact_stage_demo_torch_act_v1_candidate_seed0",
    "contact_stage_phase_action_head_v1_candidate_seed101",
    "contact_hold_weighted_torch_act_v1_candidate_seed0",
    "gripper_timing_contact_probe_v1_candidate_seed0",
    "preference_ranked_trajectory_post_training_v1_candidate_seed0",
)

SHOWCASE_VIDEO_STEMS = (
    "core_methods_grid",
    "all_registered_methods_grid",
    "language_generalization_grid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify experiment registry, metrics, reports, and fixed rollout videos.")
    parser.add_argument("--versions", type=Path, default=ROOT / "docs" / "experiment_versions.json")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "evaluation_summary.csv")
    parser.add_argument("--core-task-comparison", type=Path, default=ROOT / "docs" / "core_task_comparison_matrix.md")
    parser.add_argument("--core-task-comparison-csv", type=Path, default=ROOT / "docs" / "core_task_comparison_matrix.csv")
    parser.add_argument("--core-task-comparison-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_task_comparison_matrix_v1.json")
    parser.add_argument("--core-v2-comparison", type=Path, default=ROOT / "docs" / "core_v2_holdout_comparison_matrix.md")
    parser.add_argument("--core-v2-comparison-csv", type=Path, default=ROOT / "docs" / "core_v2_holdout_comparison_matrix.csv")
    parser.add_argument("--core-v2-comparison-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_v2_holdout_comparison_matrix_v1.json")
    parser.add_argument("--core-v2-pretrained-vlm-report", type=Path, default=ROOT / "docs" / "core_v2_pretrained_vlm_action_head_report.md")
    parser.add_argument("--core-v2-pretrained-vlm-csv", type=Path, default=ROOT / "docs" / "core_v2_pretrained_vlm_action_head_report.csv")
    parser.add_argument("--core-v2-pretrained-vlm-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_v2_pretrained_vlm_action_head_v1.json")
    parser.add_argument("--core-v2-pretrained-vlm-model", type=Path, default=ROOT / "outputs" / "clip_action_head" / "clip_core_v2_multitask_v1_20260721_104743.npz")
    parser.add_argument("--core-v2-pretrained-vlm-video", type=Path, default=ROOT / "outputs" / "videos" / "clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4")
    parser.add_argument("--core-v2-clip-semantic-report", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_waypoint_report.md")
    parser.add_argument("--core-v2-clip-semantic-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_waypoint_report.csv")
    parser.add_argument("--core-v2-clip-semantic-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_waypoint_v1.json")
    parser.add_argument("--core-v2-clip-semantic-model", type=Path, default=ROOT / "outputs" / "clip_semantic_waypoint" / "clip_semantic_waypoint_core_v2_v1_20260721_110325.npz")
    parser.add_argument("--core-v2-clip-semantic-video", type=Path, default=ROOT / "outputs" / "videos" / "clip_semantic_waypoint_core_v2_v1_leftmost_cube_seed420.mp4")
    parser.add_argument("--core-v2-clip-semantic-efficiency-report", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_data_efficiency.md")
    parser.add_argument("--core-v2-clip-semantic-efficiency-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_data_efficiency.csv")
    parser.add_argument("--core-v2-clip-semantic-efficiency-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_data_efficiency_v1.json")
    parser.add_argument("--core-v2-clip-semantic-ood-report", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_ood_generalization.md")
    parser.add_argument("--core-v2-clip-semantic-ood-csv", type=Path, default=ROOT / "docs" / "core_v2_clip_semantic_ood_generalization.csv")
    parser.add_argument("--core-v2-clip-semantic-ood-json", type=Path, default=ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_generalization_v1.json")
    parser.add_argument("--core-v2-clip-semantic-ood-success-video", type=Path, default=ROOT / "outputs" / "videos" / "clip_semantic_ood_hard_leftmost_cube_seed1300.mp4")
    parser.add_argument("--core-v2-clip-semantic-ood-failure-video", type=Path, default=ROOT / "outputs" / "videos" / "clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4")
    parser.add_argument("--language-summary", type=Path, default=ROOT / "docs" / "language_generalization_summary.csv")
    parser.add_argument("--resource-summary", type=Path, default=ROOT / "docs" / "model_resource_summary.csv")
    parser.add_argument("--resource-report", type=Path, default=ROOT / "docs" / "model_resource_summary.md")
    parser.add_argument("--data-efficiency-summary", type=Path, default=ROOT / "docs" / "data_efficiency_summary.csv")
    parser.add_argument("--data-efficiency-report", type=Path, default=ROOT / "docs" / "data_efficiency_summary.md")
    parser.add_argument("--data-efficiency-json", type=Path, default=ROOT / "outputs" / "evaluations" / "data_efficiency_v2.json")
    parser.add_argument("--domain-randomization-summary", type=Path, default=ROOT / "docs" / "domain_randomization_summary.csv")
    parser.add_argument("--domain-randomization-report", type=Path, default=ROOT / "docs" / "domain_randomization_summary.md")
    parser.add_argument("--domain-randomization-json", type=Path, default=ROOT / "outputs" / "evaluations" / "domain_randomization_eval_v1.json")
    parser.add_argument("--isaac-handoff-report", type=Path, default=ROOT / "docs" / "isaac_domain_randomization_handoff.md")
    parser.add_argument("--isaac-handoff-csv", type=Path, default=ROOT / "docs" / "isaac_domain_randomization_handoff.csv")
    parser.add_argument("--isaac-handoff-json", type=Path, default=ROOT / "outputs" / "evaluations" / "isaac_domain_randomization_handoff_v1.json")
    parser.add_argument("--real-widowx-handoff-report", type=Path, default=ROOT / "docs" / "real_widowx_validation_handoff.md")
    parser.add_argument("--real-widowx-handoff-csv", type=Path, default=ROOT / "docs" / "real_widowx_validation_handoff.csv")
    parser.add_argument("--real-widowx-handoff-json", type=Path, default=ROOT / "outputs" / "evaluations" / "real_widowx_validation_handoff_v1.json")
    parser.add_argument("--real-widowx-trial-template", type=Path, default=ROOT / "outputs" / "real_robot" / "real_widowx_validation_v1_trial_template.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "evaluation_report.md")
    parser.add_argument("--package", type=Path, default=ROOT / "docs" / "final_experiment_package.md")
    parser.add_argument("--final-showcase-handoff", type=Path, default=ROOT / "docs" / "final_showcase_handoff.md")
    parser.add_argument("--final-showcase-handoff-csv", type=Path, default=ROOT / "docs" / "final_showcase_handoff.csv")
    parser.add_argument("--final-defense-narrative", type=Path, default=ROOT / "docs" / "final_defense_narrative_script.md")
    parser.add_argument("--final-defense-narrative-csv", type=Path, default=ROOT / "docs" / "final_defense_narrative_script.csv")
    parser.add_argument("--remaining-experiment-board", type=Path, default=ROOT / "docs" / "remaining_experiment_execution_board.md")
    parser.add_argument("--remaining-experiment-board-csv", type=Path, default=ROOT / "docs" / "remaining_experiment_execution_board.csv")
    parser.add_argument("--artifact-manifest", type=Path, default=ROOT / "docs" / "final_artifact_manifest.md")
    parser.add_argument("--artifact-manifest-json", type=Path, default=ROOT / "docs" / "final_artifact_manifest.json")
    parser.add_argument("--defense-evidence-pack", type=Path, default=ROOT / "docs" / "defense_evidence_pack.md")
    parser.add_argument("--defense-evidence-pack-json", type=Path, default=ROOT / "outputs" / "evaluations" / "defense_evidence_pack_v1.json")
    parser.add_argument("--defense-evidence-pack-archive", type=Path, default=ROOT / "outputs" / "defense_evidence_pack" / "defense_evidence_pack_v1.zip")
    parser.add_argument("--defense-evidence-pack-dir", type=Path, default=ROOT / "outputs" / "defense_evidence_pack" / "defense_evidence_pack_v1")
    parser.add_argument("--dashboard", type=Path, default=ROOT / "docs" / "experiment_dashboard.html")
    parser.add_argument("--storyboard", type=Path, default=ROOT / "docs" / "defense_storyboard.md")
    parser.add_argument("--slide-outline", type=Path, default=ROOT / "docs" / "defense_slide_outline.md")
    parser.add_argument("--defense-deck", type=Path, default=ROOT / "docs" / "defense_deck.html")
    parser.add_argument("--defense-live-runbook", type=Path, default=ROOT / "docs" / "defense_live_runbook.md")
    parser.add_argument("--defense-live-runbook-csv", type=Path, default=ROOT / "docs" / "defense_live_runbook.csv")
    parser.add_argument("--defense-video-playlist", type=Path, default=ROOT / "docs" / "defense_video_playlist.md")
    parser.add_argument("--defense-video-playlist-html", type=Path, default=ROOT / "docs" / "defense_video_playlist.html")
    parser.add_argument("--defense-video-playlist-csv", type=Path, default=ROOT / "docs" / "defense_video_playlist.csv")
    parser.add_argument("--defense-video-cue-sheet", type=Path, default=ROOT / "docs" / "defense_video_cue_sheet.md")
    parser.add_argument("--defense-video-cue-sheet-csv", type=Path, default=ROOT / "docs" / "defense_video_cue_sheet.csv")
    parser.add_argument("--figures-doc", type=Path, default=ROOT / "docs" / "experiment_figures.md")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "outputs" / "figures")
    parser.add_argument("--runtime-json", type=Path, default=ROOT / "outputs" / "evaluations" / "runtime_capability_v1.json")
    parser.add_argument("--runtime-report", type=Path, default=ROOT / "docs" / "runtime_capability_report.md")
    parser.add_argument("--next-phase", type=Path, default=ROOT / "docs" / "next_phase_implementation.md")
    parser.add_argument("--next-experiment-registry", type=Path, default=ROOT / "docs" / "next_experiment_registry.md")
    parser.add_argument("--next-experiment-registry-csv", type=Path, default=ROOT / "docs" / "next_experiment_registry.csv")
    parser.add_argument("--external-dependency-readiness-audit", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.md")
    parser.add_argument("--external-dependency-readiness-csv", type=Path, default=ROOT / "docs" / "external_dependency_readiness_audit.csv")
    parser.add_argument("--external-dependency-readiness-json", type=Path, default=ROOT / "outputs" / "evaluations" / "external_dependency_readiness_audit_v1.json")
    parser.add_argument("--openvla-bridge-report", type=Path, default=ROOT / "docs" / "openvla_dataset_bridge_report.md")
    parser.add_argument("--openvla-bridge-json", type=Path, default=ROOT / "outputs" / "evaluations" / "openvla_dataset_bridge_v1.json")
    parser.add_argument("--openvla-bridge-samples", type=Path, default=ROOT / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "samples.jsonl")
    parser.add_argument("--openvla-bridge-manifest", type=Path, default=ROOT / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "manifest.json")
    parser.add_argument("--openvla-bridge-preview", type=Path, default=ROOT / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "preview_grid.png")
    parser.add_argument("--openvla-bridge-gallery", type=Path, default=ROOT / "docs" / "openvla_bridge_gallery.html")
    parser.add_argument("--openvla-feasibility-report", type=Path, default=ROOT / "docs" / "openvla_feasibility_report.md")
    parser.add_argument("--openvla-feasibility-json", type=Path, default=ROOT / "outputs" / "evaluations" / "openvla_feasibility_check_v1.json")
    parser.add_argument("--robot-vla-handoff-report", type=Path, default=ROOT / "docs" / "robot_vla_action_head_handoff.md")
    parser.add_argument("--robot-vla-handoff-json", type=Path, default=ROOT / "outputs" / "evaluations" / "robot_vla_action_head_handoff_v1.json")
    parser.add_argument("--robot-vla-remote-pack-report", type=Path, default=ROOT / "docs" / "robot_vla_remote_run_pack.md")
    parser.add_argument("--robot-vla-remote-pack-json", type=Path, default=ROOT / "outputs" / "evaluations" / "robot_vla_remote_run_pack_v1.json")
    parser.add_argument("--robot-vla-remote-pack-archive", type=Path, default=ROOT / "outputs" / "robot_vla_remote_run_pack" / "robot_vla_remote_run_pack_v1.zip")
    parser.add_argument("--robot-vla-remote-pack-dir", type=Path, default=ROOT / "outputs" / "robot_vla_remote_run_pack" / "robot_vla_remote_run_pack_v1")
    parser.add_argument("--robot-vla-remote-intake-report", type=Path, default=ROOT / "docs" / "robot_vla_remote_result_intake.md")
    parser.add_argument("--robot-vla-remote-intake-csv", type=Path, default=ROOT / "docs" / "robot_vla_remote_result_intake.csv")
    parser.add_argument("--robot-vla-remote-intake-json", type=Path, default=ROOT / "outputs" / "evaluations" / "robot_vla_remote_result_intake_v1.json")
    parser.add_argument("--command-index", type=Path, default=ROOT / "docs" / "reproducible_command_index.md")
    parser.add_argument("--method-cards", type=Path, default=ROOT / "docs" / "method_cards.md")
    parser.add_argument("--result-matrix", type=Path, default=ROOT / "docs" / "result_matrix.md")
    parser.add_argument("--method-comparison-dashboard", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.md")
    parser.add_argument("--method-comparison-dashboard-html", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.html")
    parser.add_argument("--method-comparison-dashboard-csv", type=Path, default=ROOT / "docs" / "method_comparison_dashboard.csv")
    parser.add_argument("--stage-comparison-report", type=Path, default=ROOT / "docs" / "stage_comparison_report.md")
    parser.add_argument("--stage-comparison-csv", type=Path, default=ROOT / "docs" / "stage_comparison_report.csv")
    parser.add_argument("--task-bc-stage-report", type=Path, default=ROOT / "docs" / "task_bc_stage_report.md")
    parser.add_argument("--task-bc-stage-csv", type=Path, default=ROOT / "docs" / "task_bc_stage_report.csv")
    parser.add_argument("--trajectory-act-stage-report", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.md")
    parser.add_argument("--trajectory-act-stage-csv", type=Path, default=ROOT / "docs" / "trajectory_act_stage_report.csv")
    parser.add_argument("--trajectory-act-experiment-record", type=Path, default=ROOT / "docs" / "trajectory_act_experiment_record.md")
    parser.add_argument("--trajectory-act-experiment-record-csv", type=Path, default=ROOT / "docs" / "trajectory_act_experiment_record.csv")
    parser.add_argument("--trajectory-act-diagnosis", type=Path, default=ROOT / "docs" / "trajectory_act_failure_diagnosis.md")
    parser.add_argument("--trajectory-act-diagnosis-csv", type=Path, default=ROOT / "docs" / "trajectory_act_failure_diagnosis.csv")
    parser.add_argument("--trajectory-act-conclusion-brief", type=Path, default=ROOT / "docs" / "trajectory_act_conclusion_brief.md")
    parser.add_argument("--trajectory-act-conclusion-csv", type=Path, default=ROOT / "docs" / "trajectory_act_conclusion_brief.csv")
    parser.add_argument("--trajectory-act-slow-viewer-guide", type=Path, default=ROOT / "docs" / "trajectory_act_slow_viewer_guide.md")
    parser.add_argument("--trajectory-act-slow-viewer-csv", type=Path, default=ROOT / "docs" / "trajectory_act_slow_viewer_guide.csv")
    parser.add_argument("--trajectory-phase-template-report", type=Path, default=ROOT / "docs" / "trajectory_phase_template_bc_report.md")
    parser.add_argument("--trajectory-phase-template-csv", type=Path, default=ROOT / "docs" / "trajectory_phase_template_bc_report.csv")
    parser.add_argument("--trajectory-phase-template-json", type=Path, default=ROOT / "outputs" / "evaluations" / "trajectory_phase_template_bc_v1.json")
    parser.add_argument("--trajectory-phase-template-model", type=Path, default=ROOT / "outputs" / "trajectory_phase_template_bc" / "trajectory_phase_template_bc_20260720_160007.npz")
    parser.add_argument("--grasp-gated-trajectory-knn-report", type=Path, default=ROOT / "docs" / "grasp_gated_trajectory_knn_report.md")
    parser.add_argument("--grasp-gated-trajectory-knn-csv", type=Path, default=ROOT / "docs" / "grasp_gated_trajectory_knn_report.csv")
    parser.add_argument("--grasp-gated-trajectory-knn-json", type=Path, default=ROOT / "outputs" / "evaluations" / "grasp_gated_trajectory_knn_v1.json")
    parser.add_argument("--grasp-gated-trajectory-knn-runner", type=Path, default=ROOT / "scripts" / "run_grasp_gated_trajectory_knn_policy.py")
    parser.add_argument("--grasp-gated-trajectory-knn-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_grasp_gated_trajectory_knn.py")
    parser.add_argument("--preference-trajectory-post-training-report", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_report.md")
    parser.add_argument("--preference-trajectory-post-training-csv", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_report.csv")
    parser.add_argument("--preference-trajectory-post-training-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_trajectory_post_training_v1.json")
    parser.add_argument("--preference-trajectory-post-training-model", type=Path, default=ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_20260720_165005.npz")
    parser.add_argument("--preference-trajectory-post-training-trainer", type=Path, default=ROOT / "scripts" / "train_preference_trajectory_post_training.py")
    parser.add_argument("--preference-trajectory-post-training-runner", type=Path, default=ROOT / "scripts" / "run_preference_trajectory_post_training_policy.py")
    parser.add_argument("--preference-trajectory-post-training-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_preference_trajectory_post_training.py")
    parser.add_argument("--preference-ranked-objective-summary", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_objective_summary.md")
    parser.add_argument("--preference-ranked-objective-report", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_objective_report.md")
    parser.add_argument("--preference-ranked-objective-csv", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_objective_report.csv")
    parser.add_argument("--preference-ranked-objective-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_trajectory_post_training_v1_ranked_objective_candidate.json")
    parser.add_argument("--preference-ranked-objective-model", type=Path, default=ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz")
    parser.add_argument("--preference-ranked-objective-video", type=Path, default=ROOT / "outputs" / "videos" / "preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4")
    parser.add_argument("--preference-ranked-fast-summary", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_fast_summary.md")
    parser.add_argument("--preference-ranked-fast-report", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_fast_report.md")
    parser.add_argument("--preference-ranked-fast-csv", type=Path, default=ROOT / "docs" / "preference_trajectory_post_training_v1_ranked_fast_report.csv")
    parser.add_argument("--preference-ranked-fast-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_trajectory_post_training_v1_ranked_fast_candidate.json")
    parser.add_argument("--preference-ranked-fast-model", type=Path, default=ROOT / "outputs" / "preference_post_training" / "preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz")
    parser.add_argument("--preference-ranked-fast-video", type=Path, default=ROOT / "outputs" / "videos" / "preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4")
    parser.add_argument("--preference-contact-aware-trajectory-post-training-report", type=Path, default=ROOT / "docs" / "preference_contact_aware_trajectory_post_training_report.md")
    parser.add_argument("--preference-contact-aware-trajectory-post-training-csv", type=Path, default=ROOT / "docs" / "preference_contact_aware_trajectory_post_training_report.csv")
    parser.add_argument("--preference-contact-aware-trajectory-post-training-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_contact_aware_trajectory_post_training_v1_candidate.json")
    parser.add_argument("--preference-contact-aware-trajectory-post-training-model", type=Path, default=ROOT / "outputs" / "preference_post_training" / "preference_contact_aware_trajectory_post_training_20260721_000449.npz")
    parser.add_argument("--preference-contact-aware-trajectory-post-training-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_preference_contact_aware_trajectory_post_training.py")
    parser.add_argument("--preference-ranked-trajectory-post-training-report", type=Path, default=ROOT / "docs" / "preference_ranked_trajectory_post_training_report.md")
    parser.add_argument("--preference-ranked-trajectory-post-training-csv", type=Path, default=ROOT / "docs" / "preference_ranked_trajectory_post_training_report.csv")
    parser.add_argument("--preference-ranked-trajectory-post-training-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_ranked_trajectory_post_training_v1_candidate.json")
    parser.add_argument("--preference-ranked-trajectory-post-training-model", type=Path, default=ROOT / "outputs" / "preference_post_training" / "preference_ranked_trajectory_post_training_20260721_031024.npz")
    parser.add_argument("--preference-ranked-trajectory-post-training-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_preference_ranked_trajectory_post_training.py")
    parser.add_argument("--preference-post-training-upgrade-gate", type=Path, default=ROOT / "docs" / "preference_post_training_upgrade_gate.md")
    parser.add_argument("--preference-post-training-upgrade-gate-csv", type=Path, default=ROOT / "docs" / "preference_post_training_upgrade_gate.csv")
    parser.add_argument("--preference-post-training-upgrade-gate-json", type=Path, default=ROOT / "outputs" / "evaluations" / "preference_post_training_upgrade_gate_v1.json")
    parser.add_argument("--preference-post-training-upgrade-gate-script", type=Path, default=ROOT / "scripts" / "build_preference_post_training_upgrade_gate.py")
    parser.add_argument("--preference-post-training-ablation", type=Path, default=ROOT / "docs" / "preference_post_training_ablation_matrix.md")
    parser.add_argument("--preference-post-training-ablation-csv", type=Path, default=ROOT / "docs" / "preference_post_training_ablation_matrix.csv")
    parser.add_argument("--preference-post-training-ablation-script", type=Path, default=ROOT / "scripts" / "build_preference_post_training_ablation_matrix.py")
    parser.add_argument("--contact-phase-gated-torch-act-report", type=Path, default=ROOT / "docs" / "contact_phase_gated_torch_act_report.md")
    parser.add_argument("--contact-phase-gated-torch-act-csv", type=Path, default=ROOT / "docs" / "contact_phase_gated_torch_act_report.csv")
    parser.add_argument("--contact-phase-gated-torch-act-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_phase_gated_torch_act_v1_candidate.json")
    parser.add_argument("--contact-phase-gated-torch-act-model", type=Path, default=ROOT / "outputs" / "torch_act" / "contact_phase_gated_torch_act_v1_candidate_20260721_003304.pt")
    parser.add_argument("--contact-phase-gated-torch-act-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_contact_phase_gated_torch_act.py")
    parser.add_argument("--contact-aware-phase-gated-torch-act-report", type=Path, default=ROOT / "docs" / "contact_aware_phase_gated_torch_act_report.md")
    parser.add_argument("--contact-aware-phase-gated-torch-act-csv", type=Path, default=ROOT / "docs" / "contact_aware_phase_gated_torch_act_report.csv")
    parser.add_argument("--contact-aware-phase-gated-torch-act-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_aware_phase_gated_torch_act_v1_candidate.json")
    parser.add_argument("--contact-aware-phase-gated-torch-act-model", type=Path, default=ROOT / "outputs" / "torch_act" / "contact_aware_phase_gated_torch_act_v1_candidate_20260721_004944.pt")
    parser.add_argument("--contact-aware-phase-gated-torch-act-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_contact_aware_phase_gated_torch_act.py")
    parser.add_argument("--contact-stage-subpolicy-report", type=Path, default=ROOT / "docs" / "contact_stage_subpolicy_report.md")
    parser.add_argument("--contact-stage-subpolicy-csv", type=Path, default=ROOT / "docs" / "contact_stage_subpolicy_report.csv")
    parser.add_argument("--contact-stage-subpolicy-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_stage_subpolicy_v1_candidate.json")
    parser.add_argument("--contact-stage-subpolicy-runner", type=Path, default=ROOT / "scripts" / "run_contact_stage_subpolicy.py")
    parser.add_argument("--contact-stage-subpolicy-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_contact_stage_subpolicy.py")
    parser.add_argument("--contact-stage-demo-torch-act-report", type=Path, default=ROOT / "docs" / "contact_stage_demo_torch_act_report.md")
    parser.add_argument("--contact-stage-demo-torch-act-csv", type=Path, default=ROOT / "docs" / "contact_stage_demo_torch_act_report.csv")
    parser.add_argument("--contact-stage-demo-torch-act-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_stage_demo_torch_act_v1_candidate.json")
    parser.add_argument("--contact-stage-demo-torch-act-model", type=Path, default=ROOT / "outputs" / "torch_act" / "contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt")
    parser.add_argument("--contact-stage-demo-torch-act-collector", type=Path, default=ROOT / "scripts" / "collect_contact_stage_demos.py")
    parser.add_argument("--contact-stage-demo-torch-act-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_contact_stage_demo_torch_act.py")
    parser.add_argument("--contact-stage-demo-run-dir", type=Path, default=ROOT / "data" / "demos" / "contact_stage_demo_place_blue_cube_blue_pad_medium_v1")
    parser.add_argument("--contact-stage-phase-action-head-report", type=Path, default=ROOT / "docs" / "contact_stage_phase_action_head_report.md")
    parser.add_argument("--contact-stage-phase-action-head-csv", type=Path, default=ROOT / "docs" / "contact_stage_phase_action_head_report.csv")
    parser.add_argument("--contact-stage-phase-action-head-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_stage_phase_action_head_v1_candidate.json")
    parser.add_argument("--contact-stage-phase-action-head-model", type=Path, default=ROOT / "outputs" / "phase_action_head" / "contact_stage_phase_action_head_v1_candidate_20260721_020941.npz")
    parser.add_argument("--contact-stage-phase-action-head-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_contact_stage_phase_action_head.py")
    parser.add_argument("--contact-hold-weighted-torch-act-report", type=Path, default=ROOT / "docs" / "contact_hold_weighted_torch_act_report.md")
    parser.add_argument("--contact-hold-weighted-torch-act-csv", type=Path, default=ROOT / "docs" / "contact_hold_weighted_torch_act_report.csv")
    parser.add_argument("--contact-hold-weighted-torch-act-json", type=Path, default=ROOT / "outputs" / "evaluations" / "contact_hold_weighted_torch_act_v1_candidate.json")
    parser.add_argument("--contact-hold-weighted-torch-act-model", type=Path, default=ROOT / "outputs" / "torch_act" / "contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt")
    parser.add_argument("--contact-hold-weighted-torch-act-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_contact_hold_weighted_torch_act.py")
    parser.add_argument("--gripper-timing-contact-probe-report", type=Path, default=ROOT / "docs" / "gripper_timing_contact_probe_report.md")
    parser.add_argument("--gripper-timing-contact-probe-csv", type=Path, default=ROOT / "docs" / "gripper_timing_contact_probe_report.csv")
    parser.add_argument("--gripper-timing-contact-probe-json", type=Path, default=ROOT / "outputs" / "evaluations" / "gripper_timing_contact_probe_v1_candidate.json")
    parser.add_argument("--gripper-timing-contact-probe-video", type=Path, default=ROOT / "outputs" / "videos" / "gripper_timing_contact_probe_v1_candidate_seed0.mp4")
    parser.add_argument("--gripper-timing-contact-probe-runner", type=Path, default=ROOT / "scripts" / "run_gripper_timing_probe.py")
    parser.add_argument("--gripper-timing-contact-probe-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_gripper_timing_probe.py")
    parser.add_argument("--timing-aware-trajectory-prior-residual-report", type=Path, default=ROOT / "docs" / "timing_aware_trajectory_prior_residual_bc_report.md")
    parser.add_argument("--timing-aware-trajectory-prior-residual-csv", type=Path, default=ROOT / "docs" / "timing_aware_trajectory_prior_residual_bc_report.csv")
    parser.add_argument("--timing-aware-trajectory-prior-residual-json", type=Path, default=ROOT / "outputs" / "evaluations" / "timing_aware_trajectory_prior_residual_bc_v1_candidate.json")
    parser.add_argument("--timing-aware-trajectory-prior-residual-video", type=Path, default=ROOT / "outputs" / "videos" / "timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4")
    parser.add_argument("--timing-aware-trajectory-prior-residual-model", type=Path, default=ROOT / "outputs" / "timing_aware_trajectory_prior_residual_bc" / "timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz")
    parser.add_argument("--timing-aware-trajectory-prior-residual-common", type=Path, default=ROOT / "scripts" / "timing_aware_trajectory_prior_residual_common.py")
    parser.add_argument("--timing-aware-trajectory-prior-residual-trainer", type=Path, default=ROOT / "scripts" / "train_timing_aware_trajectory_prior_residual_bc.py")
    parser.add_argument("--timing-aware-trajectory-prior-residual-runner", type=Path, default=ROOT / "scripts" / "run_timing_aware_trajectory_prior_residual_policy.py")
    parser.add_argument("--timing-aware-trajectory-prior-residual-evaluator", type=Path, default=ROOT / "scripts" / "evaluate_timing_aware_trajectory_prior_residual_bc.py")
    parser.add_argument("--control-safety-sweep", type=Path, default=ROOT / "docs" / "control_safety_sweep.md")
    parser.add_argument("--control-safety-sweep-csv", type=Path, default=ROOT / "docs" / "control_safety_sweep.csv")
    parser.add_argument("--control-safety-sweep-json", type=Path, default=ROOT / "outputs" / "evaluations" / "control_safety_sweep_v1.json")
    parser.add_argument("--action-head-stage-report", type=Path, default=ROOT / "docs" / "action_head_stage_report.md")
    parser.add_argument("--action-head-stage-csv", type=Path, default=ROOT / "docs" / "action_head_stage_report.csv")
    parser.add_argument("--action-head-control-safety-sweep", type=Path, default=ROOT / "docs" / "action_head_control_safety_sweep.md")
    parser.add_argument("--action-head-control-safety-sweep-csv", type=Path, default=ROOT / "docs" / "action_head_control_safety_sweep.csv")
    parser.add_argument("--action-head-control-safety-sweep-json", type=Path, default=ROOT / "outputs" / "evaluations" / "action_head_control_safety_sweep_v1.json")
    parser.add_argument("--strict-grasp-audit", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.md")
    parser.add_argument("--strict-grasp-audit-csv", type=Path, default=ROOT / "docs" / "strict_grasp_success_audit.csv")
    parser.add_argument("--strict-grasp-audit-json", type=Path, default=ROOT / "outputs" / "evaluations" / "strict_grasp_success_audit_v1.json")
    parser.add_argument("--stage-evidence-index", type=Path, default=ROOT / "docs" / "stage_evidence_index.md")
    parser.add_argument("--stage-evidence-csv", type=Path, default=ROOT / "docs" / "stage_evidence_index.csv")
    parser.add_argument("--stage-showcase-index", type=Path, default=ROOT / "docs" / "stage_showcase_index.md")
    parser.add_argument("--stage-showcase-html", type=Path, default=ROOT / "docs" / "stage_showcase_index.html")
    parser.add_argument("--stage-reproduction-runbook", type=Path, default=ROOT / "docs" / "stage_reproduction_runbook.md")
    parser.add_argument("--stage-reproduction-csv", type=Path, default=ROOT / "docs" / "stage_reproduction_runbook.csv")
    parser.add_argument("--research-evidence-map", type=Path, default=ROOT / "docs" / "research_evidence_map.md")
    parser.add_argument("--research-evidence-csv", type=Path, default=ROOT / "docs" / "research_evidence_map.csv")
    parser.add_argument("--research-showcase-plan", type=Path, default=ROOT / "docs" / "research_question_showcase_plan.md")
    parser.add_argument("--research-showcase-csv", type=Path, default=ROOT / "docs" / "research_question_showcase_plan.csv")
    parser.add_argument("--claim-evidence-traceability", type=Path, default=ROOT / "docs" / "claim_evidence_traceability.md")
    parser.add_argument("--claim-evidence-csv", type=Path, default=ROOT / "docs" / "claim_evidence_traceability.csv")
    parser.add_argument("--claim-video-playback-index", type=Path, default=ROOT / "docs" / "claim_video_playback_index.md")
    parser.add_argument("--claim-video-playback-csv", type=Path, default=ROOT / "docs" / "claim_video_playback_index.csv")
    parser.add_argument("--goal-completion-audit", type=Path, default=ROOT / "docs" / "goal_completion_audit.md")
    parser.add_argument("--goal-completion-csv", type=Path, default=ROOT / "docs" / "goal_completion_audit.csv")
    parser.add_argument("--method-stage-audit", type=Path, default=ROOT / "docs" / "method_stage_audit.md")
    parser.add_argument("--method-stage-audit-csv", type=Path, default=ROOT / "docs" / "method_stage_audit.csv")
    parser.add_argument("--method-evidence-gate", type=Path, default=ROOT / "docs" / "method_evidence_gate.md")
    parser.add_argument("--method-evidence-csv", type=Path, default=ROOT / "docs" / "method_evidence_gate.csv")
    parser.add_argument("--version-naming-spec", type=Path, default=ROOT / "docs" / "version_naming_and_gate_spec.md")
    parser.add_argument("--version-naming-spec-csv", type=Path, default=ROOT / "docs" / "version_naming_and_gate_spec.csv")
    parser.add_argument("--version-naming-spec-json", type=Path, default=ROOT / "outputs" / "evaluations" / "version_naming_and_gate_spec_v1.json")
    parser.add_argument("--final-method-index", type=Path, default=ROOT / "docs" / "final_method_version_index.md")
    parser.add_argument("--final-method-index-csv", type=Path, default=ROOT / "docs" / "final_method_version_index.csv")
    parser.add_argument("--thesis-results-chapter", type=Path, default=ROOT / "docs" / "thesis_results_chapter_draft.md")
    parser.add_argument("--thesis-appendix", type=Path, default=ROOT / "docs" / "thesis_appendix_tables.md")
    parser.add_argument("--thesis-method-table", type=Path, default=ROOT / "docs" / "thesis_method_comparison_table.csv")
    parser.add_argument("--thesis-domain-table", type=Path, default=ROOT / "docs" / "thesis_domain_randomization_table.csv")
    parser.add_argument("--thesis-visual-evidence", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.md")
    parser.add_argument("--thesis-visual-evidence-csv", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.csv")
    parser.add_argument("--thesis-visual-evidence-html", type=Path, default=ROOT / "docs" / "thesis_visual_evidence_index.html")
    parser.add_argument("--defense-qa-playbook", type=Path, default=ROOT / "docs" / "defense_qa_playbook.md")
    parser.add_argument("--defense-qa-playbook-csv", type=Path, default=ROOT / "docs" / "defense_qa_playbook.csv")
    parser.add_argument("--defense-qa-playbook-html", type=Path, default=ROOT / "docs" / "defense_qa_playbook.html")
    parser.add_argument("--version-lineage", type=Path, default=ROOT / "docs" / "version_lineage_index.md")
    parser.add_argument("--version-lineage-csv", type=Path, default=ROOT / "docs" / "version_lineage_index.csv")
    parser.add_argument("--version-lineage-html", type=Path, default=ROOT / "docs" / "version_lineage_index.html")
    parser.add_argument("--showcase-doc", type=Path, default=ROOT / "docs" / "video_showcase.md")
    parser.add_argument("--showcase-dir", type=Path, default=ROOT / "outputs" / "showcase")
    parser.add_argument("--video-evidence-index", type=Path, default=ROOT / "docs" / "video_evidence_index.md")
    parser.add_argument("--video-evidence-csv", type=Path, default=ROOT / "docs" / "video_evidence_index.csv")
    parser.add_argument("--candidate-diagnostic-video-index", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.md")
    parser.add_argument("--candidate-diagnostic-video-csv", type=Path, default=ROOT / "docs" / "candidate_diagnostic_video_index.csv")
    parser.add_argument("--video-quality-audit", type=Path, default=ROOT / "docs" / "video_quality_audit.md")
    parser.add_argument("--video-quality-audit-csv", type=Path, default=ROOT / "docs" / "video_quality_audit.csv")
    parser.add_argument("--video-evidence-gallery", type=Path, default=ROOT / "docs" / "video_evidence_gallery.html")
    parser.add_argument("--failure-mode-taxonomy", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.md")
    parser.add_argument("--failure-mode-taxonomy-csv", type=Path, default=ROOT / "docs" / "failure_mode_taxonomy.csv")
    parser.add_argument("--showcase-launcher", type=Path, default=ROOT / "scripts" / "showcase_launcher.py")
    parser.add_argument("--showcase-launcher-guide", type=Path, default=ROOT / "docs" / "showcase_launcher_guide.md")
    parser.add_argument("--presentation-pack-doc", type=Path, default=ROOT / "docs" / "presentation_video_pack.md")
    parser.add_argument("--presentation-pack-dir", type=Path, default=ROOT / "outputs" / "presentation_clips")
    parser.add_argument("--presentation-pack-manifest", type=Path, default=ROOT / "outputs" / "presentation_clips" / "presentation_video_pack_manifest.json")
    parser.add_argument("--video-presentation-storyboard", type=Path, default=ROOT / "docs" / "video_presentation_storyboard.md")
    parser.add_argument("--video-presentation-storyboard-html", type=Path, default=ROOT / "docs" / "video_presentation_storyboard.html")
    parser.add_argument("--videos", type=Path, default=ROOT / "outputs" / "videos")
    parser.add_argument("--min-methods", type=int, default=25)
    parser.add_argument("--min-language-rows", type=int, default=22)
    parser.add_argument("--min-data-efficiency-rows", type=int, default=24)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_video(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found: {path}")
    stream = streams[0]
    if int(stream.get("width", 0)) <= 0 or int(stream.get("height", 0)) <= 0:
        raise RuntimeError(f"invalid video dimensions: {path}")
    return {key: str(value) for key, value in stream.items()}


def verify_registered_methods(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    versions = read_json(args.versions)
    methods = versions["methods"]
    if len(methods) < args.min_methods:
        raise RuntimeError(f"expected at least {args.min_methods} methods, found {len(methods)}")

    summary_rows = {row["version"]: row for row in read_csv(args.summary)}
    missing_summary = [method["version"] for method in methods if method["version"] not in summary_rows]
    if missing_summary:
        raise RuntimeError(f"missing summary rows: {missing_summary}")

    checked_videos: list[str] = []
    for method in methods:
        artifact = ROOT / method["artifact"]
        if method["artifact"] and not artifact.exists():
            raise FileNotFoundError(f"missing artifact for {method['version']}: {artifact}")

        clip = ROOT / method["clip"]
        metadata = clip.with_suffix(".json")
        ffprobe_video(clip)
        read_json(metadata)
        checked_videos.append(clip.as_posix())

    return [method["version"] for method in methods], checked_videos


def verify_language(args: argparse.Namespace) -> list[str]:
    rows = read_csv(args.language_summary)
    if len(rows) < args.min_language_rows:
        raise RuntimeError(f"expected at least {args.min_language_rows} language rows, found {len(rows)}")
    versions = [row["version"] for row in rows]
    if "expert_scripted_language_v1" not in versions:
        raise RuntimeError("language summary is missing expert_scripted_language_v1")
    if "structured_waypoint_policy_v1" not in versions:
        raise RuntimeError("language summary is missing structured_waypoint_policy_v1")
    if "object_language_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing object_language_action_head_lite_v1")
    if "reward_weighted_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing reward_weighted_action_head_lite_v1")
    if "phase_conditioned_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing phase_conditioned_action_head_lite_v1")
    if "trajectory_knn_chunk_bc_v1" not in versions:
        raise RuntimeError("language summary is missing trajectory_knn_chunk_bc_v1")
    if "torch_act_state_chunk_v1" not in versions:
        raise RuntimeError("language summary is missing torch_act_state_chunk_v1")
    if "torch_act_state_chunk_cuda_v1" not in versions:
        raise RuntimeError("language summary is missing torch_act_state_chunk_cuda_v1")
    if "torch_act_cvae_state_chunk_v1" not in versions:
        raise RuntimeError("language summary is missing torch_act_cvae_state_chunk_v1")
    if "torch_diffusion_policy_state_chunk_v1" not in versions:
        raise RuntimeError("language summary is missing torch_diffusion_policy_state_chunk_v1")
    if "visual_feature_act_lite_v1" not in versions:
        raise RuntimeError("language summary is missing visual_feature_act_lite_v1")
    if "vision_language_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing vision_language_action_head_lite_v1")
    if "clip_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing clip_action_head_lite_v1")
    if "adapter_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing adapter_action_head_lite_v1")
    if "lora_action_head_lite_v1" not in versions:
        raise RuntimeError("language summary is missing lora_action_head_lite_v1")
    return versions


def verify_resources(args: argparse.Namespace) -> list[str]:
    rows = read_csv(args.resource_summary)
    versions = [row["version"] for row in rows]
    if len(rows) < args.min_methods:
        raise RuntimeError(f"expected at least {args.min_methods} resource rows, found {len(rows)}")
    if "trajectory_conditioned_chunk_bc_v2" not in versions:
        raise RuntimeError("resource summary is missing trajectory_conditioned_chunk_bc_v2")
    if "trajectory_knn_chunk_bc_v1" not in versions:
        raise RuntimeError("resource summary is missing trajectory_knn_chunk_bc_v1")
    if "torch_act_state_chunk_v1" not in versions:
        raise RuntimeError("resource summary is missing torch_act_state_chunk_v1")
    if "torch_act_state_chunk_cuda_v1" not in versions:
        raise RuntimeError("resource summary is missing torch_act_state_chunk_cuda_v1")
    if "torch_act_cvae_state_chunk_v1" not in versions:
        raise RuntimeError("resource summary is missing torch_act_cvae_state_chunk_v1")
    if "torch_diffusion_policy_state_chunk_v1" not in versions:
        raise RuntimeError("resource summary is missing torch_diffusion_policy_state_chunk_v1")
    if "visual_feature_act_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing visual_feature_act_lite_v1")
    if "visual_act_cnn_cvae_v1" not in versions:
        raise RuntimeError("resource summary is missing visual_act_cnn_cvae_v1")
    if "multi_task_object_action_head_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing multi_task_object_action_head_lite_v1")
    if "reward_weighted_action_head_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing reward_weighted_action_head_lite_v1")
    if "phase_conditioned_action_head_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing phase_conditioned_action_head_lite_v1")
    if "clip_action_head_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing clip_action_head_lite_v1")
    if "adapter_action_head_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing adapter_action_head_lite_v1")
    if "lora_action_head_lite_v1" not in versions:
        raise RuntimeError("resource summary is missing lora_action_head_lite_v1")
    for row in rows:
        int(row["trainable_params"])
        float(row["artifact_size_mb"])
    if not args.resource_report.exists():
        raise FileNotFoundError(args.resource_report)
    text = args.resource_report.read_text(encoding="utf-8")
    required = ("模型资源与规模汇总", "可训练参数", "当前可写入论文的阶段性结论")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"resource report is missing sections/terms: {missing}")
    return versions


def verify_data_efficiency(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.data_efficiency_summary)
    if len(rows) < args.min_data_efficiency_rows:
        raise RuntimeError(f"expected at least {args.min_data_efficiency_rows} data-efficiency rows, found {len(rows)}")
    methods = {row["method_key"] for row in rows}
    budgets = {row["demo_budget"] for row in rows}
    splits = {row["split"] for row in rows}
    if not {"knn_bc", "object_action_head", "trajectory_knn"}.issubset(methods):
        raise RuntimeError(f"data-efficiency summary is missing methods: {methods}")
    if not {"10", "25", "50", "92"}.issubset(budgets):
        raise RuntimeError(f"data-efficiency summary is missing budgets: {budgets}")
    if not {"train_range", "heldout"}.issubset(splits):
        raise RuntimeError(f"data-efficiency summary is missing splits: {splits}")
    for row in rows:
        float(row["success_rate"])
        float(row["mean_target_distance"])
        int(row["stored_samples"])

    if not args.data_efficiency_report.exists():
        raise FileNotFoundError(args.data_efficiency_report)
    text = args.data_efficiency_report.read_text(encoding="utf-8")
    required = ("数据效率评测 v2", "阶段结论", "object_action_head", "trajectory_knn")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"data-efficiency report is missing sections/terms: {missing}")

    data = read_json(args.data_efficiency_json)
    if data.get("version") != "data_efficiency_budget_sweep_v2":
        raise RuntimeError("data-efficiency json has unexpected version")
    if len(data.get("rows", [])) < args.min_data_efficiency_rows:
        raise RuntimeError("data-efficiency json has too few rows")
    return rows


def verify_domain_randomization(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.domain_randomization_report.exists():
        raise FileNotFoundError(args.domain_randomization_report)
    if not args.domain_randomization_summary.exists():
        raise FileNotFoundError(args.domain_randomization_summary)
    if not args.domain_randomization_json.exists():
        raise FileNotFoundError(args.domain_randomization_json)

    text = args.domain_randomization_report.read_text(encoding="utf-8-sig")
    required = (
        "MuJoCo Domain Randomization 代理评测",
        "domain_randomization_eval_v1",
        "isaacsim",
        "False",
        "mujoco",
        "True",
        "low_friction_soft_grip",
        "high_friction_stiff_arm",
        "structured_waypoint_policy",
        "trajectory_knn_bc",
        "visual_act_cnn_cvae",
        "outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "不能写：Isaac domain randomization 已完成",
        "不能写：真实 WidowX 或真实机械臂迁移成功/失败已经验证",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"domain randomization report is missing required terms: {missing}")

    rows = read_csv(args.domain_randomization_summary)
    if len(rows) < 18:
        raise RuntimeError(f"domain randomization csv has too few rows: {len(rows)}")
    required_columns = {
        "version",
        "method_key",
        "method_version",
        "domain",
        "seed",
        "success",
        "target_distance",
        "arm_kp",
        "arm_force",
        "gripper_kp",
        "gripper_force",
        "friction",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"domain randomization csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    methods = {row["method_key"] for row in rows}
    domains = {row["domain"] for row in rows}
    if not {"structured_waypoint_policy", "trajectory_knn_bc", "visual_act_cnn_cvae"}.issubset(methods):
        raise RuntimeError(f"domain randomization csv is missing methods: {methods}")
    if not {"nominal", "low_friction_soft_grip", "high_friction_stiff_arm"}.issubset(domains):
        raise RuntimeError(f"domain randomization csv is missing domains: {domains}")
    for row in rows:
        if row["version"] != "domain_randomization_eval_v1":
            raise RuntimeError(f"unexpected domain randomization version: {row['version']}")
        float(row["target_distance"])
        float(row["arm_kp"])
        float(row["arm_force"])
        float(row["gripper_kp"])
        float(row["gripper_force"])
        float(row["friction"])

    data = read_json(args.domain_randomization_json)
    if data.get("version") != "domain_randomization_eval_v1":
        raise RuntimeError("domain randomization json has unexpected version")
    capabilities = data.get("capabilities", {})
    if capabilities.get("mujoco") is not True:
        raise RuntimeError("domain randomization json does not confirm MuJoCo availability")
    if capabilities.get("isaacsim") is not False:
        raise RuntimeError("domain randomization json should preserve local Isaac unavailability")
    if len(data.get("rows", [])) < 18:
        raise RuntimeError("domain randomization json has too few episode rows")
    if len(data.get("summary", [])) < 9:
        raise RuntimeError("domain randomization json has too few summary rows")
    return rows


def verify_isaac_domain_randomization_handoff(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (args.isaac_handoff_report, args.isaac_handoff_csv, args.isaac_handoff_json):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.isaac_handoff_report.read_text(encoding="utf-8-sig")
    required = (
        "Isaac Domain Randomization 运行交接门禁",
        "isaac_domain_randomization_handoff_v1",
        "isaac_domain_randomization_v1",
        "domain_randomization_eval_v1",
        "completed_prerequisite",
        "place_blue_cube_blue_pad",
        "nominal",
        "low_friction_soft_grip",
        "high_friction_stiff_arm",
        "success",
        "target_distance",
        "grasp_success",
        "sim_to_sim_gap",
        "outputs/evaluations/isaac_domain_randomization_v1.json",
        "outputs/videos/isaac_domain_randomization_v1_seed0.mp4",
        "不能写成 Isaac domain randomization 已完成",
        "不能写成真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Isaac handoff report is missing terms: {missing}")

    rows = read_csv(args.isaac_handoff_csv)
    if len(rows) < 24:
        raise RuntimeError(f"Isaac handoff csv has too few rows: {len(rows)}")
    required_columns = {"category", "key", "source", "required_value", "status", "note"}
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"Isaac handoff csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    domain_rows = [row for row in rows if row["category"] == "domain"]
    domains = {row["key"] for row in domain_rows}
    if not {"nominal", "low_friction_soft_grip", "high_friction_stiff_arm"}.issubset(domains):
        raise RuntimeError(f"Isaac handoff csv is missing domains: {domains}")
    metric_rows = [row for row in rows if row["category"] == "required_metric"]
    metrics = {row["key"] for row in metric_rows}
    if not {"success", "target_distance", "grasp_success", "object_z", "contact_count", "sim_to_sim_gap", "seed", "method_version"}.issubset(metrics):
        raise RuntimeError(f"Isaac handoff csv is missing metrics: {metrics}")
    return_file_rows = [row for row in rows if row["category"] == "required_return_file"]
    if len(return_file_rows) < 6:
        raise RuntimeError("Isaac handoff csv has too few required return files")
    boundary_rows = [row for row in rows if row["category"] == "paper_boundary"]
    if not any("不能写成 Isaac domain randomization 已完成" in row["key"] for row in boundary_rows):
        raise RuntimeError("Isaac handoff csv is missing Isaac paper boundary")

    data = read_json(args.isaac_handoff_json)
    if data.get("version") != "isaac_domain_randomization_handoff_v1":
        raise RuntimeError("Isaac handoff json has unexpected version")
    if data.get("status") != "completed_prerequisite":
        raise RuntimeError("Isaac handoff json has unexpected status")
    if data.get("source_version") != "domain_randomization_eval_v1":
        raise RuntimeError("Isaac handoff json has unexpected source version")
    if data.get("target_planned_version") != "isaac_domain_randomization_v1":
        raise RuntimeError("Isaac handoff json has unexpected target version")
    if data.get("can_register_completed_isaac_method") is not False:
        raise RuntimeError("Isaac handoff json must not allow completed Isaac registration")
    summary = data.get("source_summary", {})
    if int(summary.get("source_rows", 0)) < 18:
        raise RuntimeError("Isaac handoff source row count is too small")
    if len(data.get("required_return_files", [])) < 6:
        raise RuntimeError("Isaac handoff json has too few required return files")
    if "low_friction_soft_grip" not in data.get("domain_specs", {}):
        raise RuntimeError("Isaac handoff json is missing source domain specs")
    return rows


def verify_real_widowx_validation_handoff(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.real_widowx_handoff_report,
        args.real_widowx_handoff_csv,
        args.real_widowx_handoff_json,
        args.real_widowx_trial_template,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.real_widowx_handoff_report.read_text(encoding="utf-8-sig")
    required = (
        "Real WidowX Validation 运行交接门禁",
        "real_widowx_validation_handoff_v1",
        "real_widowx_validation_v1",
        "completed_prerequisite",
        "20-50 次真实 trial",
        "真实 trial 数",
        "急停按钮",
        "trial_id",
        "method_version",
        "object_start_pose",
        "target_distance_m",
        "video_path",
        "outputs/real_robot/real_widowx_validation_v1.csv",
        "outputs/videos/real_widowx_validation_v1_trial001.mp4",
        "不能写成真实 WidowX 验证已经完成",
        "不能用 MuJoCo 或 Isaac 视频代替真实相机视频",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"real WidowX handoff report is missing terms: {missing}")

    rows = read_csv(args.real_widowx_handoff_csv)
    if len(rows) < 45:
        raise RuntimeError(f"real WidowX handoff csv has too few rows: {len(rows)}")
    required_columns = {"category", "key", "source", "required_value", "status", "note"}
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"real WidowX handoff csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    safety_rows = [row for row in rows if row["category"] == "safety_gate"]
    if len(safety_rows) < 8:
        raise RuntimeError("real WidowX handoff has too few safety gates")
    trial_field_rows = [row for row in rows if row["category"] == "trial_field"]
    trial_fields = {row["key"] for row in trial_field_rows}
    required_fields = {
        "trial_id",
        "method_version",
        "task",
        "instruction",
        "robot_model",
        "camera_model",
        "calibration_file",
        "object_start_pose",
        "target_pose",
        "success",
        "target_distance_m",
        "grasp_success",
        "failure_reason",
        "video_path",
    }
    if not required_fields.issubset(trial_fields):
        raise RuntimeError(f"real WidowX handoff csv is missing trial fields: {sorted(required_fields - trial_fields)}")
    return_file_rows = [row for row in rows if row["category"] == "required_return_file"]
    if len(return_file_rows) < 7:
        raise RuntimeError("real WidowX handoff csv has too few required return files")
    if not any("不能用 MuJoCo 或 Isaac 视频代替真实相机视频" in row["key"] for row in rows if row["category"] == "paper_boundary"):
        raise RuntimeError("real WidowX handoff csv is missing simulation-substitution paper boundary")

    trial_rows = read_csv(args.real_widowx_trial_template)
    if len(trial_rows) != 50:
        raise RuntimeError(f"real WidowX trial template should have 50 rows, found {len(trial_rows)}")
    if not required_fields.issubset(trial_rows[0]):
        raise RuntimeError(f"real WidowX trial template is missing fields: {sorted(required_fields - set(trial_rows[0]))}")
    if not all(row["success"] == "pending_real_robot_run" for row in trial_rows):
        raise RuntimeError("real WidowX trial template should not contain completed success values")
    if len({row["planned_block"] for row in trial_rows}) < 5:
        raise RuntimeError("real WidowX trial template has too few planned blocks")

    data = read_json(args.real_widowx_handoff_json)
    if data.get("version") != "real_widowx_validation_handoff_v1":
        raise RuntimeError("real WidowX handoff json has unexpected version")
    if data.get("status") != "completed_prerequisite":
        raise RuntimeError("real WidowX handoff json has unexpected status")
    if data.get("target_planned_version") != "real_widowx_validation_v1":
        raise RuntimeError("real WidowX handoff json has unexpected target version")
    if data.get("can_register_completed_real_robot_validation") is not False:
        raise RuntimeError("real WidowX handoff json must not allow completed real-robot registration")
    if int(data.get("trial_template_rows", 0)) != 50:
        raise RuntimeError("real WidowX handoff json has unexpected trial template row count")
    if len(data.get("safety_gates", [])) < 8:
        raise RuntimeError("real WidowX handoff json has too few safety gates")
    if len(data.get("required_return_files", [])) < 7:
        raise RuntimeError("real WidowX handoff json has too few required return files")
    return rows


def verify_extra_videos(args: argparse.Namespace) -> list[str]:
    checked = []
    for stem in EXTRA_VIDEO_STEMS:
        path = args.videos / f"{stem}.mp4"
        metadata = path.with_suffix(".json")
        ffprobe_video(path)
        read_json(metadata)
        checked.append(path.as_posix())
    return checked


def verify_showcase(args: argparse.Namespace) -> list[str]:
    if not args.showcase_doc.exists():
        raise FileNotFoundError(args.showcase_doc)
    text = args.showcase_doc.read_text(encoding="utf-8")
    required = ("video_showcase_v1", "core_methods_grid.mp4", "all_registered_methods_grid.mp4", "language_generalization_grid.mp4", "00_defense_video_reel.mp4", "docs/presentation_video_pack.md")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"showcase doc is missing required terms: {missing}")

    manifest = args.showcase_dir / "video_showcase_manifest.json"
    data = read_json(manifest)
    if data.get("version") != "video_showcase_v1":
        raise RuntimeError("showcase manifest has unexpected version")

    checked = []
    for stem in SHOWCASE_VIDEO_STEMS:
        path = args.showcase_dir / f"{stem}.mp4"
        ffprobe_video(path)
        checked.append(path.as_posix())
    return checked


def verify_presentation_pack(args: argparse.Namespace) -> list[str]:
    manifest = read_json(args.presentation_pack_manifest)
    if manifest.get("version") != "presentation_video_pack_v1":
        raise RuntimeError("presentation video pack manifest has unexpected version")
    stages = manifest.get("stages", [])
    if len(stages) < 7:
        raise RuntimeError(f"presentation video pack has too few stages: {len(stages)}")
    required_stage_keys = {
        "01_task_data_oracle",
        "02_basic_bc_baselines",
        "03_trajectory_act_diffusion",
        "04_action_head_peft_proxy",
        "05_language_generalization",
        "06_domain_randomization_proxy",
        "07_candidate_diagnostics",
    }
    actual_stage_keys = {stage.get("key") for stage in stages}
    missing_stage_keys = required_stage_keys - actual_stage_keys
    if missing_stage_keys:
        raise RuntimeError(f"presentation video pack is missing stages: {sorted(missing_stage_keys)}")

    checked: list[str] = []
    master = manifest.get("master")
    if not master:
        raise RuntimeError("presentation video pack manifest is missing master reel")
    for item in [master, *stages]:
        output = ROOT / item["output"]
        stream = ffprobe_video(output)
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        duration = float(stream.get("duration", item.get("duration", 0.0)))
        if width != 1280 or height != 720:
            raise RuntimeError(f"presentation video has unexpected dimensions: {output} {width}x{height}")
        if duration < 9.0:
            raise RuntimeError(f"presentation video is too short: {output} {duration}")
        checked.append(output.as_posix())

    if not args.presentation_pack_doc.exists():
        raise FileNotFoundError(args.presentation_pack_doc)
    text = args.presentation_pack_doc.read_text(encoding="utf-8-sig")
    required = (
        "答辩视频片段包",
        "presentation_video_pack_v1",
        "00_defense_video_reel.mp4",
        "阶段 1：任务、示范与结构化上界",
        "阶段 3：Trajectory / ACT / Diffusion",
        "阶段 5：语言 / 空间泛化",
        "阶段 6：MuJoCo Domain Randomization 代理",
        "阶段 7：候选诊断与失败模式",
        "candidate_diagnostic_video_index_v1",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "domain_randomization_trajectory_knn_low_friction_v1",
        "video_presentation_storyboard_v1",
        "docs\\video_presentation_storyboard.html",
        "docs/method_stage_audit.md",
        "outputs/videos",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"presentation video pack doc is missing required terms: {missing}")
    return checked


def verify_video_presentation_storyboard(args: argparse.Namespace) -> None:
    if not args.video_presentation_storyboard.exists():
        raise FileNotFoundError(args.video_presentation_storyboard)
    if not args.video_presentation_storyboard_html.exists():
        raise FileNotFoundError(args.video_presentation_storyboard_html)

    md_text = args.video_presentation_storyboard.read_text(encoding="utf-8-sig")
    html_text = args.video_presentation_storyboard_html.read_text(encoding="utf-8-sig")
    required = (
        "视频展示讲稿与时间线",
        "video_presentation_storyboard_v1",
        "总览 Reel 时间线",
        "0-10s",
        "50-60s",
        "60-70s",
        "讲稿提示",
        "量化证据",
        "论文红线",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "OpenVLA、Isaac、真实 WidowX 仍是后续阶段",
        "候选诊断视频只证明失败模式和局部现象",
    )
    missing_md = [item for item in required if item not in md_text]
    if missing_md:
        raise RuntimeError(f"video presentation storyboard markdown is missing terms: {missing_md}")
    missing_html = [item for item in required[:8] if item not in html_text]
    if missing_html:
        raise RuntimeError(f"video presentation storyboard html is missing terms: {missing_html}")
    if md_text.count("### 阶段 ") < 6:
        raise RuntimeError("video presentation storyboard has too few stage sections")
    semantic_checks = {
        "02_basic_bc_baselines.mp4": "普通 BC 不能写成语言理解或泛化策略",
        "03_trajectory_act_diffusion.mp4": "不能写成完整官方 ACT",
        "04_action_head_peft_proxy.mp4": "不能写成真实 pretrained VLA 后训练",
        "05_language_generalization.mp4": "语言 token、对象特征或 CLIP 代理不能等同于真实 VLA 语言理解",
        "06_domain_randomization_proxy.mp4": "不能写成 Isaac domain randomization",
        "07_candidate_diagnostics.mp4": "候选诊断视频只证明失败模式和局部现象",
    }
    for marker, expected in semantic_checks.items():
        position = md_text.find(marker)
        if position < 0:
            raise RuntimeError(f"video presentation storyboard is missing marker: {marker}")
        window = md_text[position : position + 1400]
        if expected not in window:
            raise RuntimeError(f"video presentation storyboard has mismatched stage evidence near {marker}: missing {expected}")
    if html_text.count("<video ") < 7:
        raise RuntimeError("video presentation storyboard html has too few videos")
    html_dir = args.video_presentation_storyboard_html.parent
    refs = []
    for part in html_text.split('src="')[1:]:
        refs.append(part.split('"', 1)[0])
    for ref in refs:
        path = (html_dir / ref).resolve()
        if not path.exists():
            raise FileNotFoundError(path)


def verify_defense_video_playlist(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.defense_video_playlist.exists():
        raise FileNotFoundError(args.defense_video_playlist)
    if not args.defense_video_playlist_html.exists():
        raise FileNotFoundError(args.defense_video_playlist_html)
    if not args.defense_video_playlist_csv.exists():
        raise FileNotFoundError(args.defense_video_playlist_csv)

    md_text = args.defense_video_playlist.read_text(encoding="utf-8-sig")
    html_text = args.defense_video_playlist_html.read_text(encoding="utf-8-sig")
    rows = read_csv(args.defense_video_playlist_csv)
    if len(rows) < 15:
        raise RuntimeError("defense video playlist has too few rows")
    required_terms = (
        "defense_video_playlist_v1",
        "阶段 claim 播放顺序",
        "Core V2 OOD 对照播放顺序",
        "候选诊断负例播放顺序",
        "讲解提示",
        "论文红线",
        "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "candidate_diagnostic_montage_v1",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "Start-Process",
        "完整 viewer 命令",
    )
    missing_md = [item for item in required_terms if item not in md_text]
    if missing_md:
        raise RuntimeError(f"defense video playlist markdown is missing terms: {missing_md}")
    missing_html = [item for item in required_terms[:5] if item not in html_text]
    if missing_html:
        raise RuntimeError(f"defense video playlist html is missing terms: {missing_html}")
    if html_text.count("<video ") < 14:
        raise RuntimeError("defense video playlist html has too few video elements")
    if html_text.count("<img ") < 1:
        raise RuntimeError("defense video playlist html is missing image evidence")

    sections = {row["section"] for row in rows}
    if {
        "阶段 claim 播放顺序",
        "Core V2 OOD 对照播放顺序",
        "候选诊断负例播放顺序",
    } - sections:
        raise RuntimeError("defense video playlist CSV is missing required sections")
    ids = {row["id"] for row in rows}
    required_ids = {
        "C03",
        "C09",
        "core_v2_ood_hard_distractors_success_v1",
        "core_v2_ood_paraphrase_failure_v1",
        "candidate_diagnostic_montage_v1",
        "trajectory_prior_residual_bc_v1_candidate",
        "grasp_gated_torch_act_state_chunk_v1_candidate",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
    }
    missing_ids = sorted(required_ids - ids)
    if missing_ids:
        raise RuntimeError(f"defense video playlist CSV is missing ids: {missing_ids}")

    html_dir = args.defense_video_playlist_html.parent
    refs = []
    for token in ('src="', 'href="'):
        for part in html_text.split(token)[1:]:
            ref = part.split('"', 1)[0]
            if ref.startswith(("http://", "https://", "#")):
                continue
            refs.append(ref)
    for ref in refs:
        path = (html_dir / ref).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
    return rows


def verify_defense_video_cue_sheet(
    args: argparse.Namespace,
    playlist_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    if not args.defense_video_cue_sheet.exists():
        raise FileNotFoundError(args.defense_video_cue_sheet)
    if not args.defense_video_cue_sheet_csv.exists():
        raise FileNotFoundError(args.defense_video_cue_sheet_csv)

    md_text = args.defense_video_cue_sheet.read_text(encoding="utf-8-sig")
    rows = read_csv(args.defense_video_cue_sheet_csv)
    if len(rows) != len(playlist_rows):
        raise RuntimeError("defense video cue sheet row count does not match playlist")

    required_terms = (
        "defense_video_cue_sheet_v1",
        "答辩视频 Cue Sheet",
        "建议起点秒",
        "建议终点秒",
        "备用 viewer 命令",
        "讲解提示",
        "论文红线",
        "C08",
        "C09",
        "core_v2_ood_hard_distractors_success_v1",
        "core_v2_ood_paraphrase_failure_v1",
        "candidate_diagnostic_montage_v1",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "真实 OpenVLA、Isaac 和真实 WidowX",
    )
    missing_terms = [item for item in required_terms if item not in md_text]
    if missing_terms:
        raise RuntimeError(f"defense video cue sheet markdown is missing terms: {missing_terms}")

    required_columns = {
        "顺序",
        "分组",
        "cue_id",
        "标题",
        "媒体类型",
        "媒体文件",
        "建议起点秒",
        "建议终点秒",
        "时长秒",
        "打开命令",
        "备用viewer命令",
        "讲解提示",
        "证据引用",
        "论文红线",
    }
    missing_columns = sorted(required_columns - set(rows[0].keys()))
    if missing_columns:
        raise RuntimeError(f"defense video cue sheet CSV is missing columns: {missing_columns}")

    ids = {row["cue_id"] for row in rows}
    required_ids = {
        "C08",
        "C09",
        "core_v2_ood_hard_distractors_success_v1",
        "core_v2_ood_paraphrase_failure_v1",
        "candidate_diagnostic_montage_v1",
        "preference_ranked_trajectory_post_training_v1_candidate",
    }
    missing_ids = sorted(required_ids - ids)
    if missing_ids:
        raise RuntimeError(f"defense video cue sheet CSV is missing ids: {missing_ids}")

    if not any(row["备用viewer命令"].strip() for row in rows):
        raise RuntimeError("defense video cue sheet has no fallback viewer commands")
    for row in rows:
        if not row["打开命令"].strip() or not row["讲解提示"].strip() or not row["论文红线"].strip():
            raise RuntimeError(f"defense video cue sheet row lacks command, prompt, or redline: {row['cue_id']}")
        if row["媒体类型"] == "video":
            if not row["建议起点秒"].strip() or not row["建议终点秒"].strip() or not row["时长秒"].strip():
                raise RuntimeError(f"defense video cue sheet video row lacks timing: {row['cue_id']}")
            if row["建议终点秒"] == "全片" or row["时长秒"] == "不适用":
                raise RuntimeError(f"defense video cue sheet video row lacks concrete duration: {row['cue_id']}")
        elif row["媒体类型"] == "image":
            if row["建议起点秒"] != "不适用" or row["建议终点秒"] != "不适用":
                raise RuntimeError(f"defense video cue sheet image row should not have timing: {row['cue_id']}")
    return rows


def verify_video_evidence_index(args: argparse.Namespace, versions: list[str]) -> list[str]:
    if not args.video_evidence_index.exists():
        raise FileNotFoundError(args.video_evidence_index)
    text = args.video_evidence_index.read_text(encoding="utf-8-sig")
    required = (
        "视频证据索引",
        "video_evidence_index_v1",
        "主任务固定片段",
        "语言/空间泛化片段",
        "推荐展示入口",
        "docs/video_evidence_gallery.html",
        "docs/video_presentation_storyboard.html",
        "docs/defense_deck.html",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/showcase/all_registered_methods_grid.mp4",
        "trajectory_conditioned_chunk_bc_v2",
        "torch_act_state_chunk_v1",
        "clip_action_head_lite_v1",
        "domain_randomization_structured_low_friction_v1",
        "domain_randomization_trajectory_knn_low_friction_v1",
        "domain_randomization_visual_act_cnn_cvae_low_friction_v1",
        "候选诊断片段",
        "trajectory_phase_template_bc_v1_candidate",
        "grasp_gated_trajectory_knn_v1_candidate",
        "preference_trajectory_post_training_v1_candidate",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "OpenVLA",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"video evidence index is missing required terms: {missing}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"video evidence index is missing versions: {missing_versions}")

    rows = read_csv(args.video_evidence_csv)
    if len(rows) < len(versions):
        raise RuntimeError(f"video evidence csv has too few rows: {len(rows)}")
    required_columns = (
        "视频类型",
        "版本",
        "阶段",
        "结果",
        "视频文件",
        "元数据文件",
        "证据用途",
        "论文红线",
    )
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"video evidence csv is missing column: {column}")

    checked: list[str] = []
    for row in rows:
        video_path = ROOT / row["视频文件"]
        metadata_path = ROOT / row["元数据文件"]
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        checked.append(video_path.as_posix())
    return checked


def verify_candidate_diagnostic_video_index(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.candidate_diagnostic_video_index.exists():
        raise FileNotFoundError(args.candidate_diagnostic_video_index)
    if not args.candidate_diagnostic_video_csv.exists():
        raise FileNotFoundError(args.candidate_diagnostic_video_csv)

    text = args.candidate_diagnostic_video_index.read_text(encoding="utf-8-sig")
    required = (
        "候选方法诊断视频索引",
        "candidate_diagnostic_video_index_v1",
        "trajectory_phase_template_bc_v1_candidate",
        "trajectory_prior_residual_bc_v1_candidate",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "grasp_gated_trajectory_knn_v1_candidate",
        "grasp_gated_torch_act_state_chunk_v1_candidate",
        "preference_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "phase_weighted_torch_act_v1_candidate",
        "contact_phase_gated_torch_act_v1_candidate",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "contact_aware_trajectory_knn_v1_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "contact_aware_phase_gated_torch_act_v1_candidate",
        "contact_stage_subpolicy_v1_candidate",
        "contact_stage_demo_torch_act_v1_candidate",
        "contact_stage_phase_action_head_v1_candidate",
        "contact_hold_weighted_torch_act_v1_candidate",
        "gripper_timing_contact_probe_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.mp4",
        "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "outputs/videos/contact_aware_trajectory_knn_v1_candidate_seed0.mp4",
        "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4",
        "outputs/videos/contact_aware_phase_gated_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4",
        "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.mp4",
        "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "--viewer",
        "--method trajectory_phase_template_bc",
        "--method trajectory_prior_residual_bc",
        "--method timing_aware_trajectory_prior_residual_bc",
        "--method grasp_gated_trajectory_chunk_bc",
        "--method grasp_gated_trajectory_knn",
        "--method grasp_gated_torch_act",
        "--method preference_trajectory_post_training",
        "--method phase_weighted_torch_act",
        "--method grasp_lift_subpolicy_probe",
        "--method contact_aware_trajectory_knn",
        "--method phase_action_head",
        "--method gripper_timing_contact_probe",
        "不能写成可靠 ACT baseline",
        "不能写成已稳定抓取",
        "不能写成 trajectory-kNN 已解决真实 grasp",
        "不能写成在线 RL 或真实偏好优化成功",
        "不能写成完整 ACT 或稳定抓取成功",
        "不能写成完整官方 ACT",
        "不能写成 learned BC/ACT/VLA baseline 成功",
        "不能写成完整官方 ACT、稳定抓取或 VLA 后训练成功",
        "不能写成可靠 learned grasp",
        "不能写成在线 RL、真实人类偏好优化",
        "不能写成 learned BC",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"candidate diagnostic video index is missing required terms: {missing}")

    rows = read_csv(args.candidate_diagnostic_video_csv)
    if len(rows) != 22:
        raise RuntimeError(f"candidate diagnostic video csv should have 22 rows, found {len(rows)}")
    required_columns = (
        "版本",
        "方法定位",
        "seed",
        "结果",
        "视频文件",
        "元数据文件",
        "实验结论",
        "论文边界",
        "完整viewer命令",
        "重新导出视频命令",
    )
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"candidate diagnostic video csv is missing column: {column}")
    versions = {row["版本"] for row in rows}
    if versions != {
        "trajectory_phase_template_bc_v1_candidate",
        "trajectory_prior_residual_bc_v1_candidate",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "grasp_gated_trajectory_knn_v1_candidate",
        "grasp_gated_torch_act_state_chunk_v1_candidate",
        "preference_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "phase_weighted_torch_act_v1_candidate",
        "contact_phase_gated_torch_act_v1_candidate",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "contact_aware_trajectory_knn_v1_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "contact_aware_phase_gated_torch_act_v1_candidate",
        "contact_stage_subpolicy_v1_candidate",
        "contact_stage_demo_torch_act_v1_candidate",
        "contact_stage_phase_action_head_v1_candidate",
        "contact_hold_weighted_torch_act_v1_candidate",
        "gripper_timing_contact_probe_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
    }:
        raise RuntimeError(f"candidate diagnostic versions differ: {sorted(versions)}")
    for row in rows:
        if "--viewer" not in row["完整viewer命令"]:
            raise RuntimeError(f"candidate viewer command is missing --viewer: {row['版本']}")
        if "export_video.py" not in row["重新导出视频命令"]:
            raise RuntimeError(f"candidate export command is missing export_video.py: {row['版本']}")
        video_path = ROOT / row["视频文件"]
        metadata_path = ROOT / row["元数据文件"]
        ffprobe_video(video_path)
        metadata = read_json(metadata_path)
        if metadata.get("summary", {}).get("grasp_success") is not False:
            raise RuntimeError(f"candidate video should document failed grasp_success: {row['版本']}")
    return rows


def verify_video_evidence_gallery(args: argparse.Namespace, min_rows: int) -> None:
    if not args.video_evidence_gallery.exists():
        raise FileNotFoundError(args.video_evidence_gallery)
    text = args.video_evidence_gallery.read_text(encoding="utf-8-sig")
    required = (
        "视频证据浏览页",
        "data-filter=\"type\"",
        "data-filter=\"stage\"",
        "data-filter=\"result\"",
        "domain_randomization_trajectory_knn_low_friction_v1",
        "domain_randomization_visual_act_cnn_cvae_low_friction_v1",
        "clip_action_head_lite_v1",
        "../outputs/videos/expert_scripted_v1_seed0.mp4",
        "../outputs/videos/domain_randomization_trajectory_knn_low_friction_seed0.mp4",
        "论文红线",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"video evidence gallery is missing required terms: {missing}")
    if text.count('class="video-card"') < min_rows:
        raise RuntimeError("video evidence gallery has too few video cards")
    if text.count("<video ") < min_rows:
        raise RuntimeError("video evidence gallery has too few video elements")


def verify_video_quality_audit(args: argparse.Namespace, min_rows: int) -> list[dict[str, str]]:
    if not args.video_quality_audit.exists():
        raise FileNotFoundError(args.video_quality_audit)
    if not args.video_quality_audit_csv.exists():
        raise FileNotFoundError(args.video_quality_audit_csv)

    text = args.video_quality_audit.read_text(encoding="utf-8-sig")
    required = (
        "视频质量审计",
        "video_quality_audit_v1",
        "docs/video_evidence_index.csv",
        "ffprobe",
        "不是成功率评测",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"video quality audit markdown is missing terms: {missing}")

    rows = read_csv(args.video_quality_audit_csv)
    if len(rows) < min_rows:
        raise RuntimeError(f"video quality audit csv has too few rows: {len(rows)}")
    required_columns = (
        "视频类型",
        "版本",
        "结果",
        "视频文件",
        "元数据文件",
        "ffprobe可播放",
        "元数据存在",
        "宽度",
        "高度",
        "fps",
        "视频时长秒",
        "索引时长秒",
        "时长通过",
        "分辨率通过",
        "审计状态",
    )
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"video quality audit csv is missing column: {column}")

    for row in rows:
        if row["审计状态"] != "通过":
            raise RuntimeError(f"video quality audit row did not pass: {row['版本']}")
        if row["ffprobe可播放"] != "是" or row["元数据存在"] != "是":
            raise RuntimeError(f"video quality audit row is not playable or has no metadata: {row['版本']}")
        if row["时长通过"] != "是" or row["分辨率通过"] != "是":
            raise RuntimeError(f"video quality audit row failed duration/resolution checks: {row['版本']}")
        video_path = ROOT / row["视频文件"]
        metadata_path = ROOT / row["元数据文件"]
        stream = ffprobe_video(video_path)
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        if width != int(row["宽度"]) or height != int(row["高度"]):
            raise RuntimeError(f"video quality audit dimensions are stale: {row['视频文件']}")
        if float(row["fps"]) <= 0 or float(row["视频时长秒"]) <= 0:
            raise RuntimeError(f"video quality audit row has invalid timing: {row['版本']}")
    return rows


def verify_failure_mode_taxonomy(args: argparse.Namespace, min_rows: int) -> list[dict[str, str]]:
    if not args.failure_mode_taxonomy.exists():
        raise FileNotFoundError(args.failure_mode_taxonomy)
    if not args.failure_mode_taxonomy_csv.exists():
        raise FileNotFoundError(args.failure_mode_taxonomy_csv)

    text = args.failure_mode_taxonomy.read_text(encoding="utf-8-sig")
    required = (
        "失败模式分类记录",
        "failure_mode_taxonomy_v1",
        "trajectory-conditioned BC / ACT",
        "数据回放/可复现",
        "未形成有效抓取/未抬升",
        "语言/空间泛化失败",
        "扰动域接触鲁棒性不足",
        "docs/video_evidence_gallery.html",
        "不能写成真实机器人验证",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
        "clip_action_head_lite_v1",
        "domain_randomization_trajectory_knn_low_friction_v1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"failure mode taxonomy is missing required terms: {missing}")

    rows = read_csv(args.failure_mode_taxonomy_csv)
    if len(rows) < min_rows:
        raise RuntimeError(f"failure mode taxonomy csv has too few rows: {len(rows)}")
    required_columns = (
        "版本",
        "方法",
        "阶段",
        "视频类型",
        "结果",
        "失败模式",
        "证据用途",
        "视频文件",
        "元数据文件",
        "论文可写",
        "论文红线",
    )
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"failure mode taxonomy csv is missing column: {column}")

    versions = {row["版本"] for row in rows}
    required_versions = {
        "linear_bc_v1",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
        "clip_action_head_lite_v1",
        "domain_randomization_trajectory_knn_low_friction_v1",
    }
    missing_versions = sorted(required_versions - versions)
    if missing_versions:
        raise RuntimeError(f"failure mode taxonomy csv is missing versions: {missing_versions}")

    modes = {row["失败模式"] for row in rows}
    required_modes = {"成功样例", "数据回放/可复现", "未形成有效抓取/未抬升", "语言/空间泛化失败", "扰动域接触鲁棒性不足"}
    missing_modes = sorted(required_modes - modes)
    if missing_modes:
        raise RuntimeError(f"failure mode taxonomy csv is missing modes: {missing_modes}")
    return rows


def verify_figures(args: argparse.Namespace) -> list[str]:
    if not args.figures_doc.exists():
        raise FileNotFoundError(args.figures_doc)
    text = args.figures_doc.read_text(encoding="utf-8")
    required = ("experiment_figures_v1", "main_task_success.svg", "language_success.svg", "resource_vs_success.svg", "data_efficiency.svg")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"figures doc is missing required terms: {missing}")

    checked = []
    for stem in ("main_task_success", "language_success", "resource_vs_success", "data_efficiency"):
        path = args.figures_dir / f"{stem}.svg"
        if not path.exists():
            raise FileNotFoundError(path)
        svg = path.read_text(encoding="utf-8")
        if "<svg" not in svg or "</svg>" not in svg:
            raise RuntimeError(f"invalid svg figure: {path}")
        checked.append(path.as_posix())
    return checked


def verify_method_cards(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.method_cards.exists():
        raise FileNotFoundError(args.method_cards)
    text = args.method_cards.read_text(encoding="utf-8")
    required = ("method_cards_v1", "推荐讲解顺序", "structured_waypoint_policy_v1", "clip_action_head_lite_v1")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"method cards are missing required terms: {missing}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"method cards are missing versions: {missing_versions}")


def verify_result_matrix(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.result_matrix.exists():
        raise FileNotFoundError(args.result_matrix)
    text = args.result_matrix.read_text(encoding="utf-8")
    video_rows = read_csv(args.video_evidence_csv)
    required = (
        "result_matrix_v1",
        "分阶段方法矩阵",
        "研究问题对应证据",
        "视频展示矩阵",
        "论文表述边界",
        "trajectory_conditioned_chunk_bc_v2",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "torch_act_cvae_state_chunk_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "phase_conditioned_action_head_lite_v1",
        "clip_action_head_lite_v1",
        f"视频证据：`{len(video_rows)}` 条",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"result matrix is missing required terms: {missing}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"result matrix is missing versions: {missing_versions}")
    if "### 其它已登记版本" in text:
        raise RuntimeError("result matrix has formal methods without a stage group")


def verify_method_comparison_dashboard(args: argparse.Namespace, versions: list[str]) -> list[dict[str, str]]:
    if not args.method_comparison_dashboard.exists():
        raise FileNotFoundError(args.method_comparison_dashboard)
    if not args.method_comparison_dashboard_html.exists():
        raise FileNotFoundError(args.method_comparison_dashboard_html)
    if not args.method_comparison_dashboard_csv.exists():
        raise FileNotFoundError(args.method_comparison_dashboard_csv)

    md_text = args.method_comparison_dashboard.read_text(encoding="utf-8-sig")
    html_text = args.method_comparison_dashboard_html.read_text(encoding="utf-8-sig")
    rows = read_csv(args.method_comparison_dashboard_csv)
    if len(rows) != len(versions):
        raise RuntimeError(f"method comparison dashboard csv should have {len(versions)} rows, found {len(rows)}")
    required_terms = (
        "method_comparison_dashboard_v1",
        "方法评测比较看板",
        "当前完成的是 MuJoCo 实验包",
        "真实 OpenVLA、Isaac 和真实 WidowX",
        "trajectory_conditioned_chunk_bc_v2",
        "torch_act_state_chunk_v1",
        "clip_action_head_lite_v1",
        "docs/evaluation_summary.csv",
        "docs/language_generalization_summary.csv",
        "docs/model_resource_summary.csv",
    )
    missing_md = [item for item in required_terms if item not in md_text]
    if missing_md:
        raise RuntimeError(f"method comparison dashboard markdown is missing terms: {missing_md}")
    html_required = (
        "method_comparison_dashboard_v1",
        "方法评测比较看板",
        "全部阶段",
        "全部结果",
        "固定视频",
        "慢速 viewer 命令",
    )
    missing_html = [item for item in html_required if item not in html_text]
    if missing_html:
        raise RuntimeError(f"method comparison dashboard html is missing terms: {missing_html}")
    if html_text.count("<tr data-stage=") != len(versions):
        raise RuntimeError("method comparison dashboard html has wrong row count")

    csv_versions = {row["version"] for row in rows}
    missing_versions = [version for version in versions if version not in csv_versions]
    if missing_versions:
        raise RuntimeError(f"method comparison dashboard csv is missing versions: {missing_versions}")
    required_columns = {
        "version",
        "stage",
        "stage_label",
        "method",
        "artifact",
        "train",
        "heldout",
        "language",
        "result_bucket",
        "trainable_params",
        "fixed_video",
        "paper_redline",
        "viewer_command",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"method comparison dashboard csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    if not any(row["result_bucket"] == "partial" for row in rows):
        raise RuntimeError("method comparison dashboard lacks partial-result methods")
    if not any(row["stage"] == "peft_action_head_proxy" and int(float(row["trainable_params"])) <= 2119 for row in rows):
        raise RuntimeError("method comparison dashboard lacks PEFT parameter comparison")
    for row in rows:
        if not row["paper_redline"]:
            raise RuntimeError(f"method comparison dashboard has empty redline: {row['version']}")
        if "--viewer" not in row["viewer_command"] or "--duration 60" not in row["viewer_command"]:
            raise RuntimeError(f"method comparison dashboard viewer command is incomplete: {row['version']}")
        for key in ("artifact", "fixed_video"):
            path = ROOT / row[key]
            if not path.exists():
                raise FileNotFoundError(path)
    html_dir = args.method_comparison_dashboard_html.parent
    for token in ('href="',):
        for part in html_text.split(token)[1:]:
            ref = part.split('"', 1)[0]
            if ref.startswith(("http://", "https://", "#")):
                continue
            path = (html_dir / ref).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
    return rows


def verify_core_task_comparison_matrix(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (args.core_task_comparison, args.core_task_comparison_csv, args.core_task_comparison_json):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.core_task_comparison.read_text(encoding="utf-8-sig")
    required_terms = (
        "核心多任务对比矩阵",
        "core_task_comparison_matrix_v1",
        "蓝色立方体 -> 蓝色盘",
        "蓝色立方体 -> 红色盘",
        "红色立方体 -> 红色盘",
        "最左物体 -> 碗",
        "视频只作为定性片段，不替代数据表",
        "重复失败候选不进入主展示",
    )
    missing = [item for item in required_terms if item not in text]
    if missing:
        raise RuntimeError(f"core task comparison matrix is missing terms: {missing}")

    rows = read_csv(args.core_task_comparison_csv)
    if len(rows) != 24:
        raise RuntimeError(f"core task comparison matrix should have 24 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "方法",
        "方法key",
        "阶段",
        "任务",
        "任务key",
        "任务定位",
        "成功",
        "成功率",
        "平均目标距离",
        "seeds",
        "主要失败seed",
        "证据CSV",
        "证据JSON",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(f"core task comparison csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    tasks = {row["任务key"] for row in rows}
    methods = {row["方法key"] for row in rows}
    if tasks != {"blue_cube_blue_pad", "blue_cube_red_pad", "red_cube_red_pad", "leftmost_to_bowl"}:
        raise RuntimeError(f"core task comparison tasks differ: {sorted(tasks)}")
    if methods != {"expert", "structured_waypoint_policy", "linear_bc", "knn_bc", "trajectory_knn", "object_action_head"}:
        raise RuntimeError(f"core task comparison methods differ: {sorted(methods)}")
    by_pair = {(row["方法key"], row["任务key"]): row for row in rows}
    if by_pair[("trajectory_knn", "blue_cube_blue_pad")]["成功"] != "3/3":
        raise RuntimeError("core task comparison should preserve trajectory-kNN blue task success")
    if by_pair[("linear_bc", "blue_cube_blue_pad")]["成功"] != "0/3":
        raise RuntimeError("core task comparison should preserve linear BC blue task failure")
    if by_pair[("structured_waypoint_policy", "red_cube_red_pad")]["成功"] != "3/3":
        raise RuntimeError("core task comparison should preserve structured red-cube success")
    if by_pair[("object_action_head", "leftmost_to_bowl")]["成功"] != "0/3":
        raise RuntimeError("core task comparison should preserve action-head spatial failure")
    for row in rows:
        if not (ROOT / row["证据CSV"]).exists():
            raise FileNotFoundError(ROOT / row["证据CSV"])
        if not (ROOT / row["证据JSON"]).exists():
            raise FileNotFoundError(ROOT / row["证据JSON"])

    data = read_json(args.core_task_comparison_json)
    if data.get("version") != "core_task_comparison_matrix_v1":
        raise RuntimeError("core task comparison json has unexpected version")
    if len(data.get("rows", [])) != 24:
        raise RuntimeError("core task comparison json should have 24 rows")
    if len(data.get("method_summary", [])) != 6 or len(data.get("task_summary", [])) != 4:
        raise RuntimeError("core task comparison json has wrong summary sizes")
    return rows


def verify_core_v2_comparison_matrix(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = (args.core_v2_comparison, args.core_v2_comparison_csv, args.core_v2_comparison_json)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.core_v2_comparison.read_text(encoding="utf-8-sig")
    required_terms = (
        "Core V2 留出集对比矩阵",
        "core_v2_holdout_comparison_matrix_v1",
        "每项任务前 20 个 episode 训练，最后 5 个 episode 留出",
        "不能把 object-language action head 称为真实 VLM/VLA 后训练成功",
        "不能把 trajectory-prior residual 称为端到端学习策略",
    )
    missing = [item for item in required_terms if item not in text]
    if missing:
        raise RuntimeError(f"core v2 comparison matrix is missing terms: {missing}")

    rows = read_csv(args.core_v2_comparison_csv)
    if len(rows) != 28:
        raise RuntimeError(f"core v2 comparison matrix should have 28 rows, found {len(rows)}")
    tasks = {row["任务key"] for row in rows}
    methods = {row["方法key"] for row in rows}
    if tasks != {"blue_to_blue", "blue_to_red", "red_to_red", "leftmost_cube"}:
        raise RuntimeError(f"core v2 comparison tasks differ: {sorted(tasks)}")
    expected_methods = {
        "expert", "structured_waypoint_policy", "linear_bc", "knn_bc", "trajectory_knn",
        "object_action_head", "trajectory_prior_residual",
    }
    if methods != expected_methods:
        raise RuntimeError(f"core v2 comparison methods differ: {sorted(methods)}")
    by_pair = {(row["方法key"], row["任务key"]): row for row in rows}
    if by_pair[("expert", "blue_to_blue")]["成功"] != "5/5":
        raise RuntimeError("core v2 expert upper bound is not 5/5")
    if by_pair[("object_action_head", "leftmost_cube")]["成功"] != "0/5":
        raise RuntimeError("core v2 action-head spatial result is unexpected")
    if by_pair[("trajectory_prior_residual", "leftmost_cube")]["成功"] != "5/5":
        raise RuntimeError("core v2 trajectory-prior spatial result is unexpected")
    for row in rows:
        for key in ("证据CSV", "证据JSON"):
            if not (ROOT / row[key]).exists():
                raise FileNotFoundError(ROOT / row[key])

    data = read_json(args.core_v2_comparison_json)
    if data.get("version") != "core_v2_holdout_comparison_matrix_v1":
        raise RuntimeError("core v2 comparison json has unexpected version")
    if len(data.get("rows", [])) != 28:
        raise RuntimeError("core v2 comparison json should have 28 rows")
    if len(data.get("method_summary", [])) != 7 or len(data.get("task_summary", [])) != 4:
        raise RuntimeError("core v2 comparison json has wrong summary sizes")
    if data.get("protocol", {}).get("workspace_profile") != "core_v2":
        raise RuntimeError("core v2 comparison json has wrong workspace profile")
    return rows


def verify_core_v2_pretrained_vlm_report(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = (
        args.core_v2_pretrained_vlm_report,
        args.core_v2_pretrained_vlm_csv,
        args.core_v2_pretrained_vlm_json,
        args.core_v2_pretrained_vlm_model,
        args.core_v2_pretrained_vlm_video,
        args.core_v2_pretrained_vlm_video.with_suffix(".json"),
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.core_v2_pretrained_vlm_report.read_text(encoding="utf-8-sig")
    required_terms = (
        "Core V2 预训练 VLM 动作头报告",
        "core_v2_pretrained_vlm_action_head_v1",
        "openai/clip-vit-base-patch32",
        "冻结编码器参数",
        "0/20",
        "不是 OpenVLA、RT-2、LoRA 或端到端 VLA",
        "clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4",
    )
    missing = [item for item in required_terms if item not in text]
    if missing:
        raise RuntimeError(f"core v2 pretrained VLM report is missing terms: {missing}")

    rows = read_csv(args.core_v2_pretrained_vlm_csv)
    if len(rows) != 4:
        raise RuntimeError(f"core v2 pretrained VLM csv should have 4 rows, found {len(rows)}")
    if {row["任务key"] for row in rows} != {"blue_to_blue", "blue_to_red", "red_to_red", "leftmost_cube"}:
        raise RuntimeError("core v2 pretrained VLM task keys differ")
    for row in rows:
        if row["成功"] != "0/5" or float(row["成功率"]) != 0.0:
            raise RuntimeError("core v2 pretrained VLM held-out results should be 0/5 per task")
        for key in ("证据CSV", "证据JSON"):
            if not (ROOT / row[key]).exists():
                raise FileNotFoundError(ROOT / row[key])

    data = read_json(args.core_v2_pretrained_vlm_json)
    if data.get("version") != "core_v2_pretrained_vlm_action_head_v1":
        raise RuntimeError("core v2 pretrained VLM json has unexpected version")
    if data.get("summary", {}).get("success") != "0/20":
        raise RuntimeError("core v2 pretrained VLM total success should be 0/20")
    metadata = data.get("model_metadata", {})
    if metadata.get("clip_model") != "openai/clip-vit-base-patch32":
        raise RuntimeError("core v2 pretrained VLM should record the CLIP model")
    if int(metadata.get("frozen_encoder_params", 0)) < 100_000_000:
        raise RuntimeError("core v2 pretrained VLM frozen encoder parameter count is implausible")
    if float(metadata.get("peak_vram_mb", 0.0)) <= 0.0:
        raise RuntimeError("core v2 pretrained VLM should record CUDA memory usage")

    ffprobe_video(args.core_v2_pretrained_vlm_video)
    video = read_json(args.core_v2_pretrained_vlm_video.with_suffix(".json"))
    if video.get("version") != "clip_core_v2_multitask_v1" or video.get("method") != "clip_action_head":
        raise RuntimeError("core v2 pretrained VLM video metadata is inconsistent")
    if video.get("summary", {}).get("success") is not False:
        raise RuntimeError("core v2 pretrained VLM video should document the representative failure")
    return rows


def verify_core_v2_clip_semantic_waypoint(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = (
        args.core_v2_clip_semantic_report,
        args.core_v2_clip_semantic_csv,
        args.core_v2_clip_semantic_json,
        args.core_v2_clip_semantic_model,
        args.core_v2_clip_semantic_video,
        args.core_v2_clip_semantic_video.with_suffix(".json"),
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    text = args.core_v2_clip_semantic_report.read_text(encoding="utf-8-sig")
    required = (
        "Core V2 CLIP 语义-结构化执行报告",
        "core_v2_clip_semantic_waypoint_v1",
        "20/20",
        "不是端到端 VLA、连续 action-head、OpenVLA、LoRA 或真实机器人结果",
        "scripted waypoint expert",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"core v2 CLIP semantic report is missing terms: {missing}")
    rows = read_csv(args.core_v2_clip_semantic_csv)
    if len(rows) != 4 or any(row["成功"] != "5/5" or row["语义正确"] != "5/5" or row["严格抓取"] != "5/5" for row in rows):
        raise RuntimeError("core v2 CLIP semantic report should be 5/5 on all four tasks")
    data = read_json(args.core_v2_clip_semantic_json)
    if data.get("version") != "core_v2_clip_semantic_waypoint_v1":
        raise RuntimeError("core v2 CLIP semantic json has unexpected version")
    if data.get("summary", {}).get("success") != "20/20" or data.get("summary", {}).get("semantic_correct") != "20/20" or data.get("summary", {}).get("strict_grasp_success") != "20/20":
        raise RuntimeError("core v2 CLIP semantic summary is unexpected")
    ffprobe_video(args.core_v2_clip_semantic_video)
    video = read_json(args.core_v2_clip_semantic_video.with_suffix(".json"))
    if video.get("summary", {}).get("success") is not True or video.get("summary", {}).get("semantic_correct") is not True or video.get("summary", {}).get("strict_grasp_success") is not True:
        raise RuntimeError("core v2 CLIP semantic video should document a successful semantic rollout")
    return rows


def verify_core_v2_clip_semantic_data_efficiency(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = (
        args.core_v2_clip_semantic_efficiency_report,
        args.core_v2_clip_semantic_efficiency_csv,
        args.core_v2_clip_semantic_efficiency_json,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    text = args.core_v2_clip_semantic_efficiency_report.read_text(encoding="utf-8-sig")
    required = (
        "Core V2 CLIP 语义-结构化执行数据效率报告",
        "core_v2_clip_semantic_data_efficiency_v1",
        "20/40/79",
        "严格抓取定义",
        "端到端 VLA 控制",
        "0/20",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"core v2 semantic data-efficiency report is missing terms: {missing}")

    rows = read_csv(args.core_v2_clip_semantic_efficiency_csv)
    if len(rows) != 12:
        raise RuntimeError(f"core v2 semantic data-efficiency csv should have 12 rows, found {len(rows)}")
    expected_samples = {"5": "20", "10": "40", "20": "79"}
    expected_tasks = {"blue_to_blue", "blue_to_red", "red_to_red", "leftmost_cube"}
    for budget, samples in expected_samples.items():
        selected = [row for row in rows if row["demo_budget_per_task"] == budget]
        if len(selected) != 4 or {row["task_key"] for row in selected} != expected_tasks:
            raise RuntimeError(f"core v2 semantic data-efficiency rows are incomplete for budget {budget}")
        if any(row["stored_samples"] != samples or row["success"] != "5/5" or row["semantic_correct"] != "5/5" or row["strict_grasp_success"] != "5/5" for row in selected):
            raise RuntimeError(f"core v2 semantic data-efficiency rows are inconsistent for budget {budget}")
        for row in selected:
            if not (ROOT / row["model"]).exists():
                raise FileNotFoundError(ROOT / row["model"])

    data = read_json(args.core_v2_clip_semantic_efficiency_json)
    if data.get("version") != "core_v2_clip_semantic_data_efficiency_v1" or len(data.get("rows", [])) != 12:
        raise RuntimeError("core v2 semantic data-efficiency json is inconsistent")
    return rows


def verify_core_v2_clip_semantic_ood_generalization(args: argparse.Namespace) -> list[dict[str, str]]:
    paths = (
        args.core_v2_clip_semantic_ood_report,
        args.core_v2_clip_semantic_ood_csv,
        args.core_v2_clip_semantic_ood_json,
        args.core_v2_clip_semantic_ood_success_video,
        args.core_v2_clip_semantic_ood_success_video.with_suffix(".json"),
        args.core_v2_clip_semantic_ood_failure_video,
        args.core_v2_clip_semantic_ood_failure_video.with_suffix(".json"),
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    text = args.core_v2_clip_semantic_ood_report.read_text(encoding="utf-8-sig")
    required = (
        "Core V2 CLIP 语义-结构化执行 OOD 泛化报告",
        "core_v2_clip_semantic_ood_generalization_v1",
        "51/60",
        "20/20",
        "azure block",
        "不能证明端到端 VLA",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"core v2 semantic OOD report is missing terms: {missing}")

    rows = read_csv(args.core_v2_clip_semantic_ood_csv)
    if len(rows) != 80:
        raise RuntimeError(f"core v2 semantic OOD csv should have 80 rows, found {len(rows)}")
    expected_conditions = {"paraphrase": 60, "hard_distractors": 20}
    for condition, count in expected_conditions.items():
        if sum(row["condition"] == condition for row in rows) != count:
            raise RuntimeError(f"core v2 semantic OOD csv has wrong count for {condition}")
    if not any(row["instruction"] == "put the azure block on the red disk" and row["task_success"] == "False" for row in rows):
        raise RuntimeError("core v2 semantic OOD csv is missing the representative paraphrase failure")

    data = read_json(args.core_v2_clip_semantic_ood_json)
    if data.get("version") != "core_v2_clip_semantic_ood_generalization_v1" or len(data.get("rows", [])) != 80:
        raise RuntimeError("core v2 semantic OOD json is inconsistent")
    expected_summary = {
        ("paraphrase", "blue_to_blue"): "15/15",
        ("paraphrase", "blue_to_red"): "8/15",
        ("paraphrase", "red_to_red"): "13/15",
        ("paraphrase", "leftmost_cube"): "15/15",
        ("hard_distractors", "blue_to_blue"): "5/5",
        ("hard_distractors", "blue_to_red"): "5/5",
        ("hard_distractors", "red_to_red"): "5/5",
        ("hard_distractors", "leftmost_cube"): "5/5",
    }
    observed_summary = {(row["condition"], row["task_key"]): row["task_success"] for row in data.get("summary", [])}
    if observed_summary != expected_summary:
        raise RuntimeError(f"core v2 semantic OOD summary is unexpected: {observed_summary}")

    ffprobe_video(args.core_v2_clip_semantic_ood_success_video)
    success_video = read_json(args.core_v2_clip_semantic_ood_success_video.with_suffix(".json"))
    success_summary = success_video.get("summary", {})
    if success_video.get("complexity") != "hard" or success_summary.get("task_success") is not True or success_summary.get("strict_grasp_success") is not True:
        raise RuntimeError("core v2 semantic OOD success video metadata is inconsistent")
    ffprobe_video(args.core_v2_clip_semantic_ood_failure_video)
    failure_video = read_json(args.core_v2_clip_semantic_ood_failure_video.with_suffix(".json"))
    failure_summary = failure_video.get("summary", {})
    if failure_summary.get("instruction") != "put the azure block on the red disk" or failure_summary.get("task_success") is not False or failure_summary.get("semantic_correct") is not False:
        raise RuntimeError("core v2 semantic OOD failure video metadata is inconsistent")
    return rows


def split_sources(value: str) -> list[str]:
    return [part.strip() for part in value.replace("；", "\n").splitlines() if part.strip()]


def verify_thesis_visual_evidence_index(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.thesis_visual_evidence.exists():
        raise FileNotFoundError(args.thesis_visual_evidence)
    if not args.thesis_visual_evidence_csv.exists():
        raise FileNotFoundError(args.thesis_visual_evidence_csv)
    if not args.thesis_visual_evidence_html.exists():
        raise FileNotFoundError(args.thesis_visual_evidence_html)

    md_text = args.thesis_visual_evidence.read_text(encoding="utf-8-sig")
    html_text = args.thesis_visual_evidence_html.read_text(encoding="utf-8-sig")
    rows = read_csv(args.thesis_visual_evidence_csv)
    if len(rows) < 22:
        raise RuntimeError(f"thesis visual evidence index has too few rows: {len(rows)}")
    required_terms = (
        "thesis_visual_evidence_index_v1",
        "论文图表与视频证据索引",
        "method_comparison_dashboard_v1",
        "defense_video_playlist_v1",
        "trajectory_act_experiment_record_v1",
        "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
        "不能写成完整官方 ACT",
        "MuJoCo domain randomization 代理",
        "不能写成 Isaac domain randomization 已完成",
        "grasp_successes=0",
        "真实 OpenVLA、RT-2 或 OpenVLA-OFT",
    )
    missing_md = [item for item in required_terms if item not in md_text]
    if missing_md:
        raise RuntimeError(f"thesis visual evidence markdown is missing terms: {missing_md}")
    html_required = (
        "thesis_visual_evidence_index_v1",
        "论文图表与视频证据索引",
        "图注/表注/讲解",
        "论文红线",
    )
    missing_html = [item for item in html_required if item not in html_text]
    if missing_html:
        raise RuntimeError(f"thesis visual evidence html is missing terms: {missing_html}")
    if html_text.count("<video ") < 12:
        raise RuntimeError("thesis visual evidence html has too few video elements")
    if html_text.count("<img ") < 4:
        raise RuntimeError("thesis visual evidence html has too few image elements")

    required_ids = {"F01", "F02", "F03", "F04", "T01", "T02", "H01", "H02", "V03", "V06", "C03", "C05", "C06"}
    row_ids = {row["编号"] for row in rows}
    missing_ids = sorted(required_ids - row_ids)
    if missing_ids:
        raise RuntimeError(f"thesis visual evidence csv is missing ids: {missing_ids}")

    required_columns = {
        "编号",
        "论文或答辩位置",
        "类型",
        "建议标题",
        "证据文件",
        "配套证据",
        "中文图注/表注/讲解",
        "可支撑结论",
        "论文红线",
        "打开命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"thesis visual evidence csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    for row in rows:
        if not row["论文红线"] or not row["中文图注/表注/讲解"]:
            raise RuntimeError(f"thesis visual evidence row lacks Chinese note/redline: {row['编号']}")
        if "Start-Process" not in row["打开命令"]:
            raise RuntimeError(f"thesis visual evidence row lacks open command: {row['编号']}")
        for path_text in [row["证据文件"], *split_sources(row["配套证据"])]:
            path = ROOT / path_text
            if not path.exists():
                raise FileNotFoundError(path)

    html_dir = args.thesis_visual_evidence_html.parent
    refs = []
    for token in ('src="', 'href="'):
        for part in html_text.split(token)[1:]:
            ref = part.split('"', 1)[0]
            if ref.startswith(("http://", "https://", "#")):
                continue
            refs.append(ref)
    for ref in refs:
        path = (html_dir / ref).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
    return rows


def verify_defense_qa_playbook(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.defense_qa_playbook.exists():
        raise FileNotFoundError(args.defense_qa_playbook)
    if not args.defense_qa_playbook_csv.exists():
        raise FileNotFoundError(args.defense_qa_playbook_csv)
    if not args.defense_qa_playbook_html.exists():
        raise FileNotFoundError(args.defense_qa_playbook_html)

    md_text = args.defense_qa_playbook.read_text(encoding="utf-8-sig")
    html_text = args.defense_qa_playbook_html.read_text(encoding="utf-8-sig")
    rows = read_csv(args.defense_qa_playbook_csv)
    if len(rows) < 14:
        raise RuntimeError(f"defense Q&A playbook has too few rows: {len(rows)}")
    required_terms = (
        "defense_qa_playbook_v1",
        "答辩追问 Q&A Playbook",
        "普通 BC",
        "trajectory-kNN",
        "完整官方 ACT",
        "严格抓取成功为 0/53",
        "OpenVLA LoRA",
        "MuJoCo domain randomization",
        "showcase_launcher.py",
        "不能把 MuJoCo proxy、候选负例、OpenVLA bridge、Isaac handoff 或真实 WidowX handoff 写成已完成的真实 VLA/真实机器人结果",
    )
    missing_md = [item for item in required_terms if item not in md_text]
    if missing_md:
        raise RuntimeError(f"defense Q&A markdown is missing terms: {missing_md}")
    html_required = (
        "defense_qa_playbook_v1",
        "答辩追问 Q&A Playbook",
        "推荐回答",
        "必须坚持的边界",
        "现场打开命令",
    )
    missing_html = [item for item in html_required if item not in html_text]
    if missing_html:
        raise RuntimeError(f"defense Q&A html is missing terms: {missing_html}")
    if html_text.count("<video ") < 8:
        raise RuntimeError("defense Q&A html has too few video elements")
    if html_text.count("<img ") < 1:
        raise RuntimeError("defense Q&A html has too few image elements")

    required_columns = {
        "编号",
        "追问主题",
        "适用阶段",
        "可能问题",
        "推荐回答",
        "首选证据",
        "首选图表或视频",
        "现场打开命令",
        "必须坚持的边界",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"defense Q&A csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_ids = {"Q01", "Q03", "Q05", "Q07", "Q09", "Q10", "Q12", "Q13", "Q14"}
    row_ids = {row["编号"] for row in rows}
    missing_ids = sorted(required_ids - row_ids)
    if missing_ids:
        raise RuntimeError(f"defense Q&A csv is missing ids: {missing_ids}")
    for row in rows:
        if not row["推荐回答"] or not row["必须坚持的边界"]:
            raise RuntimeError(f"defense Q&A row lacks answer/redline: {row['编号']}")
        if "Start-Process" not in row["现场打开命令"] and "showcase_launcher.py" not in row["现场打开命令"]:
            raise RuntimeError(f"defense Q&A row lacks a local open command: {row['编号']}")
        for path_text in [*split_sources(row["首选证据"]), row["首选图表或视频"]]:
            path = ROOT / path_text
            if not path.exists():
                raise FileNotFoundError(path)

    html_dir = args.defense_qa_playbook_html.parent
    refs = []
    for token in ('src="', 'href="'):
        for part in html_text.split(token)[1:]:
            ref = part.split('"', 1)[0]
            if ref.startswith(("http://", "https://", "#")):
                continue
            refs.append(ref)
    for ref in refs:
        path = (html_dir / ref).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
    return rows


def verify_version_lineage_index(args: argparse.Namespace, versions: list[str]) -> list[dict[str, str]]:
    if not args.version_lineage.exists():
        raise FileNotFoundError(args.version_lineage)
    if not args.version_lineage_csv.exists():
        raise FileNotFoundError(args.version_lineage_csv)
    if not args.version_lineage_html.exists():
        raise FileNotFoundError(args.version_lineage_html)

    md_text = args.version_lineage.read_text(encoding="utf-8-sig")
    html_text = args.version_lineage_html.read_text(encoding="utf-8-sig")
    rows = read_csv(args.version_lineage_csv)
    if len(rows) < 41:
        raise RuntimeError(f"version lineage index has too few rows: {len(rows)}")
    required_terms = (
        "version_lineage_index_v1",
        "实验版本谱系索引",
        "demo_place_blue_cube_blue_pad_v1",
        "formal_current",
        "completed_prerequisite",
        "completed_diagnostic",
        "planned_external_dependency",
        "robot_vla_action_head_lite_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
        "不能用当前 MuJoCo 视频替代",
    )
    missing_md = [item for item in required_terms if item not in md_text]
    if missing_md:
        raise RuntimeError(f"version lineage markdown is missing terms: {missing_md}")
    html_required = (
        "version_lineage_index_v1",
        "实验版本谱系索引",
        "正式方法",
        "前置/诊断",
        "计划版本",
    )
    missing_html = [item for item in html_required if item not in html_text]
    if missing_html:
        raise RuntimeError(f"version lineage html is missing terms: {missing_html}")

    required_columns = {
        "版本",
        "谱系层级",
        "状态",
        "类别",
        "父级/依赖",
        "阶段或登记阶段",
        "方法或对象",
        "artifact或输出",
        "量化结果或成功门槛",
        "首选视频或展示",
        "关系说明",
        "论文边界",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"version lineage csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    row_versions = {row["版本"] for row in rows}
    required_versions = {
        "demo_place_blue_cube_blue_pad_v1",
        "openvla_dataset_bridge_v1",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_action_head_lite_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
    }
    missing_versions = sorted((set(versions) | required_versions) - row_versions)
    if missing_versions:
        raise RuntimeError(f"version lineage csv is missing versions: {missing_versions}")
    statuses = {row["状态"] for row in rows}
    required_statuses = {"current_dataset", "formal_current", "completed_prerequisite", "completed_diagnostic", "planned", "planned_external_dependency"}
    missing_statuses = sorted(required_statuses - statuses)
    if missing_statuses:
        raise RuntimeError(f"version lineage csv is missing statuses: {missing_statuses}")
    formal_count = sum(1 for row in rows if row["状态"] == "formal_current")
    if formal_count != len(versions):
        raise RuntimeError(f"version lineage formal count should be {len(versions)}, found {formal_count}")
    for row in rows:
        if not row["父级/依赖"] or not row["论文边界"]:
            raise RuntimeError(f"version lineage row lacks parent/redline: {row['版本']}")
        if row["状态"] == "formal_current":
            for key in ("artifact或输出", "首选视频或展示"):
                for path_text in split_sources(row[key]):
                    if not (ROOT / path_text).exists():
                        raise FileNotFoundError(ROOT / path_text)

    html_dir = args.version_lineage_html.parent
    refs = []
    for token in ('href="',):
        for part in html_text.split(token)[1:]:
            ref = part.split('"', 1)[0]
            if ref.startswith(("http://", "https://", "#")):
                continue
            refs.append(ref)
    for ref in refs:
        path = (html_dir / ref).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
    return rows


def verify_method_stage_audit(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.method_stage_audit.exists():
        raise FileNotFoundError(args.method_stage_audit)
    text = args.method_stage_audit.read_text(encoding="utf-8-sig")
    required = (
        "方法阶段审计表",
        "论文红线",
        "总体边界",
        "视频证据索引",
        "OpenVLA",
        "真实 OpenVLA/RT-2/机器人 VLA 后训练",
        "不能写成完整视觉 ACT",
        "不能写成完整视觉 Diffusion Policy",
        "不能写成 pretrained VLA LoRA/Adapter",
        "clip_action_head_lite_v1",
        "torch_act_state_chunk_v1",
        "torch_diffusion_policy_state_chunk_v1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"method stage audit is missing required terms: {missing}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"method stage audit is missing versions: {missing_versions}")

    rows = read_csv(args.method_stage_audit_csv)
    if len(rows) < len(versions):
        raise RuntimeError(f"method stage audit csv has too few rows: {len(rows)}")
    csv_versions = {row["版本"] for row in rows}
    missing_csv_versions = [version for version in versions if version not in csv_versions]
    if missing_csv_versions:
        raise RuntimeError(f"method stage audit csv is missing versions: {missing_csv_versions}")
    for row in rows:
        for column in ("阶段分组", "方法性质", "视觉/状态输入", "语言输入", "训练方式", "论文可写", "论文红线"):
            if not row.get(column):
                raise RuntimeError(f"method stage audit csv has empty {column}: {row.get('版本')}")


def verify_method_evidence_gate(args: argparse.Namespace, versions: list[str]) -> list[dict[str, str]]:
    if not args.method_evidence_gate.exists():
        raise FileNotFoundError(args.method_evidence_gate)
    if not args.method_evidence_csv.exists():
        raise FileNotFoundError(args.method_evidence_csv)

    text = args.method_evidence_gate.read_text(encoding="utf-8-sig")
    required = (
        "方法证据门禁",
        "method_evidence_gate_v1",
        "25 个正式方法版本",
        "artifact",
        "视频质量审计",
        "慢速 viewer 命令",
        "训练/采集命令",
        "论文红线",
        "trajectory_conditioned_chunk_bc_v2",
        "torch_act_state_chunk_v1",
        "clip_action_head_lite_v1",
        "不能写成真实 OpenVLA/RT-2 后训练",
        "真实 OpenVLA/机器人 VLA、Isaac domain randomization 和真实 WidowX 验证仍需按下一阶段注册表继续执行",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"method evidence gate markdown is missing required terms: {missing}")

    rows = read_csv(args.method_evidence_csv)
    if len(rows) != len(versions):
        raise RuntimeError(f"method evidence gate csv should have {len(versions)} rows, found {len(rows)}")
    required_columns = {
        "版本",
        "阶段",
        "方法",
        "artifact",
        "artifact存在",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "可训练参数",
        "模型大小MB",
        "固定视频",
        "视频审计通过",
        "viewer命令存在",
        "训练或采集命令存在",
        "论文红线",
        "入包状态",
        "需补齐",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"method evidence gate csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    csv_versions = {row["版本"] for row in rows}
    missing_versions = [version for version in versions if version not in csv_versions]
    if missing_versions:
        raise RuntimeError(f"method evidence gate csv is missing versions: {missing_versions}")

    for row in rows:
        if row["入包状态"] != "通过" or row["需补齐"] != "无":
            raise RuntimeError(f"method evidence gate did not pass: {row['版本']} -> {row['需补齐']}")
        if row["artifact存在"] != "是":
            raise RuntimeError(f"method evidence gate artifact is missing: {row['版本']}")
        if row["视频审计通过"] != "是" or row["viewer命令存在"] != "是":
            raise RuntimeError(f"method evidence gate video/viewer check failed: {row['版本']}")
        if row["训练或采集命令存在"] not in {"是", "不适用"}:
            raise RuntimeError(f"method evidence gate command check failed: {row['版本']}")
        if not row["论文红线"]:
            raise RuntimeError(f"method evidence gate redline is empty: {row['版本']}")
        if not (ROOT / row["artifact"]).exists():
            raise FileNotFoundError(ROOT / row["artifact"])
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
    return rows


def verify_version_naming_and_gate_spec(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.version_naming_spec.exists():
        raise FileNotFoundError(args.version_naming_spec)
    if not args.version_naming_spec_csv.exists():
        raise FileNotFoundError(args.version_naming_spec_csv)
    if not args.version_naming_spec_json.exists():
        raise FileNotFoundError(args.version_naming_spec_json)

    text = args.version_naming_spec.read_text(encoding="utf-8-sig")
    required = (
        "版本命名与入包门禁规范",
        "version_naming_and_gate_spec_v1",
        "正式方法版本",
        "候选诊断版本",
        "前置门禁版本",
        "planned 外部版本",
        "视频命名",
        "资源与评测",
        "阶段归属",
        "planned 到 formal",
        "method_evidence_gate_v1",
        "external_dependency_readiness_audit_v1",
        "不能写成策略成功率结果",
        "verify_experiment_artifacts.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"version naming spec markdown is missing terms: {missing}")

    rows = read_csv(args.version_naming_spec_csv)
    if len(rows) != 8:
        raise RuntimeError(f"version naming spec csv should have 8 rows, found {len(rows)}")
    required_columns = {"类别", "规则编号", "规则", "示例", "入包门禁"}
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"version naming spec csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    rule_ids = [row["规则编号"] for row in rows]
    if rule_ids != [f"N{index:02d}" for index in range(1, 9)]:
        raise RuntimeError(f"version naming rule ids differ: {rule_ids}")
    for row in rows:
        if not row["类别"] or not row["规则"] or not row["入包门禁"]:
            raise RuntimeError(f"version naming rule has empty required field: {row.get('规则编号')}")

    data = read_json(args.version_naming_spec_json)
    if data.get("version") != "version_naming_and_gate_spec_v1":
        raise RuntimeError(f"version naming json has wrong version: {data.get('version')}")
    if int(data.get("rule_count", 0)) != len(rows):
        raise RuntimeError(f"version naming json rule_count differs: {data.get('rule_count')} vs {len(rows)}")
    json_rule_ids = [row["规则编号"] for row in data.get("rules", [])]
    if json_rule_ids != rule_ids:
        raise RuntimeError(f"version naming json rule ids differ: {json_rule_ids}")
    return rows


def verify_final_method_version_index(args: argparse.Namespace, versions: list[str]) -> list[dict[str, str]]:
    if not args.final_method_index.exists():
        raise FileNotFoundError(args.final_method_index)
    if not args.final_method_index_csv.exists():
        raise FileNotFoundError(args.final_method_index_csv)

    text = args.final_method_index.read_text(encoding="utf-8-sig")
    required = (
        "最终方法版本索引",
        "final_method_version_index_v1",
        "逐方法最终版说明与启动命令",
        "主任务慢速 viewer 命令",
        "语言/空间慢速 viewer 命令",
        "robot_vla_action_head_handoff_v1",
        "docs/reproducible_command_index.md",
        "--viewer --duration 60 --speed 0.05",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"final method version index markdown is missing terms: {missing}")

    rows = read_csv(args.final_method_index_csv)
    if len(rows) != len(versions):
        raise RuntimeError(f"final method version index csv should have {len(versions)} rows, found {len(rows)}")
    if not rows:
        raise RuntimeError("final method version index csv is empty")
    required_columns = {
        "序号",
        "版本",
        "阶段",
        "方法",
        "artifact",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "可训练参数",
        "模型大小MB",
        "固定视频",
        "入包状态",
        "简介",
        "论文边界",
        "主任务viewer命令",
        "语言viewer命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"final method version index csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    csv_versions = [row["版本"] for row in rows]
    if csv_versions != versions:
        raise RuntimeError(f"final method version index order differs from experiment_versions.json: {csv_versions}")

    for row in rows:
        command = row["主任务viewer命令"]
        if row["入包状态"] != "通过":
            raise RuntimeError(f"final method version index row did not pass: {row['版本']}")
        if "--viewer" not in command or "--duration 60" not in command or "--speed 0.05" not in command:
            raise RuntimeError(f"final method version index lacks slow viewer command: {row['版本']}")
        if not (ROOT / row["artifact"]).exists():
            raise FileNotFoundError(ROOT / row["artifact"])
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        if not row["简介"] or not row["论文边界"]:
            raise RuntimeError(f"final method version index lacks Chinese description or paper boundary: {row['版本']}")
    return rows


def verify_stage_comparison_report(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.stage_comparison_report.exists():
        raise FileNotFoundError(args.stage_comparison_report)
    text = args.stage_comparison_report.read_text(encoding="utf-8-sig")
    required = (
        "阶段对比报告",
        "stage_comparison_report_v1",
        "阶段总览",
        "阶段详表",
        "可写结论",
        "视频证据入口",
        "docs/defense_deck.html",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/showcase/all_registered_methods_grid.mp4",
        "OpenVLA",
        "trajectory_conditioned_chunk_bc_v2",
        "torch_act_state_chunk_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "clip_action_head_lite_v1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"stage comparison report is missing required terms: {missing}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"stage comparison report is missing versions: {missing_versions}")
    if text.count("### ") < 12:
        raise RuntimeError("stage comparison report has too few stage sections")

    rows = read_csv(args.stage_comparison_csv)
    if len(rows) < 12:
        raise RuntimeError(f"stage comparison csv has too few rows: {len(rows)}")
    required_columns = (
        "阶段分组",
        "方法数",
        "代表版本",
        "最好训练范围",
        "最好留出范围",
        "最好语言泛化",
        "可训练参数范围",
        "阶段目的",
        "阶段结论",
        "讲解建议",
        "视频证据",
    )
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"stage comparison csv is missing column: {column}")
    groups = {row["阶段分组"] for row in rows}
    required_groups = {"普通模仿学习", "动作块 / 轨迹条件", "ACT-style", "Diffusion Policy", "VLA/action-head 代理", "参数高效后训练代理"}
    if not required_groups.issubset(groups):
        raise RuntimeError(f"stage comparison csv is missing groups: {sorted(required_groups - groups)}")


def verify_task_bc_stage_report(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.task_bc_stage_report.exists():
        raise FileNotFoundError(args.task_bc_stage_report)
    if not args.task_bc_stage_csv.exists():
        raise FileNotFoundError(args.task_bc_stage_csv)

    text = args.task_bc_stage_report.read_text(encoding="utf-8-sig")
    required = (
        "任务 / 数据 / 普通 BC 阶段报告",
        "task_bc_stage_report_v1",
        "毕业设计前 3 层",
        "outputs/presentation_clips/01_task_data_oracle.mp4",
        "outputs/presentation_clips/02_basic_bc_baselines.mp4",
        "docs/video_evidence_gallery.html",
        "数据回放/可复现",
        "未形成有效抓取/未抬升",
        "data_efficiency_v2",
        "不能写成语言理解或 VLA 泛化",
        "不能写成 learned policy",
        "不能写成 learned VLA",
        "主任务慢速 Viewer 命令",
        "语言/空间泛化慢速 Viewer 命令",
        "训练/采集/重建命令",
        "expert_scripted_v1",
        "structured_waypoint_policy_v1",
        "replay_demo_v1",
        "linear_bc_v1",
        "knn_bc_v1",
        "mlp_bc_v1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"task/BC stage report is missing required terms: {missing}")
    if text.count("--viewer") < 6:
        raise RuntimeError("task/BC stage report has too few viewer commands")

    rows = read_csv(args.task_bc_stage_csv)
    if len(rows) != 6:
        raise RuntimeError(f"task/BC stage csv should have 6 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "阶段",
        "方法",
        "结构定位",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "可训练参数",
        "模型大小MB",
        "主任务视频",
        "失败模式",
        "论文结论",
        "论文红线",
        "主任务viewer命令",
        "训练/采集命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"task/BC stage csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_versions = {
        "expert_scripted_v1",
        "structured_waypoint_policy_v1",
        "replay_demo_v1",
        "linear_bc_v1",
        "knn_bc_v1",
        "mlp_bc_v1",
    }
    csv_versions = {row["版本"] for row in rows}
    if csv_versions != required_versions:
        raise RuntimeError(f"task/BC stage csv versions differ: {sorted(required_versions - csv_versions)}")
    for row in rows:
        if "--viewer" not in row["主任务viewer命令"]:
            raise RuntimeError(f"task/BC stage row has no main viewer command: {row['版本']}")
        if not (ROOT / row["主任务视频"]).exists():
            raise FileNotFoundError(ROOT / row["主任务视频"])
        int(row["可训练参数"])
        float(row["模型大小MB"])
    return rows


def verify_trajectory_act_stage_report(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.trajectory_act_stage_report.exists():
        raise FileNotFoundError(args.trajectory_act_stage_report)
    if not args.trajectory_act_stage_csv.exists():
        raise FileNotFoundError(args.trajectory_act_stage_csv)

    text = args.trajectory_act_stage_report.read_text(encoding="utf-8-sig")
    required = (
        "Trajectory / ACT / Diffusion 阶段报告",
        "trajectory_act_stage_report_v1",
        "阶段展示视频",
        "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
        "docs/video_evidence_gallery.html",
        "未形成有效抓取/未抬升",
        "语言/空间泛化失败",
        "domain_randomization_eval_v1",
        "不能写成完整官方 ACT",
        "不能写成完整视觉 Diffusion Policy",
        "主任务慢速 Viewer 命令",
        "语言/空间泛化慢速 Viewer 命令",
        "训练/重建命令",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "phase_conditioned_torch_act_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "visual_act_cnn_cvae_v1",
        "torch_diffusion_policy_state_chunk_v1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"trajectory/ACT stage report is missing required terms: {missing}")
    if text.count("--viewer") < 11:
        raise RuntimeError("trajectory/ACT stage report has too few viewer commands")

    rows = read_csv(args.trajectory_act_stage_csv)
    if len(rows) != 11:
        raise RuntimeError(f"trajectory/ACT stage csv should have 11 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "阶段",
        "方法",
        "结构定位",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "可训练参数",
        "模型大小MB",
        "主任务视频",
        "失败模式",
        "论文结论",
        "论文红线",
        "主任务viewer命令",
        "训练命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"trajectory/ACT stage csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_versions = {
        "act_lite_chunk_bc_v1",
        "diffusion_policy_lite_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "phase_conditioned_torch_act_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "visual_act_cnn_cvae_v1",
    }
    csv_versions = {row["版本"] for row in rows}
    if csv_versions != required_versions:
        raise RuntimeError(f"trajectory/ACT stage csv versions differ: {sorted(required_versions - csv_versions)}")
    for row in rows:
        if "--viewer" not in row["主任务viewer命令"]:
            raise RuntimeError(f"trajectory/ACT stage row has no viewer command: {row['版本']}")
        if not (ROOT / row["主任务视频"]).exists():
            raise FileNotFoundError(ROOT / row["主任务视频"])
        int(row["可训练参数"])
        float(row["模型大小MB"])
    return rows


def verify_trajectory_act_experiment_record(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.trajectory_act_experiment_record.exists():
        raise FileNotFoundError(args.trajectory_act_experiment_record)
    if not args.trajectory_act_experiment_record_csv.exists():
        raise FileNotFoundError(args.trajectory_act_experiment_record_csv)

    text = args.trajectory_act_experiment_record.read_text(encoding="utf-8-sig")
    required = (
        "Trajectory-conditioned BC / ACT 中文实验台账",
        "trajectory_act_experiment_record_v1",
        "主任务训练范围成功率等价总和",
        "严格抓取审计",
        "原始放置成功 `14/53`",
        "严格抓取成功 `0/53`",
        "--viewer --duration 60 --speed 0.05",
        "主任务慢速 viewer",
        "语言/空间泛化慢速 viewer",
        "训练命令",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
        "candidate:grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "candidate:grasp_gated_torch_act_state_chunk_v1_candidate",
        "candidate:contact_aware_trajectory_knn_v1_candidate",
        "不能写：当前结果已经证明完整官方 ACT 成功",
        "不能写：当前结果已经证明稳定抓取成功",
        "不能写：当前结果已经是 OpenVLA、RT-2 或真实机器人 VLA 后训练结果",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"trajectory/ACT experiment record is missing required terms: {missing}")
    if text.count("--viewer") < 18:
        raise RuntimeError("trajectory/ACT experiment record has too few viewer commands")

    rows = read_csv(args.trajectory_act_experiment_record_csv)
    if len(rows) != 9:
        raise RuntimeError(f"trajectory/ACT experiment record csv should have 9 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "方法",
        "阶段",
        "输入形式",
        "动作形式",
        "最终模型",
        "训练范围成功率",
        "留出范围成功率",
        "语言/空间泛化",
        "可训练参数",
        "模型大小MB",
        "固定视频",
        "固定视频结果",
        "抓取标志",
        "物体高度",
        "实验结论",
        "论文红线",
        "主任务Viewer命令",
        "语言Viewer命令",
        "训练命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"trajectory/ACT experiment record csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_versions = {
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
    csv_versions = {row["版本"] for row in rows}
    if csv_versions != required_versions:
        raise RuntimeError(f"trajectory/ACT experiment record csv versions differ: {sorted(required_versions - csv_versions)}")
    for row in rows:
        if "--viewer" not in row["主任务Viewer命令"] or "--duration 60" not in row["主任务Viewer命令"]:
            raise RuntimeError(f"trajectory/ACT experiment record row has no slow main viewer command: {row['版本']}")
        if "--viewer" not in row["语言Viewer命令"] or "--duration 60" not in row["语言Viewer命令"]:
            raise RuntimeError(f"trajectory/ACT experiment record row has no slow language viewer command: {row['版本']}")
        if not row["训练命令"]:
            raise RuntimeError(f"trajectory/ACT experiment record row has no training command: {row['版本']}")
        if not (ROOT / row["最终模型"]).exists():
            raise FileNotFoundError(ROOT / row["最终模型"])
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        int(row["可训练参数"])
        float(row["模型大小MB"])
    return rows


def verify_trajectory_act_failure_diagnosis(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.trajectory_act_diagnosis.exists():
        raise FileNotFoundError(args.trajectory_act_diagnosis)
    if not args.trajectory_act_diagnosis_csv.exists():
        raise FileNotFoundError(args.trajectory_act_diagnosis_csv)

    text = args.trajectory_act_diagnosis.read_text(encoding="utf-8-sig")
    required = (
        "Trajectory / ACT 失败诊断矩阵",
        "trajectory_act_failure_diagnosis_v1",
        "--duration 60 --speed 0.05",
        "不能简单写成“播放速度太快”",
        "闭环接触、夹紧、抬升和泛化",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
        "动作速度诊断",
        "Start-Process",
        "docs/trajectory_act_stage_report.csv",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"trajectory/ACT failure diagnosis is missing required terms: {missing}")

    rows = read_csv(args.trajectory_act_diagnosis_csv)
    if len(rows) != 11:
        raise RuntimeError(f"trajectory/ACT failure diagnosis csv should have 11 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "方法",
        "阶段",
        "结构定位",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "固定视频",
        "success",
        "target_distance",
        "ee_object_distance",
        "object_z",
        "grasp_success",
        "contact_count",
        "mean_action_norm",
        "max_action_norm",
        "接触诊断",
        "夹紧/抬升诊断",
        "泛化诊断",
        "动作速度诊断",
        "可写结论",
        "论文红线",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"trajectory/ACT failure diagnosis csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    versions = {row["版本"] for row in rows}
    required_versions = {
        "act_lite_chunk_bc_v1",
        "diffusion_policy_lite_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "phase_conditioned_torch_act_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "visual_act_cnn_cvae_v1",
    }
    if versions != required_versions:
        raise RuntimeError(f"trajectory/ACT failure diagnosis versions differ: {sorted(required_versions - versions)}")
    for row in rows:
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        float(row["target_distance"])
        float(row["ee_object_distance"])
        float(row["object_z"])
        float(row["contact_count"])
        float(row["mean_action_norm"])
        float(row["max_action_norm"])
        if not row["接触诊断"] or not row["夹紧/抬升诊断"] or not row["泛化诊断"] or not row["动作速度诊断"]:
            raise RuntimeError(f"trajectory/ACT failure diagnosis row lacks diagnosis text: {row['版本']}")
    return rows


def verify_trajectory_act_conclusion_brief(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.trajectory_act_conclusion_brief.exists():
        raise FileNotFoundError(args.trajectory_act_conclusion_brief)
    if not args.trajectory_act_conclusion_csv.exists():
        raise FileNotFoundError(args.trajectory_act_conclusion_csv)

    text = args.trajectory_act_conclusion_brief.read_text(encoding="utf-8-sig")
    required = (
        "Trajectory / ACT 论文结论摘要",
        "trajectory_act_conclusion_brief_v1",
        "一句话结论",
        "论文可写口径",
        "推荐展示顺序",
        "trajectory-conditioned BC / ACT 正式对照",
        "Diffusion 相邻对照",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
        "grasp_success=True",
        "不能证明完整官方 ACT 无效",
        "docs/trajectory_act_failure_diagnosis.md",
        "docs/strict_grasp_success_audit.md",
        "--target trajectory-act-brief",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"trajectory/ACT conclusion brief is missing required terms: {missing}")

    rows = read_csv(args.trajectory_act_conclusion_csv)
    if len(rows) != 11:
        raise RuntimeError(f"trajectory/ACT conclusion brief csv should have 11 rows, found {len(rows)}")
    required_columns = {
        "分组",
        "版本",
        "方法",
        "结构定位",
        "训练范围",
        "留出范围",
        "语言/空间泛化",
        "固定视频",
        "固定视频结果",
        "抓取标志",
        "物体高度",
        "可写结论",
        "论文红线",
        "推荐展示命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"trajectory/ACT conclusion brief csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    required_versions = {
        "act_lite_chunk_bc_v1",
        "diffusion_policy_lite_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_state_chunk_cuda_v1",
        "phase_conditioned_torch_act_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_feature_act_lite_v1",
        "visual_act_cnn_cvae_v1",
    }
    versions = {row["版本"] for row in rows}
    if versions != required_versions:
        raise RuntimeError(f"trajectory/ACT conclusion brief versions differ: {sorted(required_versions - versions)}")
    core_rows = [row for row in rows if row["分组"] == "trajectory-conditioned BC / ACT 正式对照"]
    adjacent_rows = [row for row in rows if row["分组"] == "Diffusion 相邻对照"]
    if len(core_rows) != 9 or len(adjacent_rows) != 2:
        raise RuntimeError("trajectory/ACT conclusion brief has unexpected group counts")
    for row in rows:
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        if "--viewer" not in row["推荐展示命令"]:
            raise RuntimeError(f"trajectory/ACT conclusion row lacks viewer command: {row['版本']}")
        if not row["可写结论"] or not row["论文红线"]:
            raise RuntimeError(f"trajectory/ACT conclusion row lacks conclusion or redline: {row['版本']}")
        if row["分组"] == "trajectory-conditioned BC / ACT 正式对照" and row["抓取标志"] not in {"False", "success=False"}:
            raise RuntimeError(f"trajectory/ACT core row should preserve strict grasp failure evidence: {row['版本']}")
    return rows


def verify_trajectory_act_slow_viewer_guide(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (args.trajectory_act_slow_viewer_guide, args.trajectory_act_slow_viewer_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.trajectory_act_slow_viewer_guide.read_text(encoding="utf-8-sig")
    required = (
        "Trajectory-conditioned BC / ACT 超慢可视化指南",
        "trajectory_act_slow_viewer_guide_v1",
        "--target trajectory-act-slow",
        "--viewer-speed 0.02",
        "--viewer-duration 90",
        "标准慢速 viewer",
        "超慢学习 viewer",
        "不新增量化评测",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"trajectory/ACT slow viewer guide is missing required terms: {missing}")

    rows = read_csv(args.trajectory_act_slow_viewer_csv)
    if len(rows) != 5:
        raise RuntimeError(f"trajectory/ACT slow viewer guide should have 5 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "定位",
        "观看目的",
        "训练范围成功率",
        "留出范围成功率",
        "语言/空间泛化",
        "固定视频",
        "标准慢速viewer命令",
        "超慢学习viewer命令",
        "launcher超慢命令",
        "训练命令",
        "论文口径",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"trajectory/ACT slow viewer csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_versions = {
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "torch_act_cvae_state_chunk_v1",
        "visual_act_cnn_cvae_v1",
    }
    versions = {row["版本"] for row in rows}
    if versions != required_versions:
        raise RuntimeError(f"trajectory/ACT slow viewer guide versions differ: {sorted(required_versions - versions)}")
    for row in rows:
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        if "--viewer --duration 90 --speed 0.02" not in row["超慢学习viewer命令"]:
            raise RuntimeError(f"trajectory/ACT slow viewer row lacks super-slow command: {row['版本']}")
        if "--viewer --duration 60 --speed 0.05" not in row["标准慢速viewer命令"]:
            raise RuntimeError(f"trajectory/ACT slow viewer row lacks standard slow command: {row['版本']}")
        if f"--target method:{row['版本']}" not in row["launcher超慢命令"]:
            raise RuntimeError(f"trajectory/ACT slow viewer row lacks launcher command: {row['版本']}")
        if "--viewer-speed 0.02" not in row["launcher超慢命令"] or "--viewer-duration 90" not in row["launcher超慢命令"]:
            raise RuntimeError(f"trajectory/ACT slow viewer launcher timing is wrong: {row['版本']}")
        if not row["观看目的"] or not row["论文口径"]:
            raise RuntimeError(f"trajectory/ACT slow viewer row lacks purpose or boundary: {row['版本']}")
    return rows


def verify_final_defense_narrative(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.final_defense_narrative.exists():
        raise FileNotFoundError(args.final_defense_narrative)
    if not args.final_defense_narrative_csv.exists():
        raise FileNotFoundError(args.final_defense_narrative_csv)

    text = args.final_defense_narrative.read_text(encoding="utf-8-sig")
    required = (
        "最终答辩讲解脚本",
        "final_defense_narrative_script_v1",
        "--target narrative-script",
        "trajectory-conditioned BC / ACT / Diffusion 对照",
        "action head / Adapter / LoRA / VLM proxy",
        "语言/空间泛化能力",
        "MuJoCo domain randomization 与外部依赖 readiness",
        "视频证据、cue sheet 和可视化演示",
        "论文红线",
        "不能把 readiness、handoff、remote pack 或数据桥接写成已经完成的真实实验结果",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"final defense narrative is missing required terms: {missing}")

    rows = read_csv(args.final_defense_narrative_csv)
    if len(rows) != 10:
        raise RuntimeError(f"final defense narrative csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "顺序",
        "讲解段落",
        "对应研究问题/阶段",
        "建议时长",
        "推荐打开命令",
        "推荐证据",
        "可说结论",
        "讲解稿",
        "论文红线",
        "承接下一步",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"final defense narrative csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    expected_titles = {
        "总目标、证据范围和论文边界",
        "任务、数据和可复现实验链路",
        "普通 BC baseline 和失败模式",
        "trajectory-conditioned BC / ACT / Diffusion 对照",
        "action head / Adapter / LoRA / VLM proxy",
        "语言/空间泛化能力",
        "数据效率、资源消耗和横向比较",
        "MuJoCo domain randomization 与外部依赖 readiness",
        "视频证据、cue sheet 和可视化演示",
        "最终结论和下一阶段计划",
    }
    titles = {row["讲解段落"] for row in rows}
    if titles != expected_titles:
        raise RuntimeError(f"final defense narrative titles differ: {sorted(expected_titles - titles)}")
    for row in rows:
        if "showcase_launcher.py" not in row["推荐打开命令"]:
            raise RuntimeError(f"final defense narrative row lacks launcher command: {row['讲解段落']}")
        if not row["推荐证据"] or not row["讲解稿"] or not row["论文红线"]:
            raise RuntimeError(f"final defense narrative row lacks evidence, script, or redline: {row['讲解段落']}")
    return rows


def verify_remaining_experiment_board(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.remaining_experiment_board.exists():
        raise FileNotFoundError(args.remaining_experiment_board)
    if not args.remaining_experiment_board_csv.exists():
        raise FileNotFoundError(args.remaining_experiment_board_csv)

    text = args.remaining_experiment_board.read_text(encoding="utf-8-sig")
    required = (
        "剩余实验执行看板",
        "remaining_experiment_execution_board_v1",
        "--target remaining-board",
        "robot_vla_action_head_lite_v1",
        "robot_vla_adapter_lite_v1",
        "robot_vla_lora_lite_v1",
        "preference_trajectory_post_training_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
        "任何 planned 版本只有在回填了评测 CSV/JSON",
        "Isaac 和真实 WidowX 结果不能用 MuJoCo 视频替代",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"remaining experiment board is missing required terms: {missing}")

    rows = read_csv(args.remaining_experiment_board_csv)
    if len(rows) != 6:
        raise RuntimeError(f"remaining experiment board csv should have 6 rows, found {len(rows)}")
    required_columns = {
        "优先级",
        "版本",
        "类别",
        "当前状态",
        "执行环境",
        "阻塞条件",
        "下一步动作",
        "必需回填工件",
        "成功/升级门槛",
        "完成后重建命令",
        "完成后验证命令",
        "论文红线",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"remaining experiment board csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    expected_versions = {
        "preference_trajectory_post_training_v1",
        "robot_vla_action_head_lite_v1",
        "robot_vla_adapter_lite_v1",
        "robot_vla_lora_lite_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
    }
    versions = {row["版本"] for row in rows}
    if versions != expected_versions:
        raise RuntimeError(f"remaining experiment board versions differ: {sorted(expected_versions - versions)}")
    for row in rows:
        if "verify_experiment_artifacts.py" not in row["完成后验证命令"]:
            raise RuntimeError(f"remaining experiment board row lacks verification command: {row['版本']}")
        if not row["阻塞条件"] or not row["必需回填工件"] or not row["论文红线"]:
            raise RuntimeError(f"remaining experiment board row lacks blocker, return artifacts, or redline: {row['版本']}")
    return rows


def verify_trajectory_phase_template_bc(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.trajectory_phase_template_report.exists():
        raise FileNotFoundError(args.trajectory_phase_template_report)
    if not args.trajectory_phase_template_csv.exists():
        raise FileNotFoundError(args.trajectory_phase_template_csv)
    if not args.trajectory_phase_template_json.exists():
        raise FileNotFoundError(args.trajectory_phase_template_json)
    if not args.trajectory_phase_template_model.exists():
        raise FileNotFoundError(args.trajectory_phase_template_model)

    text = args.trajectory_phase_template_report.read_text(encoding="utf-8-sig")
    required = (
        "Trajectory Phase Template BC 候选实验",
        "trajectory_phase_template_bc_v1_candidate",
        "grasp_successes=0",
        "不能登记为可靠 ACT baseline",
        "train_range",
        "heldout",
        "train_trajectory_phase_template_bc.py",
        "run_trajectory_phase_template_policy.py",
        "evaluate_trajectory_phase_template_bc.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"trajectory phase template BC report is missing required terms: {missing}")

    rows = read_csv(args.trajectory_phase_template_csv)
    if len(rows) != 2:
        raise RuntimeError(f"trajectory phase template BC csv should have 2 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "episodes",
        "successes",
        "success_rate",
        "grasp_successes",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "out_of_table",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"trajectory phase template BC csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    splits = {row["split"] for row in rows}
    if splits != {"train_range", "heldout"}:
        raise RuntimeError(f"trajectory phase template BC splits differ: {sorted(splits)}")
    for row in rows:
        if row["version"] != "trajectory_phase_template_bc_v1_candidate":
            raise RuntimeError(f"unexpected trajectory phase template version: {row['version']}")
        int(row["seed"])
        int(row["episodes"])
        int(row["successes"])
        int(row["grasp_successes"])
        float(row["success_rate"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        int(row["out_of_table"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"trajectory phase template BC command is missing no-viewer or full steps: {row['split']}")

    data = read_json(args.trajectory_phase_template_json)
    if data.get("version") != "trajectory_phase_template_bc_v1_candidate":
        raise RuntimeError("trajectory phase template BC json has unexpected version")
    if len(data.get("rows", [])) != 2:
        raise RuntimeError("trajectory phase template BC json should have 2 rows")
    return rows


def verify_grasp_gated_trajectory_knn(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.grasp_gated_trajectory_knn_report,
        args.grasp_gated_trajectory_knn_csv,
        args.grasp_gated_trajectory_knn_json,
        args.grasp_gated_trajectory_knn_runner,
        args.grasp_gated_trajectory_knn_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.grasp_gated_trajectory_knn_report.read_text(encoding="utf-8-sig")
    required = (
        "Grasp-gated Trajectory-kNN 候选实验",
        "grasp_gated_trajectory_knn_v1_candidate",
        "grasp_successes=0",
        "不能登记为可靠 ACT baseline",
        "gripper gate",
        "run_grasp_gated_trajectory_knn_policy.py",
        "evaluate_grasp_gated_trajectory_knn.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"grasp-gated trajectory-kNN report is missing required terms: {missing}")

    rows = read_csv(args.grasp_gated_trajectory_knn_csv)
    if len(rows) != 2:
        raise RuntimeError(f"grasp-gated trajectory-kNN csv should have 2 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "episodes",
        "successes",
        "success_rate",
        "grasp_successes",
        "out_of_table",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "mean_gate_closed_steps",
        "mean_gate_open_steps",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"grasp-gated trajectory-kNN csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    splits = {row["split"] for row in rows}
    if splits != {"train_range", "heldout"}:
        raise RuntimeError(f"grasp-gated trajectory-kNN splits differ: {sorted(splits)}")
    for row in rows:
        if row["version"] != "grasp_gated_trajectory_knn_v1_candidate":
            raise RuntimeError(f"unexpected grasp-gated trajectory-kNN version: {row['version']}")
        int(row["seed"])
        int(row["episodes"])
        int(row["successes"])
        int(row["grasp_successes"])
        int(row["out_of_table"])
        float(row["success_rate"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        float(row["mean_gate_closed_steps"])
        float(row["mean_gate_open_steps"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"grasp-gated trajectory-kNN command is missing no-viewer or full steps: {row['split']}")

    data = read_json(args.grasp_gated_trajectory_knn_json)
    if data.get("version") != "grasp_gated_trajectory_knn_v1_candidate":
        raise RuntimeError("grasp-gated trajectory-kNN json has unexpected version")
    if len(data.get("rows", [])) != 2:
        raise RuntimeError("grasp-gated trajectory-kNN json should have 2 rows")
    return rows


def verify_preference_trajectory_post_training(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.preference_trajectory_post_training_report,
        args.preference_trajectory_post_training_csv,
        args.preference_trajectory_post_training_json,
        args.preference_trajectory_post_training_model,
        args.preference_trajectory_post_training_trainer,
        args.preference_trajectory_post_training_runner,
        args.preference_trajectory_post_training_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.preference_trajectory_post_training_report.read_text(encoding="utf-8-sig")
    required = (
        "Preference Trajectory Post-training 候选实验",
        "preference_trajectory_post_training_v1_candidate",
        "偏好来源",
        "权重策略",
        "preferred attempts",
        "failed attempts",
        "grasp_successes",
        "不能写成：在线 RL",
        "train_preference_trajectory_post_training.py",
        "run_preference_trajectory_post_training_policy.py",
        "evaluate_preference_trajectory_post_training.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"preference trajectory post-training report is missing required terms: {missing}")

    rows = read_csv(args.preference_trajectory_post_training_csv)
    if len(rows) != 2:
        raise RuntimeError(f"preference trajectory post-training csv should have 2 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "episodes",
        "successes",
        "success_rate",
        "grasp_successes",
        "out_of_table",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"preference trajectory post-training csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    splits = {row["split"] for row in rows}
    if splits != {"train_range", "heldout"}:
        raise RuntimeError(f"preference trajectory post-training splits differ: {sorted(splits)}")
    for row in rows:
        if row["version"] != "preference_trajectory_post_training_v1_candidate":
            raise RuntimeError(f"unexpected preference trajectory post-training version: {row['version']}")
        int(row["seed"])
        int(row["episodes"])
        int(row["successes"])
        int(row["grasp_successes"])
        int(row["out_of_table"])
        float(row["success_rate"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"preference trajectory post-training command is missing no-viewer or full steps: {row['split']}")

    data = read_json(args.preference_trajectory_post_training_json)
    if data.get("version") != "preference_trajectory_post_training_v1_candidate":
        raise RuntimeError("preference trajectory post-training json has unexpected version")
    if len(data.get("rows", [])) != 2:
        raise RuntimeError("preference trajectory post-training json should have 2 rows")
    return rows


def verify_preference_ranked_objective(args: argparse.Namespace) -> list[dict[str, str]]:
    video_metadata_path = args.preference_ranked_objective_video.with_suffix(".json")
    for path in (
        args.preference_ranked_objective_summary,
        args.preference_ranked_objective_report,
        args.preference_ranked_objective_csv,
        args.preference_ranked_objective_json,
        args.preference_ranked_objective_model,
        args.preference_ranked_objective_video,
        video_metadata_path,
        args.preference_trajectory_post_training_trainer,
        args.preference_trajectory_post_training_runner,
        args.preference_trajectory_post_training_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    summary_text = args.preference_ranked_objective_summary.read_text(encoding="utf-8-sig")
    required_summary = (
        "Ranked Objective Preference Trajectory Post-training 候选总结",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "sample-stride=8",
        "| train_range | 0 | 5 | 4/5 | 0/5 |",
        "| heldout | 100 | 5 | 0/5 | 0/5 |",
        "标准抓取仍为 0/10",
        "固定 seed0 视频仍失败",
        "不能升级为正式",
        "完整命令",
        "train_preference_trajectory_post_training.py",
        "run_preference_trajectory_post_training_policy.py",
        "evaluate_preference_trajectory_post_training.py",
        "export_video.py",
    )
    missing_summary = [item for item in required_summary if item not in summary_text]
    if missing_summary:
        raise RuntimeError(f"ranked-objective preference summary is missing required terms: {missing_summary}")

    report_text = args.preference_ranked_objective_report.read_text(encoding="utf-8-sig")
    required_report = (
        "Preference Trajectory Post-training 候选实验",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "version_id",
        "train_range",
        "heldout",
        "run_preference_trajectory_post_training_policy.py",
    )
    missing_report = [item for item in required_report if item not in report_text]
    if missing_report:
        raise RuntimeError(f"ranked-objective preference report is missing required terms: {missing_report}")

    rows = read_csv(args.preference_ranked_objective_csv)
    if len(rows) != 2:
        raise RuntimeError(f"ranked-objective preference csv should have 2 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "episodes",
        "successes",
        "success_rate",
        "grasp_successes",
        "out_of_table",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"ranked-objective preference csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    splits = {row["split"] for row in rows}
    if splits != {"train_range", "heldout"}:
        raise RuntimeError(f"ranked-objective preference splits differ: {sorted(splits)}")
    by_split = {row["split"]: row for row in rows}
    train = by_split["train_range"]
    heldout = by_split["heldout"]
    for row in rows:
        if row["version"] != "preference_trajectory_post_training_v1_ranked_objective_candidate":
            raise RuntimeError(f"unexpected ranked-objective version: {row['version']}")
        int(row["seed"])
        int(row["out_of_table"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"ranked-objective evaluation command is missing no-viewer or full steps: {row['split']}")
    if (int(train["episodes"]), int(train["successes"]), int(train["grasp_successes"])) != (5, 4, 0):
        raise RuntimeError("ranked-objective train metrics should be episodes=5 successes=4 grasp_successes=0")
    if (int(heldout["episodes"]), int(heldout["successes"]), int(heldout["grasp_successes"])) != (5, 0, 0):
        raise RuntimeError("ranked-objective heldout metrics should be episodes=5 successes=0 grasp_successes=0")
    if float(train["success_rate"]) != 0.8 or float(heldout["success_rate"]) != 0.0:
        raise RuntimeError("ranked-objective success rates are unexpected")

    data = read_json(args.preference_ranked_objective_json)
    if data.get("version") != "preference_trajectory_post_training_v1_ranked_objective_candidate":
        raise RuntimeError("ranked-objective preference json has unexpected version")
    if len(data.get("rows", [])) != 2:
        raise RuntimeError("ranked-objective preference json should have 2 rows")
    metadata = data.get("metadata", {})
    if metadata.get("version") != "preference_trajectory_post_training_v1_ranked_objective_candidate":
        raise RuntimeError("ranked-objective metadata has unexpected version")
    if int(metadata.get("samples", 0)) != 46728:
        raise RuntimeError("ranked-objective metadata should document 46728 samples")
    if metadata.get("preference_mode") != "episode_rank":
        raise RuntimeError("ranked-objective metadata should document episode_rank preference mode")
    preference_summary = metadata.get("preference_summary", {})
    if int(preference_summary.get("preferred_attempts", 0)) != 92:
        raise RuntimeError("ranked-objective metadata should document 92 preferred attempts")
    if int(preference_summary.get("failed_attempts", 0)) != 40:
        raise RuntimeError("ranked-objective metadata should document 40 failed attempts")

    ffprobe_video(args.preference_ranked_objective_video)
    video_metadata = read_json(video_metadata_path)
    if video_metadata.get("version") != "preference_trajectory_post_training_v1_ranked_objective_candidate":
        raise RuntimeError("ranked-objective video metadata has unexpected version")
    if video_metadata.get("method") != "preference_trajectory_post_training":
        raise RuntimeError("ranked-objective video metadata has unexpected method")
    video_summary = video_metadata.get("summary", {})
    if video_summary.get("success"):
        raise RuntimeError("ranked-objective fixed seed0 video should preserve failure")
    if video_summary.get("tcp_grasp_lift_success"):
        raise RuntimeError("ranked-objective fixed seed0 video should not claim tcp lift")
    if video_summary.get("grasp_success"):
        raise RuntimeError("ranked-objective fixed seed0 video should not claim standard grasp success")
    if video_summary.get("strict_grasp_lift_success"):
        raise RuntimeError("ranked-objective fixed seed0 video should not claim strict grasp success")
    return rows


def verify_preference_ranked_fast(args: argparse.Namespace) -> list[dict[str, str]]:
    video_metadata_path = args.preference_ranked_fast_video.with_suffix(".json")
    for path in (
        args.preference_ranked_fast_summary,
        args.preference_ranked_fast_report,
        args.preference_ranked_fast_csv,
        args.preference_ranked_fast_json,
        args.preference_ranked_fast_model,
        args.preference_ranked_fast_video,
        video_metadata_path,
        args.preference_trajectory_post_training_trainer,
        args.preference_trajectory_post_training_runner,
        args.preference_trajectory_post_training_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    summary_text = args.preference_ranked_fast_summary.read_text(encoding="utf-8-sig")
    required_summary = (
        "Ranked Preference Trajectory Post-training 快速候选总结",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "sample-stride=32",
        "| train_range | 0 | 5 | 2/5 | 0/5 |",
        "| heldout | 100 | 5 | 0/5 | 0/5 |",
        "标准抓取仍为 0/10",
        "不能升级为正式",
        "完整命令",
        "train_preference_trajectory_post_training.py",
        "run_preference_trajectory_post_training_policy.py",
        "evaluate_preference_trajectory_post_training.py",
        "export_video.py",
    )
    missing_summary = [item for item in required_summary if item not in summary_text]
    if missing_summary:
        raise RuntimeError(f"ranked-fast preference summary is missing required terms: {missing_summary}")

    report_text = args.preference_ranked_fast_report.read_text(encoding="utf-8-sig")
    required_report = (
        "Preference Trajectory Post-training 候选实验",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "version_id",
        "train_range",
        "heldout",
        "run_preference_trajectory_post_training_policy.py",
    )
    missing_report = [item for item in required_report if item not in report_text]
    if missing_report:
        raise RuntimeError(f"ranked-fast preference report is missing required terms: {missing_report}")

    rows = read_csv(args.preference_ranked_fast_csv)
    if len(rows) != 2:
        raise RuntimeError(f"ranked-fast preference csv should have 2 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "episodes",
        "successes",
        "success_rate",
        "grasp_successes",
        "out_of_table",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"ranked-fast preference csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    splits = {row["split"] for row in rows}
    if splits != {"train_range", "heldout"}:
        raise RuntimeError(f"ranked-fast preference splits differ: {sorted(splits)}")
    by_split = {row["split"]: row for row in rows}
    train = by_split["train_range"]
    heldout = by_split["heldout"]
    if train["version"] != "preference_trajectory_post_training_v1_ranked_fast_candidate":
        raise RuntimeError("ranked-fast train row has unexpected version")
    if heldout["version"] != "preference_trajectory_post_training_v1_ranked_fast_candidate":
        raise RuntimeError("ranked-fast heldout row has unexpected version")
    if (int(train["episodes"]), int(train["successes"]), int(train["grasp_successes"])) != (5, 2, 0):
        raise RuntimeError("ranked-fast train metrics should be episodes=5 successes=2 grasp_successes=0")
    if (int(heldout["episodes"]), int(heldout["successes"]), int(heldout["grasp_successes"])) != (5, 0, 0):
        raise RuntimeError("ranked-fast heldout metrics should be episodes=5 successes=0 grasp_successes=0")
    if float(train["success_rate"]) != 0.4 or float(heldout["success_rate"]) != 0.0:
        raise RuntimeError("ranked-fast success rates are unexpected")
    for row in rows:
        int(row["seed"])
        int(row["out_of_table"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"ranked-fast evaluation command is missing no-viewer or full steps: {row['split']}")

    data = read_json(args.preference_ranked_fast_json)
    if data.get("version") != "preference_trajectory_post_training_v1_ranked_fast_candidate":
        raise RuntimeError("ranked-fast preference json has unexpected version")
    if len(data.get("rows", [])) != 2:
        raise RuntimeError("ranked-fast preference json should have 2 rows")
    metadata = data.get("metadata", {})
    if metadata.get("version") != "preference_trajectory_post_training_v1_ranked_fast_candidate":
        raise RuntimeError("ranked-fast metadata has unexpected version")
    if int(metadata.get("samples", 0)) != 11748:
        raise RuntimeError("ranked-fast metadata should document 11748 samples")
    if metadata.get("preference_mode") != "episode_rank":
        raise RuntimeError("ranked-fast metadata should document episode_rank preference mode")
    preference_summary = metadata.get("preference_summary", {})
    if int(preference_summary.get("preferred_attempts", 0)) != 92:
        raise RuntimeError("ranked-fast metadata should document 92 preferred attempts")
    if int(preference_summary.get("failed_attempts", 0)) != 40:
        raise RuntimeError("ranked-fast metadata should document 40 failed attempts")

    ffprobe_video(args.preference_ranked_fast_video)
    video_metadata = read_json(video_metadata_path)
    if video_metadata.get("version") != "preference_trajectory_post_training_v1_ranked_fast_candidate":
        raise RuntimeError("ranked-fast video metadata has unexpected version")
    if video_metadata.get("method") != "preference_trajectory_post_training":
        raise RuntimeError("ranked-fast video metadata has unexpected method")
    video_summary = video_metadata.get("summary", {})
    if not video_summary.get("success"):
        raise RuntimeError("ranked-fast fixed video should preserve loose placement success")
    if not video_summary.get("tcp_grasp_lift_success"):
        raise RuntimeError("ranked-fast fixed video should preserve tcp lift diagnostic")
    if video_summary.get("grasp_success"):
        raise RuntimeError("ranked-fast fixed video should not claim standard grasp success")
    if video_summary.get("strict_grasp_lift_success"):
        raise RuntimeError("ranked-fast fixed video should not claim strict grasp success")
    return rows


def verify_preference_contact_aware_trajectory_post_training(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.preference_contact_aware_trajectory_post_training_report,
        args.preference_contact_aware_trajectory_post_training_csv,
        args.preference_contact_aware_trajectory_post_training_json,
        args.preference_contact_aware_trajectory_post_training_model,
        args.preference_trajectory_post_training_trainer,
        args.preference_trajectory_post_training_runner,
        args.preference_contact_aware_trajectory_post_training_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.preference_contact_aware_trajectory_post_training_report.read_text(encoding="utf-8-sig")
    required = (
        "Preference + Contact-aware Trajectory Post-training 候选诊断报告",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "偏好与数据配置",
        "汇总结果",
        "单次明细",
        "tcp_lift",
        "standard_ever_grasp",
        "strict",
        "train_preference_trajectory_post_training.py",
        "run_preference_trajectory_post_training_policy.py",
        "evaluate_preference_contact_aware_trajectory_post_training.py",
        "export_video.py",
        "不是在线 RL",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"preference contact-aware trajectory post-training report is missing required terms: {missing}")

    rows = read_csv(args.preference_contact_aware_trajectory_post_training_csv)
    if len(rows) != 10:
        raise RuntimeError(f"preference contact-aware trajectory post-training csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "preference contact-aware trajectory post-training csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    splits = {row["split"] for row in rows}
    if splits != {"train_range", "heldout"}:
        raise RuntimeError(f"preference contact-aware trajectory post-training splits differ: {sorted(splits)}")
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("preference contact-aware trajectory post-training should have 5 train and 5 heldout rows")

    for row in rows:
        if row["version"] != "preference_contact_aware_trajectory_post_training_v1_candidate":
            raise RuntimeError(f"unexpected preference contact-aware trajectory post-training version: {row['version']}")
        int(row["seed"])
        int(row["steps_taken"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        float(row["mean_action_norm"])
        float(row["max_action_norm"])
    if sum(row["success"] == "True" for row in train) != 5:
        raise RuntimeError("preference contact-aware train-range success count should be 5/5")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in train) != 5:
        raise RuntimeError("preference contact-aware train-range tcp lift count should be 5/5")
    if sum(row["success"] == "True" for row in heldout) != 1:
        raise RuntimeError("preference contact-aware heldout success count should be 1/5")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in heldout) != 0:
        raise RuntimeError("preference contact-aware heldout tcp lift count should be 0/5")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("preference contact-aware standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("preference contact-aware strict grasp-lift count should be 0/10")

    data = read_json(args.preference_contact_aware_trajectory_post_training_json)
    if data.get("version") != "preference_contact_aware_trajectory_post_training_v1_candidate":
        raise RuntimeError("preference contact-aware trajectory post-training json has unexpected version")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("preference contact-aware trajectory post-training json should have 10 rows")
    return rows


def verify_preference_ranked_trajectory_post_training(args: argparse.Namespace) -> list[dict[str, str]]:
    video_path = ROOT / "outputs" / "videos" / "preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4"
    video_metadata_path = video_path.with_suffix(".json")
    for path in (
        args.preference_ranked_trajectory_post_training_report,
        args.preference_ranked_trajectory_post_training_csv,
        args.preference_ranked_trajectory_post_training_json,
        args.preference_ranked_trajectory_post_training_model,
        args.preference_trajectory_post_training_trainer,
        args.preference_trajectory_post_training_runner,
        args.preference_ranked_trajectory_post_training_evaluator,
        video_path,
        video_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.preference_ranked_trajectory_post_training_report.read_text(encoding="utf-8-sig")
    required = (
        "Ranked Preference Trajectory Post-training 候选诊断报告",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "preference_mode",
        "episode_rank",
        "汇总结果",
        "单次明细",
        "训练命令",
        "慢速 viewer 命令",
        "固定视频导出命令",
        "不能写",
        "真实 VLA",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"preference ranked trajectory post-training report is missing required terms: {missing}")

    rows = read_csv(args.preference_ranked_trajectory_post_training_csv)
    if len(rows) != 4:
        raise RuntimeError(f"preference ranked trajectory post-training csv should have 4 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "preference ranked trajectory post-training csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 2 or len(heldout) != 2:
        raise RuntimeError("preference ranked trajectory post-training should have 2 train and 2 heldout rows")
    for row in rows:
        if row["version"] != "preference_ranked_trajectory_post_training_v1_candidate":
            raise RuntimeError(f"unexpected preference ranked trajectory post-training version: {row['version']}")
        int(row["seed"])
        int(row["steps_taken"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        float(row["mean_action_norm"])
        float(row["max_action_norm"])
    if sum(row["success"] == "True" for row in train) != 2:
        raise RuntimeError("preference ranked train-range success count should be 2/2")
    if sum(row["success"] == "True" for row in heldout) != 1:
        raise RuntimeError("preference ranked heldout success count should be 1/2")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in train) != 2:
        raise RuntimeError("preference ranked train-range tcp lift count should be 2/2")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("preference ranked standard ever-grasp count should be 0/4")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("preference ranked strict grasp-lift count should be 0/4")

    data = read_json(args.preference_ranked_trajectory_post_training_json)
    if data.get("version") != "preference_ranked_trajectory_post_training_v1_candidate":
        raise RuntimeError("preference ranked trajectory post-training json has unexpected version")
    if data.get("fixed_video") != "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4":
        raise RuntimeError("preference ranked trajectory post-training json fixed video path is wrong")
    if len(data.get("rows", [])) != 4:
        raise RuntimeError("preference ranked trajectory post-training json should have 4 rows")

    ffprobe_video(video_path)
    video_metadata = read_json(video_metadata_path)
    summary = video_metadata.get("summary", {})
    if not summary.get("success"):
        raise RuntimeError("preference ranked fixed video should show loose placement success")
    if summary.get("grasp_success"):
        raise RuntimeError("preference ranked fixed video should not claim standard grasp success")
    if not summary.get("tcp_grasp_lift_success"):
        raise RuntimeError("preference ranked fixed video should preserve tcp lift diagnostic")
    if summary.get("strict_grasp_lift_success"):
        raise RuntimeError("preference ranked fixed video should not claim strict grasp success")
    if float(summary.get("max_object_z", 0.0)) < 0.17:
        raise RuntimeError("preference ranked fixed video max object z is lower than expected diagnostic lift")
    return rows


def verify_preference_post_training_upgrade_gate(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.preference_post_training_upgrade_gate,
        args.preference_post_training_upgrade_gate_csv,
        args.preference_post_training_upgrade_gate_json,
        args.preference_post_training_upgrade_gate_script,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.preference_post_training_upgrade_gate.read_text(encoding="utf-8-sig")
    required = (
        "Preference Post-training 正式升级门禁",
        "preference_post_training_upgrade_gate_v1",
        "preference_trajectory_post_training_v1",
        "当前允许升级 formal：`否`",
        "当前审计候选：`5`",
        "preference_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "不能写：`preference_trajectory_post_training_v1` 已作为正式后训练方法完成",
        "在线 RL",
        "OpenVLA/OFT",
        "--viewer --duration 60 --speed 0.05",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"preference post-training upgrade gate is missing required terms: {missing}")

    rows = read_csv(args.preference_post_training_upgrade_gate_csv)
    if len(rows) != 5:
        raise RuntimeError(f"preference post-training upgrade gate should have 5 rows, found {len(rows)}")
    required_columns = {
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
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"preference upgrade gate csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    versions = {row["版本"] for row in rows}
    required_versions = {
        "preference_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
    }
    if versions != required_versions:
        raise RuntimeError(f"preference upgrade gate versions differ: {sorted(required_versions - versions)}")
    for row in rows:
        if row["是否允许升级formal"] != "否":
            raise RuntimeError(f"preference upgrade gate should not allow formal upgrade now: {row['版本']}")
        if "严格抓取" not in row["不能升级原因"]:
            raise RuntimeError(f"preference upgrade gate row lacks strict-grasp reason: {row['版本']}")
        if "--viewer --duration 60 --speed 0.05" not in row["viewer命令"]:
            raise RuntimeError(f"preference upgrade gate row lacks slow viewer command: {row['版本']}")
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        if not (ROOT / row["报告"]).exists():
            raise FileNotFoundError(ROOT / row["报告"])

    data = read_json(args.preference_post_training_upgrade_gate_json)
    if data.get("version") != "preference_post_training_upgrade_gate_v1":
        raise RuntimeError("preference upgrade gate json has unexpected version")
    if data.get("target_formal_version") != "preference_trajectory_post_training_v1":
        raise RuntimeError("preference upgrade gate target formal version is wrong")
    if data.get("formal_upgrade_allowed_now") is not False:
        raise RuntimeError("preference upgrade gate should block formal upgrade now")
    if int(data.get("candidate_count", 0)) != 5:
        raise RuntimeError("preference upgrade gate json candidate count is wrong")
    if int(data.get("formal_upgrade_allowed_count", -1)) != 0:
        raise RuntimeError("preference upgrade gate json formal allowed count is wrong")
    if data.get("planned_registry_status") != "planned":
        raise RuntimeError("preference upgrade gate should preserve planned registry status")
    if len(data.get("rows", [])) != 5:
        raise RuntimeError("preference upgrade gate json should have 5 rows")
    return rows


def verify_preference_post_training_ablation(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.preference_post_training_ablation,
        args.preference_post_training_ablation_csv,
        args.preference_post_training_ablation_script,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.preference_post_training_ablation.read_text(encoding="utf-8-sig")
    required = (
        "Preference 后训练消融矩阵",
        "preference_post_training_ablation_matrix_v1",
        "--target preference-ablation",
        "当前候选数量：`5`",
        "当前严格抓取总成功数：`0`",
        "不能把 `preference_trajectory_post_training_v1` 升级为正式后训练方法",
        "preference_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "下一轮设计指向",
        "接触保持",
        "夹爪闭合时序",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"preference post-training ablation matrix is missing required terms: {missing}")

    rows = read_csv(args.preference_post_training_ablation_csv)
    if len(rows) != 5:
        raise RuntimeError(f"preference post-training ablation matrix should have 5 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "偏好来源",
        "权重策略",
        "训练范围放置",
        "留出范围放置",
        "训练范围TCP抬升",
        "留出范围TCP抬升",
        "训练范围严格抓取",
        "留出范围严格抓取",
        "固定视频结果",
        "固定视频",
        "固定视频目标距离",
        "固定视频物体高度",
        "是否允许升级formal",
        "主要失败原因",
        "下一轮设计指向",
        "完整viewer命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"preference ablation csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_versions = {
        "preference_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_tcp_lift_rank_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
    }
    versions = {row["版本"] for row in rows}
    if versions != required_versions:
        raise RuntimeError(f"preference ablation versions differ: {sorted(required_versions - versions)}")
    strict_values = []
    for row in rows:
        if row["是否允许升级formal"] != "否":
            raise RuntimeError(f"preference ablation should not allow formal upgrade: {row['版本']}")
        if "严格抓取" not in row["主要失败原因"]:
            raise RuntimeError(f"preference ablation row lacks strict-grasp failure reason: {row['版本']}")
        if not row["下一轮设计指向"]:
            raise RuntimeError(f"preference ablation row lacks design implication: {row['版本']}")
        if "--viewer --duration 60 --speed 0.05" not in row["完整viewer命令"]:
            raise RuntimeError(f"preference ablation row lacks slow viewer command: {row['版本']}")
        if not (ROOT / row["固定视频"]).exists():
            raise FileNotFoundError(ROOT / row["固定视频"])
        strict_values.append(row["训练范围严格抓取"])
        strict_values.append(row["留出范围严格抓取"])
    if any(value not in {"0/5", "0/2", "未记录"} for value in strict_values):
        raise RuntimeError("preference ablation strict-grasp values should preserve zero or unrecorded status")
    return rows


def verify_contact_phase_gated_torch_act(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.contact_phase_gated_torch_act_report,
        args.contact_phase_gated_torch_act_csv,
        args.contact_phase_gated_torch_act_json,
        args.contact_phase_gated_torch_act_model,
        args.contact_phase_gated_torch_act_evaluator,
        ROOT / "scripts" / "train_torch_act.py",
        ROOT / "scripts" / "run_torch_act_policy.py",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.contact_phase_gated_torch_act_report.read_text(encoding="utf-8-sig")
    required = (
        "Contact/Phase-gated Torch ACT 候选诊断报告",
        "contact_phase_gated_torch_act_v1_candidate",
        "phase-one-hot",
        "grasp/lift/place_release",
        "gripper loss",
        "grasp gate",
        "tcp_lift",
        "standard_ever_grasp",
        "strict",
        "train_torch_act.py",
        "run_torch_act_policy.py",
        "evaluate_contact_phase_gated_torch_act.py",
        "export_video.py",
        "不是完整官方 ACT",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"contact/phase-gated Torch ACT report is missing required terms: {missing}")

    rows = read_csv(args.contact_phase_gated_torch_act_csv)
    if len(rows) != 10:
        raise RuntimeError(f"contact/phase-gated Torch ACT csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "ever_tcp_lift_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
        "gate_open_steps",
        "gate_closed_steps",
        "gate_policy_steps",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(f"contact/phase-gated Torch ACT csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("contact/phase-gated Torch ACT should have 5 train and 5 heldout rows")
    for row in rows:
        if row["version"] != "contact_phase_gated_torch_act_v1_candidate":
            raise RuntimeError(f"unexpected contact/phase-gated Torch ACT version: {row['version']}")
        int(row["seed"])
        int(row["steps_taken"])
        int(row["gate_closed_steps"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
    if sum(row["success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact/phase-gated Torch ACT success count should be 0/10")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 1:
        raise RuntimeError("contact/phase-gated Torch ACT tcp lift count should be 1/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact/phase-gated Torch ACT standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact/phase-gated Torch ACT strict grasp-lift count should be 0/10")

    data = read_json(args.contact_phase_gated_torch_act_json)
    if data.get("version") != "contact_phase_gated_torch_act_v1_candidate":
        raise RuntimeError("contact/phase-gated Torch ACT json has unexpected version")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("contact/phase-gated Torch ACT json should have 10 rows")
    return rows


def verify_contact_aware_phase_gated_torch_act(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.contact_aware_phase_gated_torch_act_report,
        args.contact_aware_phase_gated_torch_act_csv,
        args.contact_aware_phase_gated_torch_act_json,
        args.contact_aware_phase_gated_torch_act_model,
        args.contact_aware_phase_gated_torch_act_evaluator,
        ROOT / "scripts" / "train_torch_act.py",
        ROOT / "scripts" / "run_torch_act_policy.py",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.contact_aware_phase_gated_torch_act_report.read_text(encoding="utf-8-sig")
    required = (
        "Contact-aware Phase-gated Torch ACT 候选诊断报告",
        "contact_aware_phase_gated_torch_act_v1_candidate",
        "augment_relative",
        "phase_one_hot",
        "grasp gate",
        "--augment-relative",
        "tcp_lift",
        "standard_ever_grasp",
        "strict",
        "train_torch_act.py",
        "run_torch_act_policy.py",
        "evaluate_contact_aware_phase_gated_torch_act.py",
        "export_video.py",
        "不是完整官方 ACT",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"contact-aware phase-gated Torch ACT report is missing required terms: {missing}")

    rows = read_csv(args.contact_aware_phase_gated_torch_act_csv)
    if len(rows) != 10:
        raise RuntimeError(f"contact-aware phase-gated Torch ACT csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "ever_tcp_lift_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
        "gate_open_steps",
        "gate_closed_steps",
        "gate_policy_steps",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "contact-aware phase-gated Torch ACT csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("contact-aware phase-gated Torch ACT should have 5 train and 5 heldout rows")
    for row in rows:
        if row["version"] != "contact_aware_phase_gated_torch_act_v1_candidate":
            raise RuntimeError(f"unexpected contact-aware phase-gated Torch ACT version: {row['version']}")
        int(row["seed"])
        int(row["steps_taken"])
        int(row["gate_closed_steps"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
    if sum(row["success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-aware phase-gated Torch ACT success count should be 0/10")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-aware phase-gated Torch ACT tcp lift count should be 0/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-aware phase-gated Torch ACT standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-aware phase-gated Torch ACT strict grasp-lift count should be 0/10")

    data = read_json(args.contact_aware_phase_gated_torch_act_json)
    if data.get("version") != "contact_aware_phase_gated_torch_act_v1_candidate":
        raise RuntimeError("contact-aware phase-gated Torch ACT json has unexpected version")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("contact-aware phase-gated Torch ACT json should have 10 rows")
    return rows


def verify_contact_stage_subpolicy(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.contact_stage_subpolicy_report,
        args.contact_stage_subpolicy_csv,
        args.contact_stage_subpolicy_json,
        args.contact_stage_subpolicy_runner,
        args.contact_stage_subpolicy_evaluator,
        ROOT / "scripts" / "export_video.py",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.contact_stage_subpolicy_report.read_text(encoding="utf-8-sig")
    required = (
        "Contact-stage Subpolicy",
        "contact_stage_subpolicy_v1_candidate",
        "scripted subpolicy",
        "不是学习策略",
        "run_contact_stage_subpolicy.py",
        "evaluate_contact_stage_subpolicy.py",
        "export_video.py",
        "不能写成 BC",
        "不能写成 BC、ACT、Diffusion Policy、VLA",
        "接触保持",
        "阶段切换",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"contact-stage subpolicy report is missing required terms: {missing}")

    rows = read_csv(args.contact_stage_subpolicy_csv)
    if len(rows) != 10:
        raise RuntimeError(f"contact-stage subpolicy csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "contact_count",
        "max_contact_count",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "attempts",
        "continued_to_place",
        "steps_taken",
        "first_lift_step",
        "first_tcp_lift_step",
        "stage_steps",
        "stage_min_tcp_distance",
        "stage_max_object_z",
        "stage_max_contact_count",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "contact-stage subpolicy csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("contact-stage subpolicy should have 5 train and 5 heldout rows")
    for row in rows:
        if row["version"] != "contact_stage_subpolicy_v1_candidate":
            raise RuntimeError(f"unexpected contact-stage subpolicy version: {row['version']}")
        int(row["seed"])
        int(row["attempts"])
        int(row["steps_taken"])
        int(row["first_lift_step"])
        int(row["first_tcp_lift_step"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        float(row["contact_count"])
        float(row["max_contact_count"])
        float(row["min_tcp_object_distance"])
        float(row["min_tcp_object_distance_while_lifted"])
        for stage_field in ("stage_steps", "stage_min_tcp_distance", "stage_max_object_z", "stage_max_contact_count"):
            if "approach" not in row[stage_field] or "transfer" not in row[stage_field]:
                raise RuntimeError(f"contact-stage subpolicy row is missing stage trace in {stage_field}")
    if sum(row["success"] == "True" for row in rows) != 10:
        raise RuntimeError("contact-stage subpolicy success count should be 10/10")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 10:
        raise RuntimeError("contact-stage subpolicy tcp lift count should be 10/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage subpolicy standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage subpolicy strict grasp-lift count should be 0/10")
    if sum(row["continued_to_place"] == "True" for row in rows) != 10:
        raise RuntimeError("contact-stage subpolicy continued-to-place count should be 10/10")

    data = read_json(args.contact_stage_subpolicy_json)
    if data.get("version") != "contact_stage_subpolicy_v1_candidate":
        raise RuntimeError("contact-stage subpolicy json has unexpected version")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("contact-stage subpolicy json should have 10 rows")
    return rows


def verify_contact_stage_demo_torch_act(args: argparse.Namespace) -> list[dict[str, str]]:
    demo_summary_path = args.contact_stage_demo_run_dir / "summary.json"
    demo_metadata_path = args.contact_stage_demo_run_dir / "metadata.jsonl"
    video_path = ROOT / "outputs" / "videos" / "contact_stage_demo_torch_act_v1_candidate_seed0.mp4"
    video_metadata_path = ROOT / "outputs" / "videos" / "contact_stage_demo_torch_act_v1_candidate_seed0.json"
    for path in (
        args.contact_stage_demo_torch_act_report,
        args.contact_stage_demo_torch_act_csv,
        args.contact_stage_demo_torch_act_json,
        args.contact_stage_demo_torch_act_model,
        args.contact_stage_demo_torch_act_collector,
        args.contact_stage_demo_torch_act_evaluator,
        demo_summary_path,
        demo_metadata_path,
        video_path,
        video_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    demo_summary = read_json(demo_summary_path)
    if demo_summary.get("version") != "contact_stage_demo_v1":
        raise RuntimeError("contact-stage demo summary has unexpected version")
    if int(demo_summary.get("episodes", 0)) != 12 or int(demo_summary.get("successes", 0)) != 12:
        raise RuntimeError("contact-stage demo summary should document 12/12 successful demos")
    metadata_lines = [line for line in demo_metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(metadata_lines) != 12:
        raise RuntimeError(f"contact-stage demo metadata should have 12 lines, found {len(metadata_lines)}")

    text = args.contact_stage_demo_torch_act_report.read_text(encoding="utf-8-sig")
    required = (
        "Contact-stage Demo Torch ACT",
        "contact_stage_demo_torch_act_v1_candidate",
        "contact_stage_demo_v1",
        "contact_stage_demo_place_blue_cube_blue_pad_medium_v1",
        "state-only ACT-style",
        "12/12",
        "train_range",
        "heldout",
        "run_torch_act_policy.py",
        "evaluate_contact_stage_demo_torch_act.py",
        "export_video.py",
        "--viewer",
        "--speed 0.05",
        "完整官方 ACT",
        "不能写成 ACT 已经学会 contact-stage 控制",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"contact-stage demo Torch ACT report is missing required terms: {missing}")

    rows = read_csv(args.contact_stage_demo_torch_act_csv)
    if len(rows) != 10:
        raise RuntimeError(f"contact-stage demo Torch ACT csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "ever_tcp_lift_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "stop_reason",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
        "gate_open_steps",
        "gate_closed_steps",
        "gate_policy_steps",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "contact-stage demo Torch ACT csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("contact-stage demo Torch ACT should have 5 train and 5 heldout rows")
    for row in rows:
        if row["version"] != "contact_stage_demo_torch_act_v1_candidate":
            raise RuntimeError(f"unexpected contact-stage demo Torch ACT version: {row['version']}")
        int(row["seed"])
        int(row["steps_taken"])
        int(row["gate_open_steps"])
        int(row["gate_closed_steps"])
        int(row["gate_policy_steps"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        float(row["min_tcp_object_distance"])
        if row["min_tcp_object_distance_while_lifted"].strip():
            float(row["min_tcp_object_distance_while_lifted"])
        float(row["mean_action_norm"])
        float(row["max_action_norm"])
    if sum(row["success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage demo Torch ACT success count should be 0/10")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage demo Torch ACT tcp lift count should be 0/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage demo Torch ACT standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage demo Torch ACT strict grasp-lift count should be 0/10")

    data = read_json(args.contact_stage_demo_torch_act_json)
    if data.get("version") != "contact_stage_demo_torch_act_v1_candidate":
        raise RuntimeError("contact-stage demo Torch ACT json has unexpected version")
    if data.get("demo_run") != "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1":
        raise RuntimeError("contact-stage demo Torch ACT json points to the wrong demo run")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("contact-stage demo Torch ACT json should have 10 rows")
    summary = data.get("summary", [])
    if len(summary) != 2 or sum(int(item.get("successes", -1)) for item in summary) != 0:
        raise RuntimeError("contact-stage demo Torch ACT json summary should document 0 successes")
    ffprobe_video(video_path)
    video_metadata = read_json(video_metadata_path)
    if video_metadata.get("summary", {}).get("grasp_success") is not False:
        raise RuntimeError("contact-stage demo Torch ACT video should document failed grasp_success")
    return rows


def verify_contact_stage_phase_action_head(args: argparse.Namespace) -> list[dict[str, str]]:
    video_path = ROOT / "outputs" / "videos" / "contact_stage_phase_action_head_v1_candidate_seed101.mp4"
    video_metadata_path = ROOT / "outputs" / "videos" / "contact_stage_phase_action_head_v1_candidate_seed101.json"
    for path in (
        args.contact_stage_phase_action_head_report,
        args.contact_stage_phase_action_head_csv,
        args.contact_stage_phase_action_head_json,
        args.contact_stage_phase_action_head_model,
        args.contact_stage_phase_action_head_evaluator,
        video_path,
        video_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.contact_stage_phase_action_head_report.read_text(encoding="utf-8-sig")
    required = (
        "Contact-stage Phase Action-Head",
        "contact_stage_phase_action_head_v1_candidate",
        "contact_stage_demo_v1",
        "contact_stage_demo_place_blue_cube_blue_pad_medium_v1",
        "轻量 action-head",
        "Adapter",
        "12/12",
        "train_range",
        "heldout",
        "0/5",
        "1/5",
        "train_phase_action_head.py",
        "evaluate_contact_stage_phase_action_head.py",
        "run_phase_action_head.py",
        "export_video.py",
        "--viewer",
        "--speed 0.05",
        "contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "不能写成稳定抓取成功",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"contact-stage phase action-head report is missing required terms: {missing}")

    rows = read_csv(args.contact_stage_phase_action_head_csv)
    if len(rows) != 10:
        raise RuntimeError(f"contact-stage phase action-head csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "phase_mode",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "ever_tcp_lift_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "stop_reason",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
        "phase_counts",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "contact-stage phase action-head csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("contact-stage phase action-head should have 5 train and 5 heldout rows")
    for row in rows:
        if row["version"] != "contact_stage_phase_action_head_v1_candidate":
            raise RuntimeError(f"unexpected contact-stage phase action-head version: {row['version']}")
        if row["phase_mode"] != "progress":
            raise RuntimeError(f"unexpected contact-stage phase action-head phase_mode: {row['phase_mode']}")
        int(row["seed"])
        int(row["steps_taken"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        float(row["min_tcp_object_distance"])
        if row["min_tcp_object_distance_while_lifted"].strip():
            float(row["min_tcp_object_distance_while_lifted"])
        float(row["mean_action_norm"])
        float(row["max_action_norm"])
        json.loads(row["phase_counts"])
    if sum(row["success"] == "True" for row in train) != 0:
        raise RuntimeError("contact-stage phase action-head train success count should be 0/5")
    if sum(row["success"] == "True" for row in heldout) != 1:
        raise RuntimeError("contact-stage phase action-head heldout success count should be 1/5")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage phase action-head tcp lift count should be 0/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage phase action-head standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-stage phase action-head strict grasp-lift count should be 0/10")
    successful_rows = [row for row in rows if row["success"] == "True"]
    if [row["seed"] for row in successful_rows] != ["101"]:
        raise RuntimeError(f"contact-stage phase action-head expected only heldout seed 101 success: {successful_rows}")

    data = read_json(args.contact_stage_phase_action_head_json)
    if data.get("version") != "contact_stage_phase_action_head_v1_candidate":
        raise RuntimeError("contact-stage phase action-head json has unexpected version")
    if data.get("demo_run") != "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1":
        raise RuntimeError("contact-stage phase action-head json points to the wrong demo run")
    if data.get("phase_mode") != "progress":
        raise RuntimeError("contact-stage phase action-head json has unexpected phase mode")
    if data.get("fixed_video") != "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4":
        raise RuntimeError("contact-stage phase action-head json fixed video should be seed101")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("contact-stage phase action-head json should have 10 rows")
    summary = data.get("summary", [])
    if len(summary) != 2 or sum(int(item.get("successes", -1)) for item in summary) != 1:
        raise RuntimeError("contact-stage phase action-head json summary should document 1 loose success")
    if sum(int(item.get("tcp_grasp_lift_successes", -1)) for item in summary) != 0:
        raise RuntimeError("contact-stage phase action-head json summary should document 0 tcp lift successes")
    ffprobe_video(video_path)
    video_metadata = read_json(video_metadata_path)
    video_summary = video_metadata.get("summary", {})
    if video_summary.get("success") is not True:
        raise RuntimeError("contact-stage phase action-head fixed video should document loose placement success")
    if video_summary.get("grasp_success") is not False:
        raise RuntimeError("contact-stage phase action-head video should document failed grasp_success")
    if video_summary.get("tcp_grasp_lift_success") is not False:
        raise RuntimeError("contact-stage phase action-head video should document failed tcp lift")
    if video_summary.get("strict_grasp_lift_success") is not False:
        raise RuntimeError("contact-stage phase action-head video should document failed strict grasp-lift")
    if float(video_summary.get("target_distance", 1.0)) > 0.06:
        raise RuntimeError("contact-stage phase action-head fixed video target distance is too large")
    if float(video_summary.get("max_object_z", 1.0)) >= 0.08:
        raise RuntimeError("contact-stage phase action-head fixed video should not be counted as lifted")
    return rows


def verify_contact_hold_weighted_torch_act(args: argparse.Namespace) -> list[dict[str, str]]:
    video_path = ROOT / "outputs" / "videos" / "contact_hold_weighted_torch_act_v1_candidate_seed0.mp4"
    video_metadata_path = ROOT / "outputs" / "videos" / "contact_hold_weighted_torch_act_v1_candidate_seed0.json"
    for path in (
        args.contact_hold_weighted_torch_act_report,
        args.contact_hold_weighted_torch_act_csv,
        args.contact_hold_weighted_torch_act_json,
        args.contact_hold_weighted_torch_act_model,
        args.contact_hold_weighted_torch_act_evaluator,
        video_path,
        video_metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.contact_hold_weighted_torch_act_report.read_text(encoding="utf-8-sig")
    required = (
        "Contact-hold Weighted Torch ACT",
        "contact_hold_weighted_torch_act_v1_candidate",
        "contact_stage_demo_v1",
        "contact_stage_demo_place_blue_cube_blue_pad_medium_v1",
        "train_torch_act.py",
        "evaluate_contact_hold_weighted_torch_act.py",
        "run_torch_act_policy.py",
        "export_video.py",
        "--viewer",
        "--speed 0.05",
        "grasp:10,lift:10,transfer:4,place_release:3",
        "0/5",
        "1/5",
        "才能写成稳定抓取",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"contact-hold weighted Torch ACT report is missing required terms: {missing}")

    rows = read_csv(args.contact_hold_weighted_torch_act_csv)
    if len(rows) != 10:
        raise RuntimeError(f"contact-hold weighted Torch ACT csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_grasp_success",
        "ever_tcp_lift_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "out_of_table",
        "steps_taken",
        "stop_reason",
        "min_tcp_object_distance",
        "min_tcp_object_distance_while_lifted",
        "mean_action_norm",
        "max_action_norm",
        "gate_open_steps",
        "gate_closed_steps",
        "gate_policy_steps",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(
            "contact-hold weighted Torch ACT csv is missing columns: "
            f"{sorted(required_columns - set(rows[0]))}"
        )
    train = [row for row in rows if row["split"] == "train_range"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    if len(train) != 5 or len(heldout) != 5:
        raise RuntimeError("contact-hold weighted Torch ACT should have 5 train and 5 heldout rows")
    for row in rows:
        if row["version"] != "contact_hold_weighted_torch_act_v1_candidate":
            raise RuntimeError(f"unexpected contact-hold weighted Torch ACT version: {row['version']}")
        int(row["seed"])
        int(row["steps_taken"])
        int(row["gate_open_steps"])
        int(row["gate_closed_steps"])
        int(row["gate_policy_steps"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        float(row["min_tcp_object_distance"])
        if row["min_tcp_object_distance_while_lifted"].strip():
            float(row["min_tcp_object_distance_while_lifted"])
        float(row["mean_action_norm"])
        float(row["max_action_norm"])
    if sum(row["success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-hold weighted Torch ACT loose success count should be 0/10")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 1:
        raise RuntimeError("contact-hold weighted Torch ACT tcp lift count should be 1/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-hold weighted Torch ACT standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("contact-hold weighted Torch ACT strict grasp-lift count should be 0/10")
    tcp_rows = [row for row in rows if row["tcp_grasp_lift_success"] == "True"]
    if [row["seed"] for row in tcp_rows] != ["0"] or [row["split"] for row in tcp_rows] != ["train_range"]:
        raise RuntimeError(f"contact-hold weighted Torch ACT expected only train seed 0 tcp lift: {tcp_rows}")

    data = read_json(args.contact_hold_weighted_torch_act_json)
    if data.get("version") != "contact_hold_weighted_torch_act_v1_candidate":
        raise RuntimeError("contact-hold weighted Torch ACT json has unexpected version")
    if data.get("demo_run") != "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1":
        raise RuntimeError("contact-hold weighted Torch ACT json points to the wrong demo run")
    if data.get("fixed_video") != "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4":
        raise RuntimeError("contact-hold weighted Torch ACT json fixed video should be seed0")
    if len(data.get("rows", [])) != 10:
        raise RuntimeError("contact-hold weighted Torch ACT json should have 10 rows")
    summary = data.get("summary", [])
    if len(summary) != 2 or sum(int(item.get("successes", -1)) for item in summary) != 0:
        raise RuntimeError("contact-hold weighted Torch ACT json summary should document 0 loose successes")
    if sum(int(item.get("tcp_grasp_lift_successes", -1)) for item in summary) != 1:
        raise RuntimeError("contact-hold weighted Torch ACT json summary should document 1 tcp lift success")

    ffprobe_video(video_path)
    video_metadata = read_json(video_metadata_path)
    video_summary = video_metadata.get("summary", {})
    if video_summary.get("success") is not False:
        raise RuntimeError("contact-hold weighted Torch ACT fixed video should document failed placement")
    if video_summary.get("grasp_success") is not False:
        raise RuntimeError("contact-hold weighted Torch ACT video should document failed grasp_success")
    if video_summary.get("tcp_grasp_lift_success") is not True:
        raise RuntimeError("contact-hold weighted Torch ACT video should document tcp lift success")
    if video_summary.get("strict_grasp_lift_success") is not False:
        raise RuntimeError("contact-hold weighted Torch ACT video should document failed strict grasp-lift")
    if float(video_summary.get("max_object_z", 0.0)) < 0.085:
        raise RuntimeError("contact-hold weighted Torch ACT fixed video should document lifted object height")
    return rows


def verify_gripper_timing_contact_probe(args: argparse.Namespace) -> list[dict[str, str]]:
    video_metadata_path = args.gripper_timing_contact_probe_video.with_suffix(".json")
    for path in (
        args.gripper_timing_contact_probe_report,
        args.gripper_timing_contact_probe_csv,
        args.gripper_timing_contact_probe_json,
        args.gripper_timing_contact_probe_video,
        video_metadata_path,
        args.gripper_timing_contact_probe_runner,
        args.gripper_timing_contact_probe_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.gripper_timing_contact_probe_report.read_text(encoding="utf-8-sig")
    required = (
        "Gripper Timing / Contact Probe 候选诊断报告",
        "gripper_timing_contact_probe_v1_candidate",
        "baseline",
        "tight_close_hold",
        "early_close_hold",
        "tcp_grasp_lift_success",
        "ever_standard_grasp_success",
        "strict_grasp_lift_success",
        "夹爪闭合时序",
        "接触保持",
        "无抬升",
        "run_gripper_timing_probe.py",
        "evaluate_gripper_timing_probe.py",
        "export_video.py",
        "不能写成 learned BC",
        "OpenVLA",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"gripper timing contact probe report is missing required terms: {missing}")

    rows = read_csv(args.gripper_timing_contact_probe_csv)
    if len(rows) != 12:
        raise RuntimeError(f"gripper timing contact probe csv should have 12 rows, found {len(rows)}")
    required_columns = {
        "version",
        "variant",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "object_z",
        "max_object_z",
        "height_threshold_hit",
        "grasp_success",
        "ever_standard_grasp_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "min_tcp_object_distance_while_lifted",
        "min_ee_object_distance_while_lifted",
        "stage_steps",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"gripper timing contact probe csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    expected_variants = {"baseline", "tight_close_hold", "early_close_hold"}
    variants = {row["variant"] for row in rows}
    if variants != expected_variants:
        raise RuntimeError(f"gripper timing variants differ: {sorted(variants)}")
    for row in rows:
        if row["version"] != "gripper_timing_contact_probe_v1_candidate":
            raise RuntimeError(f"unexpected gripper timing version: {row['version']}")
        int(row["seed"])
        float(row["target_distance"])
        float(row["object_z"])
        float(row["max_object_z"])
        if row["min_tcp_object_distance_while_lifted"].strip():
            float(row["min_tcp_object_distance_while_lifted"])
        if row["min_ee_object_distance_while_lifted"].strip():
            float(row["min_ee_object_distance_while_lifted"])

    expected_successes = {
        ("baseline", "train_range"): 2,
        ("baseline", "heldout"): 2,
        ("tight_close_hold", "train_range"): 2,
        ("tight_close_hold", "heldout"): 2,
        ("early_close_hold", "train_range"): 0,
        ("early_close_hold", "heldout"): 1,
    }
    for key, expected in expected_successes.items():
        items = [row for row in rows if (row["variant"], row["split"]) == key]
        if len(items) != 2:
            raise RuntimeError(f"gripper timing split should have 2 rows: {key}")
        successes = sum(row["success"] == "True" for row in items)
        if successes != expected:
            raise RuntimeError(f"gripper timing success count for {key} should be {expected}, found {successes}")

    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows if row["variant"] in {"baseline", "tight_close_hold"}) != 8:
        raise RuntimeError("baseline/tight gripper timing tcp lift count should be 8/8")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows if row["variant"] == "early_close_hold") != 0:
        raise RuntimeError("early-close gripper timing tcp lift count should be 0/4")
    if sum(row["ever_standard_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("gripper timing standard ever-grasp count should be 0/12")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("gripper timing strict grasp-lift count should be 0/12")

    data = read_json(args.gripper_timing_contact_probe_json)
    if data.get("version") != "gripper_timing_contact_probe_v1_candidate":
        raise RuntimeError("gripper timing json has unexpected version")
    if data.get("fixed_video") != "outputs/videos/gripper_timing_contact_probe_v1_candidate_seed0.mp4":
        raise RuntimeError("gripper timing json fixed video path is unexpected")
    if len(data.get("rows", [])) != 12 or len(data.get("summary", [])) != 6:
        raise RuntimeError("gripper timing json should have 12 rows and 6 summary rows")

    ffprobe_video(args.gripper_timing_contact_probe_video)
    video_metadata = read_json(video_metadata_path)
    summary = video_metadata.get("summary", {})
    if summary.get("variant") != "tight_close_hold":
        raise RuntimeError("gripper timing fixed video should use tight_close_hold")
    if summary.get("success") is not True:
        raise RuntimeError("gripper timing fixed video should document placement success")
    if summary.get("tcp_grasp_lift_success") is not True:
        raise RuntimeError("gripper timing fixed video should document tcp lift success")
    if summary.get("ever_standard_grasp_success") is not False:
        raise RuntimeError("gripper timing fixed video should document failed standard grasp")
    if summary.get("strict_grasp_lift_success") is not False:
        raise RuntimeError("gripper timing fixed video should document failed strict grasp-lift")
    return rows


def verify_timing_aware_trajectory_prior_residual(args: argparse.Namespace) -> list[dict[str, str]]:
    video_metadata_path = args.timing_aware_trajectory_prior_residual_video.with_suffix(".json")
    for path in (
        args.timing_aware_trajectory_prior_residual_report,
        args.timing_aware_trajectory_prior_residual_csv,
        args.timing_aware_trajectory_prior_residual_json,
        args.timing_aware_trajectory_prior_residual_video,
        video_metadata_path,
        args.timing_aware_trajectory_prior_residual_model,
        args.timing_aware_trajectory_prior_residual_common,
        args.timing_aware_trajectory_prior_residual_trainer,
        args.timing_aware_trajectory_prior_residual_runner,
        args.timing_aware_trajectory_prior_residual_evaluator,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    text = args.timing_aware_trajectory_prior_residual_report.read_text(encoding="utf-8-sig")
    required = (
        "Timing-aware Trajectory-prior Residual BC 候选实验报告",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "trajectory-conditioned BC / ACT",
        "强闭合",
        "闭合保持",
        "抬升保持",
        "residual-scale",
        "0.02",
        "train_range | 5/5",
        "heldout | 4/5",
        "strict_grasp_lift_success",
        "run_timing_aware_trajectory_prior_residual_policy.py",
        "evaluate_timing_aware_trajectory_prior_residual_bc.py",
        "export_video.py",
        "不能写：完整 ACT",
        "OpenVLA",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"timing-aware trajectory-prior residual report is missing required terms: {missing}")

    rows = read_csv(args.timing_aware_trajectory_prior_residual_csv)
    if len(rows) != 10:
        raise RuntimeError(f"timing-aware trajectory-prior residual csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "version",
        "split",
        "seed",
        "task",
        "complexity",
        "success",
        "target_distance",
        "max_object_z",
        "height_threshold_hit",
        "ever_grasp_success",
        "tcp_grasp_lift_success",
        "strict_grasp_lift_success",
        "continued_to_place",
        "out_of_table",
        "steps_taken",
        "mean_residual_norm",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"timing-aware trajectory-prior residual csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    for row in rows:
        if row["version"] != "timing_aware_trajectory_prior_residual_bc_v1_candidate":
            raise RuntimeError(f"unexpected timing-aware trajectory-prior residual version: {row['version']}")
        int(row["seed"])
        float(row["target_distance"])
        float(row["max_object_z"])
        float(row["mean_residual_norm"])

    train_rows = [row for row in rows if row["split"] == "train_range"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    if len(train_rows) != 5 or len(heldout_rows) != 5:
        raise RuntimeError("timing-aware trajectory-prior residual csv should have 5 train and 5 heldout rows")
    if sum(row["success"] == "True" for row in train_rows) != 5:
        raise RuntimeError("timing-aware trajectory-prior residual train success count should be 5/5")
    if sum(row["success"] == "True" for row in heldout_rows) != 4:
        raise RuntimeError("timing-aware trajectory-prior residual heldout success count should be 4/5")
    if sum(row["tcp_grasp_lift_success"] == "True" for row in rows) != 9:
        raise RuntimeError("timing-aware trajectory-prior residual tcp lift count should be 9/10")
    if sum(row["ever_grasp_success"] == "True" for row in rows) != 0:
        raise RuntimeError("timing-aware trajectory-prior residual standard ever-grasp count should be 0/10")
    if sum(row["strict_grasp_lift_success"] == "True" for row in rows) != 0:
        raise RuntimeError("timing-aware trajectory-prior residual strict grasp-lift count should be 0/10")

    data = read_json(args.timing_aware_trajectory_prior_residual_json)
    if data.get("version") != "timing_aware_trajectory_prior_residual_bc_v1_candidate":
        raise RuntimeError("timing-aware trajectory-prior residual json has unexpected version")
    if data.get("fixed_video") != "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4":
        raise RuntimeError("timing-aware trajectory-prior residual json fixed video path is unexpected")
    if len(data.get("rows", [])) != 10 or len(data.get("summary", [])) != 2:
        raise RuntimeError("timing-aware trajectory-prior residual json should have 10 rows and 2 summary rows")

    ffprobe_video(args.timing_aware_trajectory_prior_residual_video)
    video_metadata = read_json(video_metadata_path)
    if video_metadata.get("version") != "timing_aware_trajectory_prior_residual_bc_v1_candidate":
        raise RuntimeError("timing-aware trajectory-prior residual fixed video metadata has unexpected version")
    if video_metadata.get("method") != "timing_aware_trajectory_prior_residual_bc":
        raise RuntimeError("timing-aware trajectory-prior residual fixed video metadata has unexpected method")
    summary = video_metadata.get("summary", {})
    if summary.get("success") is not True:
        raise RuntimeError("timing-aware trajectory-prior residual fixed video should document placement success")
    if summary.get("tcp_grasp_lift_success") is not True:
        raise RuntimeError("timing-aware trajectory-prior residual fixed video should document tcp lift success")
    if summary.get("strict_grasp_lift_success") is not False:
        raise RuntimeError("timing-aware trajectory-prior residual fixed video should document failed strict grasp-lift")
    if abs(float(summary.get("residual_scale", -1.0)) - 0.02) > 1e-9:
        raise RuntimeError("timing-aware trajectory-prior residual fixed video should use residual_scale=0.02")
    return rows


def verify_control_safety_sweep(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.control_safety_sweep.exists():
        raise FileNotFoundError(args.control_safety_sweep)
    if not args.control_safety_sweep_csv.exists():
        raise FileNotFoundError(args.control_safety_sweep_csv)
    if not args.control_safety_sweep_json.exists():
        raise FileNotFoundError(args.control_safety_sweep_json)

    text = args.control_safety_sweep.read_text(encoding="utf-8-sig")
    required = (
        "控制限幅扫表",
        "control_safety_sweep_v1",
        "失败是否主要由动作太快导致",
        "trajectory_conditioned_chunk_bc_v2",
        "trajectory_knn_chunk_bc_v1",
        "torch_act_state_chunk_v1",
        "current_slow",
        "slower",
        "very_slow",
        "极慢控制仍未成功",
        "不能写成真实机器人控制已解决",
        "evaluate_control_safety_sweep.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"control safety sweep report is missing required terms: {missing}")

    rows = read_csv(args.control_safety_sweep_csv)
    if len(rows) != 9:
        raise RuntimeError(f"control safety sweep csv should have 9 rows, found {len(rows)}")
    required_columns = {
        "version",
        "method",
        "preset",
        "episodes",
        "successes",
        "success_rate",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "mean_contact_count",
        "mean_mean_action_norm",
        "max_action_norm",
        "grasp_successes",
        "out_of_table",
        "stop_reasons",
        "action_alpha",
        "max_arm_delta",
        "max_gripper_delta",
        "steps",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"control safety sweep csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    methods = {row["method"] for row in rows}
    presets = {row["preset"] for row in rows}
    if methods != {"trajectory_conditioned_chunk_bc_v2", "trajectory_knn_chunk_bc_v1", "torch_act_state_chunk_v1"}:
        raise RuntimeError(f"control safety sweep methods differ: {sorted(methods)}")
    if presets != {"current_slow", "slower", "very_slow"}:
        raise RuntimeError(f"control safety sweep presets differ: {sorted(presets)}")
    for row in rows:
        if row["version"] != "control_safety_sweep_v1":
            raise RuntimeError(f"unexpected control sweep version: {row['version']}")
        int(row["episodes"])
        int(row["successes"])
        float(row["success_rate"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        float(row["mean_contact_count"])
        float(row["max_action_norm"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"control safety sweep command is missing no-viewer or full steps: {row['method']} {row['preset']}")
    data = read_json(args.control_safety_sweep_json)
    if data.get("version") != "control_safety_sweep_v1":
        raise RuntimeError("control safety sweep json has unexpected version")
    if len(data.get("rows", [])) != 9:
        raise RuntimeError("control safety sweep json should have 9 rows")
    return rows


def verify_action_head_control_safety_sweep(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.action_head_control_safety_sweep.exists():
        raise FileNotFoundError(args.action_head_control_safety_sweep)
    if not args.action_head_control_safety_sweep_csv.exists():
        raise FileNotFoundError(args.action_head_control_safety_sweep_csv)
    if not args.action_head_control_safety_sweep_json.exists():
        raise FileNotFoundError(args.action_head_control_safety_sweep_json)

    text = args.action_head_control_safety_sweep.read_text(encoding="utf-8-sig")
    required = (
        "Action-head 控制限幅扫表",
        "action_head_control_safety_sweep_v1",
        "action-head/PEFT proxy 的失败是否主要由动作太快导致",
        "object_language_action_head_lite_v1",
        "adapter_action_head_lite_v1",
        "lora_action_head_lite_v1",
        "current_slow",
        "slower",
        "very_slow",
        "极慢控制仍未成功",
        "不是真实 OpenVLA LoRA、RT-2 或真实机械臂验证",
        "evaluate_action_head_control_safety_sweep.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"action-head control safety sweep report is missing required terms: {missing}")

    rows = read_csv(args.action_head_control_safety_sweep_csv)
    if len(rows) != 9:
        raise RuntimeError(f"action-head control safety sweep csv should have 9 rows, found {len(rows)}")
    required_columns = {
        "version",
        "method",
        "preset",
        "episodes",
        "successes",
        "success_rate",
        "mean_target_distance",
        "mean_ee_object_distance",
        "mean_object_z",
        "mean_contact_count",
        "mean_mean_action_norm",
        "max_action_norm",
        "grasp_successes",
        "out_of_table",
        "stop_reasons",
        "action_alpha",
        "max_arm_delta",
        "max_gripper_delta",
        "steps",
        "command",
        "interpretation",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"action-head control safety sweep csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    methods = {row["method"] for row in rows}
    presets = {row["preset"] for row in rows}
    if methods != {"object_language_action_head_lite_v1", "adapter_action_head_lite_v1", "lora_action_head_lite_v1"}:
        raise RuntimeError(f"action-head control safety sweep methods differ: {sorted(methods)}")
    if presets != {"current_slow", "slower", "very_slow"}:
        raise RuntimeError(f"action-head control safety sweep presets differ: {sorted(presets)}")
    for row in rows:
        if row["version"] != "action_head_control_safety_sweep_v1":
            raise RuntimeError(f"unexpected action-head control sweep version: {row['version']}")
        int(row["episodes"])
        int(row["successes"])
        float(row["success_rate"])
        float(row["mean_target_distance"])
        float(row["mean_ee_object_distance"])
        float(row["mean_object_z"])
        float(row["mean_contact_count"])
        float(row["max_action_norm"])
        if "--no-viewer" not in row["command"] or "--steps 2840" not in row["command"]:
            raise RuntimeError(f"action-head control safety sweep command is missing no-viewer or full steps: {row['method']} {row['preset']}")
    data = read_json(args.action_head_control_safety_sweep_json)
    if data.get("version") != "action_head_control_safety_sweep_v1":
        raise RuntimeError("action-head control safety sweep json has unexpected version")
    if len(data.get("rows", [])) != 9:
        raise RuntimeError("action-head control safety sweep json should have 9 rows")
    return rows


def verify_strict_grasp_success_audit(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.strict_grasp_audit.exists():
        raise FileNotFoundError(args.strict_grasp_audit)
    if not args.strict_grasp_audit_csv.exists():
        raise FileNotFoundError(args.strict_grasp_audit_csv)
    if not args.strict_grasp_audit_json.exists():
        raise FileNotFoundError(args.strict_grasp_audit_json)

    text = args.strict_grasp_audit.read_text(encoding="utf-8-sig")
    required = (
        "严格抓取成功口径审计",
        "strict_grasp_success_audit_v1",
        "原始放置成功",
        "严格抓取成功",
        "grasp_success",
        "object_z",
        "14/53",
        "0/53",
        "不能写成稳定抓取成功",
        "docs/stage_reproduction_runbook.md",
        "build_strict_grasp_success_audit.py",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"strict grasp success audit report is missing required terms: {missing}")

    rows = read_csv(args.strict_grasp_audit_csv)
    if len(rows) != 35:
        raise RuntimeError(f"strict grasp audit csv should have 35 rows, found {len(rows)}")
    required_columns = {
        "version",
        "source_version",
        "method",
        "preset_or_seed",
        "episodes",
        "loose_successes",
        "loose_success_rate",
        "strict_grasp_successes",
        "strict_grasp_success_rate",
        "mean_target_distance",
        "mean_object_z",
        "grasp_successes",
        "diagnosis",
        "paper_boundary",
        "evidence_path",
        "reproduction_command",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"strict grasp audit csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    source_versions = {row["source_version"] for row in rows}
    expected_sources = {
        "control_safety_sweep_v1",
        "action_head_control_safety_sweep_v1",
        "candidate_diagnostic_video_index_v1",
    }
    if source_versions != expected_sources:
        raise RuntimeError(f"strict grasp audit sources differ: {sorted(source_versions)}")

    episodes = 0
    loose_successes = 0
    strict_successes = 0
    loose_without_grasp_rows = 0
    for row in rows:
        if row["version"] != "strict_grasp_success_audit_v1":
            raise RuntimeError(f"unexpected strict grasp audit version: {row['version']}")
        episode_count = int(row["episodes"])
        loose = int(row["loose_successes"])
        strict = int(row["strict_grasp_successes"])
        episodes += episode_count
        loose_successes += loose
        strict_successes += strict
        loose_without_grasp_rows += int(loose > 0 and strict == 0)
        float(row["loose_success_rate"])
        float(row["strict_grasp_success_rate"])
        float(row["mean_target_distance"])
        float(row["mean_object_z"])
        if row["evidence_path"].strip() == "":
            raise RuntimeError(f"strict grasp audit row is missing evidence path: {row['method']}")
    if episodes != 53 or loose_successes != 14 or strict_successes != 0 or loose_without_grasp_rows != 13:
        raise RuntimeError(
            "strict grasp audit totals differ from expected completed evidence: "
            f"episodes={episodes}, loose={loose_successes}, strict={strict_successes}, loose_without_grasp={loose_without_grasp_rows}"
        )

    data = read_json(args.strict_grasp_audit_json)
    if data.get("version") != "strict_grasp_success_audit_v1":
        raise RuntimeError("strict grasp audit json has unexpected version")
    summary = data.get("summary", {})
    if int(summary.get("rows", 0)) != 35:
        raise RuntimeError("strict grasp audit json summary row count is wrong")
    if int(summary.get("episodes", 0)) != 53:
        raise RuntimeError("strict grasp audit json summary episode count is wrong")
    if int(summary.get("loose_successes", -1)) != 14:
        raise RuntimeError("strict grasp audit json loose success count is wrong")
    if int(summary.get("strict_grasp_successes", -1)) != 0:
        raise RuntimeError("strict grasp audit json strict success count is wrong")
    if len(data.get("rows", [])) != 35:
        raise RuntimeError("strict grasp audit json should have 35 rows")
    return rows


def verify_action_head_stage_report(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.action_head_stage_report.exists():
        raise FileNotFoundError(args.action_head_stage_report)
    if not args.action_head_stage_csv.exists():
        raise FileNotFoundError(args.action_head_stage_csv)

    text = args.action_head_stage_report.read_text(encoding="utf-8-sig")
    required = (
        "Action-Head / PEFT / CLIP 阶段报告",
        "action_head_stage_report_v1",
        "阶段展示视频",
        "outputs/presentation_clips/04_action_head_peft_proxy.mp4",
        "docs/video_evidence_gallery.html",
        "未形成有效抓取/未抬升",
        "语言/空间泛化失败",
        "data_efficiency_v2",
        "不能写成真实 pretrained VLA 后训练",
        "不能写成 OpenVLA/RT-2",
        "不能写成 pretrained VLA LoRA/Adapter",
        "CLIP 也不能写成机器人 VLA",
        "主任务慢速 Viewer 命令",
        "语言/空间泛化慢速 Viewer 命令",
        "训练/重建命令",
        "object_language_action_head_lite_v1",
        "reward_weighted_action_head_lite_v1",
        "phase_conditioned_action_head_lite_v1",
        "adapter_action_head_lite_v1",
        "lora_action_head_lite_v1",
        "vision_language_action_head_lite_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
        "outputs/videos/object_language_action_head_lite_v1_seed1_success_example.mp4",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"action-head stage report is missing required terms: {missing}")
    if text.count("--viewer") < 8:
        raise RuntimeError("action-head stage report has too few viewer commands")

    rows = read_csv(args.action_head_stage_csv)
    if len(rows) != 8:
        raise RuntimeError(f"action-head stage csv should have 8 rows, found {len(rows)}")
    required_columns = {
        "版本",
        "阶段",
        "方法",
        "结构定位",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "可训练参数",
        "模型大小MB",
        "主任务视频",
        "语言视频",
        "失败模式",
        "论文结论",
        "论文红线",
        "主任务viewer命令",
        "语言viewer命令",
        "训练命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"action-head stage csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_versions = {
        "object_language_action_head_lite_v1",
        "reward_weighted_action_head_lite_v1",
        "phase_conditioned_action_head_lite_v1",
        "adapter_action_head_lite_v1",
        "lora_action_head_lite_v1",
        "vision_language_action_head_lite_v1",
        "clip_action_head_lite_v1",
        "multi_task_object_action_head_lite_v1",
    }
    csv_versions = {row["版本"] for row in rows}
    if csv_versions != required_versions:
        raise RuntimeError(f"action-head stage csv versions differ: {sorted(required_versions - csv_versions)}")
    for row in rows:
        if "--viewer" not in row["主任务viewer命令"]:
            raise RuntimeError(f"action-head stage row has no main viewer command: {row['版本']}")
        if "--viewer" not in row["语言viewer命令"]:
            raise RuntimeError(f"action-head stage row has no language viewer command: {row['版本']}")
        if not (ROOT / row["主任务视频"]).exists():
            raise FileNotFoundError(ROOT / row["主任务视频"])
        if row["语言视频"] and not (ROOT / row["语言视频"]).exists():
            raise FileNotFoundError(ROOT / row["语言视频"])
        int(row["可训练参数"])
        float(row["模型大小MB"])
    return rows


def verify_stage_evidence_index(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.stage_evidence_index.exists():
        raise FileNotFoundError(args.stage_evidence_index)
    if not args.stage_evidence_csv.exists():
        raise FileNotFoundError(args.stage_evidence_csv)

    text = args.stage_evidence_index.read_text(encoding="utf-8-sig")
    required = (
        "阶段证据总表",
        "stage_evidence_index_v1",
        "任务/数据/普通 BC",
        "Trajectory / ACT / Diffusion",
        "Action-Head / PEFT / CLIP",
        "语言/空间泛化",
        "数据效率",
        "MuJoCo domain randomization",
        "最终展示/答辩入口",
        "外部依赖 readiness 门禁",
        "docs/task_bc_stage_report.md",
        "docs/trajectory_act_stage_report.md",
        "docs/action_head_stage_report.md",
        "docs/external_dependency_readiness_audit.md",
        "external_dependency_readiness_audit_v1",
        "formal_method_allowed_now=是 为 0 条",
        "docs/video_evidence_gallery.html",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "defense_video_playlist_v1",
        "candidate_diagnostic_montage_v1",
        "不是策略成功率结果",
        "不能写成真实机器人验证",
        "不能写成 OpenVLA/RT-2",
        "不能写成完整官方 ACT",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"stage evidence index is missing required terms: {missing}")

    rows = read_csv(args.stage_evidence_csv)
    if len(rows) != 8:
        raise RuntimeError(f"stage evidence csv should have 8 rows, found {len(rows)}")
    required_columns = {
        "阶段编号",
        "阶段名称",
        "覆盖数量",
        "关键版本",
        "阶段报告",
        "量化证据",
        "视频证据",
        "展示入口",
        "论文可写结论",
        "论文红线",
        "推荐讲解",
        "重建命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"stage evidence csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_stage_names = {
        "任务/数据/普通 BC",
        "Trajectory / ACT / Diffusion",
        "Action-Head / PEFT / CLIP",
        "语言/空间泛化",
        "数据效率",
        "MuJoCo domain randomization",
        "最终展示/答辩入口",
        "外部依赖 readiness 门禁",
    }
    csv_stage_names = {row["阶段名称"] for row in rows}
    if csv_stage_names != required_stage_names:
        raise RuntimeError(f"stage evidence csv stages differ: {sorted(required_stage_names - csv_stage_names)}")

    for row in rows:
        if "build_" not in row["重建命令"] and "evaluate_" not in row["重建命令"] and "verify_" not in row["重建命令"]:
            raise RuntimeError(f"stage evidence row has no rebuild command: {row['阶段名称']}")
        for field in ("阶段报告", "量化证据", "视频证据", "展示入口"):
            for part in row[field].replace("；", "\n").splitlines():
                candidate = part.strip().strip("`")
                if not candidate.startswith(("docs/", "outputs/")):
                    continue
                path = ROOT / candidate
                if not path.exists():
                    raise FileNotFoundError(path)
    return rows


def verify_stage_showcase_index(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.stage_showcase_index.exists():
        raise FileNotFoundError(args.stage_showcase_index)
    if not args.stage_showcase_html.exists():
        raise FileNotFoundError(args.stage_showcase_html)

    md_text = args.stage_showcase_index.read_text(encoding="utf-8-sig")
    html_text = args.stage_showcase_html.read_text(encoding="utf-8-sig")
    video_rows = read_csv(args.video_evidence_csv)
    expected_video_count = f"{len(video_rows)} 条视频证据"
    required = (
        "阶段展示总索引",
        "stage_showcase_index_v1",
        "版本名称",
        "阶段说明",
        "评测比较",
        "仿真视频片段展示",
        "完整启动命令",
        "docs/reproducible_command_index.md",
        "--viewer --duration 60 --speed 0.05",
        "docs/experiment_dashboard.html",
        "docs/video_evidence_gallery.html",
        "docs/defense_video_playlist.html",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "docs/defense_deck.html",
        "docs/next_experiment_registry.md",
        "docs/external_dependency_readiness_audit.md",
        "external_dependency_readiness_audit_v1",
        "不是策略成功率结果",
        "openvla_dataset_bridge_v1",
        "openvla_feasibility_check_v1",
        "robot_vla_action_head_handoff_v1",
        "不能写成 OpenVLA LoRA、`robot_vla_action_head_lite_v1`、Isaac 或真实 WidowX 验证已经完成",
    )
    missing_md = [item for item in required if item not in md_text]
    if missing_md:
        raise RuntimeError(f"stage showcase markdown is missing required terms: {missing_md}")
    missing_html = [item for item in required[:13] if item not in html_text]
    if missing_html:
        raise RuntimeError(f"stage showcase html is missing required terms: {missing_html}")
    if expected_video_count not in md_text or expected_video_count not in html_text:
        raise RuntimeError(f"stage showcase does not match current video evidence count: {expected_video_count}")
    missing_versions = [version for version in versions if version not in md_text]
    if missing_versions:
        raise RuntimeError(f"stage showcase markdown is missing versions: {missing_versions}")
    if md_text.count("### 阶段 ") < 7:
        raise RuntimeError("stage showcase markdown has too few stage sections")
    if html_text.count('class="stage-section"') < 9:
        raise RuntimeError("stage showcase html has too few stage sections")
    if html_text.count("<video ") < 10:
        raise RuntimeError("stage showcase html has too few embedded videos")

    html_dir = args.stage_showcase_html.parent
    refs = []
    for part in html_text.split('src="')[1:]:
        refs.append(part.split('"', 1)[0])
    for ref in refs:
        path = (html_dir / ref).resolve()
        if not path.exists():
            raise FileNotFoundError(path)


def verify_stage_reproduction_runbook(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.stage_reproduction_runbook.exists():
        raise FileNotFoundError(args.stage_reproduction_runbook)
    if not args.stage_reproduction_csv.exists():
        raise FileNotFoundError(args.stage_reproduction_csv)

    text = args.stage_reproduction_runbook.read_text(encoding="utf-8-sig")
    video_rows = read_csv(args.video_evidence_csv)
    expected_video_count = f"{len(video_rows)} 条视频证据"
    required = (
        "阶段复现实验手册",
        "stage_reproduction_runbook_v1",
        "阶段总览",
        "分阶段复现",
        "Trajectory / ACT / Diffusion",
        "Action-Head / PEFT / CLIP",
        "MuJoCo domain randomization",
        "外部依赖 readiness 门禁",
        "external_dependency_readiness_audit_v1",
        "不是策略成功率结果",
        "video_quality_audit_v1",
        "docs/video_quality_audit.md",
        "defense_video_playlist_v1",
        "docs/defense_video_playlist.html",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "candidate_diagnostic_montage_v1",
        "docs/reproducible_command_index.md",
        "--viewer --duration 60 --speed 0.05",
        "完整单方法命令",
        "视频质量审计不是成功率评测",
        "真实 OpenVLA/机器人 VLA、Isaac domain randomization 和真实 WidowX 验证仍在下一阶段",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"stage reproduction runbook markdown is missing required terms: {missing}")
    if expected_video_count not in text:
        raise RuntimeError(f"stage reproduction runbook does not match current video evidence count: {expected_video_count}")
    if text.count("### 阶段 ") < 8:
        raise RuntimeError("stage reproduction runbook has too few stage sections")
    if text.count("--viewer") < 10:
        raise RuntimeError("stage reproduction runbook has too few representative viewer commands")

    rows = read_csv(args.stage_reproduction_csv)
    if len(rows) != 8:
        raise RuntimeError(f"stage reproduction runbook csv should have 8 rows, found {len(rows)}")
    required_columns = {
        "阶段编号",
        "阶段名称",
        "覆盖数量",
        "关键版本",
        "代表viewer版本",
        "主任务viewer命令数",
        "语言viewer命令数",
        "量化证据",
        "视频证据",
        "展示入口",
        "论文红线",
        "重建命令",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"stage reproduction csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    stage_names = {row["阶段名称"] for row in rows}
    required_stage_names = {
        "任务/数据/普通 BC",
        "Trajectory / ACT / Diffusion",
        "Action-Head / PEFT / CLIP",
        "语言/空间泛化",
        "数据效率",
        "MuJoCo domain randomization",
        "最终展示/答辩入口",
        "外部依赖 readiness 门禁",
    }
    if stage_names != required_stage_names:
        raise RuntimeError(f"stage reproduction csv stages differ: {sorted(required_stage_names - stage_names)}")
    final_rows = [row for row in rows if row["阶段名称"] == "最终展示/答辩入口"]
    if len(final_rows) != 1 or expected_video_count not in final_rows[0]["覆盖数量"]:
        raise RuntimeError(f"stage reproduction final row does not match current video evidence count: {expected_video_count}")
    for row in rows:
        if not row["论文红线"]:
            raise RuntimeError(f"stage reproduction row has empty redline: {row['阶段名称']}")
        if "build_" not in row["重建命令"] and "evaluate_" not in row["重建命令"] and "verify_" not in row["重建命令"]:
            raise RuntimeError(f"stage reproduction row has no rebuild command: {row['阶段名称']}")
    return rows


def verify_research_evidence_map(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.research_evidence_map.exists():
        raise FileNotFoundError(args.research_evidence_map)
    if not args.research_evidence_csv.exists():
        raise FileNotFoundError(args.research_evidence_csv)

    text = args.research_evidence_map.read_text(encoding="utf-8-sig")
    video_rows = read_csv(args.video_evidence_csv)
    expected_video_count = f"{len(video_rows)} 条视频证据"
    required = (
        "研究问题证据映射",
        "research_evidence_map_v1",
        "轻量化后训练是否省算力/参数",
        "轻量化后训练是否省数据",
        "语言/空间泛化是否优于普通 BC",
        "仿真适配后能否迁移到真实机械臂",
        "trajectory-conditioned BC / ACT",
        "docs/stage_comparison_report.md",
        "docs/stage_showcase_index.md",
        "docs/video_presentation_storyboard.md",
        "docs/next_experiment_registry.md",
        "docs/video_evidence_index.md",
        "docs/final_artifact_manifest.json",
        "docs/openvla_dataset_bridge_report.md",
        "docs/robot_vla_action_head_handoff.md",
        "robot_vla_action_head_handoff_v1",
        "docs/robot_vla_remote_run_pack.md",
        "robot_vla_remote_run_pack_v1",
        "docs/robot_vla_remote_result_intake.md",
        "robot_vla_remote_result_intake_v1",
        "openvla_dataset_bridge_v1",
        "domain_randomization_eval_v1",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "docs/domain_randomization_summary.md",
        "docs/stage_showcase_index.html",
        "docs/video_presentation_storyboard.html",
        "OpenVLA",
        "Isaac",
        "真实 WidowX",
        "不能宣称",
        "不能写高保真 Isaac domain randomization 已完成",
        "不能写成完整官方 ACT",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"research evidence map is missing required terms: {missing}")
    if expected_video_count not in text:
        raise RuntimeError(f"research evidence map does not match current video evidence count: {expected_video_count}")

    rows = read_csv(args.research_evidence_csv)
    if len(rows) < 6:
        raise RuntimeError(f"research evidence csv has too few rows: {len(rows)}")
    required_columns = (
        "研究问题",
        "当前状态",
        "证据文件",
        "关键版本/方法",
        "量化摘要",
        "视频/展示入口",
        "可写结论",
        "论文红线",
        "下一步",
    )
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"research evidence csv is missing column: {column}")
    questions = {row["研究问题"] for row in rows}
    required_questions = {
        "轻量化后训练是否省算力/参数？",
        "轻量化后训练是否省数据？",
        "语言/空间泛化是否优于普通 BC？",
        "仿真适配后能否迁移到真实机械臂？",
        "不同阶段和方法能否被统一说明、评测比较和视频展示？",
        "trajectory-conditioned BC / ACT 是否已建立为可靠对照组？",
    }
    if not required_questions.issubset(questions):
        raise RuntimeError(f"research evidence csv is missing questions: {sorted(required_questions - questions)}")
    if not any("未完成" in row["当前状态"] for row in rows):
        raise RuntimeError("research evidence csv does not mark unfinished work")
    if not all(row["论文红线"] for row in rows):
        raise RuntimeError("research evidence csv has empty paper redlines")
    package_rows = [row for row in rows if row["研究问题"] == "不同阶段和方法能否被统一说明、评测比较和视频展示？"]
    if len(package_rows) != 1 or expected_video_count not in package_rows[0]["量化摘要"]:
        raise RuntimeError(f"research evidence package row does not match current video evidence count: {expected_video_count}")
    return rows


def verify_research_question_showcase_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.research_showcase_plan.exists():
        raise FileNotFoundError(args.research_showcase_plan)
    if not args.research_showcase_csv.exists():
        raise FileNotFoundError(args.research_showcase_csv)

    text = args.research_showcase_plan.read_text(encoding="utf-8-sig")
    required = (
        "研究问题展示选择表",
        "research_question_showcase_plan_v1",
        "推荐展示顺序",
        "核心图表",
        "主视频",
        "建议讲稿",
        "论文红线",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/presentation_clips/03_trajectory_act_diffusion.mp4",
        "outputs/presentation_clips/04_action_head_peft_proxy.mp4",
        "outputs/presentation_clips/05_language_generalization.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "docs/method_evidence_gate.md",
        "docs/stage_reproduction_runbook.md",
        "docs/robot_vla_action_head_handoff.md",
        "docs/real_widowx_validation_handoff.md",
        "outputs/real_robot/real_widowx_validation_v1_trial_template.csv",
        "OpenVLA、Isaac、真实 WidowX 未完成的部分必须作为后续工作",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"research question showcase plan is missing required terms: {missing}")

    rows = read_csv(args.research_showcase_csv)
    if len(rows) != 6:
        raise RuntimeError(f"research question showcase csv should have 6 rows, found {len(rows)}")
    required_columns = {
        "展示编号",
        "研究问题",
        "当前状态",
        "推荐展示顺序",
        "核心图表",
        "主视频",
        "辅助入口",
        "建议讲稿",
        "可写结论",
        "论文红线",
        "缺失证据",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"research showcase csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    if any(row["缺失证据"] != "无" for row in rows):
        missing_rows = [f"{row['研究问题']}: {row['缺失证据']}" for row in rows if row["缺失证据"] != "无"]
        raise RuntimeError(f"research showcase has missing evidence: {missing_rows}")
    questions = {row["研究问题"] for row in rows}
    required_questions = {
        "轻量化后训练是否省算力/参数？",
        "轻量化后训练是否省数据？",
        "语言/空间泛化是否优于普通 BC？",
        "仿真适配后能否迁移到真实机械臂？",
        "不同阶段和方法能否被统一说明、评测比较和视频展示？",
        "trajectory-conditioned BC / ACT 是否已建立为可靠对照组？",
    }
    if questions != required_questions:
        raise RuntimeError(f"research showcase questions differ: {sorted(required_questions - questions)}")
    for row in rows:
        if not row["论文红线"] or not row["主视频"]:
            raise RuntimeError(f"research showcase row lacks redline or video: {row['研究问题']}")
    return rows


def verify_claim_evidence_traceability(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.claim_evidence_traceability.exists():
        raise FileNotFoundError(args.claim_evidence_traceability)
    if not args.claim_evidence_csv.exists():
        raise FileNotFoundError(args.claim_evidence_csv)

    text = args.claim_evidence_traceability.read_text(encoding="utf-8-sig")
    video_rows = read_csv(args.video_evidence_csv)
    expected_video_claim = f"当前 {len(video_rows)} 条视频证据"
    required = (
        "Claim 证据追踪矩阵",
        "claim_evidence_traceability_v1",
        "可写 claim",
        "量化证据",
        "视频证据",
        "论文红线",
        "OpenVLA 前置工作",
        "robot_vla_action_head_handoff_v1",
        "docs/robot_vla_action_head_handoff.md",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_remote_result_intake_v1",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "方法与阶段统一追踪",
        "trajectory_conditioned_chunk_bc_v2",
        "clip_action_head_lite_v1",
        "video_quality_audit_v1",
        "不能写成 OpenVLA LoRA、RT-2",
        "不能写成 Isaac domain randomization",
        "视频只作为定性展示证据",
        "真实 OpenVLA、Isaac 和真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"claim evidence traceability markdown is missing required terms: {missing}")
    if expected_video_claim not in text:
        raise RuntimeError(f"claim evidence traceability does not match current video evidence count: {expected_video_claim}")

    rows = read_csv(args.claim_evidence_csv)
    if len(rows) != 10:
        raise RuntimeError(f"claim evidence csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "claim_id",
        "claim_type",
        "usable_claim",
        "primary_versions",
        "quantitative_evidence",
        "video_evidence",
        "display_entry",
        "paper_redline",
        "evidence_status",
        "missing_evidence",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"claim evidence csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    ids = {row["claim_id"] for row in rows}
    if ids != {f"C{index:02d}" for index in range(1, 11)}:
        raise RuntimeError(f"claim evidence ids differ: {sorted(ids)}")
    c08_rows = [row for row in rows if row["claim_id"] == "C08"]
    if len(c08_rows) != 1 or expected_video_claim not in c08_rows[0]["usable_claim"]:
        raise RuntimeError(f"claim C08 does not match current video evidence count: {expected_video_claim}")
    for row in rows:
        if row["evidence_status"] != "可写（有证据）" or row["missing_evidence"] != "无":
            raise RuntimeError(f"claim evidence row has missing evidence: {row['claim_id']} -> {row['missing_evidence']}")
        if not row["paper_redline"] or not row["video_evidence"] or not row["quantitative_evidence"]:
            raise RuntimeError(f"claim evidence row lacks evidence or redline: {row['claim_id']}")
    return rows


def verify_claim_video_playback_index(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.claim_video_playback_index.exists():
        raise FileNotFoundError(args.claim_video_playback_index)
    if not args.claim_video_playback_csv.exists():
        raise FileNotFoundError(args.claim_video_playback_csv)

    text = args.claim_video_playback_index.read_text(encoding="utf-8-sig")
    required = (
        "Claim 视频播放清单",
        "claim_video_playback_index_v1",
        "Start-Process",
        "C01",
        "C10",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "docs/claim_evidence_traceability.csv",
        "docs/robot_vla_action_head_handoff.md",
        "docs/real_widowx_validation_handoff.md",
        "视频是定性证据",
        "真实 OpenVLA、Isaac 和真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"claim video playback markdown is missing required terms: {missing}")

    rows = read_csv(args.claim_video_playback_csv)
    if len(rows) != 10:
        raise RuntimeError(f"claim video playback csv should have 10 rows, found {len(rows)}")
    required_columns = {
        "claim_id",
        "claim_type",
        "primary_video",
        "playback_command",
        "helper_commands",
        "quantitative_reference",
        "talk_prompt",
        "paper_redline",
        "evidence_status",
        "missing_evidence",
    }
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"claim video playback csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    ids = {row["claim_id"] for row in rows}
    if ids != {f"C{index:02d}" for index in range(1, 11)}:
        raise RuntimeError(f"claim video playback ids differ: {sorted(ids)}")
    for row in rows:
        if row["evidence_status"] != "可播放（有证据）" or row["missing_evidence"] != "无":
            raise RuntimeError(f"claim video playback row has missing evidence: {row['claim_id']} -> {row['missing_evidence']}")
        if "Start-Process" not in row["playback_command"]:
            raise RuntimeError(f"claim video playback row lacks Start-Process command: {row['claim_id']}")
        primary = ROOT / row["primary_video"]
        if not primary.exists():
            raise FileNotFoundError(f"claim video playback primary file is missing for {row['claim_id']}: {primary}")
        if not row["talk_prompt"] or not row["paper_redline"] or not row["quantitative_reference"]:
            raise RuntimeError(f"claim video playback row lacks prompt, redline, or quantitative reference: {row['claim_id']}")
    return rows


def verify_goal_completion_audit(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.goal_completion_audit.exists():
        raise FileNotFoundError(args.goal_completion_audit)
    if not args.goal_completion_csv.exists():
        raise FileNotFoundError(args.goal_completion_csv)

    text = args.goal_completion_audit.read_text(encoding="utf-8-sig")
    required = (
        "总目标完成度审计",
        "goal_completion_audit_v1",
        "保留不同方法和阶段的版本名称",
        "version_naming_and_gate_spec_v1",
        "docs/version_naming_and_gate_spec.md",
        "能够按方法和阶段进行说明",
        "能够做评测比较",
        "能够展示仿真视频片段",
        "每次运行具有可视化 viewer 命令",
        "trajectory-conditioned BC / ACT",
        "轻量 VLA 后训练路线有可比较代理实验",
        "当前 MuJoCo 阶段已完成；真实 OpenVLA/机器人 VLA 后训练不属于当前完成条件",
        "外部依赖阶段 readiness 门禁",
        "external_dependency_readiness_audit_v1",
        "docs/external_dependency_readiness_audit.md",
        "waiting_remote_result",
        "waiting_isaac_runtime",
        "waiting_real_robot_trials",
        "formal_method_allowed_now 全部为否",
        "Isaac/domain randomization 与真实机械臂验证",
        "MuJoCo domain randomization 代理",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "MuJoCo-only 正式范围已完成",
        "docs/mujoco_only_scope.md",
        "docs/reproducible_command_index.md",
        "docs/video_evidence_index.md",
        "docs/video_presentation_storyboard.md",
        "docs/trajectory_act_experiment_record.md",
        "docs/defense_live_runbook.md",
        "docs/stage_showcase_index.html",
        "docs/next_experiment_registry.md",
        "docs/final_artifact_manifest.md",
        "defense_evidence_pack_v1",
        "docs/defense_evidence_pack.md",
        "outputs/defense_evidence_pack/defense_evidence_pack_v1.zip",
        "当前 MuJoCo 实验包可用于阶段论文/答辩展示",
        "OpenVLA",
        "真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"goal completion audit is missing required terms: {missing}")

    rows = read_csv(args.goal_completion_csv)
    if len(rows) < 10:
        raise RuntimeError(f"goal completion csv has too few rows: {len(rows)}")
    required_columns = ("目标要求", "当前状态", "权威证据", "当前数量/结果", "缺口或边界", "下一步")
    for column in required_columns:
        if column not in rows[0]:
            raise RuntimeError(f"goal completion csv is missing column: {column}")
    requirements = {row["目标要求"] for row in rows}
    required_requirements = {
        "保留不同方法和阶段的版本名称",
        "能够按方法和阶段进行说明",
        "能够做评测比较",
        "能够展示仿真视频片段",
        "每次运行具有可视化 viewer 命令",
        "trajectory-conditioned BC / ACT 作为可靠对照组",
        "轻量 VLA 后训练路线有可比较代理实验",
        "外部依赖阶段 readiness 门禁",
        "实验记录尽量中文化并可追溯",
        "答辩证据包归档可复制可验证",
        "Isaac/domain randomization 与真实机械臂验证",
        "整体实验完成后可用于论文和答辩展示",
    }
    if not required_requirements.issubset(requirements):
        raise RuntimeError(f"goal completion csv is missing requirements: {sorted(required_requirements - requirements)}")
    if any("未完成" in row["当前状态"] for row in rows):
        raise RuntimeError("goal completion csv marks unfinished work despite the MuJoCo-only scope decision")
    if not any("MuJoCo-only 正式范围已完成" in row["当前状态"] for row in rows):
        raise RuntimeError("goal completion csv does not mark the MuJoCo-only scope as complete")
    defense_pack = read_json(args.defense_evidence_pack_json)
    manifest = read_json(args.artifact_manifest_json)
    manifest_pack_files = int(manifest.get("counts", {}).get("defense_evidence_pack_files", -1))
    pack_files = int(defense_pack.get("file_count", -1))
    if manifest_pack_files != pack_files:
        raise RuntimeError(f"goal completion evidence pack count differs: manifest={manifest_pack_files}, pack={pack_files}")
    pack_rows = [row for row in rows if row["目标要求"] == "答辩证据包归档可复制可验证"]
    if len(pack_rows) != 1:
        raise RuntimeError("goal completion csv should have one defense evidence pack row")
    result = pack_rows[0]["当前数量/结果"]
    if f"包内文件 {pack_files} 个" not in result or f"final manifest 记录文件数 {manifest_pack_files} 个" not in result:
        raise RuntimeError("goal completion audit has stale defense evidence pack counts")
    return rows


def verify_defense_live_runbook(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.defense_live_runbook.exists():
        raise FileNotFoundError(args.defense_live_runbook)
    if not args.defense_live_runbook_csv.exists():
        raise FileNotFoundError(args.defense_live_runbook_csv)

    text = args.defense_live_runbook.read_text(encoding="utf-8-sig")
    video_count = len(read_csv(args.video_evidence_csv))
    candidate_count = len(read_csv(args.candidate_diagnostic_video_csv))
    next_rows = read_csv(args.next_experiment_registry_csv)
    completed_next = sum(
        1 for row in next_rows if row.get("status", row.get("状态", "")).startswith("completed")
    )
    planned_next = len(next_rows) - completed_next
    required = (
        "答辩现场展示 Runbook",
        "defense_live_runbook_v1",
        "当前证据计数",
        "正式方法版本：`25`",
        f"视频证据条目：`{video_count}`",
        f"候选诊断视频：`{candidate_count}`",
        "原始放置成功 `14/53`",
        "严格抓取成功 `0/53`",
        f"completed/prerequisite/diagnostic `{completed_next}`",
        f"planned `{planned_next}`",
        "开场前检查",
        "推荐现场顺序",
        "分步命令",
        "现场必须坚持的边界",
        "最短应急展示",
        "00_defense_video_reel.mp4",
        "03_trajectory_act_diffusion.mp4",
        "07_candidate_diagnostics.mp4",
        "候选诊断总览",
        "trajectory_act_experiment_record.md",
        "strict_grasp_success_audit.md",
        "candidate:grasp_gated_torch_act_state_chunk_v1_candidate",
        "--action viewer --dry-run",
        "candidate:grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "--action viewer",
        "不能写成真实 OpenVLA/RT-2/机器人 VLA 后训练完成",
        "不能写成 Isaac domain randomization 已完成",
        "不能写成真实 WidowX 机械臂验证已完成",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"defense live runbook is missing required terms: {missing}")
    if text.count("Start-Process") < 20:
        raise RuntimeError("defense live runbook has too few Start-Process commands")
    if text.count("showcase_launcher.py") < 6:
        raise RuntimeError("defense live runbook has too few showcase launcher commands")

    rows = read_csv(args.defense_live_runbook_csv)
    if len(rows) != 10:
        raise RuntimeError(f"defense live runbook csv should have 10 rows, found {len(rows)}")
    required_columns = {"顺序", "时间", "环节", "打开内容", "执行命令", "讲解重点", "论文红线"}
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"defense live runbook csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    order = [row["顺序"] for row in rows]
    if order != [str(index) for index in range(10)]:
        raise RuntimeError(f"defense live runbook order is wrong: {order}")
    for row in rows:
        if not row["执行命令"].strip():
            raise RuntimeError(f"defense live runbook row has no command: {row['顺序']}")
        if not row["论文红线"].strip():
            raise RuntimeError(f"defense live runbook row has no paper boundary: {row['顺序']}")
    return rows


def verify_thesis_results_chapter(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.thesis_results_chapter.exists():
        raise FileNotFoundError(args.thesis_results_chapter)
    text = args.thesis_results_chapter.read_text(encoding="utf-8-sig")
    required = (
        "论文结果章节草稿",
        "thesis_results_chapter_draft_v1",
        "5.1 实验设置",
        "5.2 方法分组",
        "5.3 主任务闭环结果",
        "5.4 语言/空间泛化结果",
        "5.5 数据效率结果",
        "5.6 算力与参数效率",
        "5.7 MuJoCo Domain Randomization 代理评测",
        "5.8 可视化证据",
        "5.9 阶段性结论与边界",
        "5.10 后续工作",
        "domain_randomization_eval_v1",
        "docs/domain_randomization_summary.md",
        "docs/video_evidence_gallery.html",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "docs/method_stage_audit.md",
        "真实 pretrained VLA/OpenVLA 后训练",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_remote_result_intake_v1",
        "isaac_domain_randomization_handoff_v1",
        "real_widowx_validation_handoff_v1",
        "strict_grasp_success_audit_v1",
        "docs/strict_grasp_success_audit.md",
        "grasp_success",
        "object_z",
        "0/53",
        "不能写成稳定抓取成功",
        "不能写成完整视觉 ACT",
        "不能写成完整视觉 Diffusion Policy",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"thesis results chapter is missing required terms: {missing}")
    missing_versions = [version for version in versions if version not in text]
    if missing_versions:
        raise RuntimeError(f"thesis results chapter is missing versions: {missing_versions}")


def verify_thesis_appendix_tables(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.thesis_appendix.exists():
        raise FileNotFoundError(args.thesis_appendix)
    if not args.thesis_method_table.exists():
        raise FileNotFoundError(args.thesis_method_table)
    if not args.thesis_domain_table.exists():
        raise FileNotFoundError(args.thesis_domain_table)

    text = args.thesis_appendix.read_text(encoding="utf-8-sig")
    required = (
        "论文附录结果表",
        "thesis_appendix_tables_v1",
        "方法结果总表",
        "Domain Randomization 代理汇总",
        "thesis_method_comparison_table.csv",
        "thesis_domain_randomization_table.csv",
        "docs/video_evidence_gallery.html",
        "domain_randomization_eval_v1",
        "clip_action_head_lite_v1",
        "visual_act_cnn_cvae_v1",
        "MuJoCo 代理评测",
        "不能写成真实 OpenVLA/RT-2 后训练",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"thesis appendix tables doc is missing terms: {missing}")

    method_rows = read_csv(args.thesis_method_table)
    if len(method_rows) < len(versions):
        raise RuntimeError(f"thesis method comparison table has too few rows: {len(method_rows)}")
    required_method_columns = {
        "版本",
        "阶段分组",
        "主任务训练范围",
        "主任务留出范围",
        "语言/空间泛化",
        "可训练参数",
        "固定视频",
        "论文红线",
    }
    if not required_method_columns.issubset(method_rows[0]):
        raise RuntimeError(f"thesis method table is missing columns: {sorted(required_method_columns - set(method_rows[0]))}")
    method_versions = {row["版本"] for row in method_rows}
    missing_versions = [version for version in versions if version not in method_versions]
    if missing_versions:
        raise RuntimeError(f"thesis method comparison table is missing versions: {missing_versions}")
    if not all(row["论文红线"] for row in method_rows):
        raise RuntimeError("thesis method comparison table has empty paper redlines")

    domain_rows = read_csv(args.thesis_domain_table)
    if len(domain_rows) < 9:
        raise RuntimeError(f"thesis domain randomization table has too few rows: {len(domain_rows)}")
    domains = {row["扰动域"] for row in domain_rows}
    methods = {row["方法版本"] for row in domain_rows}
    if not {"nominal", "low_friction_soft_grip", "high_friction_stiff_arm"}.issubset(domains):
        raise RuntimeError(f"thesis domain table is missing domains: {domains}")
    if not {"structured_waypoint_policy_v1", "trajectory_knn_chunk_bc_v1", "visual_act_cnn_cvae_v1"}.issubset(methods):
        raise RuntimeError(f"thesis domain table is missing methods: {methods}")
    for row in domain_rows:
        float(row["成功率数值"])
        float(row["平均目标距离"])


def verify_report(args: argparse.Namespace) -> None:
    if not args.report.exists():
        raise FileNotFoundError(args.report)
    text = args.report.read_text(encoding="utf-8")
    required = ("方法结果总表", "语言/任务泛化评测", "数据效率评测", "Object-Language Action Head-lite")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"report is missing sections/terms: {missing}")


def verify_package(args: argparse.Namespace) -> None:
    if not args.package.exists():
        raise FileNotFoundError(args.package)
    text = args.package.read_text(encoding="utf-8")
    required = ("单任务闭环结果", "语言/空间泛化结果", "模型资源与规模", "数据效率", "MuJoCo Domain Randomization 代理评测", "OpenVLA / 机器人 VLA 数据桥接", "docs\\openvla_dataset_bridge_report.md", "docs\\robot_vla_action_head_handoff.md", "robot_vla_action_head_handoff_v1", "下一阶段实验注册表", "next_experiment_registry_v1", "docs\\next_experiment_registry.md", "docs\\next_experiment_registry.csv", "阶段展示总索引", "docs\\stage_showcase_index.md", "docs\\stage_showcase_index.html", "showcase_launcher_v1", "docs\\showcase_launcher_guide.md", "scripts\\showcase_launcher.py", "stage_reproduction_runbook_v1", "docs\\stage_reproduction_runbook.md", "docs\\stage_reproduction_runbook.csv", "defense_live_runbook_v1", "docs\\defense_live_runbook.md", "docs\\defense_live_runbook.csv", "defense_video_playlist_v1", "docs\\defense_video_playlist.html", "docs\\defense_video_playlist.md", "docs\\defense_video_playlist.csv", "defense_video_cue_sheet_v1", "docs\\defense_video_cue_sheet.md", "docs\\defense_video_cue_sheet.csv", "scripts\\build_defense_video_cue_sheet.py", "--target cue-sheet", "method_evidence_gate_v1", "docs\\method_evidence_gate.md", "docs\\method_evidence_gate.csv", "research_question_showcase_plan_v1", "docs\\research_question_showcase_plan.md", "docs\\research_question_showcase_plan.csv", "claim_evidence_traceability_v1", "docs\\claim_evidence_traceability.md", "docs\\claim_evidence_traceability.csv", "claim_video_playback_index_v1", "docs\\claim_video_playback_index.md", "docs\\claim_video_playback_index.csv", "trajectory_act_experiment_record_v1", "docs\\trajectory_act_experiment_record.md", "docs\\trajectory_act_experiment_record.csv", "trajectory_act_failure_diagnosis_v1", "docs\\trajectory_act_failure_diagnosis.md", "docs\\trajectory_act_failure_diagnosis.csv", "trajectory_act_conclusion_brief_v1", "docs\\trajectory_act_conclusion_brief.md", "docs\\trajectory_act_conclusion_brief.csv", "scripts\\build_trajectory_act_conclusion_brief.py", "final_defense_narrative_script_v1", "docs\\final_defense_narrative_script.md", "docs\\final_defense_narrative_script.csv", "scripts\\build_final_defense_narrative_script.py", "remaining_experiment_execution_board_v1", "docs\\remaining_experiment_execution_board.md", "docs\\remaining_experiment_execution_board.csv", "scripts\\build_remaining_experiment_execution_board.py", "trajectory_phase_template_bc_v1_candidate", "docs\\trajectory_phase_template_bc_report.md", "docs\\trajectory_phase_template_bc_report.csv", "outputs\\evaluations\\trajectory_phase_template_bc_v1.json", "outputs\\trajectory_phase_template_bc\\trajectory_phase_template_bc_20260720_160007.npz", "outputs\\videos\\trajectory_phase_template_bc_v1_candidate_seed1.mp4", "grasp_gated_trajectory_knn_v1_candidate", "docs\\grasp_gated_trajectory_knn_report.md", "docs\\grasp_gated_trajectory_knn_report.csv", "outputs\\evaluations\\grasp_gated_trajectory_knn_v1.json", "outputs\\videos\\grasp_gated_trajectory_knn_v1_candidate_seed0.mp4", "preference_trajectory_post_training_v1_candidate", "docs\\preference_trajectory_post_training_report.md", "docs\\preference_trajectory_post_training_report.csv", "outputs\\evaluations\\preference_trajectory_post_training_v1.json", "outputs\\preference_post_training\\preference_trajectory_post_training_20260720_165005.npz", "outputs\\videos\\preference_trajectory_post_training_v1_candidate_seed0.mp4", "scripts\\train_preference_trajectory_post_training.py", "scripts\\run_preference_trajectory_post_training_policy.py", "scripts\\evaluate_preference_trajectory_post_training.py", "docs\\candidate_diagnostic_video_index.md", "docs\\candidate_diagnostic_video_index.csv", "scripts\\build_candidate_diagnostic_video_index.py", "scripts\\run_grasp_gated_trajectory_knn_policy.py", "scripts\\evaluate_grasp_gated_trajectory_knn.py", "control_safety_sweep_v1", "docs\\control_safety_sweep.md", "docs\\control_safety_sweep.csv", "outputs\\evaluations\\control_safety_sweep_v1.json", "action_head_control_safety_sweep_v1", "docs\\action_head_control_safety_sweep.md", "docs\\action_head_control_safety_sweep.csv", "outputs\\evaluations\\action_head_control_safety_sweep_v1.json", "video_presentation_storyboard_v1", "video_quality_audit_v1", "docs\\video_presentation_storyboard.md", "docs\\video_presentation_storyboard.html", "阶段结果矩阵", "docs/result_matrix.md", "docs\\domain_randomization_summary.md", "docs\\final_artifact_manifest.md", "docs\\research_evidence_map.md", "docs\\goal_completion_audit.md", "docs\\stage_comparison_report.md", "docs\\task_bc_stage_report.md", "docs\\trajectory_act_stage_report.md", "docs\\action_head_stage_report.md", "docs\\stage_evidence_index.md", "docs\\method_stage_audit.md", "docs\\thesis_results_chapter_draft.md", "docs\\thesis_appendix_tables.md", "docs\\defense_slide_outline.md", "docs\\defense_deck.html", "docs\\presentation_video_pack.md", "docs\\video_evidence_index.md", "docs\\video_quality_audit.md", "docs\\video_quality_audit.csv", "docs\\video_evidence_gallery.html", "docs\\failure_mode_taxonomy.md", "复现命令", "后续工作")
    method_comparison_required = (
        "method_comparison_dashboard_v1",
        "docs\\method_comparison_dashboard.html",
        "docs\\method_comparison_dashboard.md",
        "docs\\method_comparison_dashboard.csv",
    )
    visual_index_required = (
        "thesis_visual_evidence_index_v1",
        "docs\\thesis_visual_evidence_index.html",
        "docs\\thesis_visual_evidence_index.md",
        "docs\\thesis_visual_evidence_index.csv",
        "scripts\\build_thesis_visual_evidence_index.py",
    )
    qa_required = (
        "defense_qa_playbook_v1",
        "docs\\defense_qa_playbook.html",
        "docs\\defense_qa_playbook.md",
        "docs\\defense_qa_playbook.csv",
        "scripts\\build_defense_qa_playbook.py",
    )
    lineage_required = (
        "version_lineage_index_v1",
        "docs\\version_lineage_index.html",
        "docs\\version_lineage_index.md",
        "docs\\version_lineage_index.csv",
        "scripts\\build_version_lineage_index.py",
    )
    final_showcase_required = (
        "final_showcase_handoff_v1",
        "docs\\final_showcase_handoff.md",
        "docs\\final_showcase_handoff.csv",
        "scripts\\build_final_showcase_handoff.py",
        "最终展示与交付 Handoff",
    )
    missing = [item for item in required + method_comparison_required + visual_index_required + qa_required + lineage_required + final_showcase_required if item not in text]
    if missing:
        raise RuntimeError(f"package index is missing sections: {missing}")
    trajectory_preference_required = (
        "trajectory_act_slow_viewer_guide_v1",
        "docs\\trajectory_act_slow_viewer_guide.md",
        "docs\\trajectory_act_slow_viewer_guide.csv",
        "scripts\\build_trajectory_act_slow_viewer_guide.py",
        "--target trajectory-act-slow",
        "preference_post_training_ablation_matrix_v1",
        "docs\\preference_post_training_ablation_matrix.md",
        "docs\\preference_post_training_ablation_matrix.csv",
        "scripts\\build_preference_post_training_ablation_matrix.py",
        "--target preference-ablation",
    )
    missing_trajectory_preference = [item for item in trajectory_preference_required if item not in text]
    if missing_trajectory_preference:
        raise RuntimeError(f"package index is missing trajectory/preference terms: {missing_trajectory_preference}")
    grasp_lift_required = (
        "grasp_lift_subpolicy_probe_v1_candidate",
        "docs\\grasp_lift_subpolicy_probe_report.md",
        "docs\\grasp_lift_subpolicy_probe_report.csv",
        "outputs\\evaluations\\grasp_lift_subpolicy_probe_v1_candidate.json",
        "outputs\\videos\\grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "scripts\\evaluate_grasp_lift_subpolicy_probe.py",
        "grasp_lift_subpolicy_probe",
        "contact_stage_subpolicy_v1_candidate",
        "docs\\contact_stage_subpolicy_report.md",
        "docs\\contact_stage_subpolicy_report.csv",
        "outputs\\evaluations\\contact_stage_subpolicy_v1_candidate.json",
        "outputs\\videos\\contact_stage_subpolicy_v1_candidate_seed0.mp4",
        "scripts\\run_contact_stage_subpolicy.py",
        "scripts\\evaluate_contact_stage_subpolicy.py",
        "contact_stage_subpolicy",
        "contact_stage_demo_torch_act_v1_candidate",
        "data\\demos\\contact_stage_demo_place_blue_cube_blue_pad_medium_v1\\metadata.jsonl",
        "data\\demos\\contact_stage_demo_place_blue_cube_blue_pad_medium_v1\\summary.json",
        "outputs\\torch_act\\contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt",
        "docs\\contact_stage_demo_torch_act_report.md",
        "docs\\contact_stage_demo_torch_act_report.csv",
        "outputs\\evaluations\\contact_stage_demo_torch_act_v1_candidate.json",
        "outputs\\videos\\contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
        "scripts\\collect_contact_stage_demos.py",
        "scripts\\evaluate_contact_stage_demo_torch_act.py",
        "contact_stage_phase_action_head_v1_candidate",
        "contact_hold_weighted_torch_act_v1_candidate",
        "outputs\\phase_action_head\\contact_stage_phase_action_head_v1_candidate_20260721_020941.npz",
        "docs\\contact_stage_phase_action_head_report.md",
        "docs\\contact_stage_phase_action_head_report.csv",
        "outputs\\evaluations\\contact_stage_phase_action_head_v1_candidate.json",
        "outputs\\videos\\contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "scripts\\evaluate_contact_stage_phase_action_head.py",
        "outputs\\torch_act\\contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt",
        "docs\\contact_hold_weighted_torch_act_report.md",
        "docs\\contact_hold_weighted_torch_act_report.csv",
        "outputs\\evaluations\\contact_hold_weighted_torch_act_v1_candidate.json",
        "outputs\\videos\\contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
        "scripts\\evaluate_contact_hold_weighted_torch_act.py",
        "gripper_timing_contact_probe_v1_candidate",
        "docs\\gripper_timing_contact_probe_report.md",
        "docs\\gripper_timing_contact_probe_report.csv",
        "outputs\\evaluations\\gripper_timing_contact_probe_v1_candidate.json",
        "outputs\\videos\\gripper_timing_contact_probe_v1_candidate_seed0.mp4",
        "scripts\\run_gripper_timing_probe.py",
        "scripts\\evaluate_gripper_timing_probe.py",
        "gripper_timing_contact_probe",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "docs\\timing_aware_trajectory_prior_residual_bc_report.md",
        "docs\\timing_aware_trajectory_prior_residual_bc_report.csv",
        "outputs\\evaluations\\timing_aware_trajectory_prior_residual_bc_v1_candidate.json",
        "outputs\\videos\\timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs\\timing_aware_trajectory_prior_residual_bc\\timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz",
        "scripts\\train_timing_aware_trajectory_prior_residual_bc.py",
        "scripts\\run_timing_aware_trajectory_prior_residual_policy.py",
        "scripts\\evaluate_timing_aware_trajectory_prior_residual_bc.py",
        "residual-scale=0.02",
        "train-range 放置 5/5",
        "held-out 放置 4/5",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "outputs\\preference_post_training\\preference_ranked_trajectory_post_training_20260721_031024.npz",
        "docs\\preference_ranked_trajectory_post_training_report.md",
        "docs\\preference_ranked_trajectory_post_training_report.csv",
        "outputs\\evaluations\\preference_ranked_trajectory_post_training_v1_candidate.json",
        "outputs\\videos\\preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "scripts\\evaluate_preference_ranked_trajectory_post_training.py",
        "contact_stage_demo_v1",
        "tcp_grasp_lift_success=10/10",
        "tcp_grasp_lift_success=0/10",
        "tcp_grasp_lift_success=True",
        "不能写成 learned BC/ACT/VLA baseline 成功",
    )
    missing_grasp_lift = [item for item in grasp_lift_required if item not in text]
    if missing_grasp_lift:
        raise RuntimeError(f"package index is missing Grasp/Lift probe terms: {missing_grasp_lift}")
    strict_grasp_required = (
        "strict_grasp_success_audit_v1",
        "docs\\strict_grasp_success_audit.md",
        "docs\\strict_grasp_success_audit.csv",
        "outputs\\evaluations\\strict_grasp_success_audit_v1.json",
        "scripts\\build_strict_grasp_success_audit.py",
        "grasp_success",
        "object_z",
        "不能写成稳定抓取成功",
    )
    missing_strict_grasp = [item for item in strict_grasp_required if item not in text]
    if missing_strict_grasp:
        raise RuntimeError(f"package index is missing strict grasp audit terms: {missing_strict_grasp}")
    final_method_required = (
        "final_method_version_index_v1",
        "docs\\final_method_version_index.md",
        "docs\\final_method_version_index.csv",
        "最终方法版本索引",
    )
    missing_final_method = [item for item in final_method_required if item not in text]
    if missing_final_method:
        raise RuntimeError(f"package index is missing final method index terms: {missing_final_method}")
    remote_pack_required = (
        "robot_vla_remote_run_pack_v1",
        "docs\\robot_vla_remote_run_pack.md",
        "outputs\\evaluations\\robot_vla_remote_run_pack_v1.json",
        "outputs\\robot_vla_remote_run_pack\\robot_vla_remote_run_pack_v1.zip",
        "robot_vla_remote_result_intake_v1",
        "docs\\robot_vla_remote_result_intake.md",
        "docs\\robot_vla_remote_result_intake.csv",
        "outputs\\evaluations\\robot_vla_remote_result_intake_v1.json",
    )
    missing_remote_pack = [item for item in remote_pack_required if item not in text]
    if missing_remote_pack:
        raise RuntimeError(f"package index is missing Robot VLA remote run pack terms: {missing_remote_pack}")
    isaac_handoff_required = (
        "isaac_domain_randomization_handoff_v1",
        "docs\\isaac_domain_randomization_handoff.md",
        "docs\\isaac_domain_randomization_handoff.csv",
        "outputs\\evaluations\\isaac_domain_randomization_handoff_v1.json",
        "只能写成 Isaac domain randomization 运行交接门禁",
    )
    missing_isaac_handoff = [item for item in isaac_handoff_required if item not in text]
    if missing_isaac_handoff:
        raise RuntimeError(f"package index is missing Isaac domain randomization handoff terms: {missing_isaac_handoff}")
    real_widowx_handoff_required = (
        "real_widowx_validation_handoff_v1",
        "docs\\real_widowx_validation_handoff.md",
        "docs\\real_widowx_validation_handoff.csv",
        "outputs\\evaluations\\real_widowx_validation_handoff_v1.json",
        "outputs\\real_robot\\real_widowx_validation_v1_trial_template.csv",
        "只能写成真实 WidowX 验证协议和 trial 模板",
    )
    missing_real_widowx_handoff = [item for item in real_widowx_handoff_required if item not in text]
    if missing_real_widowx_handoff:
        raise RuntimeError(f"package index is missing real WidowX validation handoff terms: {missing_real_widowx_handoff}")
    external_dependency_required = (
        "external_dependency_readiness_audit_v1",
        "docs\\external_dependency_readiness_audit.md",
        "docs\\external_dependency_readiness_audit.csv",
        "outputs\\evaluations\\external_dependency_readiness_audit_v1.json",
        "scripts\\build_external_dependency_readiness_audit.py",
        "它不是策略成功率结果",
    )
    missing_external_dependency = [item for item in external_dependency_required if item not in text]
    if missing_external_dependency:
        raise RuntimeError(f"package index is missing external dependency readiness terms: {missing_external_dependency}")
    defense_evidence_pack_required = (
        "defense_evidence_pack_v1",
        "docs\\defense_evidence_pack.md",
        "outputs\\defense_evidence_pack\\defense_evidence_pack_v1",
        "outputs\\defense_evidence_pack\\defense_evidence_pack_v1.zip",
        "outputs\\evaluations\\defense_evidence_pack_v1.json",
        "scripts\\build_defense_evidence_pack.py",
    )
    missing_defense_evidence_pack = [item for item in defense_evidence_pack_required if item not in text]
    if missing_defense_evidence_pack:
        raise RuntimeError(f"package index is missing defense evidence pack terms: {missing_defense_evidence_pack}")


def verify_final_showcase_handoff(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.final_showcase_handoff.exists():
        raise FileNotFoundError(args.final_showcase_handoff)
    if not args.final_showcase_handoff_csv.exists():
        raise FileNotFoundError(args.final_showcase_handoff_csv)

    manifest = read_json(args.artifact_manifest_json)
    counts = manifest.get("counts", {})
    text = args.final_showcase_handoff.read_text(encoding="utf-8-sig")
    required = (
        "最终展示与交付 Handoff 索引",
        "final_showcase_handoff_v1",
        "版本名称",
        "方法/阶段说明",
        "评测比较",
        "仿真视频片段展示",
        "可视化运行",
        "外部依赖",
        "docs/final_experiment_package.md",
        "docs/version_naming_and_gate_spec.md",
        "docs/final_method_version_index.md",
        "docs/stage_showcase_index.html",
        "docs/method_comparison_dashboard.html",
        "docs/defense_video_playlist.html",
        "docs/defense_video_cue_sheet.md",
        "docs/showcase_launcher_guide.md",
        "docs/external_dependency_readiness_audit.md",
        "docs/defense_evidence_pack.md",
        "--target handoff",
        "--target comparison",
        "--target playlist",
        "--action viewer",
        "真实 OpenVLA、Isaac 和真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"final showcase handoff is missing required terms: {missing}")
    expected_count_terms = {
        "正式方法版本": counts.get("methods", 0),
        "阶段展示行": counts.get("stage_evidence_rows", 0),
        "视频证据": counts.get("video_evidence_rows", 0),
        "答辩视频包项目": counts.get("presentation_pack_items", 0),
        "总目标审计行": counts.get("goal_completion_rows", 0),
        "证据包文件": counts.get("defense_evidence_pack_files", 0),
    }
    for label, value in expected_count_terms.items():
        if f"| {label} | {int(value)} |" not in text:
            raise RuntimeError(f"final showcase handoff has stale count for {label}: expected {value}")

    rows = read_csv(args.final_showcase_handoff_csv)
    if len(rows) != 10:
        raise RuntimeError(f"final showcase handoff csv should have 10 rows, found {len(rows)}")
    required_columns = {"目标需求", "首选入口", "辅助入口", "打开命令", "用途", "论文边界"}
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"final showcase handoff csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    requirements = {row["目标需求"] for row in rows}
    required_requirements = {
        "总入口和当前边界",
        "保留各部分版本名称",
        "逐方法说明",
        "按阶段说明",
        "横向评测比较",
        "仿真视频片段展示",
        "每次运行可视化",
        "论文写作和答辩讲稿",
        "后续 OpenVLA / Isaac / 真实 WidowX",
        "交付归档和复验",
    }
    if requirements != required_requirements:
        raise RuntimeError(f"final showcase handoff requirements differ: {sorted(requirements)}")
    by_requirement = {row["目标需求"]: row for row in rows}
    expected_usage_terms = {
        "逐方法说明": f"{int(counts.get('methods', 0))} 个正式方法",
        "按阶段说明": f"{int(counts.get('stage_evidence_rows', 0))} 个阶段",
        "仿真视频片段展示": f"{int(counts.get('video_evidence_rows', 0))} 条视频证据",
    }
    for requirement, term in expected_usage_terms.items():
        if term not in by_requirement[requirement]["用途"]:
            raise RuntimeError(f"final showcase handoff csv has stale usage count for {requirement}: expected {term}")
    narrative = by_requirement["论文写作和答辩讲稿"]
    if "docs/final_defense_narrative_script.md" not in narrative["首选入口"] or "--target narrative-script" not in narrative["打开命令"]:
        raise RuntimeError("final showcase handoff does not route writing/defense script to the final narrative script")
    remaining = by_requirement["后续 OpenVLA / Isaac / 真实 WidowX"]
    if "docs/remaining_experiment_execution_board.md" not in remaining["首选入口"] or "--target remaining-board" not in remaining["打开命令"]:
        raise RuntimeError("final showcase handoff does not route remaining external experiments to the execution board")
    for row in rows:
        if not row["打开命令"].strip() or not row["论文边界"].strip():
            raise RuntimeError(f"final showcase handoff row lacks command or boundary: {row['目标需求']}")
    return rows


def verify_final_artifact_manifest(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.artifact_manifest.exists():
        raise FileNotFoundError(args.artifact_manifest)
    if not args.artifact_manifest_json.exists():
        raise FileNotFoundError(args.artifact_manifest_json)

    data = read_json(args.artifact_manifest_json)
    if data.get("version") != "final_artifact_manifest_v1":
        raise RuntimeError("final artifact manifest has unexpected version")
    counts = data.get("counts", {})
    if int(counts.get("methods", 0)) < len(versions):
        raise RuntimeError("final artifact manifest method count is too small")
    if int(counts.get("stage_groups", 0)) < 12:
        raise RuntimeError("final artifact manifest stage group count is too small")
    if int(counts.get("method_evidence_rows", 0)) < len(versions):
        raise RuntimeError("final artifact manifest method evidence count is too small")
    if int(counts.get("final_method_index_rows", 0)) < len(versions):
        raise RuntimeError("final artifact manifest final method index count is too small")
    if int(counts.get("method_comparison_rows", 0)) < len(versions):
        raise RuntimeError("final artifact manifest method comparison count is too small")
    if int(counts.get("core_task_comparison_rows", 0)) != 24:
        raise RuntimeError("final artifact manifest core task comparison row count is wrong")
    if int(counts.get("core_v2_holdout_comparison_rows", 0)) != 28:
        raise RuntimeError("final artifact manifest core v2 comparison row count is wrong")
    if int(counts.get("thesis_visual_evidence_rows", 0)) < 22:
        raise RuntimeError("final artifact manifest thesis visual evidence count is too small")
    if int(counts.get("defense_qa_rows", 0)) < 14:
        raise RuntimeError("final artifact manifest defense Q&A row count is too small")
    if int(counts.get("version_lineage_rows", 0)) < 41:
        raise RuntimeError("final artifact manifest version lineage row count is too small")
    if int(counts.get("task_bc_stage_rows", 0)) < 6:
        raise RuntimeError("final artifact manifest task/BC stage count is too small")
    if int(counts.get("research_questions", 0)) < 6:
        raise RuntimeError("final artifact manifest research question count is too small")
    if int(counts.get("research_showcase_rows", 0)) < 6:
        raise RuntimeError("final artifact manifest research showcase count is too small")
    if int(counts.get("claim_evidence_rows", 0)) < 10:
        raise RuntimeError("final artifact manifest claim evidence count is too small")
    if int(counts.get("claim_video_playback_rows", 0)) < 10:
        raise RuntimeError("final artifact manifest claim video playback count is too small")
    if int(counts.get("goal_completion_rows", 0)) < 10:
        raise RuntimeError("final artifact manifest goal completion row count is too small")
    if int(counts.get("defense_live_runbook_rows", 0)) < 10:
        raise RuntimeError("final artifact manifest defense live runbook row count is too small")
    if int(counts.get("defense_video_playlist_rows", 0)) < 15:
        raise RuntimeError("final artifact manifest defense video playlist row count is too small")
    if int(counts.get("defense_video_cue_sheet_rows", 0)) < int(counts.get("defense_video_playlist_rows", 0)):
        raise RuntimeError("final artifact manifest defense video cue sheet row count is too small")
    if int(counts.get("final_defense_narrative_rows", 0)) != 10:
        raise RuntimeError("final artifact manifest final defense narrative row count is wrong")
    if int(counts.get("remaining_experiment_board_rows", 0)) != 6:
        raise RuntimeError("final artifact manifest remaining experiment board row count is wrong")
    if int(counts.get("domain_randomization_rows", 0)) < 18:
        raise RuntimeError("final artifact manifest domain randomization count is too small")
    if int(counts.get("isaac_handoff_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest Isaac handoff is missing")
    if int(counts.get("isaac_handoff_rows", 0)) < 24:
        raise RuntimeError("final artifact manifest Isaac handoff row count is too small")
    if int(counts.get("isaac_handoff_required_files", 0)) < 6:
        raise RuntimeError("final artifact manifest Isaac handoff required file count is too small")
    if int(counts.get("real_widowx_handoff_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest real WidowX handoff is missing")
    if int(counts.get("real_widowx_handoff_rows", 0)) < 45:
        raise RuntimeError("final artifact manifest real WidowX handoff row count is too small")
    if int(counts.get("real_widowx_trial_template_rows", 0)) < 50:
        raise RuntimeError("final artifact manifest real WidowX trial template is too small")
    if int(counts.get("real_widowx_required_files", 0)) < 7:
        raise RuntimeError("final artifact manifest real WidowX required file count is too small")
    if int(counts.get("trajectory_act_stage_rows", 0)) < 11:
        raise RuntimeError("final artifact manifest trajectory/ACT stage count is too small")
    if int(counts.get("trajectory_act_record_rows", 0)) < 9:
        raise RuntimeError("final artifact manifest trajectory/ACT experiment record count is too small")
    if int(counts.get("trajectory_act_diagnosis_rows", 0)) < 11:
        raise RuntimeError("final artifact manifest trajectory/ACT diagnosis count is too small")
    if int(counts.get("trajectory_act_conclusion_rows", 0)) < 11:
        raise RuntimeError("final artifact manifest trajectory/ACT conclusion brief count is too small")
    if int(counts.get("trajectory_act_slow_viewer_rows", 0)) != 5:
        raise RuntimeError("final artifact manifest trajectory/ACT slow viewer row count is wrong")
    if int(counts.get("control_safety_sweep_rows", 0)) < 9:
        raise RuntimeError("final artifact manifest control safety sweep count is too small")
    if int(counts.get("action_head_stage_rows", 0)) < 8:
        raise RuntimeError("final artifact manifest action-head stage count is too small")
    if int(counts.get("action_head_control_safety_sweep_rows", 0)) < 9:
        raise RuntimeError("final artifact manifest action-head control safety sweep count is too small")
    if int(counts.get("strict_grasp_audit_rows", 0)) < 34:
        raise RuntimeError("final artifact manifest strict grasp audit row count is too small")
    if int(counts.get("strict_grasp_audit_loose_successes", -1)) != 14:
        raise RuntimeError("final artifact manifest strict grasp audit loose success count is wrong")
    if int(counts.get("strict_grasp_audit_strict_successes", -1)) != 0:
        raise RuntimeError("final artifact manifest strict grasp audit strict success count is wrong")
    if int(counts.get("strict_grasp_audit_loose_without_grasp_rows", 0)) != 13:
        raise RuntimeError("final artifact manifest strict grasp audit loose-without-grasp count is too small")
    if int(counts.get("stage_evidence_rows", 0)) < 8:
        raise RuntimeError("final artifact manifest stage evidence count is too small")
    if int(counts.get("video_evidence_rows", 0)) < 42:
        raise RuntimeError("final artifact manifest video evidence count is too small")
    if int(counts.get("video_quality_rows", 0)) < 42:
        raise RuntimeError("final artifact manifest video quality count is too small")
    if int(counts.get("failure_mode_rows", 0)) < 42:
        raise RuntimeError("final artifact manifest failure mode count is too small")
    if int(counts.get("presentation_pack_items", 0)) < 8:
        raise RuntimeError("final artifact manifest presentation pack count is too small")
    if int(counts.get("defense_evidence_pack_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest defense evidence pack is missing")
    if int(counts.get("defense_evidence_pack_files", 0)) < 300:
        raise RuntimeError("final artifact manifest defense evidence pack file count is too small")
    if int(counts.get("defense_evidence_pack_archive_bytes", 0)) <= 0:
        raise RuntimeError("final artifact manifest defense evidence pack archive size is empty")
    if int(counts.get("openvla_bridge_samples", 0)) < 60:
        raise RuntimeError("final artifact manifest OpenVLA bridge sample count is too small")
    if int(counts.get("openvla_bridge_gallery", 0)) < 1:
        raise RuntimeError("final artifact manifest OpenVLA bridge gallery is missing")
    if int(counts.get("rlds_source_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest RLDS source is missing")
    if int(counts.get("rlds_source_episodes", 0)) != 79 or int(counts.get("rlds_source_steps", 0)) != 2528:
        raise RuntimeError("final artifact manifest RLDS source counts are wrong")
    if int(counts.get("rlds_source_validation_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest RLDS source validation is missing")
    if int(counts.get("openvla_feasibility_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest OpenVLA feasibility check is missing")
    if int(counts.get("robot_vla_handoff_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest Robot VLA handoff is missing")
    if int(counts.get("robot_vla_remote_pack_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest Robot VLA remote run pack is missing")
    if int(counts.get("robot_vla_remote_pack_files", 0)) < 80:
        raise RuntimeError("final artifact manifest Robot VLA remote run pack file count is too small")
    if int(counts.get("robot_vla_remote_intake_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest Robot VLA remote result intake is missing")
    if int(counts.get("preference_post_training_upgrade_gate_rows", 0)) != 5:
        raise RuntimeError("final artifact manifest preference upgrade gate row count is wrong")
    if int(counts.get("preference_post_training_ablation_rows", 0)) != 5:
        raise RuntimeError("final artifact manifest preference ablation row count is wrong")
    if int(counts.get("preference_ranked_objective_rows", 0)) != 2:
        raise RuntimeError("final artifact manifest ranked-objective preference row count is wrong")
    if int(counts.get("preference_ranked_fast_rows", 0)) != 2:
        raise RuntimeError("final artifact manifest ranked-fast preference row count is wrong")
    if int(counts.get("timing_aware_trajectory_prior_residual_rows", 0)) != 10:
        raise RuntimeError("final artifact manifest timing-aware trajectory-prior residual row count is wrong")
    if int(counts.get("preference_post_training_formal_upgrade_allowed", -1)) != 0:
        raise RuntimeError("final artifact manifest preference upgrade gate should not allow formal upgrade")
    if int(counts.get("external_dependency_readiness_checked", 0)) < 1:
        raise RuntimeError("final artifact manifest external dependency readiness audit is missing")
    if int(counts.get("external_dependency_readiness_rows", 0)) < 15:
        raise RuntimeError("final artifact manifest external dependency readiness row count is too small")
    if int(counts.get("external_dependency_waiting_remote_result", 0)) < 1:
        raise RuntimeError("final artifact manifest external dependency remote-result wait count is missing")
    if int(counts.get("external_dependency_waiting_isaac_runtime", 0)) < 1:
        raise RuntimeError("final artifact manifest external dependency Isaac-runtime wait count is missing")
    if int(counts.get("external_dependency_waiting_real_robot_trials", 0)) < 1:
        raise RuntimeError("final artifact manifest external dependency real-robot wait count is missing")

    manifest_versions = {method.get("version") for method in data.get("methods", [])}
    missing_versions = [version for version in versions if version not in manifest_versions]
    if missing_versions:
        raise RuntimeError(f"final artifact manifest is missing versions: {missing_versions}")

    required_paths = {
        "docs/final_experiment_package.md",
        "docs/final_showcase_handoff.md",
        "docs/final_showcase_handoff.csv",
        "scripts/build_final_showcase_handoff.py",
        "docs/final_defense_narrative_script.md",
        "docs/final_defense_narrative_script.csv",
        "scripts/build_final_defense_narrative_script.py",
        "docs/remaining_experiment_execution_board.md",
        "docs/remaining_experiment_execution_board.csv",
        "scripts/build_remaining_experiment_execution_board.py",
        "docs/reproducible_command_index.md",
        "docs/domain_randomization_summary.md",
        "docs/domain_randomization_summary.csv",
        "outputs/evaluations/domain_randomization_eval_v1.json",
        "docs/isaac_domain_randomization_handoff.md",
        "docs/isaac_domain_randomization_handoff.csv",
        "outputs/evaluations/isaac_domain_randomization_handoff_v1.json",
        "scripts/build_isaac_domain_randomization_handoff.py",
        "docs/real_widowx_validation_handoff.md",
        "docs/real_widowx_validation_handoff.csv",
        "outputs/evaluations/real_widowx_validation_handoff_v1.json",
        "outputs/real_robot/real_widowx_validation_v1_trial_template.csv",
        "scripts/build_real_widowx_validation_handoff.py",
        "docs/stage_comparison_report.md",
        "docs/method_evidence_gate.md",
        "docs/method_evidence_gate.csv",
        "docs/method_comparison_dashboard.md",
        "docs/method_comparison_dashboard.csv",
        "docs/method_comparison_dashboard.html",
        "scripts/build_method_comparison_dashboard.py",
        "docs/core_task_comparison_matrix.md",
        "docs/core_task_comparison_matrix.csv",
        "outputs/evaluations/core_task_comparison_matrix_v1.json",
        "scripts/build_core_task_comparison_matrix.py",
        "docs/core_v2_holdout_comparison_matrix.md",
        "docs/core_v2_holdout_comparison_matrix.csv",
        "outputs/evaluations/core_v2_holdout_comparison_matrix_v1.json",
        "scripts/build_core_v2_comparison_matrix.py",
        "scripts/create_demo_subset.py",
        "docs/core_task_blue_cube_blue_pad.csv",
        "outputs/evaluations/core_task_blue_cube_blue_pad.json",
        "docs/core_task_blue_cube_red_pad.csv",
        "outputs/evaluations/core_task_blue_cube_red_pad.json",
        "docs/core_task_red_cube_red_pad.csv",
        "outputs/evaluations/core_task_red_cube_red_pad.json",
        "docs/core_task_leftmost_to_bowl.csv",
        "outputs/evaluations/core_task_leftmost_to_bowl.json",
        "docs/thesis_visual_evidence_index.md",
        "docs/thesis_visual_evidence_index.csv",
        "docs/thesis_visual_evidence_index.html",
        "scripts/build_thesis_visual_evidence_index.py",
        "docs/defense_qa_playbook.md",
        "docs/defense_qa_playbook.csv",
        "docs/defense_qa_playbook.html",
        "scripts/build_defense_qa_playbook.py",
        "docs/version_lineage_index.md",
        "docs/version_lineage_index.csv",
        "docs/version_lineage_index.html",
        "scripts/build_version_lineage_index.py",
        "docs/task_bc_stage_report.md",
        "docs/task_bc_stage_report.csv",
        "docs/trajectory_act_stage_report.md",
        "docs/trajectory_act_stage_report.csv",
        "docs/trajectory_act_experiment_record.md",
        "docs/trajectory_act_experiment_record.csv",
        "scripts/build_trajectory_act_experiment_record.py",
        "docs/trajectory_act_failure_diagnosis.md",
        "docs/trajectory_act_failure_diagnosis.csv",
        "docs/trajectory_act_conclusion_brief.md",
        "docs/trajectory_act_conclusion_brief.csv",
        "scripts/build_trajectory_act_conclusion_brief.py",
        "docs/trajectory_act_slow_viewer_guide.md",
        "docs/trajectory_act_slow_viewer_guide.csv",
        "scripts/build_trajectory_act_slow_viewer_guide.py",
        "outputs/torch_act/phase_weighted_torch_act_v1_candidate_20260720_225108.pt",
        "docs/phase_weighted_torch_act_report.md",
        "docs/phase_weighted_torch_act_report.csv",
        "outputs/evaluations/phase_weighted_torch_act_v1_candidate.json",
        "scripts/evaluate_phase_weighted_torch_act.py",
        "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.mp4",
        "outputs/torch_act/contact_phase_gated_torch_act_v1_candidate_20260721_003304.pt",
        "docs/contact_phase_gated_torch_act_report.md",
        "docs/contact_phase_gated_torch_act_report.csv",
        "outputs/evaluations/contact_phase_gated_torch_act_v1_candidate.json",
        "scripts/evaluate_contact_phase_gated_torch_act.py",
        "outputs/videos/contact_phase_gated_torch_act_v1_candidate_seed0.mp4",
        "docs/grasp_lift_subpolicy_probe_report.md",
        "docs/grasp_lift_subpolicy_probe_report.csv",
        "outputs/evaluations/grasp_lift_subpolicy_probe_v1_candidate.json",
        "scripts/evaluate_grasp_lift_subpolicy_probe.py",
        "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "docs/contact_stage_subpolicy_report.md",
        "docs/contact_stage_subpolicy_report.csv",
        "outputs/evaluations/contact_stage_subpolicy_v1_candidate.json",
        "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4",
        "scripts/run_contact_stage_subpolicy.py",
        "scripts/evaluate_contact_stage_subpolicy.py",
        "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/metadata.jsonl",
        "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/summary.json",
        "outputs/torch_act/contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt",
        "docs/contact_stage_demo_torch_act_report.md",
        "docs/contact_stage_demo_torch_act_report.csv",
        "outputs/evaluations/contact_stage_demo_torch_act_v1_candidate.json",
        "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.json",
        "scripts/collect_contact_stage_demos.py",
        "scripts/evaluate_contact_stage_demo_torch_act.py",
        "outputs/phase_action_head/contact_stage_phase_action_head_v1_candidate_20260721_020941.npz",
        "docs/contact_stage_phase_action_head_report.md",
        "docs/contact_stage_phase_action_head_report.csv",
        "outputs/evaluations/contact_stage_phase_action_head_v1_candidate.json",
        "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.json",
        "scripts/evaluate_contact_stage_phase_action_head.py",
        "outputs/torch_act/contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt",
        "docs/contact_hold_weighted_torch_act_report.md",
        "docs/contact_hold_weighted_torch_act_report.csv",
        "outputs/evaluations/contact_hold_weighted_torch_act_v1_candidate.json",
        "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.json",
        "scripts/evaluate_contact_hold_weighted_torch_act.py",
        "outputs/trajectory_phase_template_bc/trajectory_phase_template_bc_20260720_160007.npz",
        "docs/trajectory_phase_template_bc_report.md",
        "docs/trajectory_phase_template_bc_report.csv",
        "outputs/evaluations/trajectory_phase_template_bc_v1.json",
        "outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4",
        "outputs/trajectory_prior_residual_bc/trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz",
        "docs/trajectory_prior_residual_bc_report.md",
        "docs/trajectory_prior_residual_bc_report.csv",
        "outputs/evaluations/trajectory_prior_residual_bc_v1_candidate.json",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.json",
        "scripts/trajectory_prior_residual_common.py",
        "scripts/train_trajectory_prior_residual_bc.py",
        "scripts/run_trajectory_prior_residual_policy.py",
        "scripts/evaluate_trajectory_prior_residual_bc.py",
        "outputs/timing_aware_trajectory_prior_residual_bc/timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz",
        "docs/timing_aware_trajectory_prior_residual_bc_report.md",
        "docs/timing_aware_trajectory_prior_residual_bc_report.csv",
        "outputs/evaluations/timing_aware_trajectory_prior_residual_bc_v1_candidate.json",
        "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.json",
        "scripts/timing_aware_trajectory_prior_residual_common.py",
        "scripts/train_timing_aware_trajectory_prior_residual_bc.py",
        "scripts/run_timing_aware_trajectory_prior_residual_policy.py",
        "scripts/evaluate_timing_aware_trajectory_prior_residual_bc.py",
        "docs/grasp_gated_trajectory_knn_report.md",
        "docs/grasp_gated_trajectory_knn_report.csv",
        "outputs/evaluations/grasp_gated_trajectory_knn_v1.json",
        "outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4",
        "outputs/preference_post_training/preference_trajectory_post_training_20260720_165005.npz",
        "docs/preference_trajectory_post_training_report.md",
        "docs/preference_trajectory_post_training_report.csv",
        "outputs/evaluations/preference_trajectory_post_training_v1.json",
        "outputs/videos/preference_trajectory_post_training_v1_candidate_seed0.mp4",
        "scripts/train_preference_trajectory_post_training.py",
        "scripts/run_preference_trajectory_post_training_policy.py",
        "scripts/evaluate_preference_trajectory_post_training.py",
        "docs/preference_trajectory_post_training_v1_ranked_fast_summary.md",
        "docs/preference_trajectory_post_training_v1_ranked_fast_report.md",
        "docs/preference_trajectory_post_training_v1_ranked_fast_report.csv",
        "outputs/evaluations/preference_trajectory_post_training_v1_ranked_fast_candidate.json",
        "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.json",
        "docs/preference_trajectory_post_training_v1_tcp_lift_rank_report.md",
        "docs/preference_trajectory_post_training_v1_tcp_lift_rank_report.csv",
        "outputs/evaluations/preference_trajectory_post_training_v1_tcp_lift_rank_candidate.json",
        "outputs/preference_post_training/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_20260721_090438.npz",
        "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_tcp_lift_rank_candidate_seed0.json",
        "docs/preference_post_training_upgrade_gate.md",
        "docs/preference_post_training_upgrade_gate.csv",
        "outputs/evaluations/preference_post_training_upgrade_gate_v1.json",
        "scripts/build_preference_post_training_upgrade_gate.py",
        "docs/preference_post_training_ablation_matrix.md",
        "docs/preference_post_training_ablation_matrix.csv",
        "scripts/build_preference_post_training_ablation_matrix.py",
        "outputs/preference_post_training/preference_contact_aware_trajectory_post_training_20260721_000449.npz",
        "docs/preference_contact_aware_trajectory_post_training_report.md",
        "docs/preference_contact_aware_trajectory_post_training_report.csv",
        "outputs/evaluations/preference_contact_aware_trajectory_post_training_v1_candidate.json",
        "outputs/videos/preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4",
        "scripts/evaluate_preference_contact_aware_trajectory_post_training.py",
        "outputs/preference_post_training/preference_ranked_trajectory_post_training_20260721_031024.npz",
        "docs/preference_ranked_trajectory_post_training_report.md",
        "docs/preference_ranked_trajectory_post_training_report.csv",
        "outputs/evaluations/preference_ranked_trajectory_post_training_v1_candidate.json",
        "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.json",
        "scripts/evaluate_preference_ranked_trajectory_post_training.py",
        "docs/candidate_diagnostic_video_index.md",
        "docs/candidate_diagnostic_video_index.csv",
        "scripts/build_candidate_diagnostic_video_index.py",
        "scripts/run_grasp_gated_trajectory_knn_policy.py",
        "scripts/evaluate_grasp_gated_trajectory_knn.py",
        "docs/control_safety_sweep.md",
        "docs/control_safety_sweep.csv",
        "outputs/evaluations/control_safety_sweep_v1.json",
        "docs/action_head_stage_report.md",
        "docs/action_head_stage_report.csv",
        "docs/action_head_control_safety_sweep.md",
        "docs/action_head_control_safety_sweep.csv",
        "outputs/evaluations/action_head_control_safety_sweep_v1.json",
        "docs/strict_grasp_success_audit.md",
        "docs/strict_grasp_success_audit.csv",
        "outputs/evaluations/strict_grasp_success_audit_v1.json",
        "scripts/build_strict_grasp_success_audit.py",
        "docs/stage_evidence_index.md",
        "docs/stage_evidence_index.csv",
        "docs/stage_showcase_index.md",
        "docs/stage_showcase_index.html",
        "docs/stage_reproduction_runbook.md",
        "docs/stage_reproduction_runbook.csv",
        "docs/research_evidence_map.md",
        "docs/research_question_showcase_plan.md",
        "docs/research_question_showcase_plan.csv",
        "docs/claim_evidence_traceability.md",
        "docs/claim_evidence_traceability.csv",
        "docs/claim_video_playback_index.md",
        "docs/claim_video_playback_index.csv",
        "docs/defense_live_runbook.md",
        "docs/defense_live_runbook.csv",
        "scripts/build_defense_live_runbook.py",
        "docs/defense_video_playlist.md",
        "docs/defense_video_playlist.csv",
        "docs/defense_video_playlist.html",
        "scripts/build_defense_video_playlist.py",
        "docs/defense_video_cue_sheet.md",
        "docs/defense_video_cue_sheet.csv",
        "scripts/build_defense_video_cue_sheet.py",
        "scripts/showcase_launcher.py",
        "docs/defense_evidence_pack.md",
        "outputs/evaluations/defense_evidence_pack_v1.json",
        "outputs/defense_evidence_pack/defense_evidence_pack_v1.zip",
        "scripts/build_defense_evidence_pack.py",
        "docs/showcase_launcher_guide.md",
        "docs/goal_completion_audit.md",
        "docs/video_evidence_index.md",
        "docs/video_quality_audit.md",
        "docs/video_quality_audit.csv",
        "docs/video_evidence_gallery.html",
        "docs/video_presentation_storyboard.md",
        "docs/video_presentation_storyboard.html",
        "docs/task_bc_stage_report.md",
        "docs/trajectory_act_stage_report.md",
        "docs/trajectory_act_failure_diagnosis.md",
        "docs/trajectory_phase_template_bc_report.md",
        "docs/grasp_gated_trajectory_knn_report.md",
        "docs/preference_post_training_ablation_matrix.md",
        "docs/preference_trajectory_post_training_v1_ranked_objective_summary.md",
        "docs/preference_trajectory_post_training_v1_ranked_objective_report.md",
        "docs/preference_trajectory_post_training_v1_ranked_objective_report.csv",
        "outputs/evaluations/preference_trajectory_post_training_v1_ranked_objective_candidate.json",
        "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.json",
        "docs/control_safety_sweep.md",
        "docs/action_head_stage_report.md",
        "docs/action_head_control_safety_sweep.md",
        "docs/failure_mode_taxonomy.md",
        "docs/failure_mode_taxonomy.csv",
        "docs/thesis_appendix_tables.md",
        "docs/thesis_method_comparison_table.csv",
        "docs/thesis_domain_randomization_table.csv",
        "docs/openvla_dataset_bridge_report.md",
        "outputs/evaluations/openvla_dataset_bridge_v1.json",
        "data/vla_bridge/openvla_dataset_bridge_v1/manifest.json",
        "data/vla_bridge/openvla_dataset_bridge_v1/samples.jsonl",
        "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png",
        "docs/openvla_bridge_gallery.html",
        "docs/widowx_mujoco_rlds_source_v1_report.md",
        "outputs/evaluations/widowx_mujoco_rlds_source_v1.json",
        "docs/widowx_mujoco_rlds_source_validation_v1.md",
        "outputs/evaluations/widowx_mujoco_rlds_source_validation_v1.json",
        "scripts/export_openvla_rlds_source.py",
        "scripts/validate_openvla_rlds_source.py",
        "scripts/remote_openvla/widowx_mujoco_pick_place_dataset_builder.py",
        "docs/openvla_feasibility_report.md",
        "outputs/evaluations/openvla_feasibility_check_v1.json",
        "docs/robot_vla_action_head_handoff.md",
        "outputs/evaluations/robot_vla_action_head_handoff_v1.json",
        "docs/robot_vla_remote_run_pack.md",
        "outputs/evaluations/robot_vla_remote_run_pack_v1.json",
        "outputs/robot_vla_remote_run_pack/robot_vla_remote_run_pack_v1.zip",
        "scripts/build_robot_vla_remote_run_pack.py",
        "docs/robot_vla_remote_result_intake.md",
        "docs/robot_vla_remote_result_intake.csv",
        "outputs/evaluations/robot_vla_remote_result_intake_v1.json",
        "scripts/build_robot_vla_remote_result_intake.py",
        "docs/next_experiment_registry.md",
        "docs/next_experiment_registry.csv",
        "docs/external_dependency_readiness_audit.md",
        "docs/external_dependency_readiness_audit.csv",
        "outputs/evaluations/external_dependency_readiness_audit_v1.json",
        "scripts/build_external_dependency_readiness_audit.py",
        "docs/defense_deck.html",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "outputs/showcase/all_registered_methods_grid.mp4",
    }
    required_paths.update(
        {
            "docs/final_method_version_index.md",
            "docs/final_method_version_index.csv",
        }
    )
    manifest_paths = {
        str(item.get("path"))
        for section in ("core_artifacts", "display_artifacts", "registered_method_videos")
        for item in data.get(section, [])
    }
    missing_paths = sorted(required_paths - manifest_paths)
    if missing_paths:
        raise RuntimeError(f"final artifact manifest is missing paths: {missing_paths}")
    missing_files = [
        str(item.get("path"))
        for section in ("core_artifacts", "display_artifacts", "registered_method_videos")
        for item in data.get(section, [])
        if not item.get("exists")
    ]
    if missing_files:
        raise RuntimeError(f"final artifact manifest reports missing files: {missing_files}")

    text = args.artifact_manifest.read_text(encoding="utf-8-sig")
    required_terms = (
        "最终实验 Artifact Manifest",
        "final_artifact_manifest_v1",
        "覆盖统计",
        "核心交付物",
        "展示入口",
        "方法版本",
        "总体验证命令",
        "clip_action_head_lite_v1",
        "domain_randomization_eval_v1",
        "isaac_domain_randomization_handoff_v1",
        "docs/isaac_domain_randomization_handoff.md",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "docs/video_evidence_gallery.html",
        "docs/method_evidence_gate.md",
        "docs/video_quality_audit.md",
        "docs/stage_showcase_index.html",
        "docs/stage_reproduction_runbook.md",
        "docs/research_question_showcase_plan.md",
        "docs/claim_evidence_traceability.md",
        "docs/claim_video_playback_index.md",
        "docs/showcase_launcher_guide.md",
        "final_defense_narrative_script_v1",
        "docs/final_defense_narrative_script.md",
        "scripts/build_final_defense_narrative_script.py",
        "remaining_experiment_execution_board_v1",
        "docs/remaining_experiment_execution_board.md",
        "scripts/build_remaining_experiment_execution_board.py",
        "trajectory_act_slow_viewer_guide_v1",
        "docs/trajectory_act_slow_viewer_guide.md",
        "preference_post_training_ablation_matrix_v1",
        "docs/preference_post_training_ablation_matrix.md",
        "docs/video_presentation_storyboard.html",
        "docs/failure_mode_taxonomy.md",
        "docs/thesis_appendix_tables.md",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/presentation_clips/07_candidate_diagnostics.mp4",
        "docs/defense_deck.html",
        "docs/stage_showcase_index.md",
        "docs/video_presentation_storyboard.md",
        "method_evidence_gate_v1",
        "stage_reproduction_runbook_v1",
        "trajectory_act_experiment_record_v1",
        "method_comparison_dashboard.md",
        "docs/method_comparison_dashboard.html",
        "thesis_visual_evidence_index_v1",
        "docs/thesis_visual_evidence_index.html",
        "docs/thesis_visual_evidence_index.md",
        "defense_qa_playbook_v1",
        "docs/defense_qa_playbook.html",
        "docs/defense_qa_playbook.md",
        "version_lineage_index_v1",
        "docs/version_lineage_index.html",
        "docs/version_lineage_index.md",
        "docs/trajectory_act_experiment_record.md",
        "trajectory_act_failure_diagnosis_v1",
        "trajectory_act_conclusion_brief_v1",
        "docs/trajectory_act_conclusion_brief.md",
        "trajectory_phase_template_bc_v1_candidate",
        "docs/trajectory_phase_template_bc_report.md",
        "trajectory_prior_residual_bc_v1_candidate",
        "docs/trajectory_prior_residual_bc_report.md",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "grasp_gated_trajectory_knn_v1_candidate",
        "docs/grasp_gated_trajectory_knn_report.md",
        "preference_trajectory_post_training_v1_candidate",
        "docs/preference_trajectory_post_training_report.md",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "docs/grasp_lift_subpolicy_probe_report.md",
        "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "contact_stage_subpolicy_v1_candidate",
        "docs/contact_stage_subpolicy_report.md",
        "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4",
        "contact_stage_demo_torch_act_v1_candidate",
        "docs/contact_stage_demo_torch_act_report.md",
        "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
        "contact_stage_phase_action_head_v1_candidate",
        "docs/contact_stage_phase_action_head_report.md",
        "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "contact_hold_weighted_torch_act_v1_candidate",
        "docs/contact_hold_weighted_torch_act_report.md",
        "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "docs/preference_ranked_trajectory_post_training_report.md",
        "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "docs/candidate_diagnostic_video_index.md",
        "outputs/videos/trajectory_phase_template_bc_v1_candidate_seed1.mp4",
        "outputs/videos/grasp_gated_trajectory_knn_v1_candidate_seed0.mp4",
        "control_safety_sweep_v1",
        "action_head_control_safety_sweep_v1",
        "research_question_showcase_plan_v1",
        "claim_evidence_traceability_v1",
        "claim_video_playback_index_v1",
        "defense_live_runbook_v1",
        "docs/defense_live_runbook.md",
        "defense_video_playlist.md",
        "docs/defense_video_playlist.html",
        "defense_video_cue_sheet_v1",
        "docs/defense_video_cue_sheet.md",
        "showcase_launcher_v1",
        "defense_evidence_pack_v1",
        "docs/defense_evidence_pack.md",
        "outputs/evaluations/defense_evidence_pack_v1.json",
        "outputs/defense_evidence_pack/defense_evidence_pack_v1.zip",
        "scripts/build_defense_evidence_pack.py",
        "video_quality_audit_v1",
        "docs/next_experiment_registry.md",
        "docs/robot_vla_action_head_handoff.md",
        "robot_vla_action_head_handoff_v1",
        "docs/robot_vla_remote_run_pack.md",
        "robot_vla_remote_run_pack_v1",
        "docs/robot_vla_remote_result_intake.md",
        "robot_vla_remote_result_intake_v1",
        "external_dependency_readiness_audit_v1",
        "docs/external_dependency_readiness_audit.md",
        "docs/external_dependency_readiness_audit.csv",
        "outputs/evaluations/external_dependency_readiness_audit_v1.json",
        "scripts/build_external_dependency_readiness_audit.py",
        "docs/research_evidence_map.md",
        "docs/goal_completion_audit.md",
        "verify_experiment_artifacts.py",
    )
    missing_terms = [term for term in required_terms if term not in text]
    if missing_terms:
        raise RuntimeError(f"final artifact manifest markdown is missing terms: {missing_terms}")
    final_method_terms = (
        "final_method_version_index_v1",
        "docs/final_method_version_index.md",
        "docs/final_method_version_index.csv",
    )
    missing_final_method_terms = [term for term in final_method_terms if term not in text]
    if missing_final_method_terms:
        raise RuntimeError(f"final artifact manifest markdown is missing final method index terms: {missing_final_method_terms}")


def verify_dashboard(args: argparse.Namespace) -> None:
    if not args.dashboard.exists():
        raise FileNotFoundError(args.dashboard)
    text = args.dashboard.read_text(encoding="utf-8")
    required = (
        "轻量化 VLA 机械臂实验 Dashboard",
        "result_matrix.md",
        "单任务闭环结果",
        "语言/空间泛化结果",
        "模型资源对比",
        "数据效率对比",
        "Kaggle 冻结 CLIP 适配器同协议对照",
        "kaggle_clip_semantic_adapter_core_v2_v1",
        "51/60",
        "48/60",
        "19/20",
        "frozen_clip_semantic_adapter_same_protocol_comparison.md",
        "实验图表",
        "method_comparison_dashboard_v1",
        "method_comparison_dashboard.html",
        "thesis_visual_evidence_index_v1",
        "thesis_visual_evidence_index.html",
        "defense_qa_playbook_v1",
        "defense_qa_playbook.html",
        "version_lineage_index_v1",
        "version_lineage_index.html",
        "stage_showcase_index_v1",
        "stage_showcase_index.html",
        "video_presentation_storyboard_v1",
        "video_presentation_storyboard.html",
        "OpenVLA 数据桥接与下一阶段",
        "openvla_dataset_bridge_v1",
        "openvla_bridge_gallery.html",
        "openvla_feasibility_check_v1",
        "robot_vla_action_head_handoff_v1",
        "robot_vla_action_head_handoff.md",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_remote_result_intake_v1",
        "isaac_domain_randomization_handoff_v1",
        "real_widowx_validation_handoff_v1",
        "strict_grasp_success_audit_v1",
        "严格抓取成功口径审计",
        "grasp_success",
        "object_z",
        "0/53",
        "next_experiment_registry_v1",
        "next_experiment_registry.md",
        "本页不是策略评测结果",
        "<video ",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"dashboard is missing sections/terms: {missing}")
    if text.count("<video ") < 10:
        raise RuntimeError("dashboard has too few video cards")


def verify_storyboard(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.storyboard.exists():
        raise FileNotFoundError(args.storyboard)
    text = args.storyboard.read_text(encoding="utf-8")
    required = (
        "defense_storyboard_v1",
        "推荐讲解顺序",
        "阶段结果矩阵",
        "论文表述红线",
        "下一阶段实验入口",
        "Vision-Language Action Head-lite",
        "PyTorch State Transformer ACT",
        "PyTorch State ACT-CVAE-lite",
        "Visual-Feature ACT-lite",
        "MuJoCo Domain Randomization 代理评测",
        "docs/domain_randomization_summary.md",
        "docs/robot_vla_action_head_handoff.md",
        "docs/robot_vla_remote_run_pack.md",
        "docs/robot_vla_remote_result_intake.md",
        "docs/isaac_domain_randomization_handoff.md",
        "docs/real_widowx_validation_handoff.md",
        "docs/strict_grasp_success_audit.md",
        "strict_grasp_success_audit_v1",
        "grasp_success",
        "object_z",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"storyboard is missing sections/terms: {missing}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"storyboard is missing versions: {missing_versions}")


def verify_slide_outline(args: argparse.Namespace) -> None:
    if not args.slide_outline.exists():
        raise FileNotFoundError(args.slide_outline)
    text = args.slide_outline.read_text(encoding="utf-8-sig")
    required = (
        "答辩幻灯片大纲",
        "defense_slide_outline_v1",
        "幻灯片总览",
        "页级讲解脚本",
        "讲解红线",
        "Slide 07：语言/空间泛化测试",
        "Slide 08：MuJoCo Domain Randomization 代理评测",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/presentation_clips/05_language_generalization.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/figures/main_task_success.svg",
        "outputs/figures/language_success.svg",
        "outputs/figures/resource_vs_success.svg",
        "outputs/figures/data_efficiency.svg",
        "不能写成完整视觉 ACT",
        "不能写成完整视觉 Diffusion Policy",
        "clip_action_head_lite_v1",
        "docs/openvla_bridge_gallery.html",
        "openvla_dataset_bridge_v1",
        "openvla_feasibility_check_v1",
        "robot_vla_action_head_handoff_v1",
        "docs/robot_vla_action_head_handoff.md",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_remote_result_intake_v1",
        "isaac_domain_randomization_handoff_v1",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "strict_grasp_success_audit_v1",
        "docs/strict_grasp_success_audit.md",
        "grasp_success",
        "object_z",
        "不能写成稳定抓取成功",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"slide outline is missing required terms: {missing}")
    if text.count("### Slide ") < 12:
        raise RuntimeError("slide outline has too few slides")


def verify_defense_deck(args: argparse.Namespace) -> None:
    if not args.defense_deck.exists():
        raise FileNotFoundError(args.defense_deck)
    text = args.defense_deck.read_text(encoding="utf-8-sig")
    required = (
        "轻量化 VLA 机械臂实验答辩 Deck",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/presentation_clips/05_language_generalization.mp4",
        "outputs/presentation_clips/06_domain_randomization_proxy.mp4",
        "outputs/figures/main_task_success.svg",
        "outputs/figures/language_success.svg",
        "outputs/figures/resource_vs_success.svg",
        "outputs/figures/data_efficiency.svg",
        "clip_action_head_lite_v1",
        "torch_diffusion_policy_state_chunk_v1",
        "openvla_dataset_bridge_v1",
        "openvla_feasibility_check_v1",
        "robot_vla_action_head_handoff_v1",
        "docs/robot_vla_action_head_handoff.md",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_remote_result_intake_v1",
        "isaac_domain_randomization_handoff_v1",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "strict_grasp_success_audit_v1",
        "docs/strict_grasp_success_audit.md",
        "grasp_success",
        "object_z",
        "不能写成稳定抓取成功",
        "docs/openvla_bridge_gallery.html",
        "MuJoCo Domain Randomization 代理评测",
        "class=\"slide\"",
        "<video ",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"defense deck is missing required terms: {missing}")
    if text.count("class=\"slide\"") < 12:
        raise RuntimeError("defense deck has too few slides")
    if text.count("<video ") < 5:
        raise RuntimeError("defense deck has too few videos")
    if text.count("<img ") < 4:
        raise RuntimeError("defense deck has too few figures")


def verify_runtime_capability(args: argparse.Namespace) -> None:
    data = read_json(args.runtime_json)
    if data.get("version") != "runtime_capability_v1":
        raise RuntimeError("runtime capability json has unexpected version")
    packages = data.get("packages", {})
    if not packages.get("mujoco", {}).get("available"):
        raise RuntimeError("runtime capability report says mujoco is unavailable")
    if "torch" not in packages:
        raise RuntimeError("runtime capability json is missing torch status")
    torch = packages.get("torch", {})
    if torch.get("cuda_available") and not torch.get("cuda_smoke", {}).get("ok"):
        raise RuntimeError("runtime capability json says CUDA is available but CUDA smoke test failed")
    if not args.runtime_report.exists():
        raise FileNotFoundError(args.runtime_report)
    text = args.runtime_report.read_text(encoding="utf-8")
    required = ("运行环境能力检查", "runtime_capability_v1", "Torch CUDA 可用", "CUDA smoke test", "下一阶段")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"runtime capability report is missing required terms: {missing}")


def verify_next_phase(args: argparse.Namespace) -> None:
    if not args.next_phase.exists():
        raise FileNotFoundError(args.next_phase)
    text = args.next_phase.read_text(encoding="utf-8")
    required = (
        "next_phase_implementation_v1",
        "runtime_cuda_torch_setup_v1",
        "openvla_dataset_bridge_v1",
        "robot_vla_action_head_lite_v1",
        "robot_vla_adapter_lite_v1",
        "robot_vla_lora_lite_v1",
        "visual_act_cnn_cvae_v1",
        "domain_randomization_eval_v1",
        "isaac_domain_randomization_handoff_v1",
        "docs/isaac_domain_randomization_handoff.md",
        "MuJoCo domain randomization 代理",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_handoff_v1",
        "docs/real_widowx_validation_handoff.md",
        "real_widowx_validation_v1",
        "setup_cuda_torch_runtime.ps1",
        "check_cuda_torch_runtime.ps1",
        "D:\\vla_torch_cuda_pkgs",
        "next_experiment_registry_v1",
        "docs\\next_experiment_registry.md",
        "每个新方法必须登记的字段",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"next phase implementation doc is missing required terms: {missing}")
    setup_script = ROOT / "scripts" / "setup_cuda_torch_runtime.ps1"
    if not setup_script.exists():
        raise FileNotFoundError(setup_script)
    check_cuda_script = ROOT / "scripts" / "check_cuda_torch_runtime.ps1"
    if not check_cuda_script.exists():
        raise FileNotFoundError(check_cuda_script)


def verify_next_experiment_registry(args: argparse.Namespace, completed_versions: list[str]) -> list[dict[str, str]]:
    if not args.next_experiment_registry.exists():
        raise FileNotFoundError(args.next_experiment_registry)
    if not args.next_experiment_registry_csv.exists():
        raise FileNotFoundError(args.next_experiment_registry_csv)

    text = args.next_experiment_registry.read_text(encoding="utf-8-sig")
    required = (
        "下一阶段实验注册表",
        "next_experiment_registry_v1",
        "completed_diagnostic",
        "planned_external_dependency",
        "preference_trajectory_post_training_v1_candidate",
        "trajectory_phase_template_bc_v1_candidate",
        "trajectory_prior_residual_bc_v1_candidate",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "gripper_timing_contact_probe_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "grasp_gated_trajectory_act_v1_candidate",
        "phase_weighted_torch_act_v1_candidate",
        "contact_phase_gated_torch_act_v1_candidate",
        "contact_aware_phase_gated_torch_act_v1_candidate",
        "contact_stage_subpolicy_v1_candidate",
        "contact_stage_demo_torch_act_v1_candidate",
        "contact_stage_phase_action_head_v1_candidate",
        "contact_hold_weighted_torch_act_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "contact_aware_trajectory_knn_v1_candidate",
        "trajectory_act_control_diagnostic",
        "trajectory_phase_template_diagnostic",
        "trajectory_prior_residual_diagnostic",
        "timing_aware_trajectory_prior_residual_diagnostic",
        "gripper_timing_contact_probe_diagnostic",
        "trajectory_ranked_objective_preference_diagnostic",
        "trajectory_ranked_fast_preference_diagnostic",
        "trajectory_act_loss_diagnostic",
        "trajectory_act_contact_phase_diagnostic",
        "trajectory_act_contact_geometry_diagnostic",
        "contact_stage_subpolicy_diagnostic",
        "contact_stage_demo_act_diagnostic",
        "contact_stage_phase_action_head_diagnostic",
        "contact_hold_weighted_act_diagnostic",
        "control_upper_bound_diagnostic",
        "trajectory_contact_feature_diagnostic",
        "trajectory_preference_contact_feature_diagnostic",
        "trajectory_ranked_preference_diagnostic",
        "strict_grasp_lift_success=0/10",
        "strict_grasp_lift_success=0/4",
        "tcp_grasp_lift_success=10/10",
        "tcp_grasp_lift_success=5/5",
        "contact_stage_demo_v1",
        "robot_vla_action_head_handoff_v1",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_remote_result_intake_v1",
        "isaac_domain_randomization_handoff_v1",
        "real_widowx_validation_handoff_v1",
        "robot_vla_action_head_lite_v1",
        "robot_vla_adapter_lite_v1",
        "robot_vla_lora_lite_v1",
        "preference_trajectory_post_training_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
        "新方法入包必填字段",
        "正式入包 Gate",
        "失败也必须保存视频和失败模式",
        "计划版本在真正运行前不能写成已完成实验",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"next experiment registry markdown is missing terms: {missing}")

    rows = read_csv(args.next_experiment_registry_csv)
    if len(rows) != 32:
        raise RuntimeError(f"next experiment registry should have 32 rows, found {len(rows)}")
    required_columns = {
        "version",
        "category",
        "status",
        "depends_on",
        "stage_to_register",
        "method_name",
        "primary_artifact",
        "evaluation_required",
        "resource_required",
        "video_outputs",
        "success_gate",
        "paper_boundary",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(f"next experiment registry csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    planned_versions = {
        "robot_vla_action_head_lite_v1",
        "robot_vla_adapter_lite_v1",
        "robot_vla_lora_lite_v1",
        "preference_trajectory_post_training_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
    }
    csv_versions = {row["version"] for row in rows}
    additional_completed_versions = (
        "trajectory_phase_template_bc_v1_candidate",
        "trajectory_prior_residual_bc_v1_candidate",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "gripper_timing_contact_probe_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
    )
    for version in additional_completed_versions:
        if version not in csv_versions:
            raise RuntimeError(f"next experiment registry is missing {version}")
    if "grasp_gated_trajectory_act_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing grasp_gated_trajectory_act_v1_candidate")
    if "phase_weighted_torch_act_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing phase_weighted_torch_act_v1_candidate")
    if "contact_phase_gated_torch_act_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_phase_gated_torch_act_v1_candidate")
    if "contact_aware_phase_gated_torch_act_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_aware_phase_gated_torch_act_v1_candidate")
    if "contact_stage_subpolicy_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_stage_subpolicy_v1_candidate")
    if "contact_stage_demo_torch_act_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_stage_demo_torch_act_v1_candidate")
    if "contact_stage_phase_action_head_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_stage_phase_action_head_v1_candidate")
    if "contact_hold_weighted_torch_act_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_hold_weighted_torch_act_v1_candidate")
    if "grasp_lift_subpolicy_probe_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing grasp_lift_subpolicy_probe_v1_candidate")
    if "contact_aware_trajectory_knn_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing contact_aware_trajectory_knn_v1_candidate")
    if "preference_contact_aware_trajectory_post_training_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing preference_contact_aware_trajectory_post_training_v1_candidate")
    if "preference_ranked_trajectory_post_training_v1_candidate" not in csv_versions:
        raise RuntimeError("next experiment registry is missing preference_ranked_trajectory_post_training_v1_candidate")
    if "robot_vla_action_head_handoff_v1" not in csv_versions:
        raise RuntimeError("next experiment registry is missing robot_vla_action_head_handoff_v1")
    if "robot_vla_remote_run_pack_v1" not in csv_versions:
        raise RuntimeError("next experiment registry is missing robot_vla_remote_run_pack_v1")
    if "robot_vla_remote_result_intake_v1" not in csv_versions:
        raise RuntimeError("next experiment registry is missing robot_vla_remote_result_intake_v1")
    if "isaac_domain_randomization_handoff_v1" not in csv_versions:
        raise RuntimeError("next experiment registry is missing isaac_domain_randomization_handoff_v1")
    if "real_widowx_validation_handoff_v1" not in csv_versions:
        raise RuntimeError("next experiment registry is missing real_widowx_validation_handoff_v1")
    missing_planned = sorted(planned_versions - csv_versions)
    if missing_planned:
        raise RuntimeError(f"next experiment registry is missing planned versions: {missing_planned}")
    planned_in_completed = sorted(planned_versions.intersection(completed_versions))
    if planned_in_completed:
        raise RuntimeError(f"planned next experiment versions are already registered as completed methods: {planned_in_completed}")

    status_values = {row["status"] for row in rows}
    if "completed_prerequisite" not in status_values or "completed_diagnostic" not in status_values or "planned" not in status_values or "planned_external_dependency" not in status_values:
        raise RuntimeError(f"next experiment registry status coverage is incomplete: {sorted(status_values)}")
    for row in rows:
        for field in ("evaluation_required", "resource_required", "video_outputs", "success_gate", "paper_boundary"):
            if not row[field].strip():
                raise RuntimeError(f"next experiment registry row has empty {field}: {row['version']}")
        boundary = row["paper_boundary"]
        if row["status"].startswith("planned") and all(term not in boundary for term in ("不能", "必须", "只有", "才能")):
            raise RuntimeError(f"planned registry row lacks a clear paper boundary: {row['version']}")
    return rows


def verify_external_dependency_readiness_audit(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.external_dependency_readiness_audit.exists():
        raise FileNotFoundError(args.external_dependency_readiness_audit)
    if not args.external_dependency_readiness_csv.exists():
        raise FileNotFoundError(args.external_dependency_readiness_csv)
    if not args.external_dependency_readiness_json.exists():
        raise FileNotFoundError(args.external_dependency_readiness_json)

    text = args.external_dependency_readiness_audit.read_text(encoding="utf-8-sig")
    required_terms = (
        "外部依赖阶段 Readiness Audit",
        "external_dependency_readiness_audit_v1",
        "waiting_remote_result",
        "waiting_robot_vla_action_head",
        "waiting_isaac_runtime",
        "waiting_real_robot_trials",
        "ready_for_local_redesign",
        "真实 OpenVLA",
        "Isaac",
        "真实 WidowX",
        "不是策略成功率结果",
    )
    missing_terms = [term for term in required_terms if term not in text]
    if missing_terms:
        raise RuntimeError(f"external dependency readiness audit is missing terms: {missing_terms}")

    rows = read_csv(args.external_dependency_readiness_csv)
    if len(rows) < 15:
        raise RuntimeError(f"external dependency readiness audit should have at least 15 rows, found {len(rows)}")
    required_columns = {
        "version",
        "category",
        "registry_status",
        "readiness_status",
        "formal_method_allowed_now",
        "blocking_condition",
        "required_next_action",
        "required_return_artifacts",
        "source_evidence",
        "paper_boundary",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError(f"external dependency readiness csv is missing columns: {sorted(required_columns - set(rows[0]))}")

    data = read_json(args.external_dependency_readiness_json)
    if data.get("version") != "external_dependency_readiness_audit_v1":
        raise RuntimeError("external dependency readiness json has unexpected version")
    if int(data.get("row_count", 0)) != len(rows):
        raise RuntimeError("external dependency readiness csv/json row counts differ")

    required_versions = {
        "robot_vla_action_head_lite_v1",
        "robot_vla_adapter_lite_v1",
        "robot_vla_lora_lite_v1",
        "preference_trajectory_post_training_v1",
        "isaac_domain_randomization_v1",
        "real_widowx_validation_v1",
        "robot_vla_remote_result_intake_v1",
        "isaac_domain_randomization_handoff_v1",
        "real_widowx_validation_handoff_v1",
    }
    csv_versions = {row["version"] for row in rows}
    missing_versions = sorted(required_versions - csv_versions)
    if missing_versions:
        raise RuntimeError(f"external dependency readiness audit is missing versions: {missing_versions}")

    required_statuses = {
        "supporting_evidence_ready",
        "waiting_remote_result",
        "waiting_robot_vla_action_head",
        "ready_for_local_redesign",
        "waiting_isaac_runtime",
        "waiting_real_robot_trials",
    }
    statuses = {row["readiness_status"] for row in rows}
    missing_statuses = sorted(required_statuses - statuses)
    if missing_statuses:
        raise RuntimeError(f"external dependency readiness audit is missing statuses: {missing_statuses}")

    if any(row["formal_method_allowed_now"] != "否" for row in rows):
        raise RuntimeError("external dependency readiness audit must not allow planned external methods as formal completed methods")
    for row in rows:
        for field in ("blocking_condition", "required_next_action", "required_return_artifacts", "source_evidence", "paper_boundary"):
            if not row[field].strip():
                raise RuntimeError(f"external dependency readiness row has empty {field}: {row['version']}")
        boundary_terms = ("不能", "不可", "只有", "必须", "不是真实")
        if row["readiness_status"].startswith("waiting") and not any(term in row["paper_boundary"] for term in boundary_terms):
            raise RuntimeError(f"external dependency readiness row lacks paper boundary wording: {row['version']}")
    return rows


def verify_openvla_dataset_bridge(args: argparse.Namespace) -> int:
    for path in (
        args.openvla_bridge_report,
        args.openvla_bridge_json,
        args.openvla_bridge_samples,
        args.openvla_bridge_manifest,
        args.openvla_bridge_preview,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    summary = read_json(args.openvla_bridge_json)
    manifest = read_json(args.openvla_bridge_manifest)
    if summary.get("version") != "openvla_dataset_bridge_v1":
        raise RuntimeError("OpenVLA bridge summary has unexpected version")
    if manifest.get("version") != "openvla_dataset_bridge_v1":
        raise RuntimeError("OpenVLA bridge manifest has unexpected version")
    samples = [line for line in args.openvla_bridge_samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 60:
        raise RuntimeError(f"OpenVLA bridge has too few samples: {len(samples)}")
    if int(summary.get("samples_exported", 0)) != len(samples):
        raise RuntimeError("OpenVLA bridge sample count does not match JSONL lines")
    first = json.loads(samples[0])
    required_fields = {"image", "instruction", "state", "action", "episode_index", "source_step"}
    missing_fields = sorted(required_fields - set(first))
    if missing_fields:
        raise RuntimeError(f"OpenVLA bridge sample is missing fields: {missing_fields}")
    if not (ROOT / first["image"]).exists():
        raise FileNotFoundError(ROOT / first["image"])

    text = args.openvla_bridge_report.read_text(encoding="utf-8")
    required_terms = (
        "OpenVLA 数据桥接报告",
        "openvla_dataset_bridge_v1",
        "image + instruction + state + action",
        "不能写：OpenVLA LoRA、真实机器人 VLA action head、Isaac 或真实 WidowX 验证已经完成",
    )
    missing_terms = [item for item in required_terms if item not in text]
    if missing_terms:
        raise RuntimeError(f"OpenVLA bridge report is missing terms: {missing_terms}")
    return len(samples)


def verify_openvla_bridge_gallery(args: argparse.Namespace, expected_samples: int) -> None:
    if not args.openvla_bridge_gallery.exists():
        raise FileNotFoundError(args.openvla_bridge_gallery)
    text = args.openvla_bridge_gallery.read_text(encoding="utf-8")
    required = (
        "OpenVLA 数据桥接浏览页",
        "openvla_bridge_gallery_v1",
        "本页不是策略评测结果",
        "../data/vla_bridge/openvla_dataset_bridge_v1/images/",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"OpenVLA bridge gallery is missing terms: {missing}")
    cards = text.count('class="sample-card"')
    if cards < expected_samples:
        raise RuntimeError(f"OpenVLA bridge gallery has too few sample cards: {cards}")
    image_refs = []
    for part in text.split('src="../')[1:]:
        image_refs.append(part.split('"', 1)[0])
    if len(image_refs) < expected_samples:
        raise RuntimeError(f"OpenVLA bridge gallery has too few image refs: {len(image_refs)}")
    for image_ref in image_refs:
        image_path = ROOT / image_ref
        if not image_path.exists():
            raise FileNotFoundError(image_path)


def verify_widowx_mujoco_rlds_source() -> int:
    source_dir = ROOT / "data" / "vla_bridge" / "widowx_mujoco_rlds_source_v1"
    source_json = ROOT / "outputs" / "evaluations" / "widowx_mujoco_rlds_source_v1.json"
    validation_json = ROOT / "outputs" / "evaluations" / "widowx_mujoco_rlds_source_validation_v1.json"
    report = ROOT / "docs" / "widowx_mujoco_rlds_source_v1_report.md"
    validation_report = ROOT / "docs" / "widowx_mujoco_rlds_source_validation_v1.md"
    exporter = ROOT / "scripts" / "export_openvla_rlds_source.py"
    validator = ROOT / "scripts" / "validate_openvla_rlds_source.py"
    builder = ROOT / "scripts" / "remote_openvla" / "widowx_mujoco_pick_place_dataset_builder.py"
    for path in (source_dir, source_json, validation_json, report, validation_report, exporter, validator, builder):
        if not path.exists():
            raise FileNotFoundError(path)

    source = read_json(source_json)
    validation = read_json(validation_json)
    if source.get("version") != "widowx_mujoco_rlds_source_v1":
        raise RuntimeError("RLDS source has an unexpected version")
    if source.get("status") != "rlds_source_ready_not_registered":
        raise RuntimeError("RLDS source status is wrong")
    if int(source.get("episodes_exported", 0)) != 79 or int(source.get("steps_exported", 0)) != 2528:
        raise RuntimeError("RLDS source does not contain the expected Core V2 episode or step count")
    if not source.get("state_representation", "").startswith("JOINT:"):
        raise RuntimeError("RLDS source has an unexpected state representation")
    if not source.get("action_representation", "").startswith("JOINT_POS:") or source.get("action_shape") != [8]:
        raise RuntimeError("RLDS source has an unexpected action representation")
    expected_task_counts = {
        "move_leftmost_cube_to_bowl": 19,
        "place_blue_cube_blue_pad": 20,
        "place_blue_cube_red_pad": 20,
        "place_red_cube_red_pad": 20,
    }
    if validation.get("source_version") != source["version"]:
        raise RuntimeError("RLDS source validation is linked to the wrong source version")
    if validation.get("task_episode_counts") != expected_task_counts:
        raise RuntimeError("RLDS source validation task distribution is wrong")
    if int(validation.get("episodes_validated", 0)) != 79 or int(validation.get("steps_validated", 0)) != 2528:
        raise RuntimeError("RLDS source validation counts are wrong")

    manifest = read_json(source_dir / "manifest.json")
    if manifest.get("version") != source["version"] or len(manifest.get("episode_manifest", [])) != 79:
        raise RuntimeError("RLDS source manifest is inconsistent")
    with np.load(source_dir / "episodes" / "episode_0000.npz") as first:
        if first["state"].shape != (32, 8) or first["action"].shape != (32, 8):
            raise RuntimeError("RLDS source episode shape is wrong")
        if int(first["is_first"].sum()) != 1 or int(first["is_last"].sum()) != 1 or int(first["is_terminal"].sum()) != 1:
            raise RuntimeError("RLDS source episode terminal fields are wrong")
        if not np.allclose(first["action"][:, 6], 0.0):
            raise RuntimeError("RLDS source padded joint action is not zero")
        if not (source_dir / str(first["image_paths"][0])).exists():
            raise RuntimeError("RLDS source image reference is missing")

    report_text = report.read_text(encoding="utf-8-sig")
    required_report = ("79", "2528", "ActionEncoding.JOINT_POS", "absolute actuator control target", "不能写：RLDS 已注册")
    missing_report = [item for item in required_report if item not in report_text]
    if missing_report:
        raise RuntimeError(f"RLDS source report is missing terms: {missing_report}")
    builder_text = builder.read_text(encoding="utf-8")
    for term in ("tensorflow_datasets", "WIDOWX_MUJOCO_RLDS_SOURCE_DIR", "joint_state", "language_instruction"):
        if term not in builder_text:
            raise RuntimeError(f"RLDS source builder is missing term: {term}")
    return int(source["steps_exported"])


def verify_kaggle_clip_semantic_adapter() -> int:
    version = "kaggle_clip_semantic_adapter_core_v2_v1"
    model = ROOT / "outputs" / "clip_semantic_waypoint" / f"{version}_kernel_v3.npz"
    evaluation = ROOT / "outputs" / "evaluations" / f"{version}.json"
    report = ROOT / "docs" / f"{version}_report.md"
    csv_path = ROOT / "docs" / f"{version}.csv"
    builder = ROOT / "scripts" / "build_kaggle_clip_semantic_adapter_report.py"
    kernel = ROOT / "kaggle" / "kernels" / "widowx_mujoco_clip_semantic_adapter_v1" / "train_clip_semantic_adapter.py"
    kernel_metadata = kernel.with_name("kernel-metadata.json")
    remote_dir = ROOT / "outputs" / "kaggle_remote" / f"{version}_kernel_v3"
    remote_metrics = remote_dir / f"{version}_metrics.json"
    predictions = remote_dir / f"{version}_predictions.csv"
    video = ROOT / "outputs" / "videos" / f"{version}_hard_leftmost_seed1900.mp4"
    video_metadata = video.with_suffix(".json")
    ood_version = "kaggle_clip_semantic_adapter_core_v2_ood_v1"
    ood_json = ROOT / "outputs" / "evaluations" / f"{ood_version}.json"
    ood_csv = ROOT / "docs" / f"{ood_version}.csv"
    ood_report = ROOT / "docs" / f"{ood_version}_report.md"
    ood_video = ROOT / "outputs" / "videos" / "kaggle_clip_semantic_adapter_core_v2_ood_paraphrase_blue_to_red_seed700.mp4"
    ood_video_metadata = ood_video.with_suffix(".json")
    for path in (model, evaluation, report, csv_path, builder, kernel, kernel_metadata, remote_metrics, predictions, video, video_metadata, ood_json, ood_csv, ood_report, ood_video, ood_video_metadata):
        if not path.exists():
            raise FileNotFoundError(path)

    with np.load(model) as data:
        required = {"x_mean", "x_std", "down_weight", "down_bias", "up_weight", "up_bias", "metadata"}
        if not required.issubset(set(data.files)):
            raise RuntimeError("Kaggle semantic adapter model has missing tensors")
        if data["x_mean"].shape != (1024,) or data["down_weight"].shape != (1024, 16) or data["up_weight"].shape != (16, 4):
            raise RuntimeError("Kaggle semantic adapter model has unexpected architecture")
        model_metadata = json.loads(data["metadata"].item())
    if model_metadata.get("version") != version or int(model_metadata.get("trainable_adapter_params", 0)) != 16468:
        raise RuntimeError("Kaggle semantic adapter model metadata is inconsistent")

    remote = read_json(remote_metrics).get("metadata", {})
    if remote.get("version") != version or int(remote.get("episode_samples", 0)) != 79:
        raise RuntimeError("Kaggle remote training data contract is wrong")
    if int(remote.get("train_samples", 0)) != 63 or int(remote.get("validation_samples", 0)) != 16:
        raise RuntimeError("Kaggle remote training split is wrong")
    if float(remote.get("train_accuracy", 0.0)) != 1.0 or float(remote.get("validation_accuracy", 0.0)) != 1.0:
        raise RuntimeError("Kaggle remote training accuracy is wrong")
    if remote.get("device") != "cpu" or remote.get("gpu_execution") is not False:
        raise RuntimeError("Kaggle runtime fallback must remain recorded as CPU")

    data = read_json(evaluation)
    local = data.get("local_closed_loop", {})
    if data.get("version") != version or int(local.get("episodes", 0)) != 20:
        raise RuntimeError("Kaggle local closed-loop contract is wrong")
    if int(local.get("successes", 0)) != 20 or int(local.get("strict_grasp_successes", 0)) != 20 or int(local.get("semantic_correct", 0)) != 20:
        raise RuntimeError("Kaggle local closed-loop strict results are wrong")
    baseline = data.get("same_protocol_baseline", {})
    if int(baseline.get("trainable_params", 0)) != 4100 or baseline.get("local_closed_loop", {}).get("success") != "20/20":
        raise RuntimeError("Kaggle same-protocol baseline is inconsistent")

    metadata = read_json(video_metadata)
    summary = metadata.get("summary", {})
    if metadata.get("version") != f"{version}_hard_leftmost" or int(metadata.get("seed", -1)) != 1900:
        raise RuntimeError("Kaggle video metadata has wrong version or seed")
    if metadata.get("task") != "move_leftmost_cube_to_bowl" or not summary.get("success") or not summary.get("strict_grasp_success"):
        raise RuntimeError("Kaggle video does not provide strict task success")
    if int(metadata.get("frames", 0)) < 100 or int(metadata.get("fps", 0)) != 30:
        raise RuntimeError("Kaggle video evidence is incomplete")

    ood = read_json(ood_json)
    if ood.get("version") != ood_version or len(ood.get("rows", [])) != 80:
        raise RuntimeError("Kaggle OOD evaluation contract is wrong")
    ood_totals = {}
    for condition in ("paraphrase", "hard_distractors"):
        rows = [row for row in ood["rows"] if row.get("condition") == condition]
        ood_totals[condition] = {
            "episodes": len(rows),
            "task_successes": sum(int(row.get("task_success", False)) for row in rows),
            "semantic_correct": sum(int(row.get("semantic_correct", False)) for row in rows),
            "strict_grasp_successes": sum(int(row.get("strict_grasp_success", False)) for row in rows),
        }
    if ood_totals["paraphrase"] != {"episodes": 60, "task_successes": 48, "semantic_correct": 48, "strict_grasp_successes": 51}:
        raise RuntimeError("Kaggle paraphrase OOD results are inconsistent")
    if ood_totals["hard_distractors"] != {"episodes": 20, "task_successes": 19, "semantic_correct": 19, "strict_grasp_successes": 20}:
        raise RuntimeError("Kaggle distractor OOD results are inconsistent")
    ood_metadata = read_json(ood_video_metadata)
    ood_summary = ood_metadata.get("summary", {})
    if ood_metadata.get("task") != "place_blue_cube_red_pad" or ood_metadata.get("seed") != 700:
        raise RuntimeError("Kaggle OOD failure video has wrong task or seed")
    if ood_summary.get("semantic_correct") or ood_summary.get("task_success") or ood_summary.get("predicted_intent") != "place_red_cube_red_pad":
        raise RuntimeError("Kaggle OOD failure video must retain the documented semantic error")

    text = report.read_text(encoding="utf-8")
    required_report = (version, "16,468", "20/20", "CPU fallback", "不是端到端 VLA", str(video.relative_to(ROOT)), ood_version, "48/60", "19/20", str(ood_video.relative_to(ROOT)))
    missing = [item for item in required_report if item not in text]
    if missing:
        raise RuntimeError(f"Kaggle report is missing required terms: {missing}")
    kernel_text = kernel.read_text(encoding="utf-8")
    if "KGAT_" in kernel_text or "KAGGLE_API_TOKEN" in kernel_text:
        raise RuntimeError("Kaggle training kernel must not contain credentials")
    return int(local["episodes"])


def verify_frozen_clip_semantic_adapter_comparison() -> int:
    version = "frozen_clip_semantic_adapter_same_protocol_comparison_v1"
    report = ROOT / "docs" / "frozen_clip_semantic_adapter_same_protocol_comparison.md"
    csv_path = ROOT / "docs" / "frozen_clip_semantic_adapter_same_protocol_comparison.csv"
    payload_path = ROOT / "outputs" / "evaluations" / f"{version}.json"
    builder = ROOT / "scripts" / "build_frozen_clip_semantic_adapter_comparison.py"
    for path in (report, csv_path, payload_path, builder):
        if not path.exists():
            raise FileNotFoundError(path)
    payload = read_json(payload_path)
    if payload.get("version") != version or len(payload.get("rows", [])) != 4:
        raise RuntimeError("frozen CLIP same-protocol comparison contract is wrong")
    deltas = payload.get("local_minus_kaggle_task_success_deltas", {})
    if deltas != {"paraphrase": 3, "hard_distractors": 1}:
        raise RuntimeError("frozen CLIP same-protocol comparison deltas are wrong")
    rows = read_csv(csv_path)
    if len(rows) != 4:
        raise RuntimeError("frozen CLIP same-protocol comparison csv must have four rows")
    expected = {
        ("clip_semantic_waypoint_core_v2_v1", "paraphrase"): (4100, "20/20", 60, 51, 51, 54),
        ("clip_semantic_waypoint_core_v2_v1", "hard_distractors"): (4100, "20/20", 20, 20, 20, 20),
        ("kaggle_clip_semantic_adapter_core_v2_v1", "paraphrase"): (16468, "20/20", 60, 48, 48, 51),
        ("kaggle_clip_semantic_adapter_core_v2_v1", "hard_distractors"): (16468, "20/20", 20, 19, 19, 20),
    }
    for row in rows:
        key = (row["version"], row["condition"])
        actual = (
            int(row["trainable_params"]),
            row["canonical_success"],
            int(row["episodes"]),
            int(row["task_successes"]),
            int(row["semantic_correct"]),
            int(row["strict_grasp_successes"]),
        )
        if key not in expected or actual != expected[key]:
            raise RuntimeError(f"frozen CLIP same-protocol comparison row is wrong: {key}")
    text = report.read_text(encoding="utf-8")
    required = (version, "51/60", "48/60", "20/20", "19/20", "不能写成端到端 VLA")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"frozen CLIP same-protocol comparison report is missing terms: {missing}")
    return len(rows)


def verify_openvla_feasibility(args: argparse.Namespace) -> None:
    if not args.openvla_feasibility_report.exists():
        raise FileNotFoundError(args.openvla_feasibility_report)
    if not args.openvla_feasibility_json.exists():
        raise FileNotFoundError(args.openvla_feasibility_json)

    data = read_json(args.openvla_feasibility_json)
    if data.get("version") != "openvla_feasibility_check_v1":
        raise RuntimeError("OpenVLA feasibility json has unexpected version")
    feasibility = data.get("feasibility", {})
    checks = feasibility.get("checks", {})
    if float(checks.get("gpu_memory_gb", 0)) <= 0:
        raise RuntimeError("OpenVLA feasibility report did not record GPU memory")
    if int(checks.get("bridge_samples", 0)) < 60:
        raise RuntimeError("OpenVLA feasibility report did not see bridge samples")
    text = args.openvla_feasibility_report.read_text(encoding="utf-8")
    required = (
        "OpenVLA 本地可行性检查",
        "openvla_feasibility_check_v1",
        "本机不适合直接训练真实 OpenVLA/机器人 VLA LoRA",
        "OpenVLA bridge 样本",
        "不能写：OpenVLA LoRA、真实机器人 VLA action head、Isaac 或真实 WidowX 验证已经完成",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"OpenVLA feasibility report is missing terms: {missing}")


def verify_robot_vla_handoff(args: argparse.Namespace) -> None:
    if not args.robot_vla_handoff_report.exists():
        raise FileNotFoundError(args.robot_vla_handoff_report)
    if not args.robot_vla_handoff_json.exists():
        raise FileNotFoundError(args.robot_vla_handoff_json)

    data = read_json(args.robot_vla_handoff_json)
    if data.get("version") != "robot_vla_action_head_handoff_v1":
        raise RuntimeError("Robot VLA handoff json has unexpected version")
    if data.get("status") != "completed_prerequisite":
        raise RuntimeError("Robot VLA handoff must remain a completed prerequisite, not a completed policy")
    verdict = data.get("local_verdict", {})
    if int(verdict.get("bridge_samples", 0)) < 60:
        raise RuntimeError("Robot VLA handoff did not see enough bridge samples")
    if float(verdict.get("gpu_memory_gb", 0)) <= 0:
        raise RuntimeError("Robot VLA handoff did not record local GPU memory")
    output_contract = data.get("output_contract", {})
    required_metrics = set(output_contract.get("required_metrics", []))
    if not {"train_range_success", "heldout_success", "language_success", "grasp_success", "object_z"}.issubset(required_metrics):
        raise RuntimeError("Robot VLA handoff is missing required success/grasp metrics")
    text = args.robot_vla_handoff_report.read_text(encoding="utf-8")
    required = (
        "Robot VLA Action-Head 运行交接门禁",
        "robot_vla_action_head_handoff_v1",
        "不能写：`robot_vla_action_head_lite_v1`",
        "OpenVLA LoRA",
        "OpenVLA-OFT",
        "真实 WidowX",
        "必须保持 planned",
        "真实机器人预训练 VLA/VLM 表征",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Robot VLA handoff report is missing terms: {missing}")


def verify_robot_vla_remote_run_pack(args: argparse.Namespace) -> None:
    for path in (
        args.robot_vla_remote_pack_report,
        args.robot_vla_remote_pack_json,
        args.robot_vla_remote_pack_archive,
        args.robot_vla_remote_pack_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    data = read_json(args.robot_vla_remote_pack_json)
    if data.get("version") != "robot_vla_remote_run_pack_v1":
        raise RuntimeError("Robot VLA remote run pack json has unexpected version")
    if data.get("status") != "completed_prerequisite":
        raise RuntimeError("Robot VLA remote run pack must remain a completed prerequisite")
    if data.get("target_planned_version") != "robot_vla_action_head_lite_v1":
        raise RuntimeError("Robot VLA remote run pack targets the wrong planned version")
    if int(data.get("bridge_samples", 0)) < 60:
        raise RuntimeError("Robot VLA remote run pack has too few bridge samples")
    if data.get("rlds_source_version") != "widowx_mujoco_rlds_source_v1":
        raise RuntimeError("Robot VLA remote run pack has the wrong RLDS source version")
    if data.get("rlds_source_validation_version") != "widowx_mujoco_rlds_source_validation_v1":
        raise RuntimeError("Robot VLA remote run pack has the wrong RLDS source validation version")
    if int(data.get("rlds_source_episodes", 0)) < 79 or int(data.get("rlds_source_steps", 0)) < 2528:
        raise RuntimeError("Robot VLA remote run pack copied too little validated RLDS source data")
    if int(data.get("minimum_gpu_memory_gb", 0)) < 27:
        raise RuntimeError("Robot VLA remote run pack lost the remote GPU memory gate")
    if int(data.get("packaged_file_count", 0)) < 80:
        raise RuntimeError("Robot VLA remote run pack has too few packaged files")
    required_return = set(data.get("required_remote_return_files", []))
    required_remote_files = {
        "outputs/robot_vla_action_head/robot_vla_action_head_lite_v1.*",
        "outputs/robot_vla_action_head/openvla_feature_cache_v1.*",
        "outputs/evaluations/robot_vla_action_head_lite_v1.json",
        "outputs/videos/robot_vla_action_head_lite_v1_seed0.mp4",
        "outputs/videos/robot_vla_action_head_lite_v1_language_seed200.mp4",
        "docs/robot_vla_action_head_lite_report.md",
    }
    if not required_remote_files.issubset(required_return):
        raise RuntimeError(f"Robot VLA remote run pack is missing return files: {sorted(required_remote_files - required_return)}")

    required_pack_files = (
        args.robot_vla_remote_pack_dir / "README_REMOTE_RUN.md",
        args.robot_vla_remote_pack_dir / "REMOTE_RUN_COMMANDS.md",
        args.robot_vla_remote_pack_dir / "RLDS_INTEGRATION.md",
        args.robot_vla_remote_pack_dir / "rlds_builder" / "widowx_mujoco_pick_place_dataset_builder.py",
        args.robot_vla_remote_pack_dir / "run_config.json",
        args.robot_vla_remote_pack_dir / "remote_result_schema.json",
        args.robot_vla_remote_pack_dir / "remote_result_template.json",
        args.robot_vla_remote_pack_dir / "samples_preview.jsonl",
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "samples.jsonl",
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "manifest.json",
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "preview_grid.png",
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "widowx_mujoco_rlds_source_v1" / "manifest.json",
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "widowx_mujoco_rlds_source_v1" / "preview_grid.png",
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "widowx_mujoco_rlds_source_v1" / "episodes" / "episode_0000.npz",
    )
    for path in required_pack_files:
        if not path.exists():
            raise FileNotFoundError(path)
    schema = read_json(args.robot_vla_remote_pack_dir / "remote_result_schema.json")
    required_schema_terms = {
        "rlds_dataset_name",
        "rlds_dataset_statistics",
        "action_representation",
        "dataset_adapter_commit",
        "train_range_success",
        "heldout_success",
        "language_success",
        "grasp_success",
        "object_z",
    }
    schema_text = json.dumps(schema, ensure_ascii=False)
    missing_schema_terms = [term for term in required_schema_terms if term not in schema_text]
    if missing_schema_terms:
        raise RuntimeError(f"Robot VLA remote schema is missing terms: {missing_schema_terms}")
    template = read_json(args.robot_vla_remote_pack_dir / "remote_result_template.json")
    if template.get("uses_real_robot_vla_features") is not False:
        raise RuntimeError("Robot VLA remote result template must require explicit real VLA feature confirmation")
    pack_samples_path = args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "openvla_dataset_bridge_v1" / "samples.jsonl"
    samples = [line for line in pack_samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(samples) < 60:
        raise RuntimeError(f"Robot VLA remote run pack copied too few samples: {len(samples)}")
    rlds_source_manifest = read_json(
        args.robot_vla_remote_pack_dir / "data" / "vla_bridge" / "widowx_mujoco_rlds_source_v1" / "manifest.json"
    )
    if rlds_source_manifest.get("version") != "widowx_mujoco_rlds_source_v1":
        raise RuntimeError("Robot VLA remote run pack has an invalid copied RLDS source manifest")
    if int(rlds_source_manifest.get("episodes_exported", 0)) < 79 or int(rlds_source_manifest.get("steps_exported", 0)) < 2528:
        raise RuntimeError("Robot VLA remote run pack copied incomplete RLDS source episodes")
    builder_text = (args.robot_vla_remote_pack_dir / "rlds_builder" / "widowx_mujoco_pick_place_dataset_builder.py").read_text(encoding="utf-8")
    for term in ("tensorflow_datasets", "WIDOWX_MUJOCO_RLDS_SOURCE_DIR", "language_instruction"):
        if term not in builder_text:
            raise RuntimeError(f"Robot VLA RLDS builder is missing term: {term}")

    with zipfile.ZipFile(args.robot_vla_remote_pack_archive) as zf:
        names = set(zf.namelist())
    required_zip_names = {
        "README_REMOTE_RUN.md",
        "REMOTE_RUN_COMMANDS.md",
        "RLDS_INTEGRATION.md",
        "rlds_builder/widowx_mujoco_pick_place_dataset_builder.py",
        "run_config.json",
        "remote_result_schema.json",
        "remote_result_template.json",
        "data/vla_bridge/openvla_dataset_bridge_v1/samples.jsonl",
        "data/vla_bridge/openvla_dataset_bridge_v1/manifest.json",
        "data/vla_bridge/openvla_dataset_bridge_v1/preview_grid.png",
        "data/vla_bridge/widowx_mujoco_rlds_source_v1/manifest.json",
        "data/vla_bridge/widowx_mujoco_rlds_source_v1/episodes/episode_0000.npz",
    }
    missing_zip_names = sorted(required_zip_names - names)
    if missing_zip_names:
        raise RuntimeError(f"Robot VLA remote run pack archive is missing files: {missing_zip_names}")

    text = args.robot_vla_remote_pack_report.read_text(encoding="utf-8-sig")
    required = (
        "Robot VLA 远端运行包",
        "robot_vla_remote_run_pack_v1",
        "robot_vla_action_head_lite_v1",
        "48GB+ GPU",
        "remote_result_schema",
        "pre_rlds_source_ready",
        "尚未注册为 RLDS",
        "JOINT_POS",
        "不能写：`robot_vla_action_head_lite_v1`",
        "OpenVLA-OFT",
        "真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Robot VLA remote run pack report is missing terms: {missing}")


def verify_robot_vla_remote_result_intake(args: argparse.Namespace) -> list[dict[str, str]]:
    for path in (
        args.robot_vla_remote_intake_report,
        args.robot_vla_remote_intake_csv,
        args.robot_vla_remote_intake_json,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    data = read_json(args.robot_vla_remote_intake_json)
    if data.get("version") != "robot_vla_remote_result_intake_v1":
        raise RuntimeError("Robot VLA remote result intake json has unexpected version")
    if data.get("target_planned_version") != "robot_vla_action_head_lite_v1":
        raise RuntimeError("Robot VLA remote result intake targets the wrong planned version")
    if data.get("remote_pack_version") != "robot_vla_remote_run_pack_v1":
        raise RuntimeError("Robot VLA remote result intake is not linked to the remote run pack")
    if data.get("status") != "waiting_for_remote_result":
        raise RuntimeError("Robot VLA remote result intake should remain waiting until real remote results are returned")
    if data.get("can_register_completed_method") is not False:
        raise RuntimeError("Robot VLA remote result intake must not allow formal registration without returned files")
    if int(data.get("returned_files_required", 0)) < 6:
        raise RuntimeError("Robot VLA remote result intake lost required return file count")
    if int(data.get("returned_files_present", 0)) != 0:
        raise RuntimeError("Robot VLA remote result intake unexpectedly sees returned files")
    blocking = "\n".join(data.get("blocking_reasons", []))
    if "缺少远端评测 JSON" not in blocking or "robot_vla_action_head_lite_v1.json" not in blocking:
        raise RuntimeError("Robot VLA remote result intake does not report the missing remote evaluation JSON")

    rows = read_csv(args.robot_vla_remote_intake_csv)
    if len(rows) < 6:
        raise RuntimeError(f"Robot VLA remote result intake csv has too few rows: {len(rows)}")
    required_columns = {"项目", "路径或模式", "当前存在", "入包要求"}
    if not required_columns.issubset(rows[0]):
        raise RuntimeError(f"Robot VLA remote result intake csv is missing columns: {sorted(required_columns - set(rows[0]))}")
    required_paths = {
        "outputs/robot_vla_action_head/robot_vla_action_head_lite_v1.*",
        "outputs/robot_vla_action_head/openvla_feature_cache_v1.*",
        "outputs/evaluations/robot_vla_action_head_lite_v1.json",
        "outputs/videos/robot_vla_action_head_lite_v1_seed0.mp4",
        "outputs/videos/robot_vla_action_head_lite_v1_language_seed200.mp4",
        "docs/robot_vla_action_head_lite_report.md",
    }
    csv_paths = {row["路径或模式"] for row in rows}
    missing_paths = sorted(required_paths - csv_paths)
    if missing_paths:
        raise RuntimeError(f"Robot VLA remote result intake csv is missing paths: {missing_paths}")

    text = args.robot_vla_remote_intake_report.read_text(encoding="utf-8-sig")
    required = (
        "Robot VLA 远端结果回填门禁",
        "robot_vla_remote_result_intake_v1",
        "waiting_for_remote_result",
        "robot_vla_action_head_lite_v1",
        "uses_real_robot_vla_features=true",
        "正式入包步骤",
        "不能写：`robot_vla_action_head_lite_v1`",
        "OpenVLA-OFT",
        "真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"Robot VLA remote result intake report is missing terms: {missing}")
    return rows


def verify_defense_evidence_pack(args: argparse.Namespace) -> dict[str, object]:
    if not args.defense_evidence_pack.exists():
        raise FileNotFoundError(args.defense_evidence_pack)
    if not args.defense_evidence_pack_json.exists():
        raise FileNotFoundError(args.defense_evidence_pack_json)
    if not args.defense_evidence_pack_archive.exists():
        raise FileNotFoundError(args.defense_evidence_pack_archive)
    if not args.defense_evidence_pack_dir.is_dir():
        raise FileNotFoundError(args.defense_evidence_pack_dir)

    data = read_json(args.defense_evidence_pack_json)
    if data.get("version") != "defense_evidence_pack_v1":
        raise RuntimeError("defense evidence pack has unexpected version")
    if int(data.get("file_count", 0)) < 300:
        raise RuntimeError("defense evidence pack has too few files")

    role_counts = data.get("role_counts", {})
    required_roles = (
        "核心中文文档/索引",
        "外部阶段前置证据",
        "复现脚本",
        "论文图表",
        "全量单方法视频证据",
        "答辩阶段短片",
        "宫格展示视频",
        "证据包入口",
    )
    missing_roles = [role for role in required_roles if int(role_counts.get(role, 0)) <= 0]
    if missing_roles:
        raise RuntimeError(f"defense evidence pack is missing role counts: {missing_roles}")

    archive_rel = args.defense_evidence_pack_archive.relative_to(ROOT).as_posix()
    if data.get("archive_path") != archive_rel:
        raise RuntimeError("defense evidence pack archive path is inconsistent")
    if data.get("archive_sha256") != sha256_file(args.defense_evidence_pack_archive):
        raise RuntimeError("defense evidence pack archive sha256 mismatch")
    if int(data.get("archive_size_bytes", 0)) <= 0:
        raise RuntimeError("defense evidence pack archive size is empty")

    with zipfile.ZipFile(args.defense_evidence_pack_archive) as archive:
        names = set(archive.namelist())
    required_entries = (
        "START_HERE.md",
        "PACK_MANIFEST.json",
        "docs/experiment_dashboard.html",
        "docs/defense_deck.html",
        "docs/final_showcase_handoff.md",
        "docs/final_showcase_handoff.csv",
        "docs/final_defense_narrative_script.md",
        "docs/final_defense_narrative_script.csv",
        "docs/remaining_experiment_execution_board.md",
        "docs/remaining_experiment_execution_board.csv",
        "docs/method_comparison_dashboard.html",
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
        "outputs/clip_action_head/clip_core_v2_multitask_v1_20260721_104743.npz",
        "outputs/videos/clip_core_v2_multitask_v1_seed0.mp4",
        "outputs/videos/clip_core_v2_multitask_v1_leftmost_cube_seed420.mp4",
        "docs/core_v2_clip_semantic_waypoint_report.md",
        "docs/core_v2_clip_semantic_waypoint_report.csv",
        "outputs/evaluations/core_v2_clip_semantic_waypoint_v1.json",
        "scripts/build_core_v2_clip_semantic_waypoint_report.py",
        "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_v1_20260721_110325.npz",
        "outputs/videos/clip_semantic_waypoint_core_v2_v1_leftmost_cube_seed420.mp4",
        "docs/core_v2_clip_semantic_data_efficiency.md",
        "docs/core_v2_clip_semantic_data_efficiency.csv",
        "outputs/evaluations/core_v2_clip_semantic_data_efficiency_v1.json",
        "scripts/evaluate_clip_semantic_waypoint_data_efficiency.py",
        "scripts/build_core_v2_clip_semantic_data_efficiency_report.py",
        "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_5eps_v1_20260721_111415.npz",
        "outputs/clip_semantic_waypoint/clip_semantic_waypoint_core_v2_10eps_v1_20260721_111439.npz",
        "docs/core_v2_clip_semantic_ood_generalization.md",
        "docs/core_v2_clip_semantic_ood_generalization.csv",
        "outputs/evaluations/core_v2_clip_semantic_ood_generalization_v1.json",
        "scripts/evaluate_clip_semantic_ood_generalization.py",
        "outputs/videos/clip_semantic_ood_hard_leftmost_cube_seed1300.mp4",
        "outputs/videos/clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4",
        "docs/version_lineage_index.html",
        "docs/defense_qa_playbook.html",
        "docs/final_experiment_package.md",
        "docs/final_closure_audit_v1.md",
        "docs/v4_independent_replication_v1.md",
        "docs/contact_phase_monitor_heldout_v1_analysis.md",
        "docs/counterfactual_intervention_pilot_v1_audit.md",
        "outputs/evaluations/final_closure_audit_v1.json",
        "outputs/evaluations/v4_independent_replication_v1.json",
        "outputs/evaluations/contact_phase_monitor_heldout_v1.json",
        "outputs/evaluations/counterfactual_intervention_pilot_v1.json",
        "videos/rgb_table_recovery_v4/seed4006_severe_leftmost_table_recovery_success.mp4",
        "docs/reproducible_command_index.md",
        "docs/trajectory_act_conclusion_brief.md",
        "docs/trajectory_act_conclusion_brief.csv",
        "docs/trajectory_act_slow_viewer_guide.md",
        "docs/trajectory_act_slow_viewer_guide.csv",
        "docs/trajectory_prior_residual_bc_report.md",
        "docs/trajectory_prior_residual_bc_report.csv",
        "docs/timing_aware_trajectory_prior_residual_bc_report.md",
        "docs/timing_aware_trajectory_prior_residual_bc_report.csv",
        "docs/phase_weighted_torch_act_report.md",
        "docs/grasp_lift_subpolicy_probe_report.md",
        "docs/contact_stage_subpolicy_report.md",
        "docs/contact_stage_demo_torch_act_report.md",
        "docs/contact_stage_demo_torch_act_report.csv",
        "docs/contact_stage_phase_action_head_report.md",
        "docs/contact_stage_phase_action_head_report.csv",
        "docs/contact_hold_weighted_torch_act_report.md",
        "docs/contact_hold_weighted_torch_act_report.csv",
        "docs/preference_ranked_trajectory_post_training_report.md",
        "docs/preference_ranked_trajectory_post_training_report.csv",
        "docs/preference_trajectory_post_training_v1_ranked_fast_summary.md",
        "docs/preference_trajectory_post_training_v1_ranked_fast_report.md",
        "docs/preference_trajectory_post_training_v1_ranked_fast_report.csv",
        "docs/preference_trajectory_post_training_v1_ranked_objective_summary.md",
        "docs/preference_trajectory_post_training_v1_ranked_objective_report.md",
        "docs/preference_trajectory_post_training_v1_ranked_objective_report.csv",
        "docs/preference_post_training_ablation_matrix.md",
        "docs/preference_post_training_ablation_matrix.csv",
        "data/demos/contact_stage_demo_place_blue_cube_blue_pad_medium_v1/summary.json",
        "outputs/evaluations/trajectory_prior_residual_bc_v1_candidate.json",
        "outputs/evaluations/timing_aware_trajectory_prior_residual_bc_v1_candidate.json",
        "outputs/evaluations/contact_stage_demo_torch_act_v1_candidate.json",
        "outputs/evaluations/contact_stage_phase_action_head_v1_candidate.json",
        "outputs/evaluations/contact_hold_weighted_torch_act_v1_candidate.json",
        "outputs/evaluations/preference_ranked_trajectory_post_training_v1_candidate.json",
        "outputs/evaluations/preference_trajectory_post_training_v1_ranked_fast_candidate.json",
        "outputs/evaluations/preference_trajectory_post_training_v1_ranked_objective_candidate.json",
        "outputs/trajectory_prior_residual_bc/trajectory_prior_residual_bc_v1_candidate_20260721_040050.npz",
        "outputs/timing_aware_trajectory_prior_residual_bc/timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/trajectory_prior_residual_bc_v1_candidate_seed0.json",
        "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "outputs/videos/timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.json",
        "outputs/torch_act/contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt",
        "outputs/phase_action_head/contact_stage_phase_action_head_v1_candidate_20260721_020941.npz",
        "outputs/torch_act/contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt",
        "outputs/preference_post_training/preference_ranked_trajectory_post_training_20260721_031024.npz",
        "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz",
        "outputs/preference_post_training/preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
        "outputs/videos/trajectory_knn_chunk_bc_v1_seed0.mp4",
        "outputs/videos/phase_weighted_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "outputs/videos/clip_semantic_ood_hard_leftmost_cube_seed1300.mp4",
        "outputs/videos/clip_semantic_ood_paraphrase_blue_to_red_seed700.mp4",
        "outputs/videos/contact_stage_subpolicy_v1_candidate_seed0.mp4",
        "outputs/videos/contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "outputs/videos/contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
        "outputs/videos/preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.json",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4",
        "outputs/videos/preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.json",
        "outputs/videos/clip_action_head_lite_v1_seed0.mp4",
        "docs/widowx_mujoco_rlds_source_v1_report.md",
        "docs/widowx_mujoco_rlds_source_validation_v1.md",
        "outputs/evaluations/widowx_mujoco_rlds_source_v1.json",
        "outputs/evaluations/widowx_mujoco_rlds_source_validation_v1.json",
        "scripts/export_openvla_rlds_source.py",
        "scripts/validate_openvla_rlds_source.py",
        "scripts/remote_openvla/widowx_mujoco_pick_place_dataset_builder.py",
        "outputs/showcase/all_registered_methods_grid.mp4",
        "outputs/figures/resource_vs_success.svg",
        "scripts/build_final_showcase_handoff.py",
        "scripts/build_trajectory_act_conclusion_brief.py",
        "scripts/build_trajectory_act_slow_viewer_guide.py",
        "scripts/build_preference_post_training_ablation_matrix.py",
        "scripts/build_final_defense_narrative_script.py",
        "scripts/build_remaining_experiment_execution_board.py",
        "scripts/trajectory_prior_residual_common.py",
        "scripts/train_trajectory_prior_residual_bc.py",
        "scripts/run_trajectory_prior_residual_policy.py",
        "scripts/evaluate_trajectory_prior_residual_bc.py",
        "scripts/timing_aware_trajectory_prior_residual_common.py",
        "scripts/train_timing_aware_trajectory_prior_residual_bc.py",
        "scripts/run_timing_aware_trajectory_prior_residual_policy.py",
        "scripts/evaluate_timing_aware_trajectory_prior_residual_bc.py",
        "scripts/evaluate_phase_weighted_torch_act.py",
        "scripts/evaluate_grasp_lift_subpolicy_probe.py",
        "scripts/run_contact_stage_subpolicy.py",
        "scripts/evaluate_contact_stage_subpolicy.py",
        "scripts/collect_contact_stage_demos.py",
        "scripts/evaluate_contact_stage_demo_torch_act.py",
        "scripts/evaluate_contact_stage_phase_action_head.py",
        "scripts/evaluate_contact_hold_weighted_torch_act.py",
        "scripts/evaluate_preference_ranked_trajectory_post_training.py",
        "scripts/build_defense_evidence_pack.py",
    )
    missing_entries = [entry for entry in required_entries if entry not in names]
    if missing_entries:
        raise RuntimeError(f"defense evidence pack zip is missing entries: {missing_entries}")

    text = args.defense_evidence_pack.read_text(encoding="utf-8")
    required_text = (
        "defense_evidence_pack_v1",
        "START_HERE.md",
        "PACK_MANIFEST.json",
        "final_showcase_handoff.md",
        "final_defense_narrative_script_v1",
        "final_defense_narrative_script.md",
        "remaining_experiment_execution_board_v1",
        "remaining_experiment_execution_board.md",
        "trajectory_act_conclusion_brief_v1",
        "trajectory_act_slow_viewer_guide_v1",
        "preference_post_training_ablation_matrix_v1",
        "00_defense_video_reel.mp4",
        "trajectory_prior_residual_bc_v1_candidate",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "contact_stage_subpolicy_v1_candidate",
        "contact_stage_demo_torch_act_v1_candidate",
        "contact_hold_weighted_torch_act_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "widowx_mujoco_rlds_source_v1",
        "widowx_mujoco_rlds_source_validation_v1",
        "真实 OpenVLA、Isaac 或真实 WidowX",
    )
    missing_text = [item for item in required_text if item not in text]
    if missing_text:
        raise RuntimeError(f"defense evidence pack report is missing terms: {missing_text}")
    return data


def verify_showcase_launcher(args: argparse.Namespace) -> list[str]:
    if not args.showcase_launcher.exists():
        raise FileNotFoundError(args.showcase_launcher)
    if not args.showcase_launcher_guide.exists():
        raise FileNotFoundError(args.showcase_launcher_guide)

    text = args.showcase_launcher_guide.read_text(encoding="utf-8-sig")
    required = (
        "本地展示启动器说明",
        "showcase_launcher_v1",
        "--list quick",
        "--list candidates",
        "--target handoff",
        "--target narrative-script",
        "--target remaining-board",
        "--target matrix",
        "--target trajectory-act-brief",
        "--target trajectory-act-slow",
        "--target live-runbook",
        "--target evidence-pack",
        "--target preference-ablation",
        "--target playlist",
        "--target cue-sheet",
        "--target comparison",
        "--target visual-index",
        "--target qa",
        "--target lineage",
        "--target claim:C03",
        "--target stage:2",
        "--target method:torch_act_state_chunk_v1",
        "--target candidate:grasp_gated_torch_act_state_chunk_v1_candidate",
        "--action viewer",
        "--dry-run",
        "真实 OpenVLA、Isaac 和真实 WidowX",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"showcase launcher guide is missing required terms: {missing}")

    quick = subprocess.run(
        [sys.executable, str(args.showcase_launcher), "--list", "quick"],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "dashboard" not in quick or "handoff" not in quick or "narrative-script" not in quick or "remaining-board" not in quick or "matrix" not in quick or "trajectory-act-brief" not in quick or "trajectory-act-slow" not in quick or "deck" not in quick or "gallery" not in quick or "comparison" not in quick or "playlist" not in quick or "cue-sheet" not in quick or "visual-index" not in quick or "qa" not in quick or "lineage" not in quick or "preference-gate" not in quick or "preference-ablation" not in quick or "evidence-pack" not in quick or "live-runbook" not in quick:
        raise RuntimeError("showcase launcher quick list is missing expected entries")

    handoff = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "handoff",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "final_showcase_handoff.md" not in handoff:
        raise RuntimeError("showcase launcher handoff dry-run is missing final_showcase_handoff.md")

    narrative = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "narrative-script",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "final_defense_narrative_script.md" not in narrative:
        raise RuntimeError("showcase launcher narrative-script dry-run is missing final_defense_narrative_script.md")

    remaining_board = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "remaining-board",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "remaining_experiment_execution_board.md" not in remaining_board:
        raise RuntimeError("showcase launcher remaining-board dry-run is missing remaining_experiment_execution_board.md")

    matrix = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "matrix",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "result_matrix.md" not in matrix:
        raise RuntimeError("showcase launcher matrix dry-run is missing result_matrix.md")

    trajectory_act_brief = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "trajectory-act-brief",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "trajectory_act_conclusion_brief.md" not in trajectory_act_brief:
        raise RuntimeError("showcase launcher trajectory-act-brief dry-run is missing trajectory_act_conclusion_brief.md")

    trajectory_act_slow = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "trajectory-act-slow",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "trajectory_act_slow_viewer_guide.md" not in trajectory_act_slow:
        raise RuntimeError("showcase launcher trajectory-act-slow dry-run is missing trajectory_act_slow_viewer_guide.md")

    comparison = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "comparison",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "method_comparison_dashboard.html" not in comparison:
        raise RuntimeError("showcase launcher comparison dry-run is missing method_comparison_dashboard.html")

    visual_index = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "visual-index",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "thesis_visual_evidence_index.html" not in visual_index:
        raise RuntimeError("showcase launcher visual-index dry-run is missing thesis_visual_evidence_index.html")

    qa = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "qa",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "defense_qa_playbook.html" not in qa:
        raise RuntimeError("showcase launcher qa dry-run is missing defense_qa_playbook.html")

    lineage = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "lineage",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "version_lineage_index.html" not in lineage:
        raise RuntimeError("showcase launcher lineage dry-run is missing version_lineage_index.html")

    playlist = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "playlist",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "defense_video_playlist.html" not in playlist:
        raise RuntimeError("showcase launcher playlist dry-run is missing defense_video_playlist.html")

    cue_sheet = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "cue-sheet",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "defense_video_cue_sheet.md" not in cue_sheet:
        raise RuntimeError("showcase launcher cue-sheet dry-run is missing defense_video_cue_sheet.md")

    live_runbook = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "live-runbook",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "defense_live_runbook.md" not in live_runbook:
        raise RuntimeError("showcase launcher live-runbook dry-run is missing defense_live_runbook.md")

    evidence_pack = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "evidence-pack",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "defense_evidence_pack.md" not in evidence_pack:
        raise RuntimeError("showcase launcher evidence-pack dry-run is missing defense_evidence_pack.md")

    preference_gate = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "preference-gate",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "preference_post_training_upgrade_gate.md" not in preference_gate:
        raise RuntimeError("showcase launcher preference-gate dry-run is missing upgrade gate doc")

    preference_ablation = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "preference-ablation",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    if "preference_post_training_ablation_matrix.md" not in preference_ablation:
        raise RuntimeError("showcase launcher preference-ablation dry-run is missing ablation matrix doc")

    candidates = subprocess.run(
        [sys.executable, str(args.showcase_launcher), "--list", "candidates"],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    required_candidates = (
        "grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate",
        "grasp_gated_torch_act_state_chunk_v1_candidate",
        "preference_trajectory_post_training_v1_candidate",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "contact_phase_gated_torch_act_v1_candidate",
        "contact_aware_phase_gated_torch_act_v1_candidate",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "contact_stage_subpolicy_v1_candidate",
        "contact_stage_demo_torch_act_v1_candidate",
        "contact_stage_phase_action_head_v1_candidate",
        "contact_hold_weighted_torch_act_v1_candidate",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "preference_ranked_trajectory_post_training_v1_candidate",
    )
    missing_candidates = [item for item in required_candidates if item not in candidates]
    if missing_candidates:
        raise RuntimeError(f"showcase launcher candidate list is missing terms: {missing_candidates}")

    dry_run = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "method:trajectory_knn_chunk_bc_v1",
            "--action",
            "viewer",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    required_dry_run = (
        "run_trajectory_knn_policy.py",
        "--viewer",
        "--duration 60",
        "--speed 0.05",
        "outputs\\trajectory_knn_bc",
    )
    missing_dry_run = [item for item in required_dry_run if item not in dry_run]
    if missing_dry_run:
        raise RuntimeError(f"showcase launcher dry-run is missing terms: {missing_dry_run}")

    candidate_dry_run = subprocess.run(
        [
            sys.executable,
            str(args.showcase_launcher),
            "--target",
            "candidate:grasp_gated_torch_act_state_chunk_v1_candidate",
            "--action",
            "viewer",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    required_candidate_dry_run = (
        "run_torch_act_policy.py",
        "--grasp-gate",
        "--viewer",
        "--duration 60",
        "--speed 0.05",
        "VLA_TORCH_PACKAGE_DIR",
    )
    missing_candidate_dry_run = [item for item in required_candidate_dry_run if item not in candidate_dry_run]
    if missing_candidate_dry_run:
        raise RuntimeError(f"showcase launcher candidate dry-run is missing terms: {missing_candidate_dry_run}")
    return ["quick", "handoff_dry_run", "narrative_script_dry_run", "remaining_board_dry_run", "matrix_dry_run", "trajectory_act_brief_dry_run", "trajectory_act_slow_dry_run", "comparison_dry_run", "visual_index_dry_run", "qa_dry_run", "lineage_dry_run", "playlist_dry_run", "cue_sheet_dry_run", "live_runbook_dry_run", "evidence_pack_dry_run", "preference_gate_dry_run", "preference_ablation_dry_run", "viewer_dry_run", "candidate_list", "candidate_viewer_dry_run"]


def verify_command_index(args: argparse.Namespace, versions: list[str]) -> None:
    if not args.command_index.exists():
        raise FileNotFoundError(args.command_index)
    text = args.command_index.read_text(encoding="utf-8")
    required = (
        "可复现实验命令索引",
        "主任务慢速 Viewer 命令",
        "语言/空间泛化慢速 Viewer 命令",
        "关键训练/重建命令",
        "评测命令",
        "--viewer",
        "--duration 60",
        "docs\\experiment_dashboard.html",
        "docs\\showcase_launcher_guide.md",
        "scripts\\showcase_launcher.py",
        "--target evidence-pack",
        "docs\\defense_evidence_pack.md",
        "outputs\\defense_evidence_pack\\defense_evidence_pack_v1.zip",
        "outputs\\evaluations\\defense_evidence_pack_v1.json",
        "scripts\\build_defense_evidence_pack.py",
        "docs\\method_stage_audit.md",
        "docs\\method_evidence_gate.md",
        "docs\\method_evidence_gate.csv",
        "docs\\stage_comparison_report.md",
        "docs\\stage_showcase_index.md",
        "docs\\stage_showcase_index.html",
        "docs\\task_bc_stage_report.md",
        "docs\\trajectory_act_stage_report.md",
        "docs\\trajectory_act_experiment_record.md",
        "docs\\trajectory_act_experiment_record.csv",
        "docs\\trajectory_act_failure_diagnosis.md",
        "docs\\trajectory_act_failure_diagnosis.csv",
        "docs\\trajectory_act_conclusion_brief.md",
        "docs\\trajectory_act_conclusion_brief.csv",
        "scripts\\build_trajectory_act_conclusion_brief.py",
        "--target trajectory-act-brief",
        "docs\\final_defense_narrative_script.md",
        "docs\\final_defense_narrative_script.csv",
        "scripts\\build_final_defense_narrative_script.py",
        "--target narrative-script",
        "docs\\remaining_experiment_execution_board.md",
        "docs\\remaining_experiment_execution_board.csv",
        "scripts\\build_remaining_experiment_execution_board.py",
        "--target remaining-board",
        "docs\\trajectory_phase_template_bc_report.md",
        "docs\\trajectory_phase_template_bc_report.csv",
        "outputs\\evaluations\\trajectory_phase_template_bc_v1.json",
        "outputs\\trajectory_phase_template_bc\\trajectory_phase_template_bc_20260720_160007.npz",
        "docs\\grasp_gated_trajectory_act_report.md",
        "docs\\grasp_gated_trajectory_act_report.csv",
        "outputs\\evaluations\\grasp_gated_trajectory_act_v1_candidate.json",
        "outputs\\videos\\grasp_gated_trajectory_conditioned_chunk_bc_v1_candidate_seed0.mp4",
        "outputs\\videos\\grasp_gated_torch_act_state_chunk_v1_candidate_seed0.mp4",
        "scripts\\evaluate_grasp_gated_trajectory_act.py",
        "grasp_gated_trajectory_chunk_bc",
        "grasp_gated_torch_act",
        "phase_weighted_torch_act_v1_candidate",
        "docs\\phase_weighted_torch_act_report.md",
        "docs\\phase_weighted_torch_act_report.csv",
        "outputs\\evaluations\\phase_weighted_torch_act_v1_candidate.json",
        "outputs\\videos\\phase_weighted_torch_act_v1_candidate_seed0.mp4",
        "scripts\\evaluate_phase_weighted_torch_act.py",
        "phase_weighted_torch_act",
        "contact_phase_gated_torch_act_v1_candidate",
        "outputs\\torch_act\\contact_phase_gated_torch_act_v1_candidate_20260721_003304.pt",
        "docs\\contact_phase_gated_torch_act_report.md",
        "docs\\contact_phase_gated_torch_act_report.csv",
        "outputs\\evaluations\\contact_phase_gated_torch_act_v1_candidate.json",
        "outputs\\videos\\contact_phase_gated_torch_act_v1_candidate_seed0.mp4",
        "scripts\\evaluate_contact_phase_gated_torch_act.py",
        "contact_aware_phase_gated_torch_act_v1_candidate",
        "outputs\\torch_act\\contact_aware_phase_gated_torch_act_v1_candidate_20260721_004944.pt",
        "docs\\contact_aware_phase_gated_torch_act_report.md",
        "docs\\contact_aware_phase_gated_torch_act_report.csv",
        "outputs\\evaluations\\contact_aware_phase_gated_torch_act_v1_candidate.json",
        "outputs\\videos\\contact_aware_phase_gated_torch_act_v1_candidate_seed0.mp4",
        "scripts\\evaluate_contact_aware_phase_gated_torch_act.py",
        "grasp_lift_subpolicy_probe_v1_candidate",
        "docs\\grasp_lift_subpolicy_probe_report.md",
        "docs\\grasp_lift_subpolicy_probe_report.csv",
        "outputs\\evaluations\\grasp_lift_subpolicy_probe_v1_candidate.json",
        "outputs\\videos\\grasp_lift_subpolicy_probe_v1_candidate_seed0.mp4",
        "scripts\\evaluate_grasp_lift_subpolicy_probe.py",
        "grasp_lift_subpolicy_probe",
        "打开 Grasp-gated trajectory-conditioned chunk BC 候选 viewer",
        "打开 Grasp-gated Torch ACT 候选 viewer",
        "docs\\grasp_gated_trajectory_knn_report.md",
        "docs\\grasp_gated_trajectory_knn_report.csv",
        "outputs\\evaluations\\grasp_gated_trajectory_knn_v1.json",
        "scripts\\run_grasp_gated_trajectory_knn_policy.py",
        "scripts\\evaluate_grasp_gated_trajectory_knn.py",
        "contact_stage_subpolicy_v1_candidate",
        "docs\\contact_stage_subpolicy_report.md",
        "docs\\contact_stage_subpolicy_report.csv",
        "outputs\\evaluations\\contact_stage_subpolicy_v1_candidate.json",
        "outputs\\videos\\contact_stage_subpolicy_v1_candidate_seed0.mp4",
        "scripts\\run_contact_stage_subpolicy.py",
        "scripts\\evaluate_contact_stage_subpolicy.py",
        "contact_stage_subpolicy",
        "contact_stage_demo_torch_act_v1_candidate",
        "data\\demos\\contact_stage_demo_place_blue_cube_blue_pad_medium_v1",
        "outputs\\torch_act\\contact_stage_demo_torch_act_v1_candidate_20260721_014152.pt",
        "docs\\contact_stage_demo_torch_act_report.md",
        "docs\\contact_stage_demo_torch_act_report.csv",
        "outputs\\evaluations\\contact_stage_demo_torch_act_v1_candidate.json",
        "outputs\\videos\\contact_stage_demo_torch_act_v1_candidate_seed0.mp4",
        "scripts\\collect_contact_stage_demos.py",
        "scripts\\evaluate_contact_stage_demo_torch_act.py",
        "contact_stage_phase_action_head_v1_candidate",
        "outputs\\phase_action_head\\contact_stage_phase_action_head_v1_candidate_20260721_020941.npz",
        "docs\\contact_stage_phase_action_head_report.md",
        "docs\\contact_stage_phase_action_head_report.csv",
        "outputs\\evaluations\\contact_stage_phase_action_head_v1_candidate.json",
        "outputs\\videos\\contact_stage_phase_action_head_v1_candidate_seed101.mp4",
        "scripts\\evaluate_contact_stage_phase_action_head.py",
        "contact_hold_weighted_torch_act_v1_candidate",
        "outputs\\torch_act\\contact_hold_weighted_torch_act_v1_candidate_20260721_023759.pt",
        "docs\\contact_hold_weighted_torch_act_report.md",
        "docs\\contact_hold_weighted_torch_act_report.csv",
        "outputs\\evaluations\\contact_hold_weighted_torch_act_v1_candidate.json",
        "outputs\\videos\\contact_hold_weighted_torch_act_v1_candidate_seed0.mp4",
        "scripts\\evaluate_contact_hold_weighted_torch_act.py",
        "timing_aware_trajectory_prior_residual_bc_v1_candidate",
        "outputs\\timing_aware_trajectory_prior_residual_bc\\timing_aware_trajectory_prior_residual_bc_v1_candidate_20260721_070442.npz",
        "docs\\timing_aware_trajectory_prior_residual_bc_report.md",
        "docs\\timing_aware_trajectory_prior_residual_bc_report.csv",
        "outputs\\evaluations\\timing_aware_trajectory_prior_residual_bc_v1_candidate.json",
        "outputs\\videos\\timing_aware_trajectory_prior_residual_bc_v1_candidate_seed0.mp4",
        "scripts\\train_timing_aware_trajectory_prior_residual_bc.py",
        "scripts\\run_timing_aware_trajectory_prior_residual_policy.py",
        "scripts\\evaluate_timing_aware_trajectory_prior_residual_bc.py",
        "timing_aware_trajectory_prior_residual_bc",
        "打开 Timing-aware trajectory-prior residual BC 候选 viewer",
        "preference_ranked_trajectory_post_training_v1_candidate",
        "outputs\\preference_post_training\\preference_ranked_trajectory_post_training_20260721_031024.npz",
        "docs\\preference_ranked_trajectory_post_training_report.md",
        "docs\\preference_ranked_trajectory_post_training_report.csv",
        "outputs\\evaluations\\preference_ranked_trajectory_post_training_v1_candidate.json",
        "outputs\\videos\\preference_ranked_trajectory_post_training_v1_candidate_seed0.mp4",
        "scripts\\evaluate_preference_ranked_trajectory_post_training.py",
        "outputs\\preference_post_training\\preference_trajectory_post_training_20260720_165005.npz",
        "docs\\preference_trajectory_post_training_report.md",
        "docs\\preference_trajectory_post_training_report.csv",
        "outputs\\evaluations\\preference_trajectory_post_training_v1.json",
        "scripts\\train_preference_trajectory_post_training.py",
        "scripts\\run_preference_trajectory_post_training_policy.py",
        "scripts\\evaluate_preference_trajectory_post_training.py",
        "Ranked-objective preference 候选补充命令",
        "preference_trajectory_post_training_v1_ranked_objective_candidate",
        "outputs\\preference_post_training\\preference_trajectory_post_training_v1_ranked_objective_candidate_20260721_073626.npz",
        "docs\\preference_trajectory_post_training_v1_ranked_objective_summary.md",
        "docs\\preference_trajectory_post_training_v1_ranked_objective_report.md",
        "docs\\preference_trajectory_post_training_v1_ranked_objective_report.csv",
        "outputs\\evaluations\\preference_trajectory_post_training_v1_ranked_objective_candidate.json",
        "outputs\\videos\\preference_trajectory_post_training_v1_ranked_objective_candidate_seed0.mp4",
        "preference_contact_aware_trajectory_post_training_v1_candidate",
        "outputs\\preference_post_training\\preference_contact_aware_trajectory_post_training_20260721_000449.npz",
        "docs\\preference_contact_aware_trajectory_post_training_report.md",
        "docs\\preference_contact_aware_trajectory_post_training_report.csv",
        "outputs\\evaluations\\preference_contact_aware_trajectory_post_training_v1_candidate.json",
        "outputs\\videos\\preference_contact_aware_trajectory_post_training_v1_candidate_seed0.mp4",
        "scripts\\evaluate_preference_contact_aware_trajectory_post_training.py",
        "docs\\control_safety_sweep.md",
        "docs\\control_safety_sweep.csv",
        "outputs\\evaluations\\control_safety_sweep_v1.json",
        "docs\\action_head_stage_report.md",
        "docs\\action_head_control_safety_sweep.md",
        "docs\\action_head_control_safety_sweep.csv",
        "outputs\\evaluations\\action_head_control_safety_sweep_v1.json",
        "docs\\stage_evidence_index.md",
        "docs\\domain_randomization_summary.md",
        "docs\\stage_reproduction_runbook.md",
        "docs\\stage_reproduction_runbook.csv",
        "docs\\defense_live_runbook.md",
        "docs\\defense_live_runbook.csv",
        "--target live-runbook",
        "docs\\defense_video_playlist.md",
        "docs\\defense_video_playlist.csv",
        "docs\\defense_video_playlist.html",
        "--target playlist",
        "docs\\defense_video_cue_sheet.md",
        "docs\\defense_video_cue_sheet.csv",
        "build_defense_video_cue_sheet.py",
        "--target cue-sheet",
        "docs\\method_comparison_dashboard.md",
        "docs\\method_comparison_dashboard.csv",
        "docs\\method_comparison_dashboard.html",
        "--target comparison",
        "docs\\thesis_visual_evidence_index.md",
        "docs\\thesis_visual_evidence_index.csv",
        "docs\\thesis_visual_evidence_index.html",
        "scripts\\build_thesis_visual_evidence_index.py",
        "--target visual-index",
        "docs\\defense_qa_playbook.md",
        "docs\\defense_qa_playbook.csv",
        "docs\\defense_qa_playbook.html",
        "scripts\\build_defense_qa_playbook.py",
        "--target qa",
        "docs\\version_lineage_index.md",
        "docs\\version_lineage_index.csv",
        "docs\\version_lineage_index.html",
        "scripts\\build_version_lineage_index.py",
        "--target lineage",
        "docs\\research_evidence_map.md",
        "docs\\research_question_showcase_plan.md",
        "docs\\research_question_showcase_plan.csv",
        "docs\\claim_evidence_traceability.md",
        "docs\\claim_evidence_traceability.csv",
        "docs\\claim_video_playback_index.md",
        "docs\\claim_video_playback_index.csv",
        "docs\\goal_completion_audit.md",
        "docs\\thesis_results_chapter_draft.md",
        "docs\\defense_slide_outline.md",
        "docs\\defense_deck.html",
        "docs\\presentation_video_pack.md",
        "docs\\video_evidence_index.md",
        "docs\\candidate_diagnostic_video_index.md",
        "docs\\video_quality_audit.md",
        "docs\\video_quality_audit.csv",
        "docs\\video_evidence_gallery.html",
        "docs\\video_presentation_storyboard.md",
        "docs\\video_presentation_storyboard.html",
        "docs\\failure_mode_taxonomy.md",
        "docs\\thesis_appendix_tables.md",
        "docs\\final_artifact_manifest.md",
        "docs\\openvla_dataset_bridge_report.md",
        "docs\\openvla_bridge_gallery.html",
        "docs\\openvla_feasibility_report.md",
        "docs\\next_experiment_registry.md",
        "docs\\next_experiment_registry.csv",
        "docs\\external_dependency_readiness_audit.md",
        "docs\\external_dependency_readiness_audit.csv",
        "outputs\\evaluations\\external_dependency_readiness_audit_v1.json",
        "build_external_dependency_readiness_audit.py",
        "export_openvla_dataset_bridge.py",
        "build_openvla_bridge_gallery.py",
        "build_next_experiment_registry.py",
        "showcase_launcher.py",
        "check_openvla_feasibility.py",
        "VLA_TORCH_PACKAGE_DIR",
        "build_video_evidence_gallery.py",
        "build_video_quality_audit.py",
        "build_failure_mode_taxonomy.py",
        "build_method_evidence_gate.py",
        "build_research_question_showcase_plan.py",
        "build_claim_evidence_traceability.py",
        "build_claim_video_playback_index.py",
        "build_method_comparison_dashboard.py",
        "build_thesis_visual_evidence_index.py",
        "build_defense_qa_playbook.py",
        "build_version_lineage_index.py",
        "build_task_bc_stage_report.py",
        "build_trajectory_act_stage_report.py",
        "build_trajectory_act_experiment_record.py",
        "打开 trajectory/ACT 中文实验台账",
        "build_trajectory_act_failure_diagnosis.py",
        "build_trajectory_act_conclusion_brief.py",
        "打开 Trajectory / ACT 论文结论摘要",
        "build_trajectory_act_slow_viewer_guide.py",
        "打开 Trajectory / ACT 超慢可视化指南",
        "--target trajectory-act-slow",
        "train_trajectory_phase_template_bc.py",
        "run_trajectory_phase_template_policy.py",
        "evaluate_trajectory_phase_template_bc.py",
        "run_grasp_gated_trajectory_knn_policy.py",
        "evaluate_grasp_gated_trajectory_knn.py",
        "train_preference_trajectory_post_training.py",
        "run_preference_trajectory_post_training_policy.py",
        "evaluate_preference_trajectory_post_training.py",
        "Ranked-fast preference 候选补充命令",
        "preference_trajectory_post_training_v1_ranked_fast_candidate",
        "docs\\preference_trajectory_post_training_v1_ranked_fast_summary.md",
        "docs\\preference_trajectory_post_training_v1_ranked_fast_report.md",
        "docs\\preference_trajectory_post_training_v1_ranked_fast_report.csv",
        "outputs\\evaluations\\preference_trajectory_post_training_v1_ranked_fast_candidate.json",
        "outputs\\preference_post_training\\preference_trajectory_post_training_v1_ranked_fast_candidate_20260721_075041.npz",
        "outputs\\videos\\preference_trajectory_post_training_v1_ranked_fast_candidate_seed0.mp4",
        "build_preference_post_training_ablation_matrix.py",
        "--target preference-ablation",
        "build_candidate_diagnostic_video_index.py",
        "evaluate_control_safety_sweep.py",
        "evaluate_action_head_control_safety_sweep.py",
        "build_action_head_stage_report.py",
        "build_stage_evidence_index.py",
        "build_stage_showcase_index.py",
        "build_stage_reproduction_runbook.py",
        "build_defense_live_runbook.py",
        "打开答辩现场展示 Runbook",
        "build_defense_video_playlist.py",
        "打开答辩视频播放清单",
        "打开答辩视频 Cue Sheet",
        "打开方法评测比较看板",
        "打开论文图表与视频证据索引",
        "打开答辩追问 Q&A Playbook",
        "打开实验版本谱系索引",
        "build_video_presentation_storyboard.py",
        "build_thesis_appendix_tables.py",
        "evaluate_domain_randomization.py",
        "build_isaac_domain_randomization_handoff.py",
        "docs\\isaac_domain_randomization_handoff.md",
        "docs\\isaac_domain_randomization_handoff.csv",
        "outputs\\evaluations\\isaac_domain_randomization_handoff_v1.json",
        "build_real_widowx_validation_handoff.py",
        "docs\\real_widowx_validation_handoff.md",
        "docs\\real_widowx_validation_handoff.csv",
        "outputs\\evaluations\\real_widowx_validation_handoff_v1.json",
        "outputs\\real_robot\\real_widowx_validation_v1_trial_template.csv",
        "domain_randomization_trajectory_knn_low_friction_v1",
        "domain_randomization_visual_act_cnn_cvae_low_friction_v1",
        "outputs/showcase/all_registered_methods_grid.mp4",
        "outputs/presentation_clips/00_defense_video_reel.mp4",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"command index is missing required terms: {missing}")
    final_method_required = (
        "docs\\final_method_version_index.md",
        "docs\\final_method_version_index.csv",
        "build_final_method_version_index.py",
        "打开最终方法版本索引",
    )
    missing_final_method = [item for item in final_method_required if item not in text]
    if missing_final_method:
        raise RuntimeError(f"command index is missing final method index terms: {missing_final_method}")
    remote_pack_required = (
        "build_robot_vla_remote_run_pack.py",
        "build_robot_vla_remote_result_intake.py",
        "docs\\robot_vla_remote_run_pack.md",
        "outputs\\evaluations\\robot_vla_remote_run_pack_v1.json",
        "outputs\\robot_vla_remote_run_pack\\robot_vla_remote_run_pack_v1.zip",
        "docs\\robot_vla_remote_result_intake.md",
        "outputs\\evaluations\\robot_vla_remote_result_intake_v1.json",
    )
    missing_remote_pack = [item for item in remote_pack_required if item not in text]
    if missing_remote_pack:
        raise RuntimeError(f"command index is missing Robot VLA remote run pack terms: {missing_remote_pack}")
    external_dependency_required = (
        "build_external_dependency_readiness_audit.py",
        "docs\\external_dependency_readiness_audit.md",
        "docs\\external_dependency_readiness_audit.csv",
        "outputs\\evaluations\\external_dependency_readiness_audit_v1.json",
    )
    missing_external_dependency = [item for item in external_dependency_required if item not in text]
    if missing_external_dependency:
        raise RuntimeError(f"command index is missing external dependency readiness terms: {missing_external_dependency}")
    missing_versions = [version for version in versions if f"`{version}`" not in text]
    if missing_versions:
        raise RuntimeError(f"command index is missing versions: {missing_versions}")
    if text.count("--viewer") < len(versions):
        raise RuntimeError("command index has too few viewer commands")


def verify_chinese_docs_are_clean(args: argparse.Namespace) -> list[str]:
    paths = [
        args.report,
        args.package,
        args.final_showcase_handoff,
        args.artifact_manifest,
        args.method_cards,
        args.method_comparison_dashboard,
        args.method_comparison_dashboard_html,
        args.thesis_visual_evidence,
        args.thesis_visual_evidence_html,
        args.defense_qa_playbook,
        args.defense_qa_playbook_html,
        args.version_lineage,
        args.version_lineage_html,
        args.storyboard,
        args.slide_outline,
        args.defense_deck,
        args.defense_live_runbook,
        args.defense_evidence_pack,
        args.resource_report,
        args.data_efficiency_report,
        args.domain_randomization_report,
        args.isaac_handoff_report,
        args.real_widowx_handoff_report,
        args.result_matrix,
        args.stage_comparison_report,
        args.task_bc_stage_report,
        args.trajectory_act_stage_report,
        args.trajectory_act_experiment_record,
        args.trajectory_act_diagnosis,
        args.trajectory_act_conclusion_brief,
        args.trajectory_act_slow_viewer_guide,
        args.final_defense_narrative,
        args.remaining_experiment_board,
        ROOT / "docs" / "phase_weighted_torch_act_report.md",
        ROOT / "docs" / "grasp_lift_subpolicy_probe_report.md",
        args.contact_stage_subpolicy_report,
        args.contact_stage_demo_torch_act_report,
        args.contact_stage_phase_action_head_report,
        args.contact_hold_weighted_torch_act_report,
        args.gripper_timing_contact_probe_report,
        args.trajectory_phase_template_report,
        args.grasp_gated_trajectory_knn_report,
        args.preference_trajectory_post_training_report,
        args.preference_contact_aware_trajectory_post_training_report,
        args.preference_ranked_trajectory_post_training_report,
        args.preference_post_training_upgrade_gate,
        args.preference_post_training_ablation,
        args.control_safety_sweep,
        args.action_head_stage_report,
        args.action_head_control_safety_sweep,
        args.strict_grasp_audit,
        args.stage_evidence_index,
        args.stage_showcase_index,
        args.stage_showcase_html,
        args.stage_reproduction_runbook,
        args.video_presentation_storyboard,
        args.video_presentation_storyboard_html,
        args.defense_video_playlist,
        args.defense_video_playlist_html,
        args.defense_video_cue_sheet,
        args.research_evidence_map,
        args.research_showcase_plan,
        args.claim_evidence_traceability,
        args.claim_video_playback_index,
        args.goal_completion_audit,
        args.method_stage_audit,
        args.method_evidence_gate,
        args.version_naming_spec,
        args.version_naming_spec_csv,
        args.final_method_index,
        args.thesis_results_chapter,
        args.thesis_appendix,
        args.runtime_report,
        args.next_phase,
        args.next_experiment_registry,
        args.external_dependency_readiness_audit,
        args.openvla_bridge_report,
        args.openvla_bridge_gallery,
        args.openvla_feasibility_report,
        args.robot_vla_remote_pack_report,
        args.robot_vla_remote_intake_report,
        args.command_index,
        args.presentation_pack_doc,
        args.video_evidence_index,
        args.candidate_diagnostic_video_index,
        args.video_quality_audit,
        args.video_evidence_gallery,
        args.failure_mode_taxonomy,
        args.showcase_launcher_guide,
        ROOT / "docs" / "final_closure_audit_v1.md",
        ROOT / "docs" / "experiment_log.md",
        ROOT / "docs" / "known_issues.md",
        ROOT / "docs" / "video_clips.md",
    ]
    checked = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8-sig")
        if "???" in text:
            raise RuntimeError(f"unresolved mojibake/question-mark text remains in {path}")
        markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
        if markers:
            raise RuntimeError(f"possible mojibake markers remain in {path}: {markers}")
        checked.append(path.as_posix())
    return checked


def verify_final_closure_audit() -> dict:
    report_path = ROOT / "docs" / "final_closure_audit_v1.md"
    json_path = ROOT / "outputs" / "evaluations" / "final_closure_audit_v1.json"
    if not report_path.exists() or not json_path.exists():
        raise FileNotFoundError("final closure audit artifacts are missing")
    payload = read_json(json_path)
    if payload.get("version") != "final_closure_audit_v1":
        raise RuntimeError("unexpected final closure audit version")
    pooled = payload.get("v4_replication", {}).get("pooled_descriptive", {})
    if (pooled.get("successes"), pooled.get("episodes"), pooled.get("semantic_correct"), pooled.get("visual_selection_correct")) != (278, 288, 288, 288):
        raise RuntimeError("final closure V4 replication totals are inconsistent")
    monitor = payload.get("rejected_candidates", {}).get("contact_monitor_early_regrasp", {})
    if (monitor.get("v4_success"), monitor.get("candidate_success"), monitor.get("paired_improved"), monitor.get("paired_regressed")) != ([143, 144], [127, 144], 1, 17):
        raise RuntimeError("final closure monitor rejection totals are inconsistent")
    counterfactual = payload.get("rejected_candidates", {}).get("same_state_early_deep_regrasp", {})
    if (counterfactual.get("continue_better"), counterfactual.get("early_better"), counterfactual.get("training_allowed")) != (47, 0, False):
        raise RuntimeError("final closure counterfactual gate is inconsistent")
    if payload.get("closure", {}).get("scope_closed") is not True:
        raise RuntimeError("final closure audit does not close the MuJoCo scope")
    missing = [path for path in payload.get("evidence_paths", {}).values() if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError(f"final closure evidence paths are missing: {missing}")
    required_terms = ("最终推荐", "拒绝证据", "278/288", "不训练选择器")
    report = report_path.read_text(encoding="utf-8-sig")
    absent = [term for term in required_terms if term not in report]
    if absent:
        raise RuntimeError(f"final closure report is missing terms: {absent}")
    return payload


def main() -> None:
    args = parse_args()
    versions, registered_videos = verify_registered_methods(args)
    language_versions = verify_language(args)
    resource_versions = verify_resources(args)
    data_efficiency_rows = verify_data_efficiency(args)
    domain_randomization_rows = verify_domain_randomization(args)
    isaac_handoff_rows = verify_isaac_domain_randomization_handoff(args)
    real_widowx_handoff_rows = verify_real_widowx_validation_handoff(args)
    extra_videos = verify_extra_videos(args)
    showcase_videos = verify_showcase(args)
    presentation_videos = verify_presentation_pack(args)
    verify_video_presentation_storyboard(args)
    defense_video_playlist_rows = verify_defense_video_playlist(args)
    defense_video_cue_sheet_rows = verify_defense_video_cue_sheet(args, defense_video_playlist_rows)
    video_evidence = verify_video_evidence_index(args, versions)
    candidate_video_rows = verify_candidate_diagnostic_video_index(args)
    verify_video_evidence_gallery(args, len(video_evidence))
    video_quality_rows = verify_video_quality_audit(args, len(video_evidence))
    failure_mode_rows = verify_failure_mode_taxonomy(args, len(video_evidence))
    figures = verify_figures(args)
    verify_method_cards(args, versions)
    verify_result_matrix(args, versions)
    method_comparison_rows = verify_method_comparison_dashboard(args, versions)
    core_task_comparison_rows = verify_core_task_comparison_matrix(args)
    core_v2_comparison_rows = verify_core_v2_comparison_matrix(args)
    core_v2_pretrained_vlm_rows = verify_core_v2_pretrained_vlm_report(args)
    core_v2_clip_semantic_rows = verify_core_v2_clip_semantic_waypoint(args)
    core_v2_clip_semantic_efficiency_rows = verify_core_v2_clip_semantic_data_efficiency(args)
    core_v2_clip_semantic_ood_rows = verify_core_v2_clip_semantic_ood_generalization(args)
    kaggle_clip_semantic_episodes = verify_kaggle_clip_semantic_adapter()
    frozen_clip_semantic_comparison_rows = verify_frozen_clip_semantic_adapter_comparison()
    thesis_visual_evidence_rows = verify_thesis_visual_evidence_index(args)
    defense_qa_rows = verify_defense_qa_playbook(args)
    version_lineage_rows = verify_version_lineage_index(args, versions)
    verify_method_stage_audit(args, versions)
    method_evidence_rows = verify_method_evidence_gate(args, versions)
    version_naming_rows = verify_version_naming_and_gate_spec(args)
    final_method_rows = verify_final_method_version_index(args, versions)
    verify_stage_comparison_report(args, versions)
    task_bc_rows = verify_task_bc_stage_report(args)
    trajectory_act_rows = verify_trajectory_act_stage_report(args)
    trajectory_act_record_rows = verify_trajectory_act_experiment_record(args)
    trajectory_act_diagnosis_rows = verify_trajectory_act_failure_diagnosis(args)
    trajectory_act_conclusion_rows = verify_trajectory_act_conclusion_brief(args)
    trajectory_act_slow_viewer_rows = verify_trajectory_act_slow_viewer_guide(args)
    final_defense_narrative_rows = verify_final_defense_narrative(args)
    remaining_experiment_board_rows = verify_remaining_experiment_board(args)
    trajectory_phase_template_rows = verify_trajectory_phase_template_bc(args)
    grasp_gated_trajectory_knn_rows = verify_grasp_gated_trajectory_knn(args)
    preference_trajectory_post_training_rows = verify_preference_trajectory_post_training(args)
    preference_ranked_objective_rows = verify_preference_ranked_objective(args)
    preference_ranked_fast_rows = verify_preference_ranked_fast(args)
    preference_contact_aware_trajectory_post_training_rows = verify_preference_contact_aware_trajectory_post_training(args)
    preference_ranked_trajectory_post_training_rows = verify_preference_ranked_trajectory_post_training(args)
    preference_upgrade_gate_rows = verify_preference_post_training_upgrade_gate(args)
    preference_ablation_rows = verify_preference_post_training_ablation(args)
    contact_phase_gated_torch_act_rows = verify_contact_phase_gated_torch_act(args)
    contact_aware_phase_gated_torch_act_rows = verify_contact_aware_phase_gated_torch_act(args)
    contact_stage_subpolicy_rows = verify_contact_stage_subpolicy(args)
    contact_stage_demo_torch_act_rows = verify_contact_stage_demo_torch_act(args)
    contact_stage_phase_action_head_rows = verify_contact_stage_phase_action_head(args)
    contact_hold_weighted_torch_act_rows = verify_contact_hold_weighted_torch_act(args)
    timing_aware_trajectory_prior_residual_rows = verify_timing_aware_trajectory_prior_residual(args)
    gripper_timing_contact_probe_rows = verify_gripper_timing_contact_probe(args)
    control_safety_rows = verify_control_safety_sweep(args)
    action_head_rows = verify_action_head_stage_report(args)
    action_head_control_safety_rows = verify_action_head_control_safety_sweep(args)
    strict_grasp_rows = verify_strict_grasp_success_audit(args)
    stage_evidence_rows = verify_stage_evidence_index(args)
    verify_stage_showcase_index(args, versions)
    stage_reproduction_rows = verify_stage_reproduction_runbook(args)
    research_evidence_rows = verify_research_evidence_map(args)
    research_showcase_rows = verify_research_question_showcase_plan(args)
    claim_evidence_rows = verify_claim_evidence_traceability(args)
    claim_video_playback_rows = verify_claim_video_playback_index(args)
    goal_completion_rows = verify_goal_completion_audit(args)
    defense_live_rows = verify_defense_live_runbook(args)
    verify_thesis_results_chapter(args, versions)
    verify_thesis_appendix_tables(args, versions)
    verify_report(args)
    verify_package(args)
    final_showcase_handoff_rows = verify_final_showcase_handoff(args)
    defense_evidence_pack = verify_defense_evidence_pack(args)
    verify_final_artifact_manifest(args, versions)
    verify_dashboard(args)
    verify_storyboard(args, versions)
    verify_slide_outline(args)
    verify_defense_deck(args)
    verify_runtime_capability(args)
    verify_next_phase(args)
    next_experiment_rows = verify_next_experiment_registry(args, versions)
    external_dependency_rows = verify_external_dependency_readiness_audit(args)
    openvla_bridge_samples = verify_openvla_dataset_bridge(args)
    verify_openvla_bridge_gallery(args, openvla_bridge_samples)
    rlds_source_steps = verify_widowx_mujoco_rlds_source()
    verify_openvla_feasibility(args)
    verify_robot_vla_handoff(args)
    verify_robot_vla_remote_run_pack(args)
    robot_vla_remote_intake_rows = verify_robot_vla_remote_result_intake(args)
    showcase_launcher_checks = verify_showcase_launcher(args)
    verify_command_index(args, versions)
    final_closure = verify_final_closure_audit()
    clean_docs = verify_chinese_docs_are_clean(args)

    print(f"methods_ok: {len(versions)}", flush=True)
    print(f"language_rows_ok: {len(language_versions)}", flush=True)
    print(f"resource_rows_ok: {len(resource_versions)}", flush=True)
    print(f"data_efficiency_rows_ok: {len(data_efficiency_rows)}", flush=True)
    print(f"domain_randomization_rows_ok: {len(domain_randomization_rows)}", flush=True)
    print(f"isaac_domain_randomization_handoff_ok: {len(isaac_handoff_rows)}", flush=True)
    print(f"real_widowx_validation_handoff_ok: {len(real_widowx_handoff_rows)}", flush=True)
    print(f"registered_videos_ok: {len(registered_videos)}", flush=True)
    print(f"extra_videos_ok: {len(extra_videos)}", flush=True)
    print(f"showcase_videos_ok: {len(showcase_videos)}", flush=True)
    print(f"presentation_pack_ok: {len(presentation_videos)}", flush=True)
    print("video_presentation_storyboard_ok: 1", flush=True)
    print(f"defense_video_playlist_ok: {len(defense_video_playlist_rows)}", flush=True)
    print(f"defense_video_cue_sheet_ok: {len(defense_video_cue_sheet_rows)}", flush=True)
    print(f"video_evidence_ok: {len(video_evidence)}", flush=True)
    print(f"candidate_diagnostic_video_ok: {len(candidate_video_rows)}", flush=True)
    print("video_evidence_gallery_ok: 1", flush=True)
    print(f"video_quality_audit_ok: {len(video_quality_rows)}", flush=True)
    print(f"failure_mode_taxonomy_ok: {len(failure_mode_rows)}", flush=True)
    print(f"figures_ok: {len(figures)}", flush=True)
    print("method_cards_ok: 1", flush=True)
    print("result_matrix_ok: 1", flush=True)
    print(f"method_comparison_dashboard_ok: {len(method_comparison_rows)}", flush=True)
    print(f"core_task_comparison_ok: {len(core_task_comparison_rows)}", flush=True)
    print(f"core_v2_holdout_comparison_ok: {len(core_v2_comparison_rows)}", flush=True)
    print(f"core_v2_pretrained_vlm_ok: {len(core_v2_pretrained_vlm_rows)}", flush=True)
    print(f"core_v2_clip_semantic_waypoint_ok: {len(core_v2_clip_semantic_rows)}", flush=True)
    print(f"core_v2_clip_semantic_data_efficiency_ok: {len(core_v2_clip_semantic_efficiency_rows)}", flush=True)
    print(f"core_v2_clip_semantic_ood_generalization_ok: {len(core_v2_clip_semantic_ood_rows)}", flush=True)
    print(f"kaggle_clip_semantic_adapter_ok: {kaggle_clip_semantic_episodes}", flush=True)
    print(f"frozen_clip_semantic_adapter_comparison_ok: {frozen_clip_semantic_comparison_rows}", flush=True)
    print(f"thesis_visual_evidence_index_ok: {len(thesis_visual_evidence_rows)}", flush=True)
    print(f"defense_qa_playbook_ok: {len(defense_qa_rows)}", flush=True)
    print(f"version_lineage_index_ok: {len(version_lineage_rows)}", flush=True)
    print("method_stage_audit_ok: 1", flush=True)
    print(f"method_evidence_gate_ok: {len(method_evidence_rows)}", flush=True)
    print(f"version_naming_spec_ok: {len(version_naming_rows)}", flush=True)
    print(f"final_method_version_index_ok: {len(final_method_rows)}", flush=True)
    print("stage_comparison_ok: 1", flush=True)
    print(f"task_bc_stage_ok: {len(task_bc_rows)}", flush=True)
    print(f"trajectory_act_stage_ok: {len(trajectory_act_rows)}", flush=True)
    print(f"trajectory_act_experiment_record_ok: {len(trajectory_act_record_rows)}", flush=True)
    print(f"trajectory_act_failure_diagnosis_ok: {len(trajectory_act_diagnosis_rows)}", flush=True)
    print(f"trajectory_act_conclusion_brief_ok: {len(trajectory_act_conclusion_rows)}", flush=True)
    print(f"trajectory_act_slow_viewer_guide_ok: {len(trajectory_act_slow_viewer_rows)}", flush=True)
    print(f"final_defense_narrative_ok: {len(final_defense_narrative_rows)}", flush=True)
    print(f"remaining_experiment_board_ok: {len(remaining_experiment_board_rows)}", flush=True)
    print(f"trajectory_phase_template_bc_ok: {len(trajectory_phase_template_rows)}", flush=True)
    print(f"grasp_gated_trajectory_knn_ok: {len(grasp_gated_trajectory_knn_rows)}", flush=True)
    print(f"preference_trajectory_post_training_ok: {len(preference_trajectory_post_training_rows)}", flush=True)
    print(f"preference_ranked_objective_ok: {len(preference_ranked_objective_rows)}", flush=True)
    print(f"preference_ranked_fast_ok: {len(preference_ranked_fast_rows)}", flush=True)
    print(f"preference_contact_aware_trajectory_post_training_ok: {len(preference_contact_aware_trajectory_post_training_rows)}", flush=True)
    print(f"preference_ranked_trajectory_post_training_ok: {len(preference_ranked_trajectory_post_training_rows)}", flush=True)
    print(f"preference_post_training_upgrade_gate_ok: {len(preference_upgrade_gate_rows)}", flush=True)
    print(f"preference_post_training_ablation_ok: {len(preference_ablation_rows)}", flush=True)
    print(f"contact_phase_gated_torch_act_ok: {len(contact_phase_gated_torch_act_rows)}", flush=True)
    print(f"contact_aware_phase_gated_torch_act_ok: {len(contact_aware_phase_gated_torch_act_rows)}", flush=True)
    print(f"contact_stage_subpolicy_ok: {len(contact_stage_subpolicy_rows)}", flush=True)
    print(f"contact_stage_demo_torch_act_ok: {len(contact_stage_demo_torch_act_rows)}", flush=True)
    print(f"contact_stage_phase_action_head_ok: {len(contact_stage_phase_action_head_rows)}", flush=True)
    print(f"contact_hold_weighted_torch_act_ok: {len(contact_hold_weighted_torch_act_rows)}", flush=True)
    print(f"timing_aware_trajectory_prior_residual_ok: {len(timing_aware_trajectory_prior_residual_rows)}", flush=True)
    print(f"gripper_timing_contact_probe_ok: {len(gripper_timing_contact_probe_rows)}", flush=True)
    print(f"control_safety_sweep_ok: {len(control_safety_rows)}", flush=True)
    print(f"action_head_stage_ok: {len(action_head_rows)}", flush=True)
    print(f"action_head_control_safety_sweep_ok: {len(action_head_control_safety_rows)}", flush=True)
    print(f"strict_grasp_audit_ok: {len(strict_grasp_rows)}", flush=True)
    print(f"stage_evidence_index_ok: {len(stage_evidence_rows)}", flush=True)
    print("stage_showcase_index_ok: 1", flush=True)
    print(f"stage_reproduction_runbook_ok: {len(stage_reproduction_rows)}", flush=True)
    print(f"research_evidence_ok: {len(research_evidence_rows)}", flush=True)
    print(f"research_question_showcase_plan_ok: {len(research_showcase_rows)}", flush=True)
    print(f"claim_evidence_traceability_ok: {len(claim_evidence_rows)}", flush=True)
    print(f"claim_video_playback_index_ok: {len(claim_video_playback_rows)}", flush=True)
    print(f"goal_completion_ok: {len(goal_completion_rows)}", flush=True)
    print(f"defense_live_runbook_ok: {len(defense_live_rows)}", flush=True)
    print("thesis_results_chapter_ok: 1", flush=True)
    print("thesis_appendix_tables_ok: 1", flush=True)
    print(f"final_showcase_handoff_ok: {len(final_showcase_handoff_rows)}", flush=True)
    print(f"defense_evidence_pack_ok: {defense_evidence_pack['file_count']}", flush=True)
    print("final_manifest_ok: 1", flush=True)
    print("storyboard_ok: 1", flush=True)
    print("slide_outline_ok: 1", flush=True)
    print("defense_deck_ok: 1", flush=True)
    print("runtime_capability_ok: 1", flush=True)
    print("next_phase_ok: 1", flush=True)
    print(f"next_experiment_registry_ok: {len(next_experiment_rows)}", flush=True)
    print(f"external_dependency_readiness_ok: {len(external_dependency_rows)}", flush=True)
    print(f"openvla_bridge_ok: {openvla_bridge_samples}", flush=True)
    print("openvla_bridge_gallery_ok: 1", flush=True)
    print(f"widowx_mujoco_rlds_source_ok: {rlds_source_steps}", flush=True)
    print("openvla_feasibility_ok: 1", flush=True)
    print("robot_vla_handoff_ok: 1", flush=True)
    print("robot_vla_remote_run_pack_ok: 1", flush=True)
    print(f"robot_vla_remote_result_intake_ok: {len(robot_vla_remote_intake_rows)}", flush=True)
    print(f"showcase_launcher_ok: {len(showcase_launcher_checks)}", flush=True)
    print(f"final_closure_audit_ok: {final_closure['v4_replication']['pooled_descriptive']['successes']}/{final_closure['v4_replication']['pooled_descriptive']['episodes']}", flush=True)
    print("command_index_ok: 1", flush=True)
    print(f"chinese_docs_clean_ok: {len(clean_docs)}", flush=True)
    print(f"package_ok: {args.package}", flush=True)
    print(f"dashboard_ok: {args.dashboard}", flush=True)
    print("verified_versions:", ", ".join(versions), flush=True)


if __name__ == "__main__":
    main()
