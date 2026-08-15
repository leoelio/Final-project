from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def condition_total(payload: dict, condition: str) -> tuple[int, int]:
    rows = [row for row in payload["rows"] if row["condition"] == condition]
    return sum(int(row["task_success"]) for row in rows), len(rows)


def main() -> None:
    page = ROOT / "docs" / "mujoco_research_summary_zh_en.html"
    summary = ROOT / "docs" / "mujoco_research_summary_zh_en.json"
    normalized_ood = ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_normalized_v1.json"
    alias_ood = ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_ood_alias_v1.json"
    independent_syntax = ROOT / "outputs" / "evaluations" / "core_v2_clip_semantic_independent_syntax_v1.json"
    contact_fusion = ROOT / "outputs" / "evaluations" / "clip_semantic_contact_fusion_v1.json"
    video = ROOT / "outputs" / "videos" / "clip_semantic_waypoint_core_v2_normalized_v1_azure_red_seed700.mp4"
    video_metadata = video.with_suffix(".json")
    fusion_video = ROOT / "outputs" / "videos" / "clip_semantic_contact_fusion_low_friction_red_red_seed3102.mp4"
    fusion_video_metadata = fusion_video.with_suffix(".json")
    task_success_videos = {
        "place_blue_cube_blue_pad": ROOT / "outputs" / "videos" / "clip_semantic_waypoint_core_v2_normalized_v1_blue_blue_seed3300.mp4",
        "place_red_cube_red_pad": ROOT / "outputs" / "videos" / "clip_semantic_waypoint_core_v2_normalized_v1_red_red_seed3300.mp4",
        "move_leftmost_cube_to_bowl": ROOT / "outputs" / "videos" / "clip_semantic_waypoint_core_v2_normalized_v1_leftmost_bowl_seed3300.mp4",
    }
    for path in (page, summary, normalized_ood, alias_ood, independent_syntax, contact_fusion, video, video_metadata, fusion_video, fusion_video_metadata, *task_success_videos.values()):
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".mp4" and path in task_success_videos.values() and not path.with_suffix(".json").exists():
            raise FileNotFoundError(path.with_suffix(".json"))

    core_suffixes = ("blue_to_blue", "blue_to_red", "red_to_red", "leftmost_cube")
    canonical_successes = 0
    canonical_episodes = 0
    for suffix in core_suffixes:
        payload = read_json(ROOT / "outputs" / "evaluations" / f"core_v2_clip_semantic_normalized_{suffix}.json")
        if payload["success"] != "5/5" or payload["strict_grasp_success"] != "5/5":
            raise RuntimeError(f"normalized canonical result is not strict 5/5: {suffix}")
        canonical_successes += 5
        canonical_episodes += 5

    normalized = read_json(normalized_ood)
    alias = read_json(alias_ood)
    if condition_total(normalized, "paraphrase") != (60, 60):
        raise RuntimeError("normalized paraphrase result is not 60/60")
    if condition_total(normalized, "hard_distractors") != (20, 20):
        raise RuntimeError("normalized hard-distractor result is not 20/20")
    if condition_total(alias, "paraphrase") != (48, 60):
        raise RuntimeError("alias-training negative result is not 48/60")
    if condition_total(alias, "hard_distractors") != (16, 20):
        raise RuntimeError("alias-training negative result is not 16/20")
    independent = read_json(independent_syntax)
    if independent["task_success"] != "40/40" or independent["strict_grasp_success"] != "40/40":
        raise RuntimeError("independent closed-vocabulary syntax result is not 40/40")

    data = read_json(summary)
    primary = data["primary"]
    if data["version"] != "mujoco_research_summary_v1":
        raise RuntimeError("unexpected page data version")
    if (primary["canonical"]["successes"], primary["canonical"]["episodes"]) != (canonical_successes, canonical_episodes):
        raise RuntimeError("page data canonical result is stale")
    if primary["ood"]["normalized"]["paraphrase"]["successes"] != 60:
        raise RuntimeError("page data normalized paraphrase result is stale")
    if (primary["independent_syntax"]["successes"], primary["independent_syntax"]["episodes"]) != (40, 40):
        raise RuntimeError("page data independent syntax result is stale")
    fusion = data["contact_fusion"]
    if fusion["stress"]["standard_success"] != "30/40" or fusion["stress"]["fusion_success"] != "36/40":
        raise RuntimeError("page data contact-fusion stress result is stale")
    if fusion["stress"]["fusion_sustained_transport_successes"] != 33:
        raise RuntimeError("page data contact-fusion sustained-transport metric is stale")
    if fusion["nominal"]["standard_success"] != "20/20" or fusion["nominal"]["fusion_success"] != "20/20":
        raise RuntimeError("page data contact-fusion nominal result is stale")

    video_data = read_json(video_metadata)
    result = video_data["summary"]
    if not result["task_success"] or not result["strict_grasp_success"] or result["predicted_intent"] != "place_blue_cube_red_pad":
        raise RuntimeError("normalized success video metadata is inconsistent")
    fusion_result = read_json(fusion_video_metadata)["summary"]
    if not fusion_result["task_success"] or fusion_result["contact_regrasp_attempts"] != 1 or fusion_result["contact_recovery_reason"] != "transport":
        raise RuntimeError("contact-fusion recovery video metadata is inconsistent")
    for task, success_video in task_success_videos.items():
        task_video_result = read_json(success_video.with_suffix(".json"))["summary"]
        if task_video_result["task"] != task or not task_video_result["task_success"] or not task_video_result["strict_grasp_success"]:
            raise RuntimeError(f"task-success video metadata is inconsistent: {task}")

    html = page.read_text(encoding="utf-8")
    for required in ("zh-button", "en-button", "#en", "clip_semantic_waypoint_core_v2_normalized_v1_azure_red_seed700.mp4", "训练时语义改写增强", "rejected"):
        if required not in html:
            raise RuntimeError(f"bilingual page is missing: {required}")

    print(f"mujoco_summary_page_ok: {page}")
    print(f"canonical_ok: {canonical_successes}/{canonical_episodes}")
    print("normalized_repair_ok: original_paraphrase=60/60 hard_distractors=20/20")
    print("independent_syntax_ok: 40/40 under the fixed closed vocabulary")
    print("negative_candidate_ok: paraphrase=48/60 hard_distractors=16/20")
    print("contact_fusion_ok: low_friction=30/40->36/40 nominal=20/20->20/20")
    print("task_success_videos_ok: blue_blue, red_red, leftmost_bowl")


if __name__ == "__main__":
    main()
