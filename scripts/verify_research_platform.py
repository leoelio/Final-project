from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(base_url: str, path: str, payload: dict | None = None) -> tuple[int, bytes, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(req, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def get_json(base_url: str, path: str) -> dict:
    status, body, _ = request(base_url, path)
    assert status == 200, f"GET {path}: HTTP {status}"
    return json.loads(body.decode("utf-8"))


def post_json(base_url: str, path: str, payload: dict) -> tuple[int, dict]:
    status, body, _ = request(base_url, path, payload)
    return status, json.loads(body.decode("utf-8"))


def wait_for_status(base_url: str, expected: set[str], timeout: float = 8.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        simulation = get_json(base_url, "/api/status")["simulation"]
        if simulation["status"] in expected:
            return simulation
        time.sleep(0.1)
    raise AssertionError(f"simulation did not reach {sorted(expected)}")


def wait_for_benchmark(base_url: str, expected: set[str], timeout: float = 70.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        benchmark = get_json(base_url, "/api/status")["benchmark"]
        if benchmark["status"] in expected:
            return benchmark
        time.sleep(0.2)
    raise AssertionError(f"benchmark did not reach {sorted(expected)}")


def wait_for_adaptation(base_url: str, expected: set[str], timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        adaptation = get_json(base_url, "/api/status")["adaptation"]
        if adaptation["status"] in expected:
            return adaptation
        time.sleep(0.2)
    raise AssertionError(f"adaptation did not reach {sorted(expected)}")


def verify(
    base_url: str,
    exercise_controls: bool,
    exercise_benchmark: bool,
    exercise_governance: bool,
    exercise_release: bool,
    exercise_adaptation: bool,
    exercise_arena: bool,
) -> None:
    config = get_json(base_url, "/api/config")
    assert config["platform_version"] == "3.7"
    assert config["governance_framework"] == "TRACE-1.0"
    assert config["portfolio_framework"] == "TRACE-PORTFOLIO-1.0"
    assert config["release_framework"] == "TRACE-RELEASE-1.0"
    assert config["adaptation_framework"] == "LOCAL-ADAPT-1.2"
    assert config["arena_framework"] == "PAIR-OPT-1.0"
    assert len(config["tasks"]) >= 4, "expected at least four validated tasks"
    assert all(item["available"] for item in config["datasets"].values()), "a configured dataset is missing"
    assert set(config["trainers"]) == {"mlp_bc", "action_chunk", "diffusion_lite"}

    status = get_json(base_url, "/api/status")
    assert {"simulation", "training", "adaptation", "benchmark", "analytics"} <= status.keys()

    health = get_json(base_url, "/api/health")
    assert health["status"] == "healthy"
    assert all(health["assets"].values()) and all(health["datasets"].values())
    assert health["adaptation"]["framework"] == "LOCAL-ADAPT-1.2"
    assert health["adaptation"]["arena_framework"] == "PAIR-OPT-1.0"
    assert health["adaptation"]["base_action_head"] is True

    adaptation = get_json(base_url, "/api/adaptation")
    assert adaptation["framework"] == "LOCAL-ADAPT-1.2"
    assert adaptation["arena_framework"] == "PAIR-OPT-1.0"
    assert set(adaptation["methods"]) == {"local_lora", "local_adapter", "micro_head", "registry_rgb_skill"}
    assert set(adaptation["profiles"]) == {"eco", "balanced", "research"}
    assert adaptation["hardware"]["gpu_required"] is False
    if adaptation.get("arena_efficiency"):
        efficiency = adaptation["arena_efficiency"]
        assert efficiency["fresh_run_id"] != efficiency["cached_run_id"]
        assert isinstance(efficiency["reduction_percent"], float)
        assert "observed local measurement" in efficiency["boundary"]
    if adaptation.get("latest_arena"):
        latest_arena = adaptation["latest_arena"]
        assert latest_arena["run_id"].startswith("opt-")
        assert latest_arena["metrics"]["paired_summary"]["matched_seed_sets"] is True
        assert len(latest_arena["metrics"]["candidate_results"]) >= 2
    invalid_task_status, invalid_task = post_json(base_url, "/api/adaptation/tasks", {})
    assert invalid_task_status == 400 and "error" in invalid_task
    invalid_estimate_status, invalid_estimate = post_json(base_url, "/api/adaptation/estimate", {})
    assert invalid_estimate_status == 400 and "error" in invalid_estimate
    invalid_arena_status, invalid_arena = post_json(base_url, "/api/adaptation/arena/estimate", {})
    assert invalid_arena_status == 400 and "error" in invalid_arena
    if adaptation["tasks"]:
        task_id = adaptation["tasks"][0]["task_id"]
        preview_status, preview_body, _ = request(base_url, f"/api/adaptation/tasks/{task_id}/preview.png?seed=0")
        assert preview_status == 200 and preview_body.startswith(b"\x89PNG\r\n\x1a\n")
        estimate_status, estimate = post_json(base_url, "/api/adaptation/estimate", {
            "task_id": task_id,
            "method": "local_lora",
            "profile": "eco",
            "episodes": 2,
            "epochs": 1,
            "viewer": False,
        })
        assert estimate_status == 200 and estimate["gate"]["passed"] is True
        assert estimate["trainable_params"] == 526 and estimate["gpu_required"] is False
        skill_estimate_status, skill_estimate = post_json(base_url, "/api/adaptation/estimate", {
            "task_id": task_id,
            "method": "registry_rgb_skill",
            "profile": "eco",
            "episodes": 2,
            "epochs": 1,
            "evaluation_episodes": 3,
            "viewer": False,
        })
        assert skill_estimate_status == 200 and skill_estimate["gate"]["passed"] is True
        assert skill_estimate["trainable_params"] == 0 and skill_estimate["evaluation_episodes"] == 3
        assert skill_estimate["viewer_overhead_mb"] == 0 and "RGB" in skill_estimate["truth_boundary"]
        arena_estimate_status, arena_estimate = post_json(base_url, "/api/adaptation/arena/estimate", {
            "task_id": task_id,
            "methods": ["local_lora", "registry_rgb_skill"],
            "profile": "eco",
            "episodes": 2,
            "epochs": 1,
            "evaluation_episodes": 3,
            "seed": 2400,
            "viewer": False,
            "reuse_dataset": True,
        })
        assert arena_estimate_status == 200 and arena_estimate["gate"]["passed"] is True
        assert arena_estimate["framework"] == "PAIR-OPT-1.0"
        assert arena_estimate["methods"] == ["local_lora", "registry_rgb_skill"]
        assert arena_estimate["evaluation_seeds"] == [3402, 3403, 3404]
        assert len(arena_estimate["dataset_fingerprint"]) == 64
        assert len(arena_estimate["candidate_estimates"]) == 2
        assert arena_estimate["sequential_peak_ram_mb"] == max(
            item["estimated_peak_ram_mb"] for item in arena_estimate["candidate_estimates"]
        )
        one_candidate_status, one_candidate = post_json(base_url, "/api/adaptation/arena/estimate", {
            "task_id": task_id,
            "methods": ["local_lora"],
            "episodes": 2,
            "epochs": 1,
        })
        assert one_candidate_status == 400 and "error" in one_candidate

    promotion_portfolio = get_json(base_url, "/api/adaptation/portfolio")
    assert promotion_portfolio["framework"] == "RESOURCE-PARETO-1.0"
    assert promotion_portfolio["promotion_thresholds"]["minimum_episodes"] == 3
    assert "identical held-out seed sets" in promotion_portfolio["comparison_boundary"]

    runs = get_json(base_url, "/api/runs?limit=10")
    assert {"runs", "analytics"} <= runs.keys()
    assert {"optimizations", "completed_optimizations", "adaptation_candidates"} <= runs["analytics"].keys()

    studies = get_json(base_url, "/api/studies")
    assert studies["summary"]["framework"] == "TRACE-1.0" and isinstance(studies["studies"], list)
    invalid_study_status, invalid_study = post_json(base_url, "/api/studies", {})
    assert invalid_study_status == 400 and "error" in invalid_study
    if studies["studies"]:
        study_id = studies["studies"][0]["study_id"]
        study = get_json(base_url, f"/api/studies/{study_id}")
        assert len(study["protocol_hash"]) == 64 and len(study["evaluation"]["gates"]) == 8
        memo_status, memo_body, _ = request(base_url, f"/api/studies/{study_id}/memo.md")
        assert memo_status == 200 and b"TRACE Decision Memo" in memo_body

    portfolio = get_json(base_url, "/api/portfolio")
    assert portfolio["schema"] == "trace-evidence-portfolio-v1"
    assert portfolio["framework"] == "TRACE-PORTFOLIO-1.0"
    assert portfolio["summary"]["methods"] >= 27 and portfolio["summary"]["claims"] >= 5
    assert portfolio["summary"]["integrity_passed"] == portfolio["summary"]["integrity_total"] == 5
    assert len(portfolio["source_digest"]) == 64 and all(len(source["sha256"]) == 64 for source in portfolio["sources"])
    assert {"reportable", "bounded", "negative", "blocked"} <= {claim["status"] for claim in portfolio["claims"]}
    assert {"frozen_clip_rgb_structured_controller", "linear_bc_v1", "lora_action_head_lite_v1", "contact_monitor_early_regrasp"} <= {method["id"] for method in portfolio["methods"]}
    portfolio_report_status, portfolio_report, _ = request(base_url, "/api/portfolio/report.md")
    assert portfolio_report_status == 200 and b"# TRACE Evidence Portfolio" in portfolio_report
    for source in portfolio["sources"]:
        source_status, source_body, _ = request(base_url, source["url"])
        assert source_status == 200 and len(source_body) == source["bytes"]
    missing_source_status, _, _ = request(base_url, "/api/portfolio/sources/not-a-source")
    assert missing_source_status == 404

    releases = get_json(base_url, "/api/releases")
    assert releases["summary"]["framework"] == "TRACE-RELEASE-1.0"
    assert releases["preview"]["framework"] == "TRACE-RELEASE-1.0"
    assert len(releases["preview"]["gates"]) == 5 and all(gate["status"] == "pass" for gate in releases["preview"]["gates"])
    assert releases["preview"]["ready"] is True and releases["preview"]["bundle_file_count"] == 7
    invalid_release_status, invalid_release = post_json(base_url, "/api/releases", {})
    assert invalid_release_status == 400 and "error" in invalid_release
    if releases["releases"]:
        release = releases["releases"][0]
        release_detail = get_json(base_url, f"/api/releases/{release['release_id']}")
        assert release_detail["verification"]["manifest_valid"] and release_detail["verification"]["files_valid"]
        assert release_detail["verification"]["source_snapshot_valid"] and release_detail["verification"]["ledger_snapshot_valid"]
        assert release_detail["verification"]["verified_files"] == release_detail["verification"]["total_files"] == 7
        manifest_status, manifest_body, _ = request(base_url, release_detail["manifest_url"])
        assert manifest_status == 200 and json.loads(manifest_body.decode("utf-8"))["manifest_hash"] == release_detail["manifest_hash"]
        readme_status, readme_body, _ = request(base_url, release_detail["readme_url"])
        assert readme_status == 200 and b"# Evidence Release" in readme_body
    missing_release_status, _, _ = request(base_url, "/api/releases/not-a-release")
    assert missing_release_status == 404
    removed_defense_status, _, _ = request(base_url, "/api/defense")
    assert removed_defense_status == 404

    csv_status, csv_body, csv_headers = request(base_url, "/api/runs/export.csv")
    assert csv_status == 200 and csv_body.startswith(b"\xef\xbb\xbfrun_id,kind,status")
    assert "widowx_experiment_ledger.csv" in csv_headers.get("Content-Disposition", "")

    frame_status, frame, headers = request(base_url, "/api/sim/frame.png")
    assert frame_status == 200 and frame.startswith(b"\x89PNG\r\n\x1a\n") and len(frame) > 10_000
    assert headers.get("X-Frame-Sequence") is not None

    legacy_status, legacy, _ = request(base_url, config["legacy_path"])
    assert legacy_status == 200 and b"WidowX Method-Task Evidence Console" in legacy

    traversal_status, _, _ = request(base_url, "/assets/%2e%2e/%2e%2e/README.md")
    assert traversal_status == 403, "static asset path traversal was not rejected"
    artifact_traversal_status, _, _ = request(base_url, "/platform_artifacts/%2e%2e/%2e%2e/README.md")
    assert artifact_traversal_status == 403, "artifact path traversal was not rejected"

    unknown_status, unknown = post_json(base_url, "/api/sim/command", {"command": "not a valid robot command"})
    assert unknown_status == 400 and "error" in unknown

    empty_benchmark_status, empty_benchmark = post_json(base_url, "/api/benchmark/start", {"tasks": []})
    assert empty_benchmark_status == 400 and "error" in empty_benchmark

    if exercise_controls:
        start_status, _ = post_json(
            base_url,
            "/api/sim/start",
            {
                "task": "place_blue_cube_blue_pad",
                "policy": "structured_state",
                "complexity": "medium",
                "seed": 991,
                "speed": 0.25,
            },
        )
        assert start_status == 200
        running = wait_for_status(base_url, {"running"})
        assert running["step"] >= 0

        pause_status, paused = post_json(base_url, "/api/sim/command", {"command": "暂停"})
        assert pause_status == 200 and paused["status"] == "paused"
        paused_step = paused["step"]
        time.sleep(0.4)
        assert get_json(base_url, "/api/status")["simulation"]["step"] == paused_step

        resume_status, resumed = post_json(base_url, "/api/sim/command", {"command": "resume"})
        assert resume_status == 200 and resumed["status"] == "running"
        time.sleep(0.2)

        stop_status, stopped = post_json(base_url, "/api/sim/command", {"command": "stop"})
        assert stop_status == 200
        if stopped["status"] != "stopped":
            stopped = wait_for_status(base_url, {"stopped"})
        assert stopped["status"] == "stopped"

    if exercise_benchmark:
        benchmark_status, benchmark = post_json(
            base_url,
            "/api/benchmark/start",
            {
                "tasks": ["place_blue_cube_blue_pad"],
                "policies": ["rgb_grounded", "structured_state"],
                "seed_start": 992,
                "seeds_per_task": 1,
                "speed": 3.0,
            },
        )
        assert benchmark_status == 200
        benchmark = wait_for_benchmark(base_url, {"completed", "failed"})
        assert benchmark["status"] == "completed" and benchmark["completed_episodes"] == 2
        assert len(benchmark["results"]) == 2 and all(row["status"] == "completed" for row in benchmark["results"])
        assert len(benchmark["policy_metrics"]) == 2
        assert benchmark["paired_summary"]["pairs"] == 1
        records = get_json(base_url, "/api/runs?limit=20")["runs"]
        parent = next(row for row in records if row["run_id"] == benchmark["benchmark_id"])
        children = [row for row in records if row.get("parent_id") == parent["run_id"]]
        assert parent["status"] == "completed" and len(children) == 2 and all(row["kind"] == "simulation" for row in children)
        rgb_child = next(row for row in children if row["config"]["policy"] == "rgb_grounded")
        assert isinstance(rgb_child["metrics"]["rgb_grounding_error"], float)
        assert {"initial_top", "final_front"} <= rgb_child["assets"].keys()
        for asset_path in rgb_child["assets"].values():
            asset_status, asset_body, _ = request(base_url, asset_path)
            assert asset_status == 200 and asset_body.startswith(b"\x89PNG\r\n\x1a\n")

        report = get_json(base_url, f"/api/runs/{parent['run_id']}/report.json")
        assert report["schema"] == "widowx-research-report-v1" and len(report["children"]) == 2
        report_status, report_body, _ = request(base_url, f"/api/runs/{parent['run_id']}/report.md")
        assert report_status == 200 and b"# Experiment Report" in report_body

    if exercise_governance:
        create_status, study = post_json(
            base_url,
            "/api/studies",
            {
                "title": "Automated TRACE governance verification",
                "hypothesis": "A same-seed RGB run can be audited against the state-reference expert.",
                "tasks": ["place_blue_cube_blue_pad"],
                "seed_start": 993,
                "seeds_per_task": 1,
                "speed": 3.0,
                "criteria": {
                    "min_success_rate": 0.8,
                    "max_target_error_mm": 20.0,
                    "max_grounding_error_mm": 15.0,
                    "max_ci_width": 0.5,
                },
            },
        )
        assert create_status == 201 and study["evaluation"]["verdict"] == "locked"
        launch_status, launched = post_json(base_url, f"/api/studies/{study['study_id']}/launch", {})
        assert launch_status == 200 and launched["benchmark"]["benchmark_id"]
        benchmark = wait_for_benchmark(base_url, {"completed", "failed"})
        assert benchmark["status"] == "completed"
        evaluated = get_json(base_url, f"/api/studies/{study['study_id']}")
        assert evaluated["evaluation"]["latest_benchmark_id"] == benchmark["benchmark_id"]
        assert evaluated["evaluation"]["paired_summary"]["pairs"] == 1
        assert all(gate["status"] == "pass" for gate in evaluated["evaluation"]["gates"] if gate["id"] != "uncertainty")
        assert next(gate for gate in evaluated["evaluation"]["gates"] if gate["id"] == "uncertainty")["status"] == "fail"
        assert evaluated["evaluation"]["verdict"] == "needs_more_evidence"

    if exercise_release:
        create_status, release = post_json(
            base_url,
            "/api/releases",
            {
                "label": "Automated verified evidence release",
                "note": "Verifier-created immutable bundle for the current MuJoCo thesis evidence.",
            },
        )
        assert create_status == 201 and release["status"] == "verified_current"
        assert release["verification"]["manifest_valid"] and release["verification"]["files_valid"]
        assert release["verification"]["source_snapshot_valid"] and release["verification"]["ledger_snapshot_valid"]
        assert release["verification"]["verified_files"] == release["verification"]["total_files"] == 7
        refreshed = get_json(base_url, "/api/releases")
        assert refreshed["releases"][0]["release_id"] == release["release_id"]
        assert refreshed["summary"]["total"] >= 1

    if exercise_adaptation:
        adaptation = get_json(base_url, "/api/adaptation")
        if adaptation["tasks"]:
            task = adaptation["tasks"][0]
        else:
            create_status, task = post_json(base_url, "/api/adaptation/tasks", {
                "source": "green_cube",
                "target": "target_blue_pad",
                "instruction": "place the green cube on the blue pad",
                "complexity": "medium",
            })
            assert create_status == 201
        start_status, started = post_json(base_url, "/api/adaptation/start", {
            "task_id": task["task_id"],
            "method": "registry_rgb_skill",
            "profile": "eco",
            "episodes": 2,
            "epochs": 1,
            "evaluation_episodes": 3,
            "seed": 2400,
            "viewer": False,
        })
        assert start_status == 200 and started["status"] in {"starting", "running"}
        completed = wait_for_adaptation(base_url, {"completed", "failed"})
        assert completed["status"] == "completed", completed.get("error")
        assert completed["collection_successes"] >= 2
        assert completed["trainable_params"] == 0
        assert completed["model_path"] and completed["peak_rss_mb"] > 20
        assert completed["peak_rss_mb"] <= completed["estimated"]["estimated_peak_ram_mb"]
        assert completed["evaluation_episodes"] == 3 and len(completed["evaluation_rows"]) == 3
        assert completed["evaluation_successes"] >= 2 and completed["evaluation_success_rate"] >= 2 / 3
        assert isinstance(completed["evaluation_target_error"], float) and completed["evaluation_seed"] == 3402
        assert isinstance(completed["evaluation_mean_target_error"], float) and completed["evaluation_mean_target_error"] <= 0.03
        records = get_json(base_url, "/api/runs?kind=adaptation&limit=10")["runs"]
        record = next(row for row in records if row["run_id"] == completed["run_id"])
        assert record["status"] == "completed" and record["metrics"]["trainable_params"] == 0
        assert record["metrics"]["evaluation_success"] == completed["evaluation_success"]
        assert len(record["metrics"]["evaluation_rows"]) == 3
        assert "adaptation_summary" in record["assets"] and record["artifact"].endswith(".json")
        promotion = get_json(base_url, f"/api/adaptation/portfolio?task_id={task['task_id']}")
        assert promotion["champion_method"] == "registry_rgb_skill" and promotion["promotion"] == "promoted"
        assert "registry_rgb_skill" in promotion["pareto_methods"]

    if exercise_arena:
        adaptation = get_json(base_url, "/api/adaptation")
        task = next((item for item in adaptation["tasks"] if item["task_id"] == "place_green_cube_blue_pad"), None)
        if task is None:
            create_status, task = post_json(base_url, "/api/adaptation/tasks", {
                "source": "green_cube",
                "target": "target_blue_pad",
                "instruction": "place the green cube on the blue pad",
                "complexity": "medium",
            })
            assert create_status == 201
        start_status, started = post_json(base_url, "/api/adaptation/arena/start", {
            "task_id": task["task_id"],
            "methods": ["local_lora", "registry_rgb_skill"],
            "profile": "eco",
            "episodes": 2,
            "epochs": 1,
            "evaluation_episodes": 3,
            "seed": 2400,
            "viewer": False,
            "reuse_dataset": True,
        })
        assert start_status == 200 and started["mode"] == "arena"
        completed = wait_for_adaptation(base_url, {"completed", "failed"})
        assert completed["status"] == "completed", completed.get("error")
        assert completed["framework"] == "LOCAL-ADAPT-1.2" and completed["arena_framework"] == "PAIR-OPT-1.0"
        assert completed["candidate_methods"] == ["local_lora", "registry_rgb_skill"]
        assert len(completed["candidate_results"]) == 2 and len(completed["dataset_fingerprint"]) == 64
        assert completed["paired_summary"]["matched_seed_sets"] is True
        assert completed["paired_summary"]["completed_candidates"] == 2
        assert completed["paired_summary"]["champion_method"] in completed["candidate_methods"]
        assert len(completed["paired_summary"]["comparisons"]) == 1
        assert completed["peak_rss_mb"] <= completed["estimated"]["sequential_peak_ram_mb"]
        report = get_json(base_url, f"/api/runs/{completed['run_id']}/report.json")
        assert report["run"]["kind"] == "optimization" and len(report["children"]) == 2
        assert all(child["kind"] == "adaptation_candidate" for child in report["children"])
        fingerprints = {child["config"]["dataset_fingerprint"] for child in report["children"]}
        seed_sets = {tuple(child["metrics"]["evaluation_seeds"]) for child in report["children"]}
        assert fingerprints == {completed["dataset_fingerprint"]} and seed_sets == {(3402, 3403, 3404)}
        report_status, report_body, _ = request(base_url, f"/api/runs/{completed['run_id']}/report.md")
        assert report_status == 200 and b"## Paired Protocol" in report_body and b"## Paired Candidates" in report_body
        artifact_status, artifact_body, _ = request(base_url, report["run"]["assets"]["paired_optimization"])
        assert artifact_status == 200 and json.loads(artifact_body.decode("utf-8"))["paired_summary"]["matched_seed_sets"] is True

    print("PASS: v3.7 configuration, low-resource adaptation, paired optimizer, TRACE/Portfolio/Release, health, ledger/export, live PNG and validation boundaries")
    if exercise_controls:
        print("PASS: start, pause, resume and stop control path")
    if exercise_benchmark:
        print("PASS: paired benchmark, confidence metrics, visual artifacts, reports and parent-child ledger records")
    if exercise_governance:
        print("PASS: immutable TRACE protocol, linked benchmark, evidence gates and decision verdict")
    if exercise_release:
        print("PASS: immutable evidence release, bundled source copies, manifest hash and file-level verification")
    if exercise_adaptation:
        print("PASS: custom task, resource gate, real MuJoCo demonstrations, Registry RGB skill compilation, three-seed holdout and Pareto promotion")
    if exercise_arena:
        print("PASS: one shared demonstration set, same-seed candidate comparison, paired statistics, resource gate and auditable optimization report")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a running WidowX research platform.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8050")
    parser.add_argument("--exercise-controls", action="store_true")
    parser.add_argument("--exercise-benchmark", action="store_true")
    parser.add_argument("--exercise-governance", action="store_true")
    parser.add_argument("--exercise-release", action="store_true")
    parser.add_argument("--exercise-adaptation", action="store_true")
    parser.add_argument("--exercise-arena", action="store_true")
    args = parser.parse_args()
    verify(
        args.base_url,
        args.exercise_controls,
        args.exercise_benchmark,
        args.exercise_governance,
        args.exercise_release,
        args.exercise_adaptation,
        args.exercise_arena,
    )


if __name__ == "__main__":
    main()
