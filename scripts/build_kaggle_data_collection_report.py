from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from widowx_env.demo_dataset import read_metadata  # noqa: E402


RESERVED = {
    "place_blue_cube_blue_pad": set(range(20, 25)),
    "place_blue_cube_red_pad": set(range(120, 125)),
    "place_red_cube_red_pad": set(range(220, 225)),
    "move_leftmost_cube_to_bowl": set(range(420, 425)),
}

RUN_DIRS = (
    ROOT / "data" / "demos" / "core_v2_place_blue_cube_blue_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_place_blue_cube_blue_pad_medium_80_v1",
    ROOT / "data" / "demos" / "core_v2_place_blue_cube_red_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_place_blue_cube_red_pad_medium_80_v1",
    ROOT / "data" / "demos" / "core_v2_place_red_cube_red_pad_medium_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_place_red_cube_red_pad_medium_80_v1",
    ROOT / "data" / "demos" / "core_v2_move_leftmost_cube_to_bowl_language_train20_v1",
    ROOT / "data" / "demos" / "kaggle_scale_move_leftmost_cube_to_bowl_language_80_v1",
    ROOT / "data" / "demos" / "kaggle_scale_move_leftmost_cube_to_bowl_language_80_v1_sampling_failure",
    ROOT / "data" / "demos" / "kaggle_scale_move_leftmost_cube_to_bowl_language_80_v1_resume2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit seed-disjoint MuJoCo demonstration expansion before a Kaggle spatial-pointer export.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=RUN_DIRS)
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "kaggle_spatial_data_collection_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "kaggle_spatial_data_collection_v1_report.md")
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    args = parse_args()
    task_rows: dict[str, list[dict]] = {}
    source_rows = []
    seen: set[tuple[str, int]] = set()
    duplicate_seeds = []
    reserved_hits = []
    missing_trajectories = []
    for run_dir in args.run_dirs:
        rows = read_metadata(run_dir)
        successful = [row for row in rows if row.get("success")]
        failures = [row for row in rows if not row.get("success")]
        for row in successful:
            task = str(row["task"])
            seed = int(row["seed"])
            key = (task, seed)
            if key in seen:
                duplicate_seeds.append({"task": task, "seed": seed, "run_dir": relative(run_dir)})
            seen.add(key)
            if seed in RESERVED.get(task, set()):
                reserved_hits.append({"task": task, "seed": seed, "run_dir": relative(run_dir)})
            trajectory = row.get("trajectory_file")
            if not trajectory or not (run_dir / str(trajectory)).exists():
                missing_trajectories.append({"task": task, "seed": seed, "run_dir": relative(run_dir), "trajectory_file": trajectory})
            task_rows.setdefault(task, []).append(row)
        source_rows.append(
            {
                "run_dir": relative(run_dir),
                "rows": len(rows),
                "successful_rows": len(successful),
                "failed_rows": len(failures),
                "sampling_failures": sum("collection_error" in row for row in failures),
            }
        )
    by_task = {
        task: {
            "successful_episodes": len(rows),
            "seed_min": min(int(row["seed"]) for row in rows),
            "seed_max": max(int(row["seed"]) for row in rows),
        }
        for task, rows in sorted(task_rows.items())
    }
    total_successes = sum(item["successful_episodes"] for item in by_task.values())
    ready = (
        total_successes >= 300
        and all(item["successful_episodes"] >= 60 for item in by_task.values())
        and not duplicate_seeds
        and not reserved_hits
        and not missing_trajectories
    )
    result = {
        "version": "kaggle_spatial_data_collection_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_runs": source_rows,
        "by_task": by_task,
        "total_successful_episodes": total_successes,
        "reserved_holdout_seeds": {task: sorted(seeds) for task, seeds in RESERVED.items()},
        "duplicate_success_seeds": duplicate_seeds,
        "reserved_holdout_hits": reserved_hits,
        "missing_success_trajectories": missing_trajectories,
        "kaggle_export_ready": ready,
        "decision": "Build the Kaggle spatial-pointer training pack." if ready else "Do not build the Kaggle pack until data audit failures are resolved.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    source_table = "\n".join(
        f"| `{item['run_dir']}` | {item['rows']} | {item['successful_rows']} | {item['failed_rows']} | {item['sampling_failures']} |"
        for item in source_rows
    )
    task_table = "\n".join(
        f"| {task} | {item['successful_episodes']} | {item['seed_min']} | {item['seed_max']} |"
        for task, item in by_task.items()
    )
    markdown = f"""# Kaggle 空间动作头数据扩展审计

版本：`kaggle_spatial_data_collection_v1`

## 源目录

| 目录 | metadata 行 | 成功 | 失败 | 场景采样失败 |
| --- | ---: | ---: | ---: | ---: |
{source_table}

## 成功示范训练池

| 任务 | 成功示范 | 最小 seed | 最大 seed |
| --- | ---: | ---: | ---: |
{task_table}

- 成功示范总数：`{total_successes}`。
- 固定留出：蓝到蓝 `20-24`、蓝到红 `120-124`、红到红 `220-224`、最左到碗 `420-424`。
- 成功 seed 重复数：`{len(duplicate_seeds)}`；留出 seed 命中数：`{len(reserved_hits)}`；缺失成功轨迹数：`{len(missing_trajectories)}`。
- Kaggle 导出门槛：每任务至少 60 条且总数至少 300。结论：`{'通过' if ready else '未通过'}`。

## 处理说明

最左任务的 seed 1326 无法生成不重叠的对象布局，已作为一条 `collection_error` 原始失败记录保留；它没有 trajectory，不参与成功示范训练池。其余执行失败也保留在 raw metadata，Kaggle 导出只读取 `success=true` 且轨迹文件存在的行。

## 决策

{result['decision']}
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "kaggle_export_ready": ready, "by_task": by_task}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
