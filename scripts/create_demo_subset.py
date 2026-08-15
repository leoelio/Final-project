from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a metadata-preserving demonstration subset with hard-linked trajectories.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episode-index", type=int, required=True)
    parser.add_argument("--successful-only", action="store_true")
    return parser.parse_args()


def link_or_copy(source: Path, output: Path) -> str:
    try:
        os.link(source, output)
        return "hardlink"
    except OSError:
        shutil.copy2(source, output)
        return "copy"


def main() -> None:
    args = parse_args()
    metadata_path = args.source / "metadata.jsonl"
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
    records = [item for item in records if int(item["episode_index"]) <= args.max_episode_index]
    if args.successful_only:
        records = [item for item in records if bool(item["success"])]
    if not records:
        raise ValueError("subset is empty")

    episodes_dir = args.output / "episodes"
    episodes_dir.mkdir(parents=True)
    link_modes = set()
    for record in records:
        relative = Path(record["trajectory_file"])
        source_file = args.source / relative
        output_file = args.output / relative
        output_file.parent.mkdir(parents=True, exist_ok=True)
        link_modes.add(link_or_copy(source_file, output_file))

    (args.output / "metadata.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(args.source),
        "episodes": len(records),
        "successes": sum(int(item["success"]) for item in records),
        "selection": {"max_episode_index": args.max_episode_index, "successful_only": args.successful_only},
        "trajectory_storage": "+".join(sorted(link_modes)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"subset_dir: {args.output}", flush=True)
    print(f"episodes: {summary['episodes']}", flush=True)
    print(f"successes: {summary['successes']}", flush=True)
    print(f"trajectory_storage: {summary['trajectory_storage']}", flush=True)


if __name__ == "__main__":
    main()
