from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and register an already exported Kaggle patch-pointer pack without rewriting its data.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "kaggle_patch_pointer_v2")
    parser.add_argument("--archive", type=Path, default=ROOT / "outputs" / "kaggle_patch_pointer_v2.zip")
    parser.add_argument("--audit-json", type=Path, default=ROOT / "outputs" / "evaluations" / "kaggle_spatial_data_collection_v1.json")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "evaluations" / "kaggle_patch_pointer_pack_v2.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "docs" / "kaggle_patch_pointer_pack_v2.md")
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = [line for line in (args.dataset_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {"manifest.json", "samples.jsonl", "kaggle_train_patch_pointer.ipynb", "KAGGLE_UPLOAD.md", "scripts/kaggle_train_patch_pointer.py"}
    with zipfile.ZipFile(args.archive) as handle:
        names = set(handle.namelist())
        corrupt = handle.testzip()
    missing = sorted(expected - names)
    valid = bool(audit.get("kaggle_export_ready")) and corrupt is None and not missing and len(samples) == int(manifest["samples"])
    result = {
        "version": f"{manifest['version']}_pack",
        "dataset_dir": relative(args.dataset_dir),
        "archive": relative(args.archive),
        "samples": int(manifest["samples"]),
        "per_split": manifest["per_split"],
        "per_task": manifest["per_task"],
        "dataset_content_sha256": manifest["dataset_content_sha256"],
        "audit": relative(args.audit_json),
        "kaggle_notebook": relative(args.dataset_dir / "kaggle_train_patch_pointer.ipynb"),
        "training_script": relative(args.dataset_dir / "scripts" / "kaggle_train_patch_pointer.py"),
        "archive_corrupt_member": corrupt,
        "archive_missing_required_files": missing,
        "ready": valid,
        "decision": "Ready for Kaggle frozen-CLIP patch-pointer long training; OpenVLA LoRA remains a separate unmet remote path." if valid else "Do not upload until pack validation failures are resolved.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = f"""# Kaggle 冻结 CLIP Patch 指针训练包 V2

- 场景：`{result['samples']}` 条初始 RGB；训练/验证：`{result['per_split']['train']}/{result['per_split']['validation']}`。
- 每任务：`{result['per_task']}`。
- 内容 hash：`{result['dataset_content_sha256']}`。
- ZIP：`{result['archive']}`，损坏成员：`{result['archive_corrupt_member']}`，缺失必需文件：`{result['archive_missing_required_files']}`。
- Kaggle notebook：`{result['kaggle_notebook']}`。

训练脚本在记录的验证 checkpoint 中选择最低空间 RMSE，而非最后一个 epoch。Kaggle 内部验证只能选择 checkpoint；模型回填后仍必须在固定 MuJoCo 独立 20 episode 上做闭环评测。

边界：该包训练冻结 CLIP patch-pointer，不是 OpenVLA、OpenVLA LoRA 或端到端 VLA 成果。
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
