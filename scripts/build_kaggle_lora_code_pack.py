from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "data" / "kaggle_lora_code_v1"
ARCHIVE = ROOT / "outputs" / "kaggle_lora_code_v1.zip"
UPLOAD_DIR = ROOT / "data" / "kaggle_lora_code_upload_v1"
SOURCES = (ROOT / "scripts" / "train_clip_lora_patch_pointer.py", ROOT / "scripts" / "clip_lora_utils.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    script_dir = PACK_DIR / "scripts"
    script_dir.mkdir(parents=True)
    for source in SOURCES:
        shutil.copy2(source, script_dir / source.name)
    manifest = {
        "version": "kaggle_lora_code_pack_v1",
        "method": "CLIP visual-attention LoRA plus 2D patch pointer",
        "source_files": [source.name for source in SOURCES],
        "data_dependency": "luxunyu/widowx-mujoco-patch-pointer-v2",
        "protocol": "Use the existing 393-scene pack and fixed 20-episode local holdout. The code pack contains no MuJoCo demonstrations.",
    }
    (PACK_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACK_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(PACK_DIR))
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
    UPLOAD_DIR.mkdir(parents=True)
    shutil.copy2(ARCHIVE, UPLOAD_DIR / ARCHIVE.name)
    (UPLOAD_DIR / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "WidowX MuJoCo CLIP LoRA Spatial Pointer Code V1",
                "id": "luxunyu/widowx-mujoco-clip-lora-pointer-code-v1",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    payload = {
        **manifest,
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256": sha256(ARCHIVE),
        "archive_members": zipfile.ZipFile(ARCHIVE).namelist(),
        "upload_dir": str(UPLOAD_DIR.relative_to(ROOT)),
    }
    output = ROOT / "outputs" / "evaluations" / "kaggle_lora_code_pack_v1.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
