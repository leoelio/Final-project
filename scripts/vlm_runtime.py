from __future__ import annotations

import sys
from pathlib import Path


VLM_PACKAGE_DIR = Path(r"C:\vla_vlm_pkgs")


def ensure_vlm_path() -> None:
    if VLM_PACKAGE_DIR.exists():
        path = str(VLM_PACKAGE_DIR)
        if path not in sys.path:
            sys.path.append(path)
