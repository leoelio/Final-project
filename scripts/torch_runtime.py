from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_TORCH_PACKAGE_DIR = Path(r"C:\vla_torch_pkgs")
TORCH_PACKAGE_DIR = Path(os.environ.get("VLA_TORCH_PACKAGE_DIR", str(DEFAULT_TORCH_PACKAGE_DIR)))


def ensure_torch_path() -> None:
    if TORCH_PACKAGE_DIR.exists():
        path = str(TORCH_PACKAGE_DIR)
        if path not in sys.path:
            sys.path.insert(0, path)
