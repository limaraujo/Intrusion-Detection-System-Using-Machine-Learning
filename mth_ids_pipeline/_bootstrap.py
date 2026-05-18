"""Garante imports (ex.: FCBF_module) a partir da raiz do repositório."""

import sys
from pathlib import Path

from .config import REPO_ROOT


def ensure_repo_on_path() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
