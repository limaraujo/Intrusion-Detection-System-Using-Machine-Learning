"""Controle de reprodutibilidade: seeds globais, versões e registro de execução."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_RANDOM_STATE = 0

PACKAGE_VERSIONS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "imbalanced-learn",
    "hyperopt",
    "scikit-optimize",
)


def set_global_seeds(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Define seeds globais para Python, NumPy e variável de ambiente PYTHONHASHSEED."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    for pkg in PACKAGE_VERSIONS:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def log_run_config(
    report_dir: Path,
    *,
    run_name: str,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persiste parâmetros e versões de bibliotecas de uma execução."""
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "timestamp": int(time.time()),
        "environment": collect_environment_versions(),
        "config": config,
        "non_deterministic_components": [
            "Hyperopt fmin (sem random_state explícito no notebook)",
            "KernelPCA (sklearn, sem random_state)",
            "SMOTE (imbalanced-learn, sem random_state no notebook)",
            "MiniBatchKMeans no ramo anomaly (sem seed no notebook)",
            "Amostragem benigna PortScan (random_state=None no notebook)",
        ],
    }
    if extra:
        payload.update(extra)
    out = report_dir / f"{run_name}_config.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return out
