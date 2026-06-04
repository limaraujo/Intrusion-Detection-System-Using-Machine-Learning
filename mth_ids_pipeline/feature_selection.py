"""Seleção de atributos: Information Gain + FCBF (notebook e artigo)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_selection import mutual_info_classif

if TYPE_CHECKING:
    import pandas as pd


def information_gain_feature_subset(
    X_train: np.ndarray,
    feature_names: list[str],
    y_train: np.ndarray,
    *,
    cumulative: float = 0.9,
    random_state: int = 0,
) -> list[str]:
    """Seleciona features até atingir `cumulative` da importância MI normalizada."""
    importances = mutual_info_classif(X_train, y_train, random_state=random_state)
    total = float(np.sum(importances)) or 1.0
    ranked = sorted(zip(importances / total, feature_names), reverse=True)
    acc = 0.0
    selected: list[str] = []
    for score, name in ranked:
        acc += float(score)
        selected.append(name)
        if acc >= cumulative:
            break
    return selected


def numeric_feature_names(df: pd.DataFrame, label_col: str = "Label") -> list[str]:
    return [c for c in df.columns if c != label_col and __import__("pandas").api.types.is_numeric_dtype(df[c])]


def apply_fcbf(X: np.ndarray, y: np.ndarray, *, k: int = 20):
    """Aplica FCBFK(k) — requer FCBF_module no path."""
    from mth_ids_pipeline.utils.FCBF_module import FCBFK

    fcbf = FCBFK(k=k)
    return fcbf.fit_transform(X, y)
