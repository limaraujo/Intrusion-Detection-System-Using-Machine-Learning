"""Utilitários compartilhados do ramo anomaly (fases 7–11)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from .config import A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO


def build_anomaly_binary_split(
    df: pd.DataFrame,
    attack_label: int,
    *,
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conjunto sem o ataque zero-day (binário) e conjunto só com esse ataque (rótulo 1).

    Alinhado ao artigo: leave-one-attack-out — o ataque escolhido é o desconhecido.
    """
    df1 = df[df[label_col] != attack_label].copy()
    df1.loc[df1[label_col] > 0, label_col] = 1

    df2 = df[df[label_col] == attack_label].copy()
    df2.loc[:, label_col] = 1
    return df1, df2


def discover_attack_labels(df: pd.DataFrame, *, label_col: str = "Label") -> list[int]:
    """Rótulos de ataque (>0) presentes no dataframe."""
    return sorted(int(x) for x in df[label_col].unique() if int(x) > 0)


def load_anomaly_splits(
    input_dir: Path,
    *,
    smote_target: int | None,
    random_state: int,
    label_col: str = "Label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Carrega KPCA + meta de slice; aplica SMOTE no treino se necessário."""
    df = pd.read_parquet(input_dir / A04_AFTER_KPCA)
    meta = json.loads((input_dir / A06_TEST_SLICE_INFO).read_text(encoding="utf-8"))
    n_df1 = int(meta["n_df1_rows"])

    X_all = df.drop(columns=[label_col]).values
    y_all = np.ravel(df[label_col].values)
    X_train = X_all[:n_df1]
    y_train = y_all[:n_df1]
    X_test = X_all[n_df1:]
    y_test = y_all[n_df1:]

    train_path = input_dir / A05_TRAIN_SMOTE
    if train_path.exists():
        tr = pd.read_parquet(train_path)
        X_train = tr.drop(columns=[label_col]).values
        y_train = np.ravel(tr[label_col].values)
        return X_train, X_test, y_train, y_test, True

    counts = pd.Series(y_train).value_counts()
    target = smote_target if smote_target is not None else int(counts.max())
    did_smote = False
    if 1 in counts.index and target > int(counts[1]):
        kw: dict = {"sampling_strategy": {1: target}}
        if "random_state" in inspect.signature(SMOTE.__init__).parameters:
            kw["random_state"] = random_state
        if "n_jobs" in inspect.signature(SMOTE.__init__).parameters:
            kw["n_jobs"] = -1
        smote = SMOTE(**kw)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        did_smote = True
    return X_train, X_test, y_train, y_test, did_smote
