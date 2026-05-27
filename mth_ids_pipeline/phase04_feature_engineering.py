"""
Fase 4: seleção por Information Gain (90% acumulado) + FCBF (k=20), segundo split 80/20.

Requer `FCBF_module.py` na raiz do repositório (ver README).

Saída: 04_train_after_fcbf.csv, 04_test_after_fcbf.csv, 04_selected_features.txt
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

try:
    from ._bootstrap import ensure_repo_on_path
except ImportError:
    from _bootstrap import ensure_repo_on_path

try:
    from .config import (
        INTERMEDIATE_DIR,
        P02_SAMPLED_KMEANS,
        P04_SELECTED_FEATURES,
        P04_TEST_FSS,
        P04_TRAIN_FSS,
        ensure_intermediate_dirs,
    )
except ImportError:
    from config import (
    INTERMEDIATE_DIR,
    P02_SAMPLED_KMEANS,
    P04_SELECTED_FEATURES,
    P04_TEST_FSS,
    P04_TRAIN_FSS,
    ensure_intermediate_dirs,
    )


def _numeric_feature_names(df: pd.DataFrame, label_col: str) -> list[str]:
    return [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]


def information_gain_feature_subset(
    X_train: np.ndarray, feature_names: list[str], y_train: np.ndarray, *, cumulative: float = 0.9
) -> list[str]:
    importances = mutual_info_classif(X_train, y_train, random_state=0)
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


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()
    try:
        from mth_ids_pipeline.utils.FCBF_module import FCBFK
    except ImportError as e:
        raise ImportError(
            "Coloque FCBF_module.py na raiz do repositório (clone de "
            "https://github.com/SantiagoEG/FCBF_module)."
        ) from e

    parser = argparse.ArgumentParser(description="Fase 4 — IG + FCBF + split")
    parser.add_argument("--input", type=Path, default=None, help="Parquet amostrado (default fase 2)")
    parser.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    ensure_intermediate_dirs()

    path_df = args.input or (args.output_dir / P02_SAMPLED_KMEANS.replace(".csv", ".parquet"))
    df = pd.read_parquet(path_df)
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    feature_names = _numeric_feature_names(df, label_col)
    X = df[feature_names].values
    y = np.ravel(df[label_col].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=args.random_state, stratify=y
    )

    ig_features = information_gain_feature_subset(X_train, feature_names, y_train, cumulative=0.9)
    (args.output_dir / P04_SELECTED_FEATURES).write_text("\n".join(ig_features), encoding="utf-8")

    X_fs = df[ig_features].values
    fcbf = FCBFK(k=args.fcbf_k)
    X_fss = fcbf.fit_transform(X_fs, y)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_fss, y, train_size=0.8, test_size=0.2, random_state=args.random_state, stratify=y
    )

    cols = [f"f{i}" for i in range(X_train2.shape[1])]
    train_out = pd.DataFrame(X_train2, columns=cols)
    train_out[label_col] = y_train2
    test_out = pd.DataFrame(X_test2, columns=cols)
    test_out[label_col] = y_test2

    train_out.to_csv(args.output_dir / P04_TRAIN_FSS, index=False)
    test_out.to_csv(args.output_dir / P04_TEST_FSS, index=False)
    print(f"Salvo: {args.output_dir / P04_TRAIN_FSS} {train_out.shape}")
    print(f"Salvo: {args.output_dir / P04_TEST_FSS} {test_out.shape}")
    print(f"Features IG (pré-FCBF): {len(ig_features)}, colunas pós-FCBF: {len(cols)}")


if __name__ == "__main__":
    main()
