"""
Fase 5: SMOTE no conjunto de treino (estratégia igual ao notebook), teste inalterado.

Saída: 05_train_after_smote.csv, 05_test_unchanged.csv
"""

from __future__ import annotations

import argparse
import inspect
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE

from .config import INTERMEDIATE_DIR, P04_TEST_FSS, P04_TRAIN_FSS, P05_TEST, P05_TRAIN_SMOTE, ensure_intermediate_dirs


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 5 — SMOTE")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR)
    args = parser.parse_args()
    ensure_intermediate_dirs()

    tr_path = args.train or (args.output_dir / P04_TRAIN_FSS)
    te_path = args.test or (args.output_dir / P04_TEST_FSS)
    label_col = "Label"

    train_df = pd.read_csv(tr_path)
    test_df = pd.read_csv(te_path)

    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values

    kw: dict = {"sampling_strategy": {2: 1000, 4: 1000}}
    if "n_jobs" in inspect.signature(SMOTE.__init__).parameters:
        kw["n_jobs"] = -1
    smote = SMOTE(**kw)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    fcols = [c for c in train_df.columns if c != label_col]
    out_train = pd.DataFrame(X_res, columns=fcols)
    out_train[label_col] = y_res
    out_train.to_csv(args.output_dir / P05_TRAIN_SMOTE, index=False)
    test_df.to_csv(args.output_dir / P05_TEST, index=False)
    print(f"Salvo: {args.output_dir / P05_TRAIN_SMOTE} shape={out_train.shape}")
    print(f"Salvo: {args.output_dir / P05_TEST} shape={test_df.shape}")


if __name__ == "__main__":
    main()
