"""
Fase 3: primeiro train_test_split estratificado (80/20) sobre o conjunto amostrado.

Saida: 03_train.parquet, 03_test.parquet
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from .config import (
        INTERMEDIATE_DIR,
        P02_SAMPLED_KMEANS,
        P03_TEST,
        P03_TRAIN,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )
except ImportError:
    from config import (
        INTERMEDIATE_DIR,
        P02_SAMPLED_KMEANS,
        P03_TEST,
        P03_TRAIN,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )

try:
    from .reporting import dataset_report, write_report
except ImportError:
    from reporting import dataset_report, write_report


def split_train_test(
    df: pd.DataFrame,
    *,
    random_state: int = 0,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    y = np.asarray(df[label_col]).ravel()

    train_df, test_df = train_test_split(
        df,
        train_size=1 - test_size,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return train_df, test_df


def main() -> None:
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Fase 3 — train/test split")
    parser.add_argument("--input", type=Path, default=None, help="Parquet da fase 2")
    parser.add_argument(
        "--train-out",
        "--train-output",
        dest="train_out",
        type=Path,
        default=None,
        help="Parquet de treino (default: fase 3)",
    )
    parser.add_argument(
        "--test-out",
        "--test-output",
        dest="test_out",
        type=Path,
        default=None,
        help="Parquet de teste (default: fase 3)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fracao para teste")
    parser.add_argument("--random-state", type=int, default=0, help="Seed para split")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()

    ensure_intermediate_dirs()

    inp = args.input or (INTERMEDIATE_DIR / P02_SAMPLED_KMEANS.replace(".csv", ".parquet"))

    df = pd.read_parquet(inp)

    tr, te = split_train_test(df, random_state=args.random_state, test_size=args.test_size)

    tr_path = args.train_out or (INTERMEDIATE_DIR / P03_TRAIN.replace(".csv", ".parquet"))
    te_path = args.test_out or (INTERMEDIATE_DIR / P03_TEST.replace(".csv", ".parquet"))

    tr.to_parquet(tr_path, index=False)
    te.to_parquet(te_path, index=False)

    print(f"Salvo: {tr_path} ({tr.shape}), {te_path} ({te.shape})")

    label_col = "Label" if "Label" in tr.columns else tr.columns[-1]
    report = {
        "input": str(inp),
        "train_output": str(tr_path),
        "test_output": str(te_path),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train": dataset_report(tr, label_col),
        "test": dataset_report(te, label_col),
    }
    report_path = write_report(args.report_dir, "phase03_train_test_split", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()