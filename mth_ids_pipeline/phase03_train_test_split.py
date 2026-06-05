"""
Fase 3: train_test_split estratificado sobre o conjunto amostrado (artigo: 70/30).

Saida: 03_train.parquet, 03_test.parquet
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from .cli import init_paths, phase_parser, supervised_path
    from .config import DEFAULT_TEST_SIZE, P02_SAMPLED_KMEANS, P03_TEST, P03_TRAIN
except ImportError:
    from cli import init_paths, phase_parser, supervised_path
    from config import DEFAULT_TEST_SIZE, P02_SAMPLED_KMEANS, P03_TEST, P03_TRAIN

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

    parser = phase_parser("Fase 3 — train/test split")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Artigo: 0.3 (70/30)")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    paths = init_paths(args)
    df = pd.read_parquet(supervised_path(paths, P02_SAMPLED_KMEANS))
    tr, te = split_train_test(df, random_state=args.random_state, test_size=args.test_size)
    tr_path = supervised_path(paths, P03_TRAIN)
    te_path = supervised_path(paths, P03_TEST)

    tr.to_parquet(tr_path, index=False)
    te.to_parquet(te_path, index=False)

    print(f"Salvo: {tr_path} ({tr.shape}), {te_path} ({te.shape})")

    label_col = "Label" if "Label" in tr.columns else tr.columns[-1]
    report = {
        "input": str(supervised_path(paths, P02_SAMPLED_KMEANS)),
        "train_output": str(tr_path),
        "test_output": str(te_path),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train": dataset_report(tr, label_col),
        "test": dataset_report(te, label_col),
    }
    report_path = write_report(paths.reports, "phase03_train_test_split", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()