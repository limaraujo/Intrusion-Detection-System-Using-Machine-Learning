"""
Fase 3: primeiro train_test_split estratificado (80/20) sobre o conjunto amostrado.

Saída: 03_train.csv, 03_test.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import INTERMEDIATE_DIR, P02_SAMPLED_KMEANS, P03_TEST, P03_TRAIN, ensure_intermediate_dirs


def split_train_test(df: pd.DataFrame, *, random_state: int = 0, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    y = df[label_col].values
    train_df, test_df = train_test_split(df, train_size=1 - test_size, test_size=test_size, random_state=random_state, stratify=y)
    return train_df, test_df


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 3 — split inicial")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    ensure_intermediate_dirs()
    inp = args.input or (args.output_dir / P02_SAMPLED_KMEANS)
    df = pd.read_csv(inp)
    tr, te = split_train_test(df, random_state=args.random_state, test_size=args.test_size)
    tr_path = args.output_dir / P03_TRAIN
    te_path = args.output_dir / P03_TEST
    tr.to_csv(tr_path, index=False)
    te.to_csv(te_path, index=False)
    print(f"Salvo: {tr_path} ({tr.shape}), {te_path} ({te.shape})")


if __name__ == "__main__":
    main()
