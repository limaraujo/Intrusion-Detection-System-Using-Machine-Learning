"""
Fase 3: primeiro train_test_split estratificado (80/20) sobre o conjunto amostrado.

Saída: 03_train.csv, 03_test.csv
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from .config import (
    INTERMEDIATE_DIR,
    P02_SAMPLED_KMEANS,
    P03_TEST,
    P03_TRAIN,
    ensure_intermediate_dirs,
)


def split_train_test(
    df: pd.DataFrame,
    *,
    random_state: int = 0,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    X = df.drop(columns=[label_col],axis=1).values
    y = df[label_col].values
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

    ensure_intermediate_dirs()

    inp = INTERMEDIATE_DIR / P02_SAMPLED_KMEANS

    df = pd.read_csv(inp)

    tr, te = split_train_test(df)

    tr_path = INTERMEDIATE_DIR / P03_TRAIN
    te_path = INTERMEDIATE_DIR / P03_TEST

    tr.to_csv(tr_path, index=False)
    te.to_csv(te_path, index=False)

    print(f"Salvo: {tr_path} ({tr.shape}), {te_path} ({te.shape})")


if __name__ == "__main__":
    main()
