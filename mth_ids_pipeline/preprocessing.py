"""Pré-processamento alinhado ao notebook MTH_IDS_IoTJ.ipynb."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def zscore_normalize(df: pd.DataFrame, *, label_col: str = "Label") -> pd.DataFrame:
    """
    Z-score nas features numéricas (notebook: ``dtypes != 'object'`` com Label em texto).
    Em pandas moderno, restringe a ``is_numeric_dtype`` para evitar colunas ``string``.
    """
    out = df.copy()
    feature_cols = [
        c
        for c in out.columns
        if c != label_col and pd.api.types.is_numeric_dtype(out[c])
    ]
    out[feature_cols] = out[feature_cols].apply(lambda x: (x - x.mean()) / (x.std()))
    return out.fillna(0)


def encode_labels(df: pd.DataFrame, *, label_col: str = "Label") -> tuple[pd.DataFrame, LabelEncoder]:
    out = df.copy()
    encoder = LabelEncoder()
    out[label_col] = encoder.fit_transform(out[label_col].astype(str)).astype("int64")
    return out, encoder


def load_and_preprocess(raw_csv: Path, *, label_col: str = "Label", verbose: bool = True) -> pd.DataFrame:
    """Fase 1: load CSV + Z-score + fillna(0)."""
    warnings.filterwarnings("ignore")
    if verbose:
        print(f"Carregando dataset: {raw_csv}")
    start = time.time()
    df = pd.read_csv(raw_csv, index_col=False)
    if verbose:
        print(df[label_col].value_counts())
        print(f"Dataset carregado em {time.time() - start:.2f}s")
        print(f"Shape original: {df.shape}")
    df = zscore_normalize(df, label_col=label_col)
    if verbose:
        print(f"Pré-processamento concluído. Shape final: {df.shape}")
    return df
