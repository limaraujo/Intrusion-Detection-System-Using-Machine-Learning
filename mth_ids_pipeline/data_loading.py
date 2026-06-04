"""Carregamento de datasets tabulares para o pipeline MTH-IDS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: Path | str, *, label_col: str = "Label") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=False)
    if label_col not in df.columns:
        raise ValueError(f"Coluna '{label_col}' não encontrada em {path}")
    return df


def load_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
