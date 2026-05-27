"""
Fase 1: carrega CICIDS2017, normalização Z-score nas colunas numéricas,
preenche NaN com 0. Mantém a coluna Label como no CSV original.

Saída:
data/pipeline_mth_ids/01_preprocessed.parquet
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

try:
    from .config import (
        DEFAULT_RAW_CSV,
        INTERMEDIATE_DIR,
        P01_PREPROCESSED,
        ensure_intermediate_dirs,
    )
except ImportError:
    from config import (
        DEFAULT_RAW_CSV,
        INTERMEDIATE_DIR,
        P01_PREPROCESSED,
        ensure_intermediate_dirs,
    )


def load_and_preprocess(raw_csv: Path) -> pd.DataFrame:
    print(f"Carregando dataset: {raw_csv}")

    start = time.time()

    # Ignora warnings durante normalização
    warnings.filterwarnings("ignore")

    # Lê CSV
    df = pd.read_csv(raw_csv, index_col=False)
    print(df.Label.value_counts())

    print(f"Dataset carregado em {time.time() - start:.2f}s")
    print(f"Shape original: {df.shape}")

    # Detecta coluna Label
    label_col = "Label" if "Label" in df.columns else df.columns[-1]

    print(f"Coluna Label detectada: {label_col}")

    # Seleciona colunas numéricas
    numeric = df.select_dtypes(include="number").columns

    # Remove Label das features
    feature_cols = [c for c in numeric if c != label_col]

    print(f"Total de features numéricas: {len(feature_cols)}")

    # Normalização Z-score
    print("Iniciando normalização Z-score...")

    norm_start = time.time()

    df[feature_cols] = df[feature_cols].apply(
        lambda x: (x - x.mean()) / (x.std())
    )

    print(f"Normalização concluída em {time.time() - norm_start:.2f}s")

    # Substitui NaN
    print("Substituindo NaN por 0...")

    df = df.fillna(0)

    print("Pré-processamento concluído.")
    print(f"Shape final: {df.shape}")

    return df


def main() -> None:
    total_start = time.time()

    # Cria diretórios
    ensure_intermediate_dirs()

    # Nome do arquivo parquet
    out = INTERMEDIATE_DIR / P01_PREPROCESSED.replace(".csv", ".parquet")

    # Executa pipeline
    df = load_and_preprocess(DEFAULT_RAW_CSV)

    # Salva parquet
    print("Salvando arquivo parquet...")

    save_start = time.time()

    df.to_parquet(out, index=False)

    print(f"Arquivo salvo em: {out}")
    print(f"Tempo de salvamento: {time.time() - save_start:.2f}s")
    print(f"Tempo total do pipeline: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()