"""
Fase 1: carrega CICIDS2017, normalização Z-score nas colunas numéricas,
preenche NaN com 0. Mantém a coluna Label como no CSV original.

Saída:
data/pipeline_mth_ids/01_preprocessed.parquet
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

try:
    from .config import (
        DEFAULT_RAW_CSV,
        INTERMEDIATE_DIR,
        P01_PREPROCESSED,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )
except ImportError:
    from config import (
        DEFAULT_RAW_CSV,
        INTERMEDIATE_DIR,
        P01_PREPROCESSED,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )

try:
    from .reporting import dataset_report, write_report
except ImportError:
    from reporting import dataset_report, write_report


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

    parser = argparse.ArgumentParser(description="Fase 1 — load + preprocess")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_CSV, help="CSV bruto de entrada")
    parser.add_argument("--output", type=Path, default=None, help="Parquet de saida (default: fase 1)")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()

    # Cria diretórios
    ensure_intermediate_dirs()

    # Nome do arquivo parquet
    out = args.output or (INTERMEDIATE_DIR / P01_PREPROCESSED.replace(".csv", ".parquet"))

    # Executa pipeline
    df = load_and_preprocess(args.input)

    # Salva parquet
    print("Salvando arquivo parquet...")

    save_start = time.time()

    df.to_parquet(out, index=False)

    print(f"Arquivo salvo em: {out}")
    print(f"Tempo de salvamento: {time.time() - save_start:.2f}s")
    print(f"Tempo total do pipeline: {time.time() - total_start:.2f}s")

    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    numeric = df.select_dtypes(include="number").columns
    feature_cols = [c for c in numeric if c != label_col]
    report = dataset_report(df, label_col)
    report.update(
        {
            "input": str(args.input),
            "output": str(out),
            "feature_count": len(feature_cols),
            "duration_s": round(time.time() - total_start, 4),
        }
    )
    report_path = write_report(args.report_dir, "phase01_load_preprocess", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()