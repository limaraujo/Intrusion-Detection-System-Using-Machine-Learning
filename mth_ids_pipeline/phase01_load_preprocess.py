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


try:
    from .preprocessing import load_and_preprocess as _load_and_preprocess
except ImportError:
    from preprocessing import load_and_preprocess as _load_and_preprocess


def load_and_preprocess(raw_csv: Path) -> pd.DataFrame:
    return _load_and_preprocess(raw_csv)


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