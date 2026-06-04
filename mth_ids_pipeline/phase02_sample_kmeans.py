"""
Fase 2:
- LabelEncoder na coluna Label
- Separação entre classes minoritárias e majoritárias
- MiniBatchKMeans nas majoritárias
- Amostragem proporcional por cluster
- Concatena novamente com as minoritárias

Saída:
data/pipeline_mth_ids/02_sampled_kmeans.parquet
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import pandas as pd

try:
    from .config import (
        INTERMEDIATE_DIR,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )
except ImportError:
    from config import (
        INTERMEDIATE_DIR,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )

try:
    from .reporting import dataset_report, write_report
except ImportError:
    from reporting import dataset_report, write_report


try:
    from .clustering import sample_kmeans as _sample_kmeans
except ImportError:
    from clustering import sample_kmeans as _sample_kmeans


def sample_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 1000,
    random_state: int = 0,
    frac: float = 0.008,
) -> pd.DataFrame:
    print("Iniciando Fase 2 — Sampling com MiniBatchKMeans")
    start = time.time()
    combined = _sample_kmeans(
        df, n_clusters=n_clusters, random_state=random_state, frac=frac
    )
    label_col = "Label" if "Label" in combined.columns else combined.columns[-1]
    print(combined[label_col].value_counts())
    print(f"Shape final após sampling: {combined.shape}")
    print(f"Fase 2 concluída em {time.time() - start:.2f}s")
    return combined


def main() -> None:
    warnings.filterwarnings("ignore")

    total_start = time.time()

    parser = argparse.ArgumentParser(description="Fase 2 — sampling k-means")
    parser.add_argument("--input", type=Path, default=None, help="Parquet da fase 1")
    parser.add_argument("--output", type=Path, default=None, help="Parquet de saida (default: fase 2)")
    parser.add_argument("--n-clusters", type=int, default=1000, help="Numero de clusters")
    parser.add_argument("--frac", type=float, default=0.008, help="Fracao amostrada por cluster")
    parser.add_argument("--random-state", type=int, default=0, help="Seed para amostragem")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()

    # Cria diretórios
    ensure_intermediate_dirs()

    # Arquivo de entrada da Fase 1
    inp = args.input or (INTERMEDIATE_DIR / P01_PREPROCESSED.replace(".csv", ".parquet"))

    # Arquivo de saída da Fase 2
    out = args.output or (INTERMEDIATE_DIR / P02_SAMPLED_KMEANS.replace(".csv", ".parquet"))

    print(f"Lendo arquivo: {inp}")

    load_start = time.time()

    # Lê parquet
    df = pd.read_parquet(inp)

    print(f"Arquivo carregado em {time.time() - load_start:.2f}s")
    print(f"Shape original: {df.shape}")

    # Executa sampling
    sampled = sample_kmeans(
        df,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        frac=args.frac,
    )

    # Salva parquet
    print("Salvando parquet...")

    save_start = time.time()

    sampled.to_parquet(out, index=False)

    print(f"Arquivo salvo: {out}")
    print(f"Tempo salvando: {time.time() - save_start:.2f}s")
    print(f"Tempo total: {time.time() - total_start:.2f}s")

    label_col = "Label" if "Label" in sampled.columns else sampled.columns[-1]
    report = dataset_report(sampled, label_col)
    report.update(
        {
            "input": str(inp),
            "output": str(out),
            "n_clusters": args.n_clusters,
            "frac": args.frac,
            "random_state": args.random_state,
            "duration_s": round(time.time() - total_start, 4),
        }
    )
    report_path = write_report(args.report_dir, "phase02_sample_kmeans", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()