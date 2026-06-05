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

import time
import warnings
from pathlib import Path

import pandas as pd

try:
    from .cli import init_paths, phase_parser, supervised_path
    from .config import (
        DEFAULT_MINORITY_LABELS,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        parse_minority_labels,
    )
except ImportError:
    from cli import init_paths, phase_parser, supervised_path
    from config import (
        DEFAULT_MINORITY_LABELS,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        parse_minority_labels,
    )

try:
    from .reporting import dataset_report, write_report
except ImportError:
    from reporting import dataset_report, write_report


try:
    from .clustering import sample_kmeans as _sample_kmeans
    from .label_profiles import minority_labels_all_attacks
except ImportError:
    from clustering import sample_kmeans as _sample_kmeans
    from label_profiles import minority_labels_all_attacks


def sample_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 1000,
    random_state: int = 0,
    frac: float = 0.008,
    minority_labels: tuple[int, ...] = DEFAULT_MINORITY_LABELS,
) -> pd.DataFrame:
    print("Iniciando Fase 2 — Sampling com MiniBatchKMeans")
    start = time.time()
    combined = _sample_kmeans(
        df,
        n_clusters=n_clusters,
        random_state=random_state,
        frac=frac,
        minority_labels=minority_labels,
    )
    label_col = "Label" if "Label" in combined.columns else combined.columns[-1]
    print(combined[label_col].value_counts())
    print(f"Shape final após sampling: {combined.shape}")
    print(f"Fase 2 concluída em {time.time() - start:.2f}s")
    return combined


def main() -> None:
    warnings.filterwarnings("ignore")

    total_start = time.time()

    parser = phase_parser("Fase 2 — sampling k-means")
    parser.add_argument("--n-clusters", type=int, default=1000)
    parser.add_argument("--frac", type=float, default=0.008)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--minority-labels",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MINORITY_LABELS),
        help="Classes preservadas (ex.: 6,1,4); perfil merged",
    )
    parser.add_argument("--auto-minority", action="store_true", help="Preservar todos os ataques (perfil fine)")
    args = parser.parse_args()

    paths = init_paths(args)
    inp = supervised_path(paths, P01_PREPROCESSED)
    out = supervised_path(paths, P02_SAMPLED_KMEANS)

    print(f"Lendo arquivo: {inp}")

    load_start = time.time()

    # Lê parquet
    df = pd.read_parquet(inp)

    if args.auto_minority:
        minority_labels = minority_labels_all_attacks(df)
        print(f"auto-minority: preservando {len(minority_labels)} classes de ataque")
    else:
        minority_labels = parse_minority_labels(args.minority_labels)

    print(f"Arquivo carregado em {time.time() - load_start:.2f}s")
    print(f"Shape original: {df.shape}")

    # Executa sampling
    sampled = sample_kmeans(
        df,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        frac=args.frac,
        minority_labels=minority_labels,
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
            "minority_labels": list(minority_labels),
            "auto_minority": args.auto_minority,
            "duration_s": round(time.time() - total_start, 4),
        }
    )
    report_path = write_report(paths.reports, "phase02_sample_kmeans", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()