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

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder

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


def sample_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 1000,
    random_state: int = 0,
    frac: float = 0.008,
) -> pd.DataFrame:

    print("Iniciando Fase 2 — Sampling com MiniBatchKMeans")

    start = time.time()


    # Detecta coluna Label
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    labelencoder = LabelEncoder()

    # Substitui a coluna inteira para evitar setitem em dtype string do PyArrow
    df[label_col] = labelencoder.fit_transform(df[label_col].astype(str)).astype("int64")
    print(f"Contagem de valores na coluna Label: {label_col}")

    print("Classes encontradas:")
    print(df[label_col].value_counts())

    # Classes minoritárias (6, 1, 4)
    df_minor = df[df[label_col].isin([6, 1, 4])]
    
    # Classes majoritárias
    df_major = df.drop(df_minor.index)

    print(f"Majoritárias shape: {df_major.shape}")
    print(f"Minoritárias shape: {df_minor.shape}")

    # Remove Label para clustering
    X = df_major.drop(columns=[label_col]).to_numpy(
        dtype=np.float32
    )
    
    print("Treinando MiniBatchKMeans...")

    cluster_start = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state, 
    )

    kmeans.fit(X)

    print(f"KMeans concluído em {time.time() - cluster_start:.2f}s")

    # Adiciona labels dos clusters
    df_major["klabel"] = kmeans.labels_
    
    print("Contagem de valores na coluna klabel:")
    print(df_major['klabel'].value_counts())

    cols = list(df_major)
    insert_at = min(78, len(cols))
    cols.insert(insert_at, cols.pop(cols.index(label_col)))
    df_major = df_major.loc[:, cols]

    print(df_major)


    sample_start = time.time()
    
    # Sampling proporcional por cluster
    def typicalSampling(group):
        name = group.name # Não faz nada
        return group.sample(frac=frac, random_state=random_state)

    result = df_major.groupby(
        'klabel', group_keys=False
    ).apply(typicalSampling)
    
    print(result['Label'].value_counts())

    print(f"Sampling concluído em {time.time() - sample_start:.2f}s")

    # Remove coluna auxiliar
    result = result.drop(columns=["klabel"], errors="ignore")

    # Junta novamente com minoritárias
    print("Concatenando classes minoritárias...")

    combined = pd.concat(
        [result, df_minor],
        ignore_index=True,
    )

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