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
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder

from config import (
    INTERMEDIATE_DIR,
    P01_PREPROCESSED,
    P02_SAMPLED_KMEANS,
    ensure_intermediate_dirs,
)


def sample_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 1000,
    frac: float = 0.008,
    random_state: int = 0,
) -> pd.DataFrame:

    print("Iniciando Fase 2 — Sampling com MiniBatchKMeans")

    start = time.time()

    # Detecta coluna Label
    
    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])



    # Copia dataframe
    df = df.copy()

    # Label Encoding
    print("Aplicando LabelEncoder...")

    le = LabelEncoder()

    df[label_col] = df[label_col].astype("object")
    df[label_col] = le.fit_transform(df[label_col]).astype(int)

    print("Classes encontradas:")
    print(dict(enumerate(le.classes_)))

    # Classes minoritárias
    # conforme notebook original
    print("Separando classes minoritárias...")

    df_minor = df[
        (df[label_col] == 6)
        | (df[label_col] == 1)
        | (df[label_col] == 4)
    ]

    # Classes majoritárias
    df_major = df.drop(df_minor.index)

    print(f"Majoritárias shape: {df_major.shape}")
    print(f"Minoritárias shape: {df_minor.shape}")

    # Remove Label para clustering
    X = df_major.drop([label_col], axis=1)

    print("Treinando MiniBatchKMeans...")

    cluster_start = time.time()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        verbose=0,
    )

    kmeans.fit(X)

    print(f"KMeans concluído em {time.time() - cluster_start:.2f}s")

    # Adiciona labels dos clusters
    df_major = df_major.copy()
    df_major["klabel"] = kmeans.labels_

    print("Executando amostragem por cluster...")

    sample_start = time.time()

    # Sampling proporcional por cluster
    def typical_sampling(group: pd.DataFrame) -> pd.DataFrame:
        return group.sample(
            frac=frac,
            random_state=random_state,
        )

    result = (
        df_major
        .groupby("klabel", group_keys=False)
        .apply(typical_sampling)
    )

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

    # Cria diretórios
    ensure_intermediate_dirs()

    # Arquivo de entrada da Fase 1
    inp = INTERMEDIATE_DIR / P01_PREPROCESSED.replace(".csv", ".parquet")

    # Arquivo de saída da Fase 2
    out = INTERMEDIATE_DIR / P02_SAMPLED_KMEANS.replace(".csv", ".parquet")

    print(f"Lendo arquivo: {inp}")

    load_start = time.time()

    # Lê parquet
    df = pd.read_parquet(inp)

    print(f"Arquivo carregado em {time.time() - load_start:.2f}s")
    print(f"Shape original: {df.shape}")

    # Executa sampling
    sampled = sample_kmeans(
        df,
        n_clusters=1000,
        frac=0.008,
        random_state=0,
    )

    # Salva parquet
    print("Salvando parquet...")

    save_start = time.time()

    sampled.to_parquet(out, index=False)

    print(f"Arquivo salvo: {out}")
    print(f"Tempo salvando: {time.time() - save_start:.2f}s")
    print(f"Tempo total: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()