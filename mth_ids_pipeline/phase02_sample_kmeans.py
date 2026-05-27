"""
Fase 2:
- LabelEncoder na coluna Label
- Separação entre classes minoritárias e majoritárias
- MiniBatchKMeans nas majoritárias
- Amostragem proporcional por cluster
- Concatena novamente com as minoritárias

Saída:
data/pipeline_mth_ids/02_sampled_kmeans.csv
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder


from .config import (
    INTERMEDIATE_DIR,
    P01_PREPROCESSED,
    P02_SAMPLED_KMEANS,
    ensure_intermediate_dirs,
)


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
    cols.insert(78, cols.pop(cols.index(label_col)))
    df_major = df_major.loc[:, cols]

    print(df_major)


    sample_start = time.time()
    
    # Sampling proporcional por cluster
    def typicalSampling(group):
        name = group.name # Não faz nada
        frac = 0.008
        return group.sample(frac=frac)

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

    # Cria diretórios
    ensure_intermediate_dirs()

    # Arquivo de entrada da Fase 1
    inp = INTERMEDIATE_DIR / P01_PREPROCESSED

    # Arquivo de saída da Fase 2
    out = INTERMEDIATE_DIR / P02_SAMPLED_KMEANS

    print(f"Lendo arquivo: {inp}")

    load_start = time.time()

    df = pd.read_csv(inp)

    print(f"Arquivo carregado em {time.time() - load_start:.2f}s")
    print(f"Shape original: {df.shape}")

    # Executa sampling
    sampled = sample_kmeans(
        df,
        n_clusters=1000,
        random_state=0,
        frac=0.008,
    )

    print("Salvando CSV...")

    save_start = time.time()

    sampled.to_csv(out, index=False)

    print(f"Arquivo salvo: {out}")
    print(f"Tempo salvando: {time.time() - save_start:.2f}s")
    print(f"Tempo total: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()
