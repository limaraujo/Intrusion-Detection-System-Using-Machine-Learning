"""Clustering: amostragem k-means (fase 2) e CL-k-means (anomaly)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics


@dataclass
class CLKmeansResult:
    y_pred: np.ndarray
    accuracy: float
    cluster_confidence: np.ndarray
    benign_clusters: set[int]
    model: MiniBatchKMeans


def sample_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 1000,
    random_state: int = 0,
    frac: float = 0.008,
    label_col: str = "Label",
    minority_labels: tuple[int, ...] = (6, 1, 4),
) -> pd.DataFrame:
    """MiniBatchKMeans na classe majoritária + amostragem 0.8% por cluster."""
    from .preprocessing import encode_labels

    encoded, _ = encode_labels(df, label_col=label_col)
    df_minor = encoded[encoded[label_col].isin(minority_labels)]
    df_major = encoded.drop(df_minor.index)

    X = df_major.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    df_major = df_major.copy()
    df_major["klabel"] = kmeans.labels_

    def typical_sampling(group: pd.DataFrame) -> pd.DataFrame:
        return group.sample(frac=frac)

    result = df_major.groupby("klabel", group_keys=False).apply(typical_sampling)
    result = result.drop(columns=["klabel"], errors="ignore")
    return pd.concat([result, df_minor], ignore_index=True)


def cl_kmeans_fit_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int | None = 0,
    metric: str = "euclidean",
) -> CLKmeansResult:
    """
    CL-k-means com probabilidade de cluster p_i (fração da classe majoritária no cluster).

    metric: 'euclidean' ou 'manhattan' (MiniBatchKMeans via metric_params se disponível;
    fallback euclidiano em versões antigas do sklearn).
    """
    kw: dict = {"n_clusters": n_clusters, "batch_size": batch_size}
    if random_state is not None:
        kw["random_state"] = random_state
    if metric == "manhattan":
        try:
            kw["metric"] = "manhattan"
        except TypeError:
            pass

    km = MiniBatchKMeans(**kw)
    train_labels = km.fit_predict(X_train)
    test_labels = km.predict(X_test)

    attack_counts = np.zeros(n_clusters, dtype=np.float64)
    benign_counts = np.zeros(n_clusters, dtype=np.float64)
    for i, cluster in enumerate(train_labels):
        if y_train[i] == 1:
            attack_counts[cluster] += 1
        else:
            benign_counts[cluster] += 1

    benign_clusters: set[int] = set()
    majority_ratio = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        total = attack_counts[c] + benign_counts[c]
        if total <= 0:
            majority_ratio[c] = 0.0
            benign_clusters.add(c)
            continue
        if attack_counts[c] <= benign_counts[c]:
            benign_clusters.add(c)
            majority_ratio[c] = benign_counts[c] / total
        else:
            majority_ratio[c] = attack_counts[c] / total

    mapped = np.zeros(len(y_test), dtype=np.int64)
    conf = np.zeros(len(y_test), dtype=np.float64)
    for i in range(len(y_test)):
        c = int(test_labels[i])
        mapped[i] = 0 if c in benign_clusters else 1
        conf[i] = majority_ratio[c]

    acc = float(metrics.accuracy_score(y_test, mapped))
    return CLKmeansResult(
        y_pred=mapped,
        accuracy=acc,
        cluster_confidence=conf,
        benign_clusters=benign_clusters,
        model=km,
    )


def cl_kmeans(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
    random_state: int | None = 0,
    metric: str = "euclidean",
) -> tuple[np.ndarray, float]:
    """API legada: retorna (predições, acurácia)."""
    res = cl_kmeans_fit_predict(
        X_train,
        X_test,
        y_train,
        y_test,
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        metric=metric,
    )
    return res.y_pred, res.accuracy
