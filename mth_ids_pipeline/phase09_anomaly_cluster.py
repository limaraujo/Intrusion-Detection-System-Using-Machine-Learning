"""
Fase 9 (anomaly): SMOTE no treino (partição df1 | df2), CL-k-means no conjunto de teste.

Lê a04_after_kpca.csv e a06_test_slice.json da fase 8.
"""

from __future__ import annotations

import argparse
import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix

from .config import ANOMALY_DIR, A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO, ensure_intermediate_dirs


def cl_kmeans(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int = 100,
) -> tuple[np.ndarray, float]:
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0)
    result = km.fit_predict(X_train)
    result2 = km.predict(X_test)

    a = np.zeros(n_clusters)
    b = np.zeros(n_clusters)
    for v in range(n_clusters):
        for i in range(len(y_train)):
            if result[i] == v:
                if y_train[i] == 1:
                    a[v] += 1
                else:
                    b[v] += 1
    list1: list[int] = []
    list2: list[int] = []
    for v in range(n_clusters):
        if a[v] <= b[v]:
            list1.append(v)
        else:
            list2.append(v)
    mapped = result2.copy()
    for v in range(len(y_test)):
        if result2[v] in list1:
            mapped[v] = 0
        elif result2[v] in list2:
            mapped[v] = 1
    acc = metrics.accuracy_score(y_test, mapped)
    return mapped, acc


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 9 — SMOTE + CL-k-means (anomaly)")
    parser.add_argument("--dir", type=Path, default=ANOMALY_DIR)
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()
    ensure_intermediate_dirs()

    df = pd.read_csv(args.dir / A04_AFTER_KPCA)
    meta = json.loads((args.dir / A06_TEST_SLICE_INFO).read_text(encoding="utf-8"))
    n_df1 = int(meta["n_df1_rows"])
    label_col = "Label"

    X_all = df.drop(columns=[label_col]).values
    y_all = np.ravel(df[label_col].values)

    X_train = X_all[:n_df1]
    y_train = y_all[:n_df1]
    X_test = X_all[n_df1:]
    y_test = y_all[n_df1:]

    kw: dict = {"sampling_strategy": {1: 18225}}
    if "n_jobs" in inspect.signature(SMOTE.__init__).parameters:
        kw["n_jobs"] = -1
    smote = SMOTE(**kw)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    cols = [c for c in df.columns if c != label_col]
    pd.DataFrame(X_train, columns=cols).assign(**{label_col: y_train}).to_csv(args.dir / A05_TRAIN_SMOTE, index=False)
    print(f"Salvo treino pós-SMOTE: {args.dir / A05_TRAIN_SMOTE}")

    pred, acc = cl_kmeans(X_train, X_test, y_train, y_test, n_clusters=args.n_clusters)
    print(classification_report(y_test, pred))
    print("Accuracy:", acc)
    print("CM:\n", confusion_matrix(y_test, pred))


if __name__ == "__main__":
    main()
