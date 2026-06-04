"""
Fase 9 (anomaly): SMOTE no treino (partição df1 | df2), CL-k-means no conjunto de teste.

Lê a04_after_kpca.parquet e a06_test_slice.json da fase 8.
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
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .clustering import cl_kmeans
    from .config import ANOMALY_DIR, A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO, REPORTS_DIR, ensure_intermediate_dirs
except ImportError:
    from clustering import cl_kmeans
    from config import ANOMALY_DIR, A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO, REPORTS_DIR, ensure_intermediate_dirs

try:
    from .reporting import write_report
except ImportError:
    from reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 9 — SMOTE + CL-k-means (anomaly)")
    parser.add_argument("--input-dir", type=Path, default=None, help="Diretorio de entrada (fase 8)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio de saida")
    parser.add_argument("--dir", type=Path, default=None, help="Alias para --input-dir e --output-dir")
    parser.add_argument("--n-clusters", type=int, default=8, help="Numero de clusters")
    parser.add_argument("--random-state", type=int, default=0, help="Seed CL-k-means e SMOTE")
    parser.add_argument("--smote-target", type=int, default=18225, help="Target SMOTE para classe 1")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()
    ensure_intermediate_dirs()

    input_dir = args.input_dir or args.dir or ANOMALY_DIR
    output_dir = args.output_dir or args.dir or ANOMALY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_dir / A04_AFTER_KPCA)
    meta = json.loads((input_dir / A06_TEST_SLICE_INFO).read_text(encoding="utf-8"))
    n_df1 = int(meta["n_df1_rows"])
    label_col = "Label"

    X_all = df.drop(columns=[label_col]).values
    y_all = np.ravel(df[label_col].values)

    X_train = X_all[:n_df1]
    y_train = y_all[:n_df1]
    X_test = X_all[n_df1:]
    y_test = y_all[n_df1:]

    counts_before = pd.Series(y_train).value_counts()
    target = int(args.smote_target)
    did_smote = False
    if 1 in counts_before and target > int(counts_before[1]):
        kw: dict = {"sampling_strategy": {1: target}}
        if "random_state" in inspect.signature(SMOTE.__init__).parameters:
            kw["random_state"] = args.random_state
        if "n_jobs" in inspect.signature(SMOTE.__init__).parameters:
            kw["n_jobs"] = -1
        smote = SMOTE(**kw)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        did_smote = True

    cols = [c for c in df.columns if c != label_col]
    train_out = output_dir / A05_TRAIN_SMOTE
    pd.DataFrame(X_train, columns=cols).assign(**{label_col: y_train}).to_parquet(train_out, index=False)
    print(f"Salvo treino pós-SMOTE: {train_out}")

    pred, acc = cl_kmeans(
        X_train, X_test, y_train, y_test,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )
    print(classification_report(y_test, pred))
    print("Accuracy:", acc)
    print("CM:\n", confusion_matrix(y_test, pred))

    counts_after = pd.Series(y_train).value_counts()
    report = {
        "input": str(input_dir / A04_AFTER_KPCA),
        "train_output": str(train_out),
        "test_slice_meta": str(input_dir / A06_TEST_SLICE_INFO),
        "n_clusters": args.n_clusters,
        "random_state": args.random_state,
        "smote_target": target,
        "did_smote": did_smote,
        "train_counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "train_counts_after": {str(k): int(v) for k, v in counts_after.items()},
        "accuracy": float(acc),
    }
    report_path = write_report(args.report_dir, "phase09_anomaly_cluster", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
