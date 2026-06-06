"""
Fase 9 (anomaly): SMOTE no treino (partição df1 | df2), CL-k-means no conjunto de teste.

Lê a04_after_kpca.parquet e a06_test_slice.json da fase 8.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

try:
    from mth_ids_pipeline.io.anomaly_io import (
        apply_notebook_anomaly_smote,
        label_value_counts_dict,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.clustering import cl_kmeans
    from mth_ids_pipeline.config import A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import (
        apply_notebook_anomaly_smote,
        label_value_counts_dict,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from mth_ids_pipeline.core.clustering import cl_kmeans
    from mth_ids_pipeline.config import A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO

try:
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 9 — SMOTE + CL-k-means (anomaly)")
    add_work_dir(parser)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--smote-target",
        type=int,
        default=None,
        help="Notebook: default = nº de BENIGN no treino (18225 no demo PortScan)",
    )
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    df = pd.read_parquet(work / A04_AFTER_KPCA)
    meta = json.loads((work / A06_TEST_SLICE_INFO).read_text(encoding="utf-8"))
    n_df1 = int(meta["n_df1_rows"])
    label_col = "Label"

    X_all = df.drop(columns=[label_col]).values
    y_all = np.ravel(df[label_col].values)

    X_train = X_all[:n_df1]
    y_train = y_all[:n_df1]
    X_test = X_all[n_df1:]
    y_test = y_all[n_df1:]

    print(
        f"Partição LOAO (fase 9): treino={X_train.shape} labels={label_value_counts_dict(pd.Series(y_train))} | "
        f"teste={X_test.shape} labels={label_value_counts_dict(pd.Series(y_test))}"
    )

    counts_before = pd.Series(y_train).value_counts()
    X_train, y_train, did_smote, resolved = apply_notebook_anomaly_smote(
        X_train,
        y_train,
        smote_target=args.smote_target,
        random_state=args.random_state,
    )
    if did_smote and resolved is not None:
        print(f"SMOTE notebook: classe 1 -> {resolved} (alvo = n benignos no treino)")

    cols = [c for c in df.columns if c != label_col]
    train_out = work / A05_TRAIN_SMOTE
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
        "input": str(work / A04_AFTER_KPCA),
        "train_output": str(train_out),
        "test_slice_meta": str(work / A06_TEST_SLICE_INFO),
        "n_clusters": args.n_clusters,
        "random_state": args.random_state,
        "smote_target": resolved if did_smote else args.smote_target,
        "smote_target_resolved": resolved,
        "did_smote": did_smote,
        "train_counts_before": {str(k): int(v) for k, v in counts_before.items()},
        "train_counts_after": {str(k): int(v) for k, v in counts_after.items()},
        "test_label_counts": label_value_counts_dict(pd.Series(y_test)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "accuracy": float(acc),
    }
    report_path = write_report(paths.reports, "phase09_anomaly_cluster", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
