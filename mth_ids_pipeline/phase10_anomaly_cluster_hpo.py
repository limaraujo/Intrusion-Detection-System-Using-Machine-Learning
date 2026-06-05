"""
Fase 10 (anomaly): BO-GP para otimizar n_clusters do CL-k-means.

Reproduz a célula do notebook com gp_minimize (skopt).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

try:
    from .anomaly_io import label_value_counts_dict, load_anomaly_splits
    from .cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from .clustering import cl_kmeans
    from .hyperparameter_optimization import optimize_cl_kmeans_clusters
    from .reporting import write_report
except ImportError:
    from anomaly_io import label_value_counts_dict, load_anomaly_splits
    from cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from clustering import cl_kmeans
    from hyperparameter_optimization import optimize_cl_kmeans_clusters
    from reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 10 — BO-GP CL-k-means")
    add_work_dir(parser)
    parser.add_argument("--n-calls", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--smote-target", type=int, default=18225)
    parser.add_argument("--skip-hpo", action="store_true")
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    X_train, X_test, y_train, y_test, did_smote = load_anomaly_splits(
        work, smote_target=args.smote_target, random_state=args.random_state
    )

    _, baseline_acc = cl_kmeans(
        X_train, X_test, y_train, y_test, n_clusters=8, random_state=args.random_state
    )
    print(f"Baseline CL-k-means (n=8): accuracy={baseline_acc:.4f}")

    best_metric = "euclidean"
    if args.skip_hpo:
        best_n, best_acc = 8, baseline_acc
    else:

        def objective(n: int, metric: str = "euclidean") -> float:
            _, acc = cl_kmeans(
                X_train,
                X_test,
                y_train,
                y_test,
                n_clusters=n,
                random_state=args.random_state,
                metric=metric,
            )
            return acc

        best_n, best_acc, best_metric = optimize_cl_kmeans_clusters(
            lambda n: objective(n),
            n_calls=args.n_calls,
            random_state=args.random_state,
        )

    print(f"Melhor n_clusters={best_n}, metric={best_metric}, accuracy={best_acc:.4f}")
    pred, final_acc = cl_kmeans(
        X_train,
        X_test,
        y_train,
        y_test,
        n_clusters=best_n,
        random_state=args.random_state,
        metric=best_metric,
    )
    print(f"Avaliação final n={best_n}: accuracy={final_acc:.4f}")

    train_label_counts = label_value_counts_dict(pd.Series(y_train))
    test_label_counts = label_value_counts_dict(pd.Series(y_test))
    print(
        f"Partição LOAO (fase 10): treino={X_train.shape} labels={train_label_counts} | "
        f"teste={X_test.shape} labels={test_label_counts}"
    )

    report = {
        "work_dir": str(work),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
        "baseline_clusters": 8,
        "baseline_accuracy": float(baseline_acc),
        "best_n_clusters": int(best_n),
        "best_metric": best_metric,
        "best_accuracy": float(best_acc),
        "final_accuracy": float(final_acc),
        "n_calls": args.n_calls,
        "random_state": args.random_state,
        "smote_target": args.smote_target,
        "did_smote": did_smote,
        "reproducible": True,
    }
    report_path = write_report(paths.reports, "phase10_anomaly_cluster_hpo", report)
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
