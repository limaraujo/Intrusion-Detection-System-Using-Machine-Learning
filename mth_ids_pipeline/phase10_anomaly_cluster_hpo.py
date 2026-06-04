"""
Fase 10 (anomaly): BO-GP para otimizar n_clusters do CL-k-means.

Reproduz a célula do notebook com gp_minimize (skopt).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

try:
    from .anomaly_io import load_anomaly_splits
    from .clustering import cl_kmeans
    from .config import ANOMALY_DIR, REPORTS_DIR, ensure_intermediate_dirs
    from .hyperparameter_optimization import optimize_cl_kmeans_clusters
    from .reporting import write_report
except ImportError:
    from anomaly_io import load_anomaly_splits
    from clustering import cl_kmeans
    from config import ANOMALY_DIR, REPORTS_DIR, ensure_intermediate_dirs
    from hyperparameter_optimization import optimize_cl_kmeans_clusters
    from reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 10 — BO-GP CL-k-means")
    parser.add_argument("--input-dir", type=Path, default=ANOMALY_DIR)
    parser.add_argument("--n-calls", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--smote-target", type=int, default=18225)
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--optimize-metric", action="store_true", help="BO-GP também otimiza métrica de distância")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    ensure_intermediate_dirs()

    X_train, X_test, y_train, y_test, did_smote = load_anomaly_splits(
        args.input_dir, smote_target=args.smote_target, random_state=args.random_state
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

        if args.optimize_metric:

            def objective_metric(n: int, metric: str) -> float:
                return objective(n, metric)

            best_n, best_acc, best_metric = optimize_cl_kmeans_clusters(
                objective_metric,
                n_calls=args.n_calls,
                random_state=args.random_state,
                optimize_metric=True,
            )
        else:
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

    report = {
        "input_dir": str(args.input_dir),
        "baseline_clusters": 8,
        "baseline_accuracy": float(baseline_acc),
        "best_n_clusters": int(best_n),
        "best_metric": best_metric,
        "best_accuracy": float(best_acc),
        "final_accuracy": float(final_acc),
        "optimize_metric": args.optimize_metric,
        "n_calls": args.n_calls,
        "random_state": args.random_state,
        "smote_target": args.smote_target,
        "did_smote": did_smote,
        "reproducible": True,
    }
    report_path = write_report(args.report_dir, "phase10_anomaly_cluster_hpo", report)
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
