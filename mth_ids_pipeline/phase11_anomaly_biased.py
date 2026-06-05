"""
Fase 11 (anomaly tier 4): CL-k-means + biased classifiers B1/B2 + threshold p* (artigo).

- k padrão: lido de phase10_anomaly_cluster_hpo.json
- B1/B2: mesma família do melhor modelo da fase 6
- Gate: aplica biased só se melhorar F1 no hold-out interno do treino
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .anomaly_io import load_anomaly_splits
    from .biased_classifiers import (
        apply_biased_refinement,
        estimator_factory_for_supervised,
        pick_best_biased_mode,
        load_best_n_clusters,
        pick_best_supervised_model,
        train_biased_pair,
    )
    from .clustering import cl_kmeans_fit_predict
    from .cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from .config import DEFAULT_BIASED_MODE
    from .evaluation import binary_dr_far_f1
    from .reporting import write_report
except ImportError:
    from anomaly_io import load_anomaly_splits
    from biased_classifiers import (
        apply_biased_refinement,
        estimator_factory_for_supervised,
        pick_best_biased_mode,
        load_best_n_clusters,
        pick_best_supervised_model,
        train_biased_pair,
    )
    from clustering import cl_kmeans_fit_predict
    from cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from config import DEFAULT_BIASED_MODE
    from evaluation import binary_dr_far_f1
    from reporting import write_report


def _resolve_n_clusters(report_dir: Path, n_clusters: int | None) -> int:
    if n_clusters is not None:
        return int(n_clusters)
    auto = load_best_n_clusters(report_dir)
    if auto is not None:
        print(f"k automático da fase 10: {auto}")
        return auto
    print("Aviso: phase10 não encontrado; usando n_clusters=8")
    return 8


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 11 — CL-k-means + biased classifiers (tier 4)")
    add_work_dir(parser)
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--p-star", type=float, default=0.933)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--smote-target", type=int, default=18225)
    parser.add_argument("--metric", choices=("euclidean", "manhattan"), default="euclidean")
    parser.add_argument(
        "--biased-mode",
        choices=("auto", "both", "b1-only", "b2-only", "none"),
        default=DEFAULT_BIASED_MODE,
        help="Artigo: both (B1+B2); auto = gate por F1 no treino",
    )
    args = parser.parse_args()
    requested = args.biased_mode

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    report_dir = paths.reports
    n_clusters = _resolve_n_clusters(report_dir, args.n_clusters)
    metrics_path = paths.intermediate / "06_supervised_metrics.json"
    best_model_name = pick_best_supervised_model(metrics_path)
    factory = estimator_factory_for_supervised(best_model_name, random_state=args.random_state)
    print(f"Biased learners: família de '{best_model_name}'")

    X_train, X_test, y_train, y_test, did_smote = load_anomaly_splits(
        work,
        smote_target=args.smote_target,
        random_state=args.random_state,
    )

    cl_res = cl_kmeans_fit_predict(
        X_train,
        X_test,
        y_train,
        y_test,
        n_clusters=n_clusters,
        random_state=args.random_state,
        metric=args.metric,
    )
    print(f"CL-k-means (n={n_clusters}): accuracy={cl_res.accuracy:.4f}")
    print(classification_report(y_test, cl_res.y_pred))
    metrics_cl = binary_dr_far_f1(y_test, cl_res.y_pred)
    print(
        f"DR={metrics_cl['detection_rate']:.4f} FAR={metrics_cl['false_alarm_rate']:.4f} "
        f"F1={metrics_cl['f1']:.4f}"
    )

    y_final = cl_res.y_pred
    metrics_final = metrics_cl
    biased_stats: dict = {}
    selection_info: dict = {}

    effective_mode, selection_info = pick_best_biased_mode(
        X_train,
        y_train,
        n_clusters=n_clusters,
        random_state=args.random_state,
        metric=args.metric,
        p_star=args.p_star,
        estimator_factory=factory,
        requested=requested,  # type: ignore[arg-type]
    )
    print(f"\nModo biased selecionado: {effective_mode}")
    if selection_info.get("scores"):
        print("F1 validação interna:", selection_info["scores"])

    if effective_mode != "none":
        train_cl = cl_kmeans_fit_predict(
            X_train,
            X_train,
            y_train,
            y_train,
            n_clusters=n_clusters,
            random_state=args.random_state,
            metric=args.metric,
        )
        b1, b2, biased_stats = train_biased_pair(
            X_train,
            y_train,
            train_cl.y_pred,
            estimator_factory=factory,
            random_state=args.random_state,
            mode=effective_mode,
        )
        y_final = apply_biased_refinement(
            cl_res.y_pred,
            cl_res.cluster_confidence,
            X_test,
            b1=b1,
            b2=b2,
            p_star=args.p_star,
            mode=effective_mode,
        )
        metrics_final = binary_dr_far_f1(y_test, y_final)
        print(f"\nMTH-IDS anomaly (biased mode={effective_mode}, p*={args.p_star}):")
        print(classification_report(y_test, y_final))
        print(
            f"Acc={metrics_final['accuracy']:.4f} DR={metrics_final['detection_rate']:.4f} "
            f"FAR={metrics_final['false_alarm_rate']:.4f} F1={metrics_final['f1']:.4f}"
        )
        print("CM:\n", confusion_matrix(y_test, y_final))
    else:
        print("\nResultado final = CL-k-means (sem biased).")

    report = {
        "work_dir": str(work),
        "n_clusters": n_clusters,
        "n_clusters_source": "cli" if args.n_clusters is not None else "phase10",
        "p_star": args.p_star,
        "metric": args.metric,
        "random_state": args.random_state,
        "smote_target": args.smote_target,
        "did_smote": did_smote,
        "biased_mode_requested": requested,
        "biased_mode_applied": effective_mode,
        "supervised_model_for_biased": best_model_name,
        "mode_selection": selection_info,
        "cl_kmeans": metrics_cl,
        "mth_ids_anomaly": metrics_final,
        "biased_training": biased_stats,
    }
    report_path = write_report(report_dir, "phase11_anomaly_biased", report)
    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
