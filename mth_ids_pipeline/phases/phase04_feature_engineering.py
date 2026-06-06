"""
Fase 4: IG (α acumulado) → FCBF (k=20) → split estratificado.

Notebook (--scale-mode phase1, --fcbf-scope full): Z-score da fase 1, FCBF no dataset completo.
Artigo (--scale-mode split, --fcbf-scope train): StandardScaler no treino, FCBF só no treino.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from mth_ids_pipeline.utils.bootstrap import ensure_repo_on_path
except ImportError:
    from mth_ids_pipeline.utils.bootstrap import ensure_repo_on_path

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import P02_SAMPLED_KMEANS, P04_SELECTED_FEATURES, P04_TEST_FSS, P04_TRAIN_FSS
    from mth_ids_pipeline.core.feature_selection import (
        fit_fcbf,
        information_gain_feature_subset,
        transform_fcbf,
    )
    from mth_ids_pipeline.io.reporting import write_report
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import P02_SAMPLED_KMEANS, P04_SELECTED_FEATURES, P04_TEST_FSS, P04_TRAIN_FSS
    from mth_ids_pipeline.core.feature_selection import fit_fcbf, information_gain_feature_subset, transform_fcbf
    from mth_ids_pipeline.io.reporting import write_report


def _scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    X_all: np.ndarray,
    *,
    scale_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """phase1 = IoTJ (sem reescalar); split = StandardScaler fit no treino."""
    if scale_mode == "phase1":
        return X_train, X_test, X_all
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_all_s = scaler.transform(X_all)
    return X_train_s, X_test_s, X_all_s


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()
    try:
        from mth_ids_pipeline.utils.FCBF_module import FCBFK
    except ImportError as e:
        raise ImportError("Coloque FCBF_module.py na raiz do repositório.") from e

    parser = phase_parser("Fase 4 — IG + FCBF + split")
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--fcbf-scope",
        choices=("train", "full"),
        default="train",
        help="train=artigo (FCBF só no treino); full=notebook",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("phase1", "split"),
        default=None,
        help="phase1=IoTJ (Z-score fase 1); split=StandardScaler no treino (artigo)",
    )
    parser.add_argument("--ig-cumulative", type=float, default=0.9)
    parser.add_argument("--optimize-ig", action="store_true", help="BO-GP para α IG (artigo)")
    parser.add_argument("--ig-hpo-calls", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=10, help="CV para BO-GP α")
    args = parser.parse_args()

    scale_mode = args.scale_mode or ("phase1" if args.fcbf_scope == "full" else "split")
    train_size = 1.0 - float(args.test_size)

    paths = init_paths(args)
    output_dir = paths.intermediate
    df = pd.read_parquet(supervised_path(paths, P02_SAMPLED_KMEANS))
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    feature_names = list(df.drop(columns=[label_col]).columns)
    X = df.drop(columns=[label_col]).values.astype(np.float64)
    y = np.ravel(df[label_col].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    X_train_s, X_test_s, X_all_s = _scale_features(X_train, X_test, X, scale_mode=scale_mode)

    ig_cumulative = float(args.ig_cumulative)
    ig_hpo_report: dict | None = None
    if args.optimize_ig:
        try:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_ig_alpha
        except ImportError:
            from mth_ids_pipeline.core.hyperparameter_optimization import optimize_ig_alpha
        hpo = optimize_ig_alpha(
            X_train_s,
            y_train,
            feature_names,
            fcbf_k=args.fcbf_k,
            n_calls=args.ig_hpo_calls,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
        )
        ig_cumulative = hpo.best_alpha
        ig_hpo_report = {
            "best_alpha": hpo.best_alpha,
            "best_cv_accuracy": hpo.best_score,
            "trials": hpo.trials,
        }
        print(f"BO-GP IG: alpha={ig_cumulative:.4f}, CV acc={hpo.best_score:.4f}")

    ig_features = information_gain_feature_subset(
        X_train_s, feature_names, y_train, cumulative=ig_cumulative
    )
    (output_dir / P04_SELECTED_FEATURES).write_text("\n".join(ig_features), encoding="utf-8")
    ig_idx = [feature_names.index(n) for n in ig_features]

    if args.fcbf_scope == "full":
        X_ig = X_all_s[:, ig_idx]
        fcbf = FCBFK(k=args.fcbf_k)
        X_fss = fcbf.fit_transform(X_ig, y)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_fss,
            y,
            train_size=train_size,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    else:
        X_train_ig = X_train_s[:, ig_idx]
        X_test_ig = X_test_s[:, ig_idx]
        fcbf = fit_fcbf(X_train_ig, y_train, k=args.fcbf_k)
        X_train2 = transform_fcbf(fcbf, X_train_ig)
        X_test2 = transform_fcbf(fcbf, X_test_ig)
        y_train2, y_test2 = y_train, y_test

    cols = [f"f{i}" for i in range(X_train2.shape[1])]
    train_out = pd.DataFrame(X_train2, columns=cols)
    train_out[label_col] = y_train2
    test_out = pd.DataFrame(X_test2, columns=cols)
    test_out[label_col] = y_test2

    train_path = output_dir / P04_TRAIN_FSS
    test_path = output_dir / P04_TEST_FSS
    train_out.to_parquet(train_path, index=False)
    test_out.to_parquet(test_path, index=False)
    print(f"Salvo: {train_path} {train_out.shape}")
    print(f"Salvo: {test_path} {test_out.shape}")

    report = {
        "ig_feature_count": len(ig_features),
        "ig_cumulative": ig_cumulative,
        "post_fcbf_feature_count": len(cols),
        "test_size": float(args.test_size),
        "fcbf_scope": args.fcbf_scope,
        "scale_mode": scale_mode,
        "optimize_ig": args.optimize_ig,
        "train_output": str(train_path),
        "test_output": str(test_path),
    }
    if ig_hpo_report:
        report["ig_hpo"] = ig_hpo_report
    write_report(paths.reports, "phase04_feature_engineering", report)


if __name__ == "__main__":
    main()
