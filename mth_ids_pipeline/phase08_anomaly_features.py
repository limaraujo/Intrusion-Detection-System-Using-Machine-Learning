"""
Fase 8 (anomaly): partição LOAO treino/teste, depois Z-score → IG → FCBF → KPCA.

Ajuste de cada etapa somente no treino; transform no teste com os mesmos parâmetros.
Sem IG/FCBF/KPCA no conjunto combinado treino+teste.

Saídas: a03_combined_normalized.parquet, a04_after_kpca.parquet, a06_test_slice.json,
        fitted_*.joblib / fitted_ig_features.txt
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import json
import numpy as np
import pandas as pd

try:
    from .anomaly_io import (
        build_loao_train_test_split,
        log_loao_partition,
        numeric_feature_columns,
        require_path,
        save_anomaly_fitted_artifacts,
        validate_loao_partition,
    )
    from .dimensionality_reduction import fit_kpca, transform_kpca
    from .feature_selection import AnomalyFeaturePipeline
except ImportError:
    from anomaly_io import (
        build_loao_train_test_split,
        log_loao_partition,
        numeric_feature_columns,
        require_path,
        save_anomaly_fitted_artifacts,
        validate_loao_partition,
    )
    from dimensionality_reduction import fit_kpca, transform_kpca
    from feature_selection import AnomalyFeaturePipeline

try:
    from ._bootstrap import ensure_repo_on_path
except ImportError:
    from _bootstrap import ensure_repo_on_path

try:
    from .cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from .config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        A03_COMBINED_NORMALIZED,
        A04_AFTER_KPCA,
    )
except ImportError:
    from cli import add_work_dir, init_paths, phase_parser, resolve_work_dir
    from config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        A03_COMBINED_NORMALIZED,
        A04_AFTER_KPCA,
    )

try:
    from .reporting import write_report
except ImportError:
    from reporting import write_report


def _df_from_scaled(
    X: np.ndarray,
    feature_names: list[str],
    y: np.ndarray,
    *,
    label_col: str = "Label",
) -> pd.DataFrame:
    out = pd.DataFrame(X, columns=feature_names)
    out[label_col] = y
    return out


def _df_from_kpca(X_kpca: np.ndarray, y: np.ndarray, *, label_col: str = "Label") -> pd.DataFrame:
    cols = [f"kpca_{i}" for i in range(X_kpca.shape[1])]
    out = pd.DataFrame(X_kpca, columns=cols)
    out[label_col] = y
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()

    parser = phase_parser("Fase 8 — features anomaly (IG+FCBF+KPCA, sem vazamento)")
    add_work_dir(parser)
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--kpca-components", type=int, default=10)
    parser.add_argument("--benign-target", type=int, default=None, help="Benignos no teste (LOAO usa 1:1)")
    parser.add_argument("--random-state", type=int, default=None)
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    start = time.time()
    total_steps = 7
    step = 0
    label_col = "Label"

    def tick(message: str) -> None:
        nonlocal step
        step += 1
        bar_len = 20
        filled = int(bar_len * step / total_steps)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start
        print(f"[{step}/{total_steps}] [{bar}] {message} (+{elapsed:.1f}s)")

    p1 = work / A01_WITHOUT_PORTSCAN
    p2 = work / A02_PORTSCAN_ONLY
    phase7_hint = (
        f"Execute a fase 7 antes:\n"
        f"  python -m mth_ids_pipeline.phase07_anomaly_datasets "
        f"--intermediate-dir {paths.intermediate} --work-dir {work} --attack-label <N>"
    )
    require_path(p1, hint=phase7_hint)
    require_path(p2, hint=phase7_hint)
    df1 = pd.read_parquet(p1)
    df2 = pd.read_parquet(p2)
    tick("Dados carregados (fase 7)")

    orig_meta: dict = {}
    round_meta_path = work / A00_LOAO_ROUND
    if round_meta_path.is_file():
        orig_meta = json.loads(round_meta_path.read_text(encoding="utf-8"))

    train_df, test_df, partition_meta = build_loao_train_test_split(
        df1,
        df2,
        label_col=label_col,
        benign_target=args.benign_target,
        random_state=args.random_state,
    )
    if orig_meta:
        partition_meta.update(
            {
                k: orig_meta[k]
                for k in (
                    "zero_day_label",
                    "train_original_label_counts",
                    "train_attack_labels_present",
                    "zero_day_fully_excluded_from_train",
                )
                if k in orig_meta
            }
        )
    validate_loao_partition(train_df, test_df, partition_meta, label_col=label_col)
    log_loao_partition(
        stage="fase 8 (pré-normalização)",
        train_df=train_df,
        test_df=test_df,
        meta=partition_meta,
        label_col=label_col,
    )
    feature_names = numeric_feature_columns(train_df, label_col=label_col)
    print(
        f"Partição LOAO: treino={train_df.shape}, teste={test_df.shape}, "
        f"features={len(feature_names)}"
    )
    tick("Partição treino/teste (antes de normalização)")

    X_train = train_df[feature_names].values
    y_train = np.ravel(train_df[label_col].values)
    X_test = test_df[feature_names].values
    y_test = np.ravel(test_df[label_col].values)
    print(f"  [bruto treino] shape={X_train.shape}")
    print(f"  [bruto teste] shape={X_test.shape}")

    pipeline = AnomalyFeaturePipeline(
        fcbf_k=args.fcbf_k,
        ig_cumulative=0.9,
        random_state=args.random_state if args.random_state is not None else 0,
    )
    X_train_fcbf = pipeline.fit(X_train, y_train, feature_names)
    X_test_fcbf = pipeline.transform(X_test, split="teste")
    tick("Z-score + IG + FCBF (fit treino, transform teste)")

    kpca = fit_kpca(X_train_fcbf, n_components=args.kpca_components, kernel="rbf")
    X_train_kpca = transform_kpca(X_train_fcbf, kpca, split="treino")
    X_test_kpca = transform_kpca(X_test_fcbf, kpca, split="teste")
    tick("KPCA (fit treino, transform teste)")

    train_z = pipeline.scaler.transform(X_train)
    test_z = pipeline.scaler.transform(X_test)
    df_norm = pd.concat(
        [
            _df_from_scaled(train_z, feature_names, y_train, label_col=label_col),
            _df_from_scaled(test_z, feature_names, y_test, label_col=label_col),
        ],
        ignore_index=True,
    )
    combined_path = work / A03_COMBINED_NORMALIZED
    df_norm.to_parquet(combined_path, index=False)

    df_kpca = pd.concat(
        [
            _df_from_kpca(X_train_kpca, y_train, label_col=label_col),
            _df_from_kpca(X_test_kpca, y_test, label_col=label_col),
        ],
        ignore_index=True,
    )
    kpca_path = work / A04_AFTER_KPCA
    df_kpca.to_parquet(kpca_path, index=False)

    artifact_paths = save_anomaly_fitted_artifacts(
        work,
        scaler=pipeline.scaler,
        ig_features=pipeline.ig_features,
        fcbf=pipeline.fcbf,
        kpca=kpca,
        partition_meta={
            **partition_meta,
            "fcbf_k": args.fcbf_k,
            "fcbf_selected_indices": pipeline.fcbf_selected_indices(),
            "kpca_components": args.kpca_components,
            "kpca_kernel": "rbf",
            "ig_cumulative": 0.9,
            "feature_count_raw": len(feature_names),
            "feature_count_ig": len(pipeline.ig_features),
            "feature_count_fcbf": int(X_train_fcbf.shape[1]),
        },
    )
    tick("Artefatos e parquets salvos")

    meta = partition_meta
    print(f"Salvo: {combined_path} shape={df_norm.shape}")
    print(f"Salvo: {kpca_path} shape={df_kpca.shape}")
    print(f"Meta/partição: {artifact_paths['partition']}")

    report = {
        "input_without_portscan": str(p1),
        "input_portscan": str(p2),
        "combined_output": str(combined_path),
        "kpca_output": str(kpca_path),
        "test_slice_meta": artifact_paths["partition"],
        "artifact_paths": artifact_paths,
        "protocol": "fit_train_transform_test",
        "fcbf_k": args.fcbf_k,
        "kpca_components": args.kpca_components,
        "zero_day_samples": meta["zero_day_samples"],
        "benign_sample_n": meta["benign_sampled"],
        "benign_available": meta["benign_available_in_train"],
        "benign_pairing_rule": meta["benign_pairing_rule"],
        "random_state": args.random_state,
        "train_shape": {"rows": int(train_df.shape[0]), "cols": int(len(feature_names))},
        "test_shape": {"rows": int(test_df.shape[0]), "cols": int(len(feature_names))},
        "kpca_shape": {"rows": int(df_kpca.shape[0]), "cols": int(df_kpca.shape[1] - 1)},
        "ig_feature_count": len(pipeline.ig_features),
        "fcbf_feature_count": int(X_train_fcbf.shape[1]),
    }
    report_path = write_report(paths.reports, "phase08_anomaly_features", report)
    print(f"Relatorio salvo em: {report_path}")
    tick("Relatorio JSON")


if __name__ == "__main__":
    main()
