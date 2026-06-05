"""
Fase 4: seleção por Information Gain (90% acumulado) + FCBF (k=20), segundo split 80/20.

Requer `FCBF_module.py` na raiz do repositório (ver README).

Saída: 04_train_after_fcbf.parquet, 04_test_after_fcbf.parquet, 04_selected_features.txt
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from ._bootstrap import ensure_repo_on_path
except ImportError:
    from _bootstrap import ensure_repo_on_path

try:
    from .cli import init_paths, phase_parser, supervised_path
    from .config import P02_SAMPLED_KMEANS, P04_SELECTED_FEATURES, P04_TEST_FSS, P04_TRAIN_FSS
    from .feature_selection import information_gain_feature_subset
except ImportError:
    from cli import init_paths, phase_parser, supervised_path
    from config import P02_SAMPLED_KMEANS, P04_SELECTED_FEATURES, P04_TEST_FSS, P04_TRAIN_FSS
    from feature_selection import information_gain_feature_subset

try:
    from .reporting import write_report
except ImportError:
    from reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()
    try:
        from mth_ids_pipeline.utils.FCBF_module import FCBFK
    except ImportError as e:
        raise ImportError(
            "Coloque FCBF_module.py na raiz do repositório (clone de "
            "https://github.com/SantiagoEG/FCBF_module)."
        ) from e

    parser = phase_parser("Fase 4 — IG + FCBF + split")
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    paths = init_paths(args)
    output_dir = paths.intermediate
    path_df = supervised_path(paths, P02_SAMPLED_KMEANS)
    df = pd.read_parquet(path_df)
    label_col = "Label" if "Label" in df.columns else df.columns[-1]
    feature_names = list(df.drop(columns=[label_col]).columns)
    X = df.drop(columns=[label_col]).values
    y = np.ravel(df.iloc[:, -1].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=args.random_state, stratify=y
    )

    ig_features = information_gain_feature_subset(X_train, feature_names, y_train, cumulative=0.9)
    (output_dir / P04_SELECTED_FEATURES).write_text("\n".join(ig_features), encoding="utf-8")

    X_fs = df[ig_features].values
    fcbf = FCBFK(k=args.fcbf_k)
    X_fss = fcbf.fit_transform(X_fs, y)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_fss, y, train_size=0.8, test_size=0.2, random_state=args.random_state, stratify=y
    )

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
    print(f"Features IG (pré-FCBF): {len(ig_features)}, colunas pós-FCBF: {len(cols)}")

    report = {
        "input": str(path_df),
        "selected_features_file": str(output_dir / P04_SELECTED_FEATURES),
        "train_output": str(train_path),
        "test_output": str(test_path),
        "ig_feature_count": len(ig_features),
        "fcbf_k": args.fcbf_k,
        "post_fcbf_feature_count": len(cols),
        "train_shape": {"rows": int(train_out.shape[0]), "cols": int(train_out.shape[1])},
        "test_shape": {"rows": int(test_out.shape[0]), "cols": int(test_out.shape[1])},
        "random_state": args.random_state,
    }
    report_path = write_report(paths.reports, "phase04_feature_engineering", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
