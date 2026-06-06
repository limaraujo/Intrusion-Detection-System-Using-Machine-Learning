"""
Fase 2: LabelEncoder + MiniBatchKMeans (k=1000) + amostragem 0.8% + minoritárias intactas.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

try:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        DEFAULT_MINORITY_LABELS,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        parse_minority_labels,
    )
    from mth_ids_pipeline.core.clustering import sample_kmeans as _sample_kmeans
    from mth_ids_pipeline.label_profiles import minority_labels_all_attacks
    from mth_ids_pipeline.io.reporting import dataset_report, write_report
except ImportError:
    from mth_ids_pipeline.cli import init_paths, phase_parser, supervised_path
    from mth_ids_pipeline.config import (
        DEFAULT_MINORITY_LABELS,
        P01_PREPROCESSED,
        P02_SAMPLED_KMEANS,
        parse_minority_labels,
    )
    from mth_ids_pipeline.core.clustering import sample_kmeans as _sample_kmeans
    from label_profiles import minority_labels_all_attacks
    from mth_ids_pipeline.io.reporting import dataset_report, write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    total_start = time.time()

    parser = phase_parser("Fase 2 — sampling k-means")
    parser.add_argument("--n-clusters", type=int, default=1000)
    parser.add_argument("--frac", type=float, default=0.008)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--minority-labels",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MINORITY_LABELS),
    )
    parser.add_argument("--auto-minority", action="store_true")
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Pula amostragem k-means e usa o dataset completo da fase 1",
    )
    args = parser.parse_args()

    paths = init_paths(args)
    df = pd.read_parquet(supervised_path(paths, P01_PREPROCESSED))

    if args.skip_sampling:
        sampled = df.copy()
        minority_labels: tuple[int, ...] = ()
    else:
        if args.auto_minority:
            minority_labels = minority_labels_all_attacks(df)
        else:
            minority_labels = parse_minority_labels(args.minority_labels)
        sampled = _sample_kmeans(
            df,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
            frac=args.frac,
            minority_labels=minority_labels,
        )

    out = supervised_path(paths, P02_SAMPLED_KMEANS)
    sampled.to_parquet(out, index=False)
    label_col = "Label" if "Label" in sampled.columns else sampled.columns[-1]
    print(sampled[label_col].value_counts())
    print(f"Salvo: {out} shape={sampled.shape}")

    report = dataset_report(sampled, label_col)
    report.update(
        {
            "input": str(supervised_path(paths, P01_PREPROCESSED)),
            "output": str(out),
            "n_clusters": args.n_clusters,
            "frac": args.frac,
            "minority_labels": list(minority_labels),
            "auto_minority": args.auto_minority,
            "skip_sampling": args.skip_sampling,
            "duration_s": round(time.time() - total_start, 4),
        }
    )
    write_report(paths.reports, "phase02_sample_kmeans", report)


if __name__ == "__main__":
    main()
