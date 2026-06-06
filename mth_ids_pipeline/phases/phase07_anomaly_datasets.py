"""
Fase 7 (anomaly branch): split leave-one-attack-out — treino sem o ataque zero-day, teste só com ele.

Saídas: a01_without_portscan.parquet, a02_portscan_only.parquet (nomes legados do notebook).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import json
import pandas as pd

try:
    from mth_ids_pipeline.io.anomaly_io import (
        _json_safe,
        build_anomaly_binary_split,
        label_value_counts_dict,
        log_loao_partition,
        loao_original_label_report,
        require_path,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir, supervised_path
    from mth_ids_pipeline.config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        P02_SAMPLED_KMEANS,
    )
except ImportError:
    from mth_ids_pipeline.io.anomaly_io import (
        _json_safe,
        build_anomaly_binary_split,
        label_value_counts_dict,
        log_loao_partition,
        loao_original_label_report,
        require_path,
    )
    from mth_ids_pipeline.cli import add_work_dir, init_paths, phase_parser, resolve_work_dir, supervised_path
    from mth_ids_pipeline.config import (
        A00_LOAO_ROUND,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        P02_SAMPLED_KMEANS,
    )

try:
    from mth_ids_pipeline.io.reporting import dataset_report, write_report
except ImportError:
    from mth_ids_pipeline.io.reporting import dataset_report, write_report

# PortScan no notebook (LabelEncoder sobre amostra CICIDS2017)
DEFAULT_ZERO_DAY_LABEL = 5


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 7 — datasets anomaly (LOAO)")
    add_work_dir(parser)
    parser.add_argument("--attack-label", type=int, default=DEFAULT_ZERO_DAY_LABEL)
    args = parser.parse_args()

    paths = init_paths(args)
    work = resolve_work_dir(args, paths)
    path_in = supervised_path(paths, P02_SAMPLED_KMEANS)
    require_path(
        path_in,
        hint=(
            "Execute as fases 1–2 antes (gera 02_sampled_kmeans.parquet).\n"
            f"  python -m mth_ids_pipeline.experiment_runner --intermediate-dir {paths.intermediate} "
            "--from-phase 1 --to-phase 2\n"
            "Depois rode a fase 7 com o mesmo --intermediate-dir e --work-dir."
        ),
    )
    df = pd.read_parquet(path_in)
    label_col = "Label"

    orig_report = loao_original_label_report(df, args.attack_label, label_col=label_col)
    df1, df2 = build_anomaly_binary_split(df, args.attack_label, label_col=label_col)

    p1 = work / A01_WITHOUT_PORTSCAN
    p2 = work / A02_PORTSCAN_ONLY
    df1.to_parquet(p1, index=False)
    df2.to_parquet(p2, index=False)
    round_meta_path = work / A00_LOAO_ROUND
    round_meta_path.write_text(
        json.dumps(_json_safe(orig_report), indent=2),
        encoding="utf-8",
    )
    print(f"Salvo: {p1} {df1.shape}, {p2} {df2.shape}")
    phase7_meta = {
        **orig_report,
        "zero_day_samples": int(len(df2)),
        "benign_sampled": 0,
        "benign_pairing_rule": "fase7_zero_day_only",
        "train_binary_label_counts": label_value_counts_dict(df1[label_col]),
        "test_binary_label_counts": label_value_counts_dict(df2[label_col]),
    }
    log_loao_partition(
        stage="fase 7 (split binário)",
        train_df=df1,
        test_df=df2,
        meta=phase7_meta,
        label_col=label_col,
    )

    report = {
        "input": str(path_in),
        "attack_label": args.attack_label,
        "without_portscan_output": str(p1),
        "portscan_output": str(p2),
        "without_portscan": dataset_report(df1, label_col),
        "portscan": dataset_report(df2, label_col),
        **orig_report,
    }
    report_path = write_report(paths.reports, "phase07_anomaly_datasets", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
