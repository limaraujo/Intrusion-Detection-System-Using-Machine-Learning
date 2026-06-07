"""
Detector anomaly global (Tabela X) — fases 7–11.

Treina um único modelo binário (benigno vs ataque) no mesmo 80%% do split
supervisionado (default 80/20). O hold-out 20%% fica reservado para ``run_eval`` (fase 13).

Use a **mesma** ``--intermediate-dir`` do supervisionado (ex.: merged).

Exemplo (CICIDS2017 / Tabela X):
  python -m mth_ids_pipeline.utils.merge_cicids --profile merged
  python -m mth_ids_pipeline.run_supervised --protocol paper --from 4 --to 6 --test-size 0.3
  python -m mth_ids_pipeline.run_global_anomaly --protocol paper --test-size 0.3
  python -m mth_ids_pipeline.run_eval \\
    --intermediate-dir data/pipeline_mth_ids_merged \\
    --work-dir data/pipeline_mth_ids_merged/anomaly/global \\
    --test-size 0.3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mth_ids_pipeline.config import (
    ANOMALY_GLOBAL_WORK_SUBDIR,
    DEFAULT_TEST_SIZE,
    INTERMEDIATE_DIR_MERGED,
)
from mth_ids_pipeline.io.subprocess_env import configure_stdio_utf8, utf8_subprocess_env
from mth_ids_pipeline.protocol import get_protocol_settings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_phase(module: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=_repo_root(), env=utf8_subprocess_env())


def main() -> None:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="MTH-IDS anomaly global (Tabela X)")
    parser.add_argument("--protocol", choices=["paper", "notebook"], default="paper")
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR_MERGED)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Default: <intermediate-dir>/anomaly/global",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--from-phase", type=int, default=7, choices=(7, 8, 9, 10, 11))
    parser.add_argument("--to-phase", type=int, default=11, choices=(7, 8, 9, 10, 11))
    parser.add_argument("--no-hpo", action="store_true")
    args = parser.parse_args()

    ps = get_protocol_settings(args.protocol)
    intermediate = Path(args.intermediate_dir)
    work = Path(args.work_dir) if args.work_dir else intermediate / ANOMALY_GLOBAL_WORK_SUBDIR
    work.mkdir(parents=True, exist_ok=True)
    report_dir = intermediate / "phase_reports"

    common = [
        "--intermediate-dir", str(intermediate),
        "--report-dir", str(report_dir),
        "--work-dir", str(work),
        "--random-state", str(args.random_state),
    ]

    phase10_extra = [
        *common,
        "--n-calls", str(ps.cl_hpo_n_calls),
        "--hpo-metric", ps.cl_hpo_metric,
        "--metrics", ",".join(ps.cl_kmeans_metrics),
    ]
    if args.no_hpo:
        phase10_extra.append("--skip-hpo")

    phases: list[tuple[int, str, list[str]]] = [
        (
            7,
            "mth_ids_pipeline.phases.phase07_anomaly_datasets",
            [*common, "--mode", "global", "--test-size", str(args.test_size)],
        ),
        (
            8,
            "mth_ids_pipeline.phases.phase08_anomaly_features",
            [
                *common,
                "--feature-fit-scope", "train",
                "--fcbf-k", str(ps.fcbf_k),
                "--kpca-components", str(ps.kpca_components),
                "--kpca-kernel", ps.kpca_kernel,
                "--ig-cumulative", str(ps.ig_cumulative),
                "--zscore-scope", ps.anomaly_zscore_scope,
            ]
            + (["--optimize-ig"] if ps.optimize_ig else [])
            + (["--optimize-kpca"] if ps.optimize_kpca else []),
        ),
        (
            9,
            "mth_ids_pipeline.phases.phase09_anomaly_cluster",
            [*common, "--n-clusters", "8"],
        ),
        (10, "mth_ids_pipeline.phases.phase10_anomaly_cluster_hpo", phase10_extra),
        (
            11,
            "mth_ids_pipeline.phases.phase11_anomaly_biased",
            [
                *common,
                "--p-star", str(ps.cl_p_star),
                "--biased-mode", ps.biased_mode,
            ]
            + (["--force-biased"] if ps.force_biased else [])
            + (
                ["--optimize-p-star", "--p-star-n-calls", str(ps.cl_hpo_n_calls)]
                if ps.optimize_p_star
                else []
            ),
        ),
    ]

    # Bootstrap 06_supervised_metrics if missing (biased tier 4)
    metrics_path = intermediate / "06_supervised_metrics.json"
    if not metrics_path.is_file():
        print(f"Aviso: {metrics_path} ausente — execute run_supervised antes (Tabela VII).")

    print(f"Anomaly global (Tabela X): intermediate={intermediate} work={work}")
    for phase_num, module, extra in phases:
        if phase_num < args.from_phase or phase_num > args.to_phase:
            continue
        _run_phase(module, extra)

    print(f"\nConcluído. Modelos em: {work / 'models' / 'anomaly'}")


if __name__ == "__main__":
    main()
