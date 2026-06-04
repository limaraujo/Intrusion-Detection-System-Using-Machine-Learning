"""
Orquestrador reprodutível do pipeline MTH-IDS.

Encapsula run_all com registro de seeds, versões e parâmetros.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DEFAULT_RAW_CSV, REPORTS_DIR, ensure_intermediate_dirs
from .reproducibility import DEFAULT_RANDOM_STATE, log_run_config, set_global_seeds


@dataclass
class ExperimentConfig:
    """Configuração reprodutível do experimento MTH-IDS."""

    random_state: int = DEFAULT_RANDOM_STATE
    raw_csv: Path = field(default_factory=lambda: DEFAULT_RAW_CSV)
    from_phase: int = 1
    to_phase: int = 6
    only_phase: int | None = None
    skip_phase6: bool = False
    run_hpo: bool = False
    report_dir: Path = field(default_factory=lambda: REPORTS_DIR)
    # Parâmetros metodológicos explícitos (notebook)
    kmeans_n_clusters: int = 1000
    kmeans_frac: float = 0.008
    test_size: float = 0.2
    ig_cumulative: float = 0.9
    fcbf_k: int = 20
    smote_strategy: dict[int, int] = field(default_factory=lambda: {2: 1000, 4: 1000})
    kpca_components: int = 10
    anomaly_benign_target: int = 1255
    anomaly_smote_target: int = 18225
    cl_kmeans_default: int = 16
    cl_kmeans_hpo: bool = True
    cl_p_star: float = 0.933
    paper_protocol: bool = False
    run_loao: bool = False


PHASE_MODULES = {
    1: "mth_ids_pipeline.phase01_load_preprocess",
    2: "mth_ids_pipeline.phase02_sample_kmeans",
    3: "mth_ids_pipeline.phase03_train_test_split",
    4: "mth_ids_pipeline.phase04_feature_engineering",
    5: "mth_ids_pipeline.phase05_smote",
    6: "mth_ids_pipeline.phase06_supervised_models",
    7: "mth_ids_pipeline.phase07_anomaly_datasets",
    8: "mth_ids_pipeline.phase08_anomaly_features",
    9: "mth_ids_pipeline.phase09_anomaly_cluster",
    10: "mth_ids_pipeline.phase10_anomaly_cluster_hpo",
    11: "mth_ids_pipeline.phase11_anomaly_biased",
    12: "mth_ids_pipeline.phase12_anomaly_loao",
}


def _phase_extra_args(phase: int, cfg: ExperimentConfig) -> list[str]:
    extra: list[str] = ["--report-dir", str(cfg.report_dir)]
    if phase == 1 and cfg.raw_csv:
        extra += ["--input", str(cfg.raw_csv)]
    if phase == 2:
        extra += [
            "--n-clusters", str(cfg.kmeans_n_clusters),
            "--frac", str(cfg.kmeans_frac),
            "--random-state", str(cfg.random_state),
        ]
    if phase == 3:
        test_size = 0.3 if cfg.paper_protocol else cfg.test_size
        extra += ["--test-size", str(test_size), "--random-state", str(cfg.random_state)]
    if phase == 4:
        extra += ["--fcbf-k", str(cfg.fcbf_k), "--random-state", str(cfg.random_state)]
    if phase == 5 and cfg.paper_protocol:
        extra.append("--paper-smote")
    if phase == 6:
        if not cfg.run_hpo:
            extra.append("--no-hpo")
        extra.append("--no-plots")
        if cfg.paper_protocol:
            extra += ["--cv-folds", "10"]
            if cfg.run_hpo:
                extra.append("--hpo-on-validation")
    if phase == 8:
        extra += [
            "--benign-target", str(cfg.anomaly_benign_target),
            "--random-state", str(cfg.random_state),
        ]
    if phase == 9:
        extra += [
            "--n-clusters", str(cfg.cl_kmeans_default),
            "--smote-target", str(cfg.anomaly_smote_target),
        ]
    if phase == 10 and not cfg.cl_kmeans_hpo:
        extra.append("--skip-hpo")
    if phase == 11:
        extra += [
            "--p-star", str(cfg.cl_p_star),
            "--random-state", str(cfg.random_state),
            "--smote-target", str(cfg.anomaly_smote_target),
            "--biased-mode", "auto",
        ]
        # n_clusters: omitido → fase 11 lê best_n_clusters da fase 10
    return extra


def run_experiment(cfg: ExperimentConfig) -> None:
    set_global_seeds(cfg.random_state)
    ensure_intermediate_dirs()

    config_path = log_run_config(
        cfg.report_dir,
        run_name="experiment_runner",
        config={
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        },
    )
    print(f"Configuração registrada em: {config_path}")

    start = cfg.only_phase or cfg.from_phase
    end = cfg.only_phase or cfg.to_phase
    repo_root = Path(__file__).resolve().parents[1]

    for phase in range(start, end + 1):
        if phase == 6 and cfg.skip_phase6:
            print(">> Pulando fase 6")
            continue
        if phase == 12 and not cfg.run_loao:
            print(">> Pulando fase 12 (use --run-loao)")
            continue
        module = PHASE_MODULES[phase]
        cmd = [sys.executable, "-m", module, *_phase_extra_args(phase, cfg)]
        print("\n>>", " ".join(cmd))
        subprocess.check_call(cmd, cwd=repo_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment runner reprodutível MTH-IDS")
    parser.add_argument("--from", dest="from_phase", type=int, default=1)
    parser.add_argument("--to", type=int, default=6)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--run-hpo", action="store_true", help="Executar HPO na fase 6 (lento)")
    parser.add_argument("--skip-phase6", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument(
        "--paper-protocol",
        action="store_true",
        help="Artigo: split 70/30, SMOTE 100k, CV 10-fold na fase 6",
    )
    parser.add_argument("--run-loao", action="store_true", help="Executar fase 12 (leave-one-attack-out)")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        random_state=args.random_state,
        raw_csv=args.raw_csv or DEFAULT_RAW_CSV,
        from_phase=args.from_phase,
        to_phase=args.to,
        only_phase=args.only,
        skip_phase6=args.skip_phase6,
        run_hpo=args.run_hpo,
        report_dir=args.report_dir,
        paper_protocol=args.paper_protocol,
        run_loao=args.run_loao,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
