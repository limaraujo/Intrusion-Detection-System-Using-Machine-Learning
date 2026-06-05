"""
Orquestrador reprodutível do pipeline MTH-IDS.

Defaults alinhados ao notebook MTH_IDS_IoTJ.ipynb (supervisionado).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import (
    DEFAULT_ANOMALY_SMOTE_TARGET,
    DEFAULT_BIASED_MODE,
    DEFAULT_CL_P_STAR,
    DEFAULT_CV_FOLDS,
    DEFAULT_HPO_ON_VALIDATION,
    DEFAULT_META_LEARNER,
    DEFAULT_MINORITY_LABELS,
    DEFAULT_RAW_CSV,
    DEFAULT_SMOTE_TARGETS,
    DEFAULT_TEST_SIZE,
    REPORTS_DIR,
    ensure_intermediate_dirs,
    get_pipeline_paths,
)
from .label_profiles import get_label_profile
from .reproducibility import DEFAULT_RANDOM_STATE, log_run_config, set_global_seeds


@dataclass
class ExperimentConfig:
    """Configuração reprodutível do experimento MTH-IDS."""

    random_state: int = DEFAULT_RANDOM_STATE
    raw_csv: Path = field(default_factory=lambda: DEFAULT_RAW_CSV)
    intermediate_dir: Path | None = None
    minority_labels: str = field(
        default_factory=lambda: ",".join(str(x) for x in DEFAULT_MINORITY_LABELS)
    )
    from_phase: int = 1
    to_phase: int = 6
    only_phase: int | None = None
    skip_phase6: bool = False
    run_hpo: bool = True
    report_dir: Path | None = None
    kmeans_n_clusters: int = 1000
    kmeans_frac: float = 0.008
    test_size: float = DEFAULT_TEST_SIZE
    ig_cumulative: float = 0.9
    fcbf_k: int = 20
    smote_strategy: dict[int, int] = field(default_factory=lambda: dict(DEFAULT_SMOTE_TARGETS))
    kpca_components: int = 10
    anomaly_benign_target: int | None = None
    anomaly_smote_target: int = DEFAULT_ANOMALY_SMOTE_TARGET
    cl_kmeans_default: int = 16
    cl_kmeans_hpo: bool = True
    cl_p_star: float = DEFAULT_CL_P_STAR
    cv_folds: int = DEFAULT_CV_FOLDS
    meta_learner: str = DEFAULT_META_LEARNER
    hpo_on_validation: bool = DEFAULT_HPO_ON_VALIDATION
    biased_mode: str = DEFAULT_BIASED_MODE
    run_loao: bool = False
    label_profile: str | None = None
    auto_minority: bool = False


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


def _resolved_report_dir(cfg: ExperimentConfig) -> Path:
    if cfg.report_dir is not None:
        return cfg.report_dir
    if cfg.intermediate_dir is not None:
        return cfg.intermediate_dir / "phase_reports"
    return REPORTS_DIR


def _phase_extra_args(phase: int, cfg: ExperimentConfig) -> list[str]:
    extra: list[str] = ["--report-dir", str(_resolved_report_dir(cfg))]
    if cfg.intermediate_dir is not None:
        extra += ["--intermediate-dir", str(cfg.intermediate_dir)]

    if phase == 1 and cfg.raw_csv:
        extra += ["--input", str(cfg.raw_csv)]
    if phase == 2:
        extra += [
            "--n-clusters", str(cfg.kmeans_n_clusters),
            "--frac", str(cfg.kmeans_frac),
            "--random-state", str(cfg.random_state),
        ]
        if cfg.auto_minority:
            extra.append("--auto-minority")
        elif cfg.minority_labels:
            extra += ["--minority-labels", cfg.minority_labels]
    if phase == 4:
        extra += ["--fcbf-k", str(cfg.fcbf_k), "--random-state", str(cfg.random_state)]
    if phase == 6:
        if not cfg.run_hpo:
            extra.append("--no-hpo")
        extra.append("--no-plots")
        extra += [
            "--cv-folds", str(cfg.cv_folds),
            "--meta-learner", cfg.meta_learner,
        ]
        if cfg.hpo_on_validation:
            extra.append("--hpo-on-validation")
    if phase == 8:
        extra += ["--random-state", str(cfg.random_state)]
        if cfg.anomaly_benign_target is not None:
            extra += ["--benign-target", str(cfg.anomaly_benign_target)]
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
            "--biased-mode", cfg.biased_mode,
        ]
    return extra


def run_experiment(cfg: ExperimentConfig) -> None:
    set_global_seeds(cfg.random_state)
    paths = get_pipeline_paths(cfg.intermediate_dir, _resolved_report_dir(cfg))
    ensure_intermediate_dirs(paths.intermediate)

    config_path = log_run_config(
        paths.reports,
        run_name="experiment_runner",
        config={
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
            "protocol": "notebook",
            "resolved_intermediate_dir": str(paths.intermediate),
            "resolved_report_dir": str(paths.reports),
        },
    )
    print("Protocolo: notebook MTH_IDS_IoTJ.ipynb")
    print(f"Configuração registrada em: {config_path}")
    print(f"Artefatos: {paths.intermediate}")

    start = cfg.only_phase or cfg.from_phase
    end = cfg.only_phase or cfg.to_phase
    repo_root = Path(__file__).resolve().parents[1]

    for phase in range(start, end + 1):
        if phase == 3:
            print(">> Pulando fase 3 (split 70/30 — ausente no notebook)")
            continue
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
    parser = argparse.ArgumentParser(
        description="Pipeline MTH-IDS — protocolo do notebook MTH_IDS_IoTJ.ipynb",
    )
    parser.add_argument(
        "--label-profile",
        choices=["merged", "fine"],
        default="merged",
        help="merged: Tabela VII (~7 famílias); fine: Tabela IX (~14 LOAO)",
    )
    parser.add_argument("--from", dest="from_phase", type=int, default=1)
    parser.add_argument("--to", type=int, default=6)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--intermediate-dir", type=Path, default=None)
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-hpo", action="store_true", help="Fase 6: hiperparâmetros fixos (sem BO-TPE)")
    parser.add_argument("--skip-phase6", action="store_true")
    parser.add_argument("--run-loao", action="store_true", help="Fase 12 (Tabela IX)")
    args = parser.parse_args()

    profile = get_label_profile(args.label_profile)
    minority = profile.minority_labels_csv() or ""

    cfg = ExperimentConfig(
        random_state=args.random_state,
        raw_csv=args.raw_csv or profile.raw_csv,
        intermediate_dir=args.intermediate_dir or profile.intermediate_dir,
        minority_labels=minority or ",".join(str(x) for x in DEFAULT_MINORITY_LABELS),
        auto_minority=profile.auto_minority,
        label_profile=args.label_profile,
        from_phase=args.from_phase,
        to_phase=args.to,
        only_phase=args.only,
        skip_phase6=args.skip_phase6,
        run_hpo=not args.no_hpo,
        run_loao=args.run_loao,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
