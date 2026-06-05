"""Caminhos e nomes de arquivos intermediários do pipeline MTH-IDS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "pipeline_mth_ids"
ANOMALY_DIR = INTERMEDIATE_DIR / "anomaly"
REPORTS_DIR = INTERMEDIATE_DIR / "phase_reports"

DEFAULT_MINORITY_LABELS: tuple[int, ...] = (6, 1, 4)

# CICIDS2017 merged — LabelEncoder alfabético:
# 0=BENIGN 1=Bot 2=BruteForce 3=DoS 4=Infiltration 5=PortScan 6=WebAttack
CICIDS2017_MERGED_LABEL_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "BruteForce",
    3: "DoS",
    4: "Infiltration",
    5: "PortScan",
    6: "WebAttack",
}

# SMOTE supervisionado — notebook MTH_IDS_IoTJ.ipynb
NOTEBOOK_SMOTE_TARGETS: dict[int, int] = {
    2: 1_000,  # BruteForce
    4: 1_000,  # Infiltration
}

# SMOTE supervisionado — artigo (Yang et al., IEEE IoT Journal 2022)
PAPER_SMOTE_TARGETS: dict[int, int] = {
    1: 100_000,  # Bot
    2: 100_000,  # BruteForce
    4: 100_000,  # Infiltration
    6: 100_000,  # WebAttack
}

DEFAULT_TEST_SIZE = 0.2  # notebook: split 80/20 na engenharia de features
DEFAULT_CV_FOLDS = 0  # notebook: HPO por acurácia no hold-out
DEFAULT_META_LEARNER = "xgb"  # notebook: stacking meta XGBoost + HPO
DEFAULT_SMOTE_TARGETS = NOTEBOOK_SMOTE_TARGETS
DEFAULT_HPO_ON_VALIDATION = False  # notebook: accuracy_score(y_test, y_pred)
DEFAULT_BIASED_MODE = "both"  # tier 4: B1 + B2
DEFAULT_ANOMALY_SMOTE_TARGET = 18225
DEFAULT_CL_P_STAR = 0.933


@dataclass(frozen=True)
class PipelinePaths:
    intermediate: Path
    anomaly: Path
    reports: Path


def parse_minority_labels(value: str) -> tuple[int, ...]:
    labels = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not labels:
        raise ValueError("minority-labels não pode ser vazio")
    return labels


def get_pipeline_paths(
    intermediate_dir: Path | None = None,
    report_dir: Path | None = None,
) -> PipelinePaths:
    inter = Path(intermediate_dir) if intermediate_dir is not None else INTERMEDIATE_DIR
    reports = Path(report_dir) if report_dir is not None else inter / "phase_reports"
    return PipelinePaths(intermediate=inter, anomaly=inter / "anomaly", reports=reports)


def ensure_pipeline_dirs(paths: PipelinePaths) -> None:
    paths.intermediate.mkdir(parents=True, exist_ok=True)
    paths.anomaly.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)

# Entrada bruta CICIDS2017 (ver label_profiles.py: merged vs fine)
DEFAULT_RAW_CSV = DATA_DIR / "CICIDS2017.csv"
DEFAULT_RAW_CSV_FINE = DATA_DIR / "CICIDS2017_fine.csv"
INTERMEDIATE_DIR_MERGED = DATA_DIR / "pipeline_mth_ids_merged"
INTERMEDIATE_DIR_FINE = DATA_DIR / "pipeline_mth_ids_fine"

# Fase supervisionada (known attacks)
P01_PREPROCESSED = "01_preprocessed.csv"
P02_SAMPLED_KMEANS = "02_sampled_kmeans.csv"
P03_TRAIN = "03_train.csv"
P03_TEST = "03_test.csv"
P04_TRAIN_FSS = "04_train_after_fcbf.parquet"
P04_TEST_FSS = "04_test_after_fcbf.parquet"
P04_SELECTED_FEATURES = "04_selected_features.txt"
P05_TRAIN_SMOTE = "05_train_after_smote.parquet"
P05_TEST = "05_test_unchanged.parquet"

# Ramo anomaly-based
A00_LOAO_ROUND = "a00_loao_round.json"
A01_WITHOUT_PORTSCAN = "a01_without_portscan.parquet"
A02_PORTSCAN_ONLY = "a02_portscan_only.parquet"
A03_COMBINED_NORMALIZED = "a03_combined_normalized.parquet"
A04_AFTER_KPCA = "a04_after_kpca.parquet"
A05_TRAIN_SMOTE = "a05_train_after_smote.parquet"
A06_TEST_SLICE_INFO = "a06_test_slice.json"


def ensure_intermediate_dirs(intermediate_dir: Path | None = None) -> PipelinePaths:
    paths = get_pipeline_paths(intermediate_dir)
    ensure_pipeline_dirs(paths)
    return paths
