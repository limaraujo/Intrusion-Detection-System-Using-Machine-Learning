"""Caminhos e nomes de arquivos intermediários do pipeline MTH-IDS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
INTERMEDIATE_DIR = DATA_DIR / "pipeline_mth_ids"
ANOMALY_DIR = INTERMEDIATE_DIR / "anomaly"
REPORTS_DIR = INTERMEDIATE_DIR / "phase_reports"

DEFAULT_MINORITY_LABELS: tuple[int, ...] = (6, 1, 4)

# Fine: ver ``label_profiles.compute_fine_minority_labels_notebook_aligned()`` (Bot/Infiltration/WebAttack)
CICIDS2017_FINE_LABEL_NAMES: dict[int, str] = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack Brute Force",
    13: "Web Attack Sql Injection",
    14: "Web Attack XSS",
}

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

# SMOTE supervisionado — artigo original (Yang et al. 2022); protocolo paper usa NOTEBOOK_SMOTE_TARGETS
PAPER_SMOTE_TARGETS: dict[int, int] = {
    1: 100_000,  # Bot
    2: 100_000,  # BruteForce
    4: 100_000,  # Infiltration
    6: 100_000,  # WebAttack
}

DEFAULT_TEST_SIZE = 0.2  # notebook e protocolo paper: split 80/20 na engenharia de features
PAPER_TEST_SIZE = 0.3  # artigo original: 70% treino / 30% teste (legado)
DEFAULT_CV_FOLDS = 0  # notebook: HPO por acurácia no hold-out
PAPER_CV_FOLDS = 10
DEFAULT_META_LEARNER = "xgb"  # notebook: stacking meta XGBoost + HPO
PAPER_META_LEARNER = "best-base"  # artigo tier 2: clone do melhor base (maior F1 no hold-out)
DEFAULT_SMOTE_TARGETS = NOTEBOOK_SMOTE_TARGETS
DEFAULT_HPO_ON_VALIDATION = False  # notebook: accuracy_score(y_test, y_pred)
PAPER_HPO_ON_VALIDATION = True  # artigo Sec. IV-F: HPO por acurácia em validação
PAPER_IG_CUMULATIVE = 0.9
PAPER_FCBF_K = 20
PAPER_KPCA_COMPONENTS = 10
PAPER_KPCA_KERNEL = "rbf"
PAPER_FEATURE_FIT_SCOPE = "combined"  # anomaly tier 3: IG/FCBF/KPCA no conjunto combinado
PAPER_FCBF_SCOPE = "train"  # supervisionado: FCBF só no treino
PAPER_SUPERVISED_SCALE = "split"  # StandardScaler após split (artigo)
NOTEBOOK_SUPERVISED_SCALE = "phase1"  # Z-score só da fase 1 (IoTJ)
NOTEBOOK_ANOMALY_ZSCORE = "per_split"  # Z-score em df1/df2 separados (IoTJ)
PAPER_ANOMALY_ZSCORE = "combined"  # Z-score no conjunto concatenado
PAPER_HPO_N_CALLS = 15
DEFAULT_BIASED_MODE = "both"  # tier 4: B1 + B2
# Notebook IoTJ: SMOTE(sampling_strategy={1: N}) com N = nº de BENIGN no treino df1 (18225 no demo PortScan).
DEFAULT_ANOMALY_SMOTE_TARGET: int | None = None
NOTEBOOK_ANOMALY_SMOTE_DEMO_BENIGN = 18225
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

# Log de sessão do ramo supervisionado (experiment_runner / run_supervised)
SUPERVISED_RUN_LOG = "supervised_run.log"

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

# Artefatos de inferência (fases 4/6/11 salvam; fase 13 carrega)
MODELS_DIR = "models"
MODELS_SUPERVISED_DIR = "models/supervised"
MODELS_ANOMALY_DIR = "models/anomaly"
MODEL_MANIFEST = "manifest.json"
SUP_SCALER = "scaler.joblib"
SUP_FCBF = "fcbf.joblib"
SUP_FEATURE_NAMES = "feature_names.txt"
SUP_IG_FEATURES = "ig_features.txt"
SUP_DT = "dt.joblib"
SUP_RF = "rf.joblib"
SUP_ET = "et.joblib"
SUP_XGB = "xgb.joblib"
SUP_STACKING_META = "stacking_meta.joblib"
ANOM_CL_STATE = "cl_kmeans_state.joblib"
ANOM_B1 = "b1.joblib"
ANOM_B2 = "b2.joblib"
DEFAULT_ANOMALY_ATTACK_PRED_LABEL = 99
GLOBAL_TABLE_X_PROTOCOL = "global_table_x"
ANOMALY_GLOBAL_WORK_SUBDIR = "anomaly/global"
PAPER_TABLE_X_REFERENCE = {
    "cicids2017": {
        "accuracy_pct": 99.88,
        "detection_rate_pct": 99.77,
        "false_alarm_rate_pct": 0.10,
        "f1": 0.9988,
    },
    "can": {
        "accuracy_pct": 99.99,
        "detection_rate_pct": 100.0,
        "false_alarm_rate_pct": 0.00005,
        "f1": 0.9999,
    },
}


def ensure_intermediate_dirs(intermediate_dir: Path | None = None) -> PipelinePaths:
    paths = get_pipeline_paths(intermediate_dir)
    ensure_pipeline_dirs(paths)
    return paths
