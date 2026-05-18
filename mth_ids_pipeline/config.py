"""Caminhos e nomes de arquivos intermediários do pipeline MTH-IDS."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "pipeline_mth_ids"
ANOMALY_DIR = INTERMEDIATE_DIR / "anomaly"

# Entrada bruta (mesmo padrão do notebook)
DEFAULT_RAW_CSV = DATA_DIR / "CICIDS2017.csv"

# Fase supervisionada (known attacks)
P01_PREPROCESSED = "01_preprocessed.csv"
P02_SAMPLED_KMEANS = "02_sampled_kmeans.csv"
P03_TRAIN = "03_train.csv"
P03_TEST = "03_test.csv"
P04_TRAIN_FSS = "04_train_after_fcbf.csv"
P04_TEST_FSS = "04_test_after_fcbf.csv"
P04_SELECTED_FEATURES = "04_selected_features.txt"
P05_TRAIN_SMOTE = "05_train_after_smote.csv"
P05_TEST = "05_test_unchanged.csv"

# Ramo anomaly-based
A01_WITHOUT_PORTSCAN = "a01_without_portscan.csv"
A02_PORTSCAN_ONLY = "a02_portscan_only.csv"
A03_COMBINED_NORMALIZED = "a03_combined_normalized.csv"
A04_AFTER_KPCA = "a04_after_kpca.csv"
A05_TRAIN_SMOTE = "a05_train_after_smote.csv"
A06_TEST_SLICE_INFO = "a06_test_slice.json"


def ensure_intermediate_dirs() -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
