"""
Fase 5: SMOTE no conjunto de treino (estratégia igual ao notebook), teste inalterado.

Saída: 05_train_after_smote.parquet, 05_test_unchanged.parquet
"""

from __future__ import annotations

import argparse
import inspect
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE

try:
    from .config import INTERMEDIATE_DIR, P04_TEST_FSS, P04_TRAIN_FSS, P05_TEST, P05_TRAIN_SMOTE, ensure_intermediate_dirs
except ImportError:
    from config import INTERMEDIATE_DIR, P04_TEST_FSS, P04_TRAIN_FSS, P05_TEST, P05_TRAIN_SMOTE, ensure_intermediate_dirs

try:
    from .config import REPORTS_DIR
except ImportError:
    from config import REPORTS_DIR

try:
    from .reporting import write_report
except ImportError:
    from reporting import write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 5 — SMOTE")
    parser.add_argument("--train", "--train-input", dest="train", type=Path, default=None, help="Parquet treino (fase 4)")
    parser.add_argument("--test", "--test-input", dest="test", type=Path, default=None, help="Parquet teste (fase 4)")
    parser.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR, help="Diretorio de saida")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()
    ensure_intermediate_dirs()

    tr_path = args.train or (args.output_dir / P04_TRAIN_FSS)
    te_path = args.test or (args.output_dir / P04_TEST_FSS)
    label_col = "Label"

    train_df = pd.read_parquet(tr_path)
    test_df = pd.read_parquet(te_path)

    y_train = train_df[label_col].values
    X_train = train_df.drop(columns=[label_col]).values

    orig_counts = pd.Series(y_train).value_counts()
    print("Contagem de rótulos no conjunto de treino original:")
    print(orig_counts)
    
    kw: dict = {"sampling_strategy": {2: 1000, 4: 1000}}
    if "n_jobs" in inspect.signature(SMOTE.__init__).parameters:
        kw["n_jobs"] = -1
    smote = SMOTE(**kw)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    smote_counts = pd.Series(y_res).value_counts()
    print("Contagem de rótulos no conjunto de treino após SMOTE:")
    print(smote_counts)

    fcols = [c for c in train_df.columns if c != label_col]
    out_train = pd.DataFrame(X_res, columns=fcols)
    out_train[label_col] = y_res
    train_out = args.output_dir / P05_TRAIN_SMOTE
    test_out = args.output_dir / P05_TEST
    out_train.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    print(f"Salvo: {train_out} shape={out_train.shape}")
    print(f"Salvo: {test_out} shape={test_df.shape}")

    report = {
        "train_input": str(tr_path),
        "test_input": str(te_path),
        "train_output": str(train_out),
        "test_output": str(test_out),
        "smote_sampling_strategy": {"2": 1000, "4": 1000},
        "train_counts_before": {str(k): int(v) for k, v in orig_counts.items()},
        "train_counts_after": {str(k): int(v) for k, v in smote_counts.items()},
    }
    report_path = write_report(args.report_dir, "phase05_smote", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
