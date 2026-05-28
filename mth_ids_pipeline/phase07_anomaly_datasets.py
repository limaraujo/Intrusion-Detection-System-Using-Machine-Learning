"""
Fase 7 (anomaly branch): gera Parquet sem PortScan e só PortScan, rótulos binários (0 benigno, 1 ataque).

Saídas em data/pipeline_mth_ids/anomaly/: a01_without_portscan.parquet, a02_portscan_only.parquet
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

try:
    from .config import ANOMALY_DIR, INTERMEDIATE_DIR, P02_SAMPLED_KMEANS, A01_WITHOUT_PORTSCAN, A02_PORTSCAN_ONLY, REPORTS_DIR, ensure_intermediate_dirs
except ImportError:
    from config import ANOMALY_DIR, INTERMEDIATE_DIR, P02_SAMPLED_KMEANS, A01_WITHOUT_PORTSCAN, A02_PORTSCAN_ONLY, REPORTS_DIR, ensure_intermediate_dirs

try:
    from .reporting import dataset_report, write_report
except ImportError:
    from reporting import dataset_report, write_report


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 7 — datasets anomaly (PortScan)")
    parser.add_argument("--input", type=Path, default=None, help="Parquet amostrado (fase 2)")
    parser.add_argument("--output-dir", type=Path, default=ANOMALY_DIR, help="Diretorio de saida")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()
    ensure_intermediate_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    path_in = args.input or (INTERMEDIATE_DIR / P02_SAMPLED_KMEANS.replace(".csv", ".parquet"))
    df = pd.read_parquet(path_in)
    label_col = "Label"

    df1 = df[df[label_col] != 5].copy()
    df1.loc[df1[label_col] > 0, label_col] = 1

    df2 = df[df[label_col] == 5].copy()
    df2.loc[:, label_col] = 1

    p1 = args.output_dir / A01_WITHOUT_PORTSCAN
    p2 = args.output_dir / A02_PORTSCAN_ONLY
    df1.to_parquet(p1, index=False)
    df2.to_parquet(p2, index=False)
    print(f"Salvo: {p1} {df1.shape}, {p2} {df2.shape}")

    report = {
        "input": str(path_in),
        "without_portscan_output": str(p1),
        "portscan_output": str(p2),
        "without_portscan": dataset_report(df1, label_col),
        "portscan": dataset_report(df2, label_col),
    }
    report_path = write_report(args.report_dir, "phase07_anomaly_datasets", report)
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()
