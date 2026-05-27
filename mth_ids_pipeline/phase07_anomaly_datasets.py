"""
Fase 7 (anomaly branch): gera CSVs sem PortScan e só PortScan, rótulos binários (0 benigno, 1 ataque).

Saídas em data/pipeline_mth_ids/anomaly/: a01_without_portscan.csv, a02_portscan_only.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

try:
    from .config import ANOMALY_DIR, INTERMEDIATE_DIR, P02_SAMPLED_KMEANS, A01_WITHOUT_PORTSCAN, A02_PORTSCAN_ONLY, ensure_intermediate_dirs
except ImportError:
    from config import ANOMALY_DIR, INTERMEDIATE_DIR, P02_SAMPLED_KMEANS, A01_WITHOUT_PORTSCAN, A02_PORTSCAN_ONLY, ensure_intermediate_dirs


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 7 — datasets anomaly (PortScan)")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ANOMALY_DIR)
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
    df1.to_csv(p1, index=False)
    df2.to_csv(p2, index=False)
    print(f"Salvo: {p1} {df1.shape}, {p2} {df2.shape}")


if __name__ == "__main__":
    main()
