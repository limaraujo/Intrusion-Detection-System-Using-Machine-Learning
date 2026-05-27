"""
Fase 8 (anomaly): normalização, mistura benigna+PortScan, IG + FCBF + Kernel PCA.

Saídas: a03_combined_normalized.csv, a04_after_kpca.csv, a06_test_slice.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import mutual_info_classif

try:
    from ._bootstrap import ensure_repo_on_path
except ImportError:
    from _bootstrap import ensure_repo_on_path

try:
    from .config import (
        ANOMALY_DIR,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        A03_COMBINED_NORMALIZED,
        A04_AFTER_KPCA,
        A06_TEST_SLICE_INFO,
        ensure_intermediate_dirs,
    )
except ImportError:
    from config import (
        ANOMALY_DIR,
        A01_WITHOUT_PORTSCAN,
        A02_PORTSCAN_ONLY,
        A03_COMBINED_NORMALIZED,
        A04_AFTER_KPCA,
        A06_TEST_SLICE_INFO,
        ensure_intermediate_dirs,
    )


def _ig_subset(importances: np.ndarray, names: list[str], *, cumulative: float = 0.9) -> list[str]:
    s = float(np.sum(importances)) or 1.0
    ranked = sorted(zip(importances / s, names), reverse=True)
    acc = 0.0
    out: list[str] = []
    for score, n in ranked:
        acc += float(score)
        out.append(n)
        if acc >= cumulative:
            break
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_repo_on_path()
    try:
        from mth_ids_pipeline.utils.FCBF_module import FCBFK
    except ImportError as e:
        raise ImportError("FCBF_module.py deve estar na raiz do repositório.") from e

    parser = argparse.ArgumentParser(description="Fase 8 — features anomaly (IG+FCBF+KPCA)")
    parser.add_argument("--dir", type=Path, default=ANOMALY_DIR)
    parser.add_argument("--fcbf-k", type=int, default=20)
    parser.add_argument("--kpca-components", type=int, default=10)
    args = parser.parse_args()
    ensure_intermediate_dirs()
    args.dir.mkdir(parents=True, exist_ok=True)

    df1 = pd.read_csv(args.dir / A01_WITHOUT_PORTSCAN)
    df2 = pd.read_csv(args.dir / A02_PORTSCAN_ONLY)
    label_col = "Label"

    features = df1.drop(columns=[label_col]).select_dtypes(include="number").columns.tolist()
    df1 = df1.copy()
    df2 = df2.copy()
    df1[features] = df1[features].apply(lambda x: (x - x.mean()) / (x.std()))
    df2[features] = df2[features].apply(lambda x: (x - x.mean()) / (x.std()))
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    # Mistura conforme notebook: amostra benigna de df1 adicionada a df2
    df2p = df1[df1[label_col] == 0]
    n_benign = len(df2p)
    frac = 1255 / 18225 if n_benign else 0
    if frac > 0 and n_benign > 0:
        df2pp = df2p.sample(frac=frac, replace=False, random_state=None)
        df2 = pd.concat([df2, df2pp], ignore_index=True)

    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(args.dir / A03_COMBINED_NORMALIZED, index=False)

    X = df[features].values
    y = np.ravel(df[label_col].values)

    importances = mutual_info_classif(X, y, random_state=0)
    ig_names = _ig_subset(importances, features, cumulative=0.9)
    X_fs = df[ig_names].values

    fcbf = FCBFK(k=args.fcbf_k)
    X_fss = fcbf.fit_transform(X_fs, y)

    kpca = KernelPCA(n_components=args.kpca_components, kernel="rbf")
    kpca.fit(X_fss, y)
    X_kpca = kpca.transform(X_fss)

    cols = [f"kpca_{i}" for i in range(X_kpca.shape[1])]
    out = pd.DataFrame(X_kpca, columns=cols)
    out[label_col] = y
    out.to_csv(args.dir / A04_AFTER_KPCA, index=False)

    meta = {"n_df1_rows": int(len(df1)), "n_total": int(len(df))}
    (args.dir / A06_TEST_SLICE_INFO).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Salvo: {args.dir / A04_AFTER_KPCA} shape={out.shape}, meta={meta}")


if __name__ == "__main__":
    main()
