"""
Fase 8 (anomaly): normalização, mistura benigna+PortScan, IG + FCBF + Kernel PCA.

Saídas: a03_combined_normalized.parquet, a04_after_kpca.parquet, a06_test_slice.json
"""

from __future__ import annotations

import argparse
import json
import time
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
        REPORTS_DIR,
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
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )

try:
    from .reporting import write_report
except ImportError:
    from reporting import write_report


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
    parser.add_argument("--input-dir", type=Path, default=None, help="Diretorio de entrada (fase 7)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretorio de saida")
    parser.add_argument("--dir", type=Path, default=None, help="Alias para --input-dir e --output-dir")
    parser.add_argument("--fcbf-k", type=int, default=20, help="Numero de features FCBF")
    parser.add_argument("--kpca-components", type=int, default=10, help="Componentes KPCA")
    parser.add_argument("--benign-frac", type=float, default=None, help="Fracao de benignos a amostrar")
    parser.add_argument("--benign-target", type=int, default=None, help="Numero de benignos a amostrar")
    parser.add_argument("--random-state", type=int, default=None, help="Seed para amostragem")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    args = parser.parse_args()
    ensure_intermediate_dirs()
    start = time.time()
    total_steps = 6
    step = 0

    def tick(message: str) -> None:
        nonlocal step
        step += 1
        bar_len = 20
        filled = int(bar_len * step / total_steps)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start
        print(f"[{step}/{total_steps}] [{bar}] {message} (+{elapsed:.1f}s)")
    input_dir = args.input_dir or args.dir or ANOMALY_DIR
    output_dir = args.output_dir or args.dir or ANOMALY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df1 = pd.read_parquet(input_dir / A01_WITHOUT_PORTSCAN)
    df2 = pd.read_parquet(input_dir / A02_PORTSCAN_ONLY)
    tick("Dados carregados")
    label_col = "Label"

    features = df1.drop(columns=[label_col]).select_dtypes(include="number").columns.tolist()
    df1 = df1.copy()
    df2 = df2.copy()
    df1[features] = df1[features].apply(lambda x: (x - x.mean()) / (x.std()))
    df2[features] = df2[features].apply(lambda x: (x - x.mean()) / (x.std()))
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)
    tick("Normalizacao concluida")

    # Mistura conforme notebook: amostra benigna de df1 adicionada a df2
    df2p = df1[df1[label_col] == 0]
    n_benign = len(df2p)
    sample_n = 0
    if args.benign_frac is not None:
        sample_n = int(round(n_benign * float(args.benign_frac)))
    elif args.benign_target is not None:
        sample_n = int(args.benign_target)
    else:
        sample_n = min(len(df2), n_benign)
    sample_n = max(0, min(sample_n, n_benign))
    if sample_n > 0:
        df2pp = df2p.sample(n=sample_n, replace=False, random_state=args.random_state)
        df2 = pd.concat([df2, df2pp], ignore_index=True)
    tick("Amostragem benignos concluida")

    df = pd.concat([df1, df2], ignore_index=True)
    combined_path = output_dir / A03_COMBINED_NORMALIZED
    df.to_parquet(combined_path, index=False)

    X = df[features].values
    y = np.ravel(df[label_col].values)

    importances = mutual_info_classif(X, y, random_state=0)
    ig_names = _ig_subset(importances, features, cumulative=0.9)
    X_fs = df[ig_names].values

    fcbf = FCBFK(k=args.fcbf_k)
    X_fss = fcbf.fit_transform(X_fs, y)
    tick("IG + FCBF concluidos")

    kpca = KernelPCA(n_components=args.kpca_components, kernel="rbf")
    kpca.fit(X_fss, y)
    X_kpca = kpca.transform(X_fss)
    tick("KPCA concluida")

    cols = [f"kpca_{i}" for i in range(X_kpca.shape[1])]
    out = pd.DataFrame(X_kpca, columns=cols)
    out[label_col] = y
    kpca_path = output_dir / A04_AFTER_KPCA
    out.to_parquet(kpca_path, index=False)

    meta = {"n_df1_rows": int(len(df1)), "n_total": int(len(df))}
    meta_path = output_dir / A06_TEST_SLICE_INFO
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Salvo: {kpca_path} shape={out.shape}, meta={meta}")

    report = {
        "input_without_portscan": str(input_dir / A01_WITHOUT_PORTSCAN),
        "input_portscan": str(input_dir / A02_PORTSCAN_ONLY),
        "combined_output": str(combined_path),
        "kpca_output": str(kpca_path),
        "test_slice_meta": str(meta_path),
        "fcbf_k": args.fcbf_k,
        "kpca_components": args.kpca_components,
        "benign_sample_n": sample_n,
        "benign_total": int(n_benign),
        "random_state": args.random_state,
        "kpca_shape": {"rows": int(out.shape[0]), "cols": int(out.shape[1])},
    }
    report_path = write_report(args.report_dir, "phase08_anomaly_features", report)
    print(f"Relatorio salvo em: {report_path}")
    tick("Saidas e relatorio salvos")


if __name__ == "__main__":
    main()
