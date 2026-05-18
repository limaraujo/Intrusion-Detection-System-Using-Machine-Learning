"""
Executa as fases 1–6 (IDS supervisionado) e opcionalmente 7–9 (ramo anomaly).

Exemplo:
  python -m mth_ids_pipeline.run_all --to 6 --skip-phase6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(module: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module]
    if extra:
        cmd.extend(extra)
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestra o pipeline MTH-IDS modular")
    parser.add_argument("--to", type=int, default=6, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="Última fase a executar")
    parser.add_argument("--skip-phase6", action="store_true", help="Não treinar modelos (fase 6)")
    parser.add_argument("--phase6-args", type=str, default="--no-hpo --no-plots", help="Argumentos extras para fase 6")
    args = parser.parse_args()

    run("mth_ids_pipeline.phase01_load_preprocess")
    if args.to < 2:
        return
    run("mth_ids_pipeline.phase02_sample_kmeans")
    if args.to < 3:
        return
    run("mth_ids_pipeline.phase03_train_test_split")
    if args.to < 4:
        return
    run("mth_ids_pipeline.phase04_feature_engineering")
    if args.to < 5:
        return
    run("mth_ids_pipeline.phase05_smote")
    if args.to < 6:
        return
    if not args.skip_phase6:
        extra = args.phase6_args.split() if args.phase6_args.strip() else []
        run("mth_ids_pipeline.phase06_supervised_models", extra)
    if args.to < 7:
        return
    run("mth_ids_pipeline.phase07_anomaly_datasets")
    if args.to < 8:
        return
    run("mth_ids_pipeline.phase08_anomaly_features")
    if args.to < 9:
        return
    run("mth_ids_pipeline.phase09_anomaly_cluster")


if __name__ == "__main__":
    main()
