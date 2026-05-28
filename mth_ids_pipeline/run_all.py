"""
Executa as fases 1–6 (IDS supervisionado) e opcionalmente 7–9 (ramo anomaly).

Exemplo:
  python -m mth_ids_pipeline.run_all --to 6 --skip-phase6
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    from .config import REPORTS_DIR
except ImportError:
    from config import REPORTS_DIR


def run(module: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, "-m", module]
    if extra:
        cmd.extend(extra)
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestra o pipeline MTH-IDS modular")
    parser.add_argument("--to", type=int, default=6, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="Última fase a executar")
    parser.add_argument("--from", dest="from_phase", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="Primeira fase a executar")
    parser.add_argument("--only", type=int, default=None, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], help="Executar apenas uma fase")
    parser.add_argument("--raw-csv", type=Path, default=None, help="CSV bruto para fase 1 (override)")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Diretorio para relatorios JSON")
    parser.add_argument("--skip-phase6", action="store_true", help="Não treinar modelos (fase 6)")
    parser.add_argument("--phase6-args", type=str, default="--no-hpo --no-plots", help="Argumentos extras para fase 6")
    parser.add_argument("--phase1-args", type=str, default="")
    parser.add_argument("--phase2-args", type=str, default="")
    parser.add_argument("--phase3-args", type=str, default="")
    parser.add_argument("--phase4-args", type=str, default="")
    parser.add_argument("--phase5-args", type=str, default="")
    parser.add_argument("--phase7-args", type=str, default="")
    parser.add_argument("--phase8-args", type=str, default="")
    parser.add_argument("--phase9-args", type=str, default="")
    args = parser.parse_args()

    phases = {
        1: "mth_ids_pipeline.phase01_load_preprocess",
        2: "mth_ids_pipeline.phase02_sample_kmeans",
        3: "mth_ids_pipeline.phase03_train_test_split",
        4: "mth_ids_pipeline.phase04_feature_engineering",
        5: "mth_ids_pipeline.phase05_smote",
        6: "mth_ids_pipeline.phase06_supervised_models",
        7: "mth_ids_pipeline.phase07_anomaly_datasets",
        8: "mth_ids_pipeline.phase08_anomaly_features",
        9: "mth_ids_pipeline.phase09_anomaly_cluster",
    }

    start_phase = args.only or args.from_phase
    end_phase = args.only or args.to
    if start_phase > end_phase:
        raise SystemExit("--from nao pode ser maior que --to")

    for phase_num in range(start_phase, end_phase + 1):
        if phase_num == 6 and args.skip_phase6:
            print("\n>> Pulando fase 6 (skip-phase6)")
            continue

        extra: list[str] = []
        if args.report_dir:
            extra += ["--report-dir", str(args.report_dir)]
        if phase_num == 1 and args.raw_csv:
            extra += ["--input", str(args.raw_csv)]

        phase_arg_key = f"phase{phase_num}_args"
        raw_extra = getattr(args, phase_arg_key, "")
        if raw_extra.strip():
            extra += shlex.split(raw_extra)

        run(phases[phase_num], extra)


if __name__ == "__main__":
    main()
