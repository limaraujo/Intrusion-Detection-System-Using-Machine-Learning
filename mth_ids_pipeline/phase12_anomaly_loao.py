"""
Fase 12: leave-one-attack-out no ramo anomaly (Tabela IX do artigo).

Para cada rótulo de ataque no dataset amostrado, executa fases 7→8→9→11 em subdiretório
e agrega DR/FAR/F1.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

try:
    from .anomaly_io import discover_attack_labels
    from .config import (
        ANOMALY_DIR,
        INTERMEDIATE_DIR,
        P02_SAMPLED_KMEANS,
        REPORTS_DIR,
        ensure_intermediate_dirs,
    )
    from .reporting import write_report
except ImportError:
    from anomaly_io import discover_attack_labels
    from config import ANOMALY_DIR, INTERMEDIATE_DIR, P02_SAMPLED_KMEANS, REPORTS_DIR, ensure_intermediate_dirs
    from reporting import write_report


def _run_phase(module: str, extra: list[str], repo_root: Path) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=repo_root)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 12 — LOAO anomaly (14 ataques / amostra)")
    parser.add_argument("--input", type=Path, default=None, help="Parquet amostrado (fase 2)")
    parser.add_argument("--output-root", type=Path, default=ANOMALY_DIR / "loao")
    parser.add_argument("--attack-labels", type=str, default=None, help="Lista '1,2,5' ou todos")
    parser.add_argument("--benign-target", type=int, default=1255)
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--skip-phase9", action="store_true", help="Pular SMOTE explícito na fase 9")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--phase8-extra", type=str, default="")
    parser.add_argument("--phase11-extra", type=str, default="")
    args = parser.parse_args()
    ensure_intermediate_dirs()
    repo_root = Path(__file__).resolve().parents[1]

    path_in = args.input or (INTERMEDIATE_DIR / P02_SAMPLED_KMEANS.replace(".csv", ".parquet"))
    df = pd.read_parquet(path_in)

    if args.attack_labels:
        attacks = [int(x.strip()) for x in args.attack_labels.split(",")]
    else:
        attacks = discover_attack_labels(df)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for attack in attacks:
        subdir = args.output_root / f"attack_{attack}"
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\nLOAO — ataque zero-day label={attack}\n{'=' * 60}")

        attack_report_dir = subdir / "reports"
        p7 = [
            "--input", str(path_in),
            "--output-dir", str(subdir),
            "--attack-label", str(attack),
            "--report-dir", str(attack_report_dir),
        ]
        _run_phase("mth_ids_pipeline.phase07_anomaly_datasets", p7, repo_root)

        p8 = [
            "--input-dir", str(subdir),
            "--output-dir", str(subdir),
            "--benign-target", str(args.benign_target),
            "--random-state", str(args.random_state),
            "--report-dir", str(attack_report_dir),
        ]
        if args.phase8_extra.strip():
            p8 += shlex.split(args.phase8_extra)
        _run_phase("mth_ids_pipeline.phase08_anomaly_features", p8, repo_root)

        if not args.skip_phase9:
            p9 = [
                "--input-dir", str(subdir),
                "--output-dir", str(subdir),
                "--n-clusters", str(args.n_clusters),
                "--random-state", str(args.random_state),
                "--report-dir", str(attack_report_dir),
            ]
            _run_phase("mth_ids_pipeline.phase09_anomaly_cluster", p9, repo_root)

        p11 = [
            "--input-dir", str(subdir),
            "--n-clusters", str(args.n_clusters),
            "--random-state", str(args.random_state),
            "--report-dir", str(attack_report_dir),
        ]
        if args.phase11_extra.strip():
            p11 += shlex.split(args.phase11_extra)
        _run_phase("mth_ids_pipeline.phase11_anomaly_biased", p11, repo_root)

        report_file = attack_report_dir / "phase11_anomaly_biased.json"
        if report_file.exists():
            rep = json.loads(report_file.read_text(encoding="utf-8"))
            m = rep.get("mth_ids_anomaly") or rep.get("cl_kmeans") or {}
            rows.append(
                {
                    "attack_label": attack,
                    "accuracy": m.get("accuracy"),
                    "detection_rate": m.get("detection_rate"),
                    "false_alarm_rate": m.get("false_alarm_rate"),
                    "f1": m.get("f1"),
                    "output_dir": str(subdir),
                }
            )

    if rows:
        dr_mean = sum(r["detection_rate"] or 0 for r in rows) / len(rows)
        far_mean = sum(r["false_alarm_rate"] or 0 for r in rows) / len(rows)
        f1_mean = sum(r["f1"] or 0 for r in rows) / len(rows)
        summary = {
            "n_attacks": len(rows),
            "mean_detection_rate": dr_mean,
            "mean_false_alarm_rate": far_mean,
            "mean_f1": f1_mean,
            "per_attack": rows,
            "paper_reference_cicids2017": {
                "mean_f1": 0.80013,
                "mean_dr_pct": 75.943,
                "mean_far_pct": 13.882,
            },
        }
        out_path = args.output_root / "loao_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nLOAO concluído: {len(rows)} ataques. Média F1={f1_mean:.4f}")
        print(f"Resumo: {out_path}")
        write_report(args.report_dir, "phase12_anomaly_loao", summary)


if __name__ == "__main__":
    main()
