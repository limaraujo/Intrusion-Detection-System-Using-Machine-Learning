"""
Fase 12: leave-one-attack-out no ramo anomaly (Tabela IX do artigo).

Para cada rótulo de ataque no dataset amostrado, executa fases 7→8→9→10→11 em subdiretório
e agrega DR/FAR/F1.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

try:
    from .anomaly_io import discover_attack_labels
    from .biased_classifiers import load_best_n_clusters
    from .cli import init_paths, phase_parser, supervised_path
    from .config import A06_TEST_SLICE_INFO, P02_SAMPLED_KMEANS
    from .reporting import write_report
except ImportError:
    from anomaly_io import discover_attack_labels
    from biased_classifiers import load_best_n_clusters
    from cli import init_paths, phase_parser, supervised_path
    from config import A06_TEST_SLICE_INFO, P02_SAMPLED_KMEANS
    from reporting import write_report


def _run_phase(module: str, extra: list[str], repo_root: Path) -> None:
    cmd = [sys.executable, "-m", module, *extra]
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=repo_root)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = phase_parser("Fase 12 — LOAO anomaly (Tabela IX)")
    parser.add_argument("--output-root", type=Path, default=None, help="Raiz LOAO (default: anomaly/loao)")
    parser.add_argument("--attack-labels", type=str, default=None, help="Ex.: 1,2,5 ou todos")
    parser.add_argument("--benign-target", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--smote-target", type=int, default=18225)
    parser.add_argument("--hpo-n-calls", type=int, default=20)
    parser.add_argument("--skip-phase9", action="store_true")
    parser.add_argument("--skip-phase10", action="store_true")
    args = parser.parse_args()

    paths = init_paths(args)
    output_root = args.output_root or (paths.anomaly / "loao")
    repo_root = Path(__file__).resolve().parents[1]
    path_in = supervised_path(paths, P02_SAMPLED_KMEANS)
    df = pd.read_parquet(path_in)

    if args.attack_labels:
        attacks = [int(x.strip()) for x in args.attack_labels.split(",")]
    else:
        attacks = discover_attack_labels(df)

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for attack in attacks:
        subdir = output_root / f"attack_{attack}"
        subdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\nLOAO — ataque zero-day label={attack}\n{'=' * 60}")

        attack_report_dir = subdir / "reports"
        common = ["--intermediate-dir", str(paths.intermediate), "--report-dir", str(attack_report_dir)]
        _run_phase(
            "mth_ids_pipeline.phase07_anomaly_datasets",
            [*common, "--work-dir", str(subdir), "--attack-label", str(attack)],
            repo_root,
        )

        p8 = [*common, "--work-dir", str(subdir), "--random-state", str(args.random_state)]
        if args.benign_target is not None:
            p8 += ["--benign-target", str(args.benign_target)]
        _run_phase("mth_ids_pipeline.phase08_anomaly_features", p8, repo_root)

        if not args.skip_phase9:
            p9 = [
                *common,
                "--work-dir",
                str(subdir),
                "--random-state",
                str(args.random_state),
                "--smote-target",
                str(args.smote_target),
            ]
            _run_phase("mth_ids_pipeline.phase09_anomaly_cluster", p9, repo_root)

        best_k: int | None = None
        if not args.skip_phase10:
            p10 = [
                *common,
                "--work-dir",
                str(subdir),
                "--random-state",
                str(args.random_state),
                "--smote-target",
                str(args.smote_target),
                "--n-calls",
                str(args.hpo_n_calls),
            ]
            _run_phase("mth_ids_pipeline.phase10_anomaly_cluster_hpo", p10, repo_root)
            best_k = load_best_n_clusters(attack_report_dir)

        p11 = [
            *common,
            "--work-dir",
            str(subdir),
            "--random-state",
            str(args.random_state),
            "--smote-target",
            str(args.smote_target),
        ]
        _run_phase("mth_ids_pipeline.phase11_anomaly_biased", p11, repo_root)

        report_file = attack_report_dir / "phase11_anomaly_biased.json"
        if report_file.exists():
            rep = json.loads(report_file.read_text(encoding="utf-8"))
            m = rep.get("mth_ids_anomaly") or rep.get("cl_kmeans") or {}
            slice_meta: dict = {}
            slice_path = subdir / A06_TEST_SLICE_INFO
            if slice_path.exists():
                slice_meta = json.loads(slice_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "attack_label": attack,
                    "n_clusters": rep.get("n_clusters") or best_k,
                    "zero_day_samples": slice_meta.get("zero_day_samples"),
                    "benign_sampled": slice_meta.get("benign_sampled"),
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
            "loao_phases": "7,8,9,10,11",
            "benign_pairing": "paper_table_ix_1_to_1 (unless --benign-target)",
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
        out_path = output_root / "loao_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nLOAO concluído: {len(rows)} ataques. Média F1={f1_mean:.4f}")
        print(f"Resumo: {out_path}")
        write_report(paths.reports, "phase12_anomaly_loao", summary)


if __name__ == "__main__":
    main()
