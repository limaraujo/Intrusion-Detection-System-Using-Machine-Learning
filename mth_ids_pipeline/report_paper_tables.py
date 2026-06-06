"""
Gera tabelas de comparação com o artigo (Tabela VII supervisionado, Tabela IX LOAO).

Uso:
  python -m mth_ids_pipeline.report_paper_tables --intermediate-dir data/pipeline_mth_ids_merged
  python -m mth_ids_pipeline.report_paper_tables --loao-root data/pipeline_mth_ids_fine/anomaly/loao
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mth_ids_pipeline.config import INTERMEDIATE_DIR
from mth_ids_pipeline.core.evaluation import (
    NOTEBOOK_REFERENCE_SUPERVISED,
    PAPER_REFERENCE_SUPERVISED,
    compare_metrics,
)


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def report_notebook_comparison(intermediate_dir: Path) -> None:
    """Comparação vs referências do notebook IoTJ."""
    metrics_path = intermediate_dir / "06_supervised_metrics.json"
    if not metrics_path.is_file():
        print(f"Notebook: métricas não encontradas em {metrics_path}")
        return
    metrics = _load_json(metrics_path)
    model_map = {
        "XGBoost (base)": "XGBoost HPO",
        "RandomForest (HPO)": "RandomForest HPO",
        "DecisionTree (HPO)": "DecisionTree HPO",
        "ExtraTrees (HPO)": "ExtraTrees HPO",
        "Stacking meta (HPO XGB)": "Stacking meta HPO",
    }
    rows: list[dict] = []
    for entry in metrics:
        ref_key = model_map.get(entry["model"])
        if not ref_key or ref_key not in NOTEBOOK_REFERENCE_SUPERVISED:
            continue
        ref = NOTEBOOK_REFERENCE_SUPERVISED[ref_key]
        for cmp in compare_metrics(entry, ref, metric_keys=("accuracy", "f1_weighted")):
            cmp["label"] = f"{entry['model']} / {cmp['metric']}"
            rows.append(cmp)
    if not rows:
        return
    print("\n" + "=" * 72)
    print("Comparação vs Notebook (hold-out)")
    print("=" * 72)
    print(f"{'Modelo/Métrica':<28} {'Ref':>10} {'Reprod':>10} {'Δ abs':>10} {'Δ %':>8}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['label']:<28} "
            f"{row['reference']:>10.6f} "
            f"{row['reproduced']:>10.6f} "
            f"{row['absolute_diff']:>10.6f} "
            f"{row['percent_diff']:>7.2f}%"
        )


def report_table_vii(intermediate_dir: Path) -> None:
    metrics_path = intermediate_dir / "06_supervised_metrics.json"
    cv_path = intermediate_dir / "phase_reports" / "phase06_supervised_models.json"
    if not metrics_path.is_file():
        print(f"Tabela VII: métricas não encontradas em {metrics_path}")
        return

    metrics = _load_json(metrics_path)
    cv_report = _load_json(cv_path) if cv_path else None
    print("\n" + "=" * 72)
    print("TABELA VII — Supervisionado (multi-class)")
    print("=" * 72)
    print(f"{'Modelo':<24} {'Acc':>10} {'F1(w)':>10} {'Ref Acc':>10} {'Δ Acc':>10}")
    print("-" * 72)

    ref = PAPER_REFERENCE_SUPERVISED
    for row in metrics:
        name = row.get("model", "?")
        acc = float(row.get("accuracy", 0))
        f1 = float(row.get("f1_weighted", 0))
        ref_acc = float(ref.get("accuracy", 0))
        delta = acc - ref_acc
        print(f"{name:<24} {acc:>10.6f} {f1:>10.6f} {ref_acc:>10.6f} {delta:>+10.6f}")

    if cv_report and cv_report.get("cv_reports"):
        print("\n10-fold CV no treino (artigo):")
        for name, rep in cv_report["cv_reports"].items():
            print(f"  {name}: {rep['mean']:.4f} ± {rep['std']:.4f}")


def report_table_ix(loao_root: Path) -> None:
    summary_path = loao_root / "loao_summary.json"
    summary = _load_json(summary_path)
    if not summary:
        alt = loao_root.parent / "phase_reports" / "phase12_anomaly_loao.json"
        summary = _load_json(alt) if alt.is_file() else None
    if not summary and loao_root.is_dir():
        try:
            from mth_ids_pipeline.config import CICIDS2017_FINE_LABEL_NAMES
            from mth_ids_pipeline.io.loao_reporting import build_loao_summary, write_loao_summary

            label_names = {k: v for k, v in CICIDS2017_FINE_LABEL_NAMES.items() if k > 0}
            summary = build_loao_summary(
                loao_root,
                label_names,
                attacks_in_dataset=len(label_names),
            )
            if summary.get("per_attack"):
                write_loao_summary(loao_root, summary)
        except ImportError:
            pass
    if not summary:
        print(f"Tabela IX: resumo LOAO nao encontrado em {loao_root}")
        return

    print("\n" + "=" * 72)
    print("TABELA IX — Anomaly LOAO (DR / FAR / F1)")
    print("=" * 72)
    ref = summary.get("paper_reference_cicids2017", {})
    print(
        f"Média F1: {summary.get('mean_f1', 0):.5f} "
        f"(ref artigo: {ref.get('mean_f1', 0.80013):.5f})"
    )
    print(
        f"Média DR: {summary.get('mean_detection_rate', 0):.4f} "
        f"(ref: {ref.get('mean_dr_pct', 75.943) / 100:.4f})"
    )
    print(
        f"Média FAR: {summary.get('mean_false_alarm_rate', 0):.4f} "
        f"(ref: {ref.get('mean_far_pct', 13.882) / 100:.4f})"
    )
    print(f"\n{'Ataque':<8} {'F1':>10} {'DR':>10} {'FAR':>10} {'N test':>10}")
    print("-" * 52)
    for row in summary.get("per_attack", []):
        print(
            f"{row.get('attack_label', '?'):<8} "
            f"{(row.get('f1') or 0):>10.4f} "
            f"{(row.get('detection_rate') or 0):>10.4f} "
            f"{(row.get('false_alarm_rate') or 0):>10.4f} "
            f"{(row.get('test_rows') or 0):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatórios Tabela VII / IX vs artigo")
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--loao-root", type=Path, default=None)
    parser.add_argument(
        "--table",
        choices=("vii", "ix", "notebook", "all"),
        default="all",
    )
    args = parser.parse_args()

    if args.table in ("notebook", "all"):
        report_notebook_comparison(args.intermediate_dir)
    if args.table in ("vii", "all"):
        report_table_vii(args.intermediate_dir)
    if args.table in ("ix", "all"):
        loao = args.loao_root or (args.intermediate_dir / "anomaly" / "loao")
        report_table_ix(loao)


if __name__ == "__main__":
    main()
