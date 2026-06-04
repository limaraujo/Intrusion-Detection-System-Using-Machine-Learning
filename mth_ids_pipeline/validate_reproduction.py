"""
Gera tabela de comparação métricas reproduzidas vs notebook vs artigo.

Uso:
  python -m mth_ids_pipeline.validate_reproduction
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import INTERMEDIATE_DIR, REPORTS_DIR
from .evaluation import NOTEBOOK_REFERENCE_SUPERVISED, PAPER_REFERENCE_SUPERVISED, compare_metrics


def _load_supervised_metrics() -> list[dict]:
    path = INTERMEDIATE_DIR / "06_supervised_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Métricas não encontradas: {path}. Execute a fase 6 primeiro.")
    return json.loads(path.read_text(encoding="utf-8"))


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print(f"{'Modelo/Métrica':<28} {'Ref':>10} {'Reprod':>10} {'Δ abs':>10} {'Δ %':>8}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row.get('label', row.get('metric', '')):<28} "
            f"{row['reference']:>10.6f} "
            f"{row['reproduced']:>10.6f} "
            f"{row['absolute_diff']:>10.6f} "
            f"{row['percent_diff']:>7.2f}%"
        )


def main() -> None:
    metrics = _load_supervised_metrics()
    model_map = {
        "XGBoost (base)": "XGBoost HPO",
        "RandomForest (HPO)": "RandomForest HPO",
        "DecisionTree (HPO)": "DecisionTree HPO",
        "ExtraTrees (HPO)": "ExtraTrees HPO",
        "Stacking meta (HPO XGB)": "Stacking meta HPO",
    }

    notebook_rows: list[dict] = []
    for entry in metrics:
        ref_key = model_map.get(entry["model"])
        if not ref_key or ref_key not in NOTEBOOK_REFERENCE_SUPERVISED:
            continue
        ref = NOTEBOOK_REFERENCE_SUPERVISED[ref_key]
        for cmp in compare_metrics(entry, ref, metric_keys=("accuracy", "f1_weighted")):
            cmp["label"] = f"{entry['model']} / {cmp['metric']}"
            notebook_rows.append(cmp)

    _print_table("Comparação vs Notebook (sampled hold-out)", notebook_rows)

    stacking = next((m for m in metrics if m["model"] == "Stacking meta (HPO XGB)"), None)
    if stacking:
        paper = PAPER_REFERENCE_SUPERVISED["MTH-IDS (Multi-Class Model)"]
        paper_rows = [
            {
                "label": "Accuracy (paper % → frac)",
                "reference": paper["accuracy_pct"] / 100.0,
                "reproduced": stacking["accuracy"],
                "absolute_diff": stacking["accuracy"] - paper["accuracy_pct"] / 100.0,
                "percent_diff": (stacking["accuracy"] - paper["accuracy_pct"] / 100.0)
                / (paper["accuracy_pct"] / 100.0)
                * 100.0,
            },
            {
                "label": "F1 (paper vs weighted)",
                "reference": paper["f1"],
                "reproduced": stacking["f1_weighted"],
                "absolute_diff": stacking["f1_weighted"] - paper["f1"],
                "percent_diff": (stacking["f1_weighted"] - paper["f1"]) / paper["f1"] * 100.0,
            },
        ]
        _print_table("Comparação vs Artigo Tabela VII (ressalva: protocolo diferente)", paper_rows)

    anomaly_report = REPORTS_DIR / "phase10_anomaly_cluster_hpo.json"
    if anomaly_report.exists():
        rep = json.loads(anomaly_report.read_text(encoding="utf-8"))
        print(f"\nAnomaly CL-k-means BO-GP: n={rep.get('best_n_clusters')}, acc={rep.get('best_accuracy', 0):.4f}")
    elif (REPORTS_DIR / "phase09_anomaly_cluster.json").exists():
        rep = json.loads((REPORTS_DIR / "phase09_anomaly_cluster.json").read_text(encoding="utf-8"))
        print(f"\nAnomaly CL-k-means baseline: n={rep.get('n_clusters')}, acc={rep.get('accuracy', 0):.4f}")


if __name__ == "__main__":
    main()
