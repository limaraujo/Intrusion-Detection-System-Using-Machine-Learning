"""Ramo anomaly — fases 7–12. Paper: fine + LOAO (Tabela IX).

Artefatos em ``data/pipeline_mth_ids_fine`` (separado de ``pipeline_mth_ids_merged``).
Bootstrap automático (se faltarem pré-requisitos):

- fases **1–2** no fine → ``02_sampled_kmeans.parquet`` (k-means 0,8%; minoritárias fine =
  equivalentes ao ``df_minor`` merged: Bot, Infiltration, WebAttack — ver ``label_profiles.py``)
- fases **1–6** no merged (Tabela VII) → ``06_supervised_metrics.json`` copiado para fine (biased tier 4)

Ver ``docs/PASTAS_E_BOOTSTRAP.md``.
"""

from __future__ import annotations

from mth_ids_pipeline.config import INTERMEDIATE_DIR_FINE
from mth_ids_pipeline.orchestration.experiment_runner import (
    build_arg_parser,
    config_from_args,
    run_experiment,
)


def main() -> None:
    parser = build_arg_parser(
        "MTH-IDS anomaly (fases 7–12; pasta separada: data/pipeline_mth_ids_fine)"
    )
    parser.set_defaults(from_phase=7, to=11, label_profile="fine")
    args = parser.parse_args()
    if args.loao:
        args.to = 12
    if args.intermediate_dir is None:
        args.intermediate_dir = INTERMEDIATE_DIR_FINE
    run_experiment(config_from_args(args, branch="anomaly"))


if __name__ == "__main__":
    main()
