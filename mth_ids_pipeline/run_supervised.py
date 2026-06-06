"""Ramo supervisionado — fases 1–6. Paper: merged (Tabela VII).

Artefatos em ``data/pipeline_mth_ids_merged`` (separado do anomaly fine).
"""

from __future__ import annotations

from mth_ids_pipeline.config import INTERMEDIATE_DIR_MERGED
from mth_ids_pipeline.orchestration.experiment_runner import build_arg_parser, config_from_args, run_experiment


def main() -> None:
    parser = build_arg_parser(
        "MTH-IDS supervisionado (fases 1–6; pasta: data/pipeline_mth_ids_merged)"
    )
    parser.set_defaults(from_phase=1, to=6, label_profile="merged")
    args = parser.parse_args()
    if args.intermediate_dir is None:
        args.intermediate_dir = INTERMEDIATE_DIR_MERGED
    run_experiment(config_from_args(args, branch="supervised"))


if __name__ == "__main__":
    main()
