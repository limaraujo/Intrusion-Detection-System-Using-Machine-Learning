"""Ramo supervisionado — fases 1–6.

Paper/CICIDS: ``data/pipeline_mth_ids_merged``
CAN: ``--protocol can_paper`` ou ``can_notebook`` → ``data/pipeline_can_merged`` (sem SMOTE)
"""

from __future__ import annotations

from mth_ids_pipeline.config import INTERMEDIATE_DIR_CAN_MERGED, INTERMEDIATE_DIR_MERGED
from mth_ids_pipeline.orchestration.experiment_runner import build_arg_parser, config_from_args, run_experiment
from mth_ids_pipeline.protocol import is_can_protocol


def main() -> None:
    parser = build_arg_parser(
        "MTH-IDS supervisionado (fases 1–6)"
    )
    parser.set_defaults(from_phase=1, to=6, label_profile="merged")
    args = parser.parse_args()
    if args.intermediate_dir is None:
        args.intermediate_dir = (
            INTERMEDIATE_DIR_CAN_MERGED if is_can_protocol(args.protocol) else INTERMEDIATE_DIR_MERGED
        )
    if is_can_protocol(args.protocol) and args.label_profile == "merged":
        args.label_profile = "can_merged"
    run_experiment(config_from_args(args, branch="supervised"))


if __name__ == "__main__":
    main()
