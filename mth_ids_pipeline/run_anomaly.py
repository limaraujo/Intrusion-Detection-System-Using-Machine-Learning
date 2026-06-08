"""Ramo anomaly — fases 7–12.

Paper/CICIDS: ``data/pipeline_mth_ids_fine`` + LOAO
CAN: ``--protocol can_paper`` / ``can_notebook`` → ``data/pipeline_can_fine``
"""

from __future__ import annotations

from mth_ids_pipeline.config import INTERMEDIATE_DIR_CAN_FINE, INTERMEDIATE_DIR_FINE
from mth_ids_pipeline.orchestration.experiment_runner import (
    build_arg_parser,
    config_from_args,
    run_experiment,
)
from mth_ids_pipeline.protocol import is_can_protocol


def main() -> None:
    parser = build_arg_parser(
        "MTH-IDS anomaly (fases 7–12)"
    )
    parser.set_defaults(from_phase=7, to=11, label_profile="fine")
    args = parser.parse_args()
    if args.loao:
        args.to = 12
    if args.intermediate_dir is None:
        args.intermediate_dir = (
            INTERMEDIATE_DIR_CAN_FINE if is_can_protocol(args.protocol) else INTERMEDIATE_DIR_FINE
        )
    if is_can_protocol(args.protocol) and args.label_profile == "fine":
        args.label_profile = "can_fine"
    run_experiment(config_from_args(args, branch="anomaly"))


if __name__ == "__main__":
    main()
