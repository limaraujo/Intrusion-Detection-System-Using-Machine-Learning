"""
Ramo anomaly (zero-day): fases 7–12 — protocolo do artigo (Tabela IX) por padrão.

Exemplos:
  python -m mth_ids_pipeline.run_anomaly
  python -m mth_ids_pipeline.run_anomaly --label-profile fine --loao
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment_runner import ExperimentConfig, run_experiment
from .label_profiles import get_label_profile
from .reproducibility import DEFAULT_RANDOM_STATE


def main() -> None:
    parser = argparse.ArgumentParser(description="MTH-IDS — ramo anomaly (fases 7–12)")
    parser.add_argument(
        "--label-profile",
        choices=["merged", "fine"],
        default="fine",
        help="Artigo Tabela IX: fine (~14 LOAO); merged: 6 famílias",
    )
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--intermediate-dir", type=Path, default=None)
    parser.add_argument("--from", dest="from_phase", type=int, default=7)
    parser.add_argument("--to", type=int, default=11)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--loao", action="store_true", help="Executar fase 12 (LOAO completo)")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args()

    profile = get_label_profile(args.label_profile)
    to_phase = 12 if args.loao else args.to
    cfg = ExperimentConfig(
        raw_csv=args.raw_csv or profile.raw_csv,
        intermediate_dir=args.intermediate_dir or profile.intermediate_dir,
        minority_labels=profile.minority_labels_csv() or "",
        auto_minority=profile.auto_minority,
        label_profile=args.label_profile,
        from_phase=args.from_phase,
        to_phase=to_phase,
        only_phase=args.only,
        random_state=args.random_state,
        run_loao=args.loao or to_phase == 12,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
