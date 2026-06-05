"""
Ramo supervisionado (signature-based): fases 1–6 — protocolo do notebook por padrão.

Exemplos:
  python -m mth_ids_pipeline.run_supervised --label-profile merged
  python -m mth_ids_pipeline.run_supervised --label-profile merged --no-hpo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiment_runner import ExperimentConfig, run_experiment
from .label_profiles import get_label_profile
from .reproducibility import DEFAULT_RANDOM_STATE


def main() -> None:
    parser = argparse.ArgumentParser(description="MTH-IDS — ramo supervisionado (fases 1–6)")
    parser.add_argument("--label-profile", choices=["merged", "fine"], default="merged")
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--intermediate-dir", type=Path, default=None)
    parser.add_argument("--from", dest="from_phase", type=int, default=1)
    parser.add_argument("--to", type=int, default=6)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--no-hpo", action="store_true", help="Fase 6: sem BO-TPE (rápido, fora do notebook)")
    parser.add_argument("--skip-phase6", action="store_true")
    args = parser.parse_args()

    profile = get_label_profile(args.label_profile)
    cfg = ExperimentConfig(
        raw_csv=args.raw_csv or profile.raw_csv,
        intermediate_dir=args.intermediate_dir or profile.intermediate_dir,
        minority_labels=profile.minority_labels_csv() or "",
        auto_minority=profile.auto_minority,
        label_profile=args.label_profile,
        from_phase=args.from_phase,
        to_phase=args.to,
        only_phase=args.only,
        random_state=args.random_state,
        run_hpo=not args.no_hpo,
        skip_phase6=args.skip_phase6,
    )
    run_experiment(cfg)


if __name__ == "__main__":
    main()
