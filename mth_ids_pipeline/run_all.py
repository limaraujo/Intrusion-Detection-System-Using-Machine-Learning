"""
Atalho para experiment_runner.

Exemplos:
  python -m mth_ids_pipeline.run_all --protocol paper --from 1 --to 6
  python -m mth_ids_pipeline.run_all --protocol paper --loao --from 7 --to 12
  python -m mth_ids_pipeline.run_all --protocol notebook --from 1 --to 6
"""

from __future__ import annotations

from mth_ids_pipeline.orchestration.experiment_runner import main

if __name__ == "__main__":
    main()
