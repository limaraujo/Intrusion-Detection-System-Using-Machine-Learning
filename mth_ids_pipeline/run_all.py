"""
Atalho para experiment_runner (evita duplicar 12× --phaseN-args).

Exemplo:
  python -m mth_ids_pipeline.run_all --label-profile merged --from 1 --to 6
  python -m mth_ids_pipeline.run_all --label-profile merged --run-loao --from 7 --to 12
"""

from __future__ import annotations

from .experiment_runner import main

if __name__ == "__main__":
    main()
