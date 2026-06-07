"""Avaliação end-to-end MTH-IDS (fase 13 / Tabela X).

Exemplo (CICIDS2017 — detector anomaly global):
  python -m mth_ids_pipeline.run_global_anomaly
  python -m mth_ids_pipeline.run_eval \\
    --intermediate-dir data/pipeline_mth_ids_merged \\
    --work-dir data/pipeline_mth_ids_merged/anomaly/global
  python -m mth_ids_pipeline.report_paper_tables --table x \\
    --intermediate-dir data/pipeline_mth_ids_merged
"""

from __future__ import annotations

from mth_ids_pipeline.phases.phase13_full_system_eval import main

if __name__ == "__main__":
    main()
