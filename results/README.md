# Resultados de execução

Saídas geradas pelos runners (`run_supervised`, `run_anomaly`, `report_paper_tables`, etc.). **Não versionar logs e métricas novas** — esta pasta é recriada localmente após cada execução.

## Layout padrão

| Caminho | Conteúdo |
|---------|----------|
| `results/` | CICIDS2017 padrão (`paper_comparison.json`, `tables_report.txt`) |
| `results/cicids2017/` | Execução CICIDS com subpastas `config/`, `logs/`, `metrics/` |
| `results/can_intrusion/` | Car-Hacking (`--protocol can`) |
| `results/can_otids/` | OTIDS (`--protocol can_otids`) |
| `results/unsw_nb15/` | UNSW-NB15 (`--protocol unsw`) |
| `results/logs/` | Logs timestampados (legado; preferir subpastas por dataset) |

## Snapshot versionado (validação)

A subpasta `results/cicids2017/` contém um **recorte mínimo** da reprodução documentada em [docs/REPRODUCAO_CICIDS2017_VALIDACAO.md](../docs/REPRODUCAO_CICIDS2017_VALIDACAO.md) (configs merged, log supervisionado, resumo LOAO, relatório de tabelas). Demais artefatos devem ser gerados localmente:

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```
