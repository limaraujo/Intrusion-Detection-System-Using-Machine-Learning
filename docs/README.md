# Documentação MTH-IDS (pipeline modular)

Índice da documentação do pacote `mth_ids_pipeline` — reprodução do artigo Yang et al. (IEEE IoT Journal 2022) e do notebook IoTJ.

## Por onde começar

| Objetivo | Documento |
|----------|-----------|
| **Comandos separados por tabela (VII / IX / X)** | [TABELAS_COMANDOS_SEPARADOS.md](TABELAS_COMANDOS_SEPARADOS.md) |
| **Passo a passo para rodar Tabelas VII, IX e X** | [COMO_RODAR_TABELAS.md](COMO_RODAR_TABELAS.md) |
| **CSE-CIC-IDS2018 — cada tabela (VII / IX / X)** | [IDS2018_TABELAS_VII_IX_X.md](IDS2018_TABELAS_VII_IX_X.md) |
| **CSE-CIC-IDS2018 — isolamento e bootstrap** | [PROTOCOLO_CSE_CIC_IDS2018.md](PROTOCOLO_CSE_CIC_IDS2018.md) |
| **Merged vs fine**, Tabelas VII/IX/X, LOAO vs global | [MERGED_VS_FINE_E_TABELAS.md](MERGED_VS_FINE_E_TABELAS.md) |
| Entender arquitetura e pastas do código | [GUIA_ARQUITETURA_MTH_IDS.md](GUIA_ARQUITETURA_MTH_IDS.md) |
| Reproduzir **Tabela VII**, **IX** e **X** | [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) |
| Por que existem `merged` e `fine` + bootstrap automático | [PASTAS_E_BOOTSTRAP.md](PASTAS_E_BOOTSTRAP.md) |
| **Rodar cada fase** (comandos CLI, retomar LOAO, ver resultados) | [PIPELINE_PHASES.md — Rodar cada fase manualmente](PIPELINE_PHASES.md#rodar-cada-fase-manualmente) |
| Referência de todas as 12 fases (entradas, saídas, tempos) | [PIPELINE_PHASES.md](PIPELINE_PHASES.md) |

## Fluxo rápido (protocolo paper)

```powershell
# 1) CSVs
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine

# 2) Tabela VII
python -m mth_ids_pipeline.run_supervised --protocol paper

# 3) Tabela IX (LOAO — fine, muitas horas)
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao

# 4) Tabela X (sistema completo — merged, não fine)
python -m mth_ids_pipeline.run_global_anomaly --protocol paper
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global

# 5) Relatórios vs artigo
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

Um único ataque LOAO (ex.: Bot, label 1):

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap `
  --attack-label 1
```

Comandos fase a fase e retomada após falha na fase 9+: [PIPELINE_PHASES.md](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais).

## Mapa de artefatos

| Pasta | Perfil | Conteúdo |
|-------|--------|----------|
| `data/pipeline_mth_ids_merged/` | merged | Fases 1–6 (Tabela VII); `anomaly/global/` (Tabela X) |
| `data/pipeline_mth_ids_merged/anomaly/global/` | merged | Detector anomaly global (fases 7–11) |
| `data/pipeline_mth_ids_merged/phase_reports/phase13_full_system_eval.json` | merged | Resultado Tabela X (fase 13) |
| `data/pipeline_mth_ids_fine/` | fine | Fases 1–2, bootstrap; LOAO em `anomaly/loao/` |
| `data/pipeline_mth_ids_fine/anomaly/loao/attack_<N>/` | fine | Uma rodada LOAO (fases 7–11), `loao_run.log` |
| `data/pipeline_mth_ids_fine/anomaly/loao/loao_summary.json` | fine | Agregado Tabela IX |

## Entrada do código

| Script | Fases | Default | Tabela |
|--------|-------|---------|--------|
| `run_supervised` | 1–6 | `merged` | VII |
| `run_anomaly` | 7–11 ou 7–12 (`--loao`) | `fine` | IX |
| `run_global_anomaly` | 7–11 (modo global) | `merged` | X (pré-requisito) |
| `run_eval` | 13 | `merged` (via `--intermediate-dir`) | X |
| `run_all` | alias de `experiment_runner` | `--protocol paper` | — |
| `report_paper_tables` | — | VII / IX / X vs artigo | — |

## Solução de problemas (resumo)

| Sintoma | Solução |
|---------|---------|
| `02_sampled_kmeans.parquet` ausente (global / Tabela X) | Fases 1–2 no **merged**: `run_supervised --from 1 --to 2` |
| `02_sampled_kmeans.parquet` ausente (LOAO) | `run_anomaly` sem `--skip-bootstrap`, ou fases 1–2 no **fine** |
| `phase13_full_system_eval.json` ausente | Rodar `run_global_anomaly` + `run_eval` após fases 4–6 merged |
| `06_supervised_metrics.json` ausente no fine | `run_supervised` no merged (bootstrap copia) |
| Fase 9: `SMOTE … unexpected keyword argument 'n_jobs'` | Atualizar código (`anomaly_io.py`); `imbalanced-learn` ≥ 0.12 remove `n_jobs` |
| BO-GP: `n_calls >= 10` | Usar `--n-calls 15` ou `--hpo-n-calls 15` (padrão paper) |
| Fase 8 LOAO ~1 h por ataque | Normal (KernelPCA ~3 GiB RAM); retomar só 9–11 se `a04_…` existe |
| `loao_summary.json` vazio após falha | Reconstruir com `build_loao_summary` — ver [PIPELINE_PHASES.md](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais) |
| Tabela IX vazia no terminal | `loao_summary.json` desatualizado; reconstruir resumo antes de `report_paper_tables` |

Detalhes: [PIPELINE_PHASES.md — Solução de problemas](PIPELINE_PHASES.md#solução-de-problemas).

## Arquivo histórico

Relatórios de auditoria e refatoração antigos: [archive/](archive/README.md). Podem não refletir o código atual.
