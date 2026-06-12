# Auditoria de conformidade — Yang et al. (2022) vs pipeline MTH-IDS

Data: 2026-06-08  
Escopo: correções metodológicas pontuais (sem refatoração de arquitetura, APIs ou nomes de arquivos).

> Documento histórico. Localização atual: `docs/archive/AUDIT_YANG2022_PROTOCOL.md` (antes em `results/`).

---

## 1. Arquivos modificados

| Arquivo | Papel |
|---------|-------|
| `mth_ids_pipeline/phases/phase02_sample_kmeans.py` | Default `--frac` → 0.008 (mantido) |
| `mth_ids_pipeline/config.py` | Constantes `PAPER_*`, `DEFAULT_KMEANS_FRAC`, `NOTEBOOK_KMEANS_FRAC` |
| `mth_ids_pipeline/protocol.py` | Preset `paper`: split 70/30, k-means 10%, `cl_hpo_metric_source` |
| `mth_ids_pipeline/orchestration/experiment_runner.py` | Propaga `cl_hpo_metric_source` à fase 10 |
| `mth_ids_pipeline/core/feature_selection.py` | `fit_fcbf` com modos `k` e `alpha` |
| `mth_ids_pipeline/core/hyperparameter_optimization.py` | `optimize_fcbf_alpha` (BO-GP) |
| `mth_ids_pipeline/phases/phase04_feature_engineering.py` | CLI `--fcbf-mode`, `--fcbf-alpha`, `--optimize-fcbf` |
| `mth_ids_pipeline/phases/phase10_anomaly_cluster_hpo.py` | Relatório `hpo_metric_source` + evidências |
| `tests/test_paper_protocol.py` | **Novo** — 9 testes de conformidade |

**Sem alteração necessária (já conformes):**

- `mth_ids_pipeline/phases/phase06_supervised_models.py` — `best-base` + HPO em CV já implementados.

---

## 7. Validação

```powershell
python -m pytest tests/test_paper_protocol.py -v
```

Resultado: **9 passed**.

Para reexecutar o protocolo paper após as mudanças:

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper --from 1 --to 6
```

Verificar em `phase_reports/experiment_runner_config.json`:
- `test_size`: 0.3
- `kmeans_frac`: 0.008
- `cv_folds`: 10
- `hpo_on_validation`: true
- `meta_learner`: "best-base"
