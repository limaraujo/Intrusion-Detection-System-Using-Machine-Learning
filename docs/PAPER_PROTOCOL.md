# Protocolo MTH-IDS — Metodologia do artigo

Default: `--protocol paper` (Yang et al., IEEE IoT Journal 2022).

> **Pastas separadas e bootstrap:** ver [PASTAS_E_BOOTSTRAP.md](PASTAS_E_BOOTSTRAP.md).

## Supervisionado (Tabela VII)

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_mth_ids_merged` |
| Dataset | `merge_cicids --profile merged` → `CICIDS2017.csv` |
| Perfil | `merged` (7 famílias) |
| Amostragem | k-means `frac=0.008` + minoritárias preservadas (igual notebook) |
| Split | 80% treino / 20% teste (igual notebook) |
| Validação | 10-fold CV no treino |
| HPO | BO-TPE com objetivo = acurácia em CV (não no teste) |
| SMOTE | BruteForce (2) e Infiltration (4) → **1 000** cada (igual notebook) |
| IG | BO-GP para α acumulado; fallback 0,9 |
| FCBF | k=20, **ajuste só no treino** |
| Z-score | Fit no treino, transform treino+teste (fase 4) |
| Stacking (tier 2) | Meta-learner = **clone do melhor base** (`best-base`; maior F1 weighted no hold-out) |

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --intermediate-dir data/pipeline_mth_ids_merged
```

`run_supervised` usa por padrão `--label-profile merged` e `--intermediate-dir data/pipeline_mth_ids_merged`.

## Anomaly LOAO (Tabela IX)

| Item | Valor |
|------|-------|
| Pasta | **`data/pipeline_mth_ids_fine`** (separada do merged) |
| Dataset | `merge_cicids --profile fine` → `CICIDS2017_fine.csv` |
| Perfil | `fine` (~14 ataques originais) |
| Bootstrap | Fases **1–2** no fine + **Tabela VII** no merged (`06_…` copiado; automático) |
| Amostragem fine (fase 2) | k-means 0,8% + minoritárias = fine equivalentes ao **`df_minor` merged** (Bot, Infiltration, WebAttack) — **não** “todos os rótulos não agregados pelo merge” (PortScan é amostrado) |
| LOAO | 14 ataques; teste = ataque zero-day + benignos 1:1 |
| Features | Z-score → IG → FCBF → KPCA no **conjunto combinado** |
| BO-GP | α IG, parâmetros KPCA, n_clusters, p* (tiers 3–4) |
| Métricas | DR, FAR, F1 binário |

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

**Um ataque ou subset** (útil para teste; ainda refaz fases 7–8 por ataque):

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap `
  --attack-label 1

# ou
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao --attack-labels 1,5,10
```

**Retomar** após falha na fase 9+ (sem repetir KPCA da fase 8): rodar fases 9–11 manualmente e reconstruir `loao_summary.json` — [PIPELINE_PHASES.md](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais).

**Comandos fase a fase:** [PIPELINE_PHASES.md — Rodar cada fase manualmente](PIPELINE_PHASES.md#rodar-cada-fase-manualmente).

`run_anomaly` usa por padrão `--label-profile fine` e `--intermediate-dir data/pipeline_mth_ids_fine`.

**Pré-requisitos (resolvidos pelo bootstrap):**

| Artefato | Onde | Motivo |
|----------|------|--------|
| `02_sampled_kmeans.parquet` | fine, fases 1–2 | Fase 7 lê a amostra k-means LOAO |
| `06_supervised_metrics.json` | merged fases 4–6 → **cópia** no fine | Fase 11: família do melhor learner Tabela VII (B1/B2) |

Flags: `--skip-bootstrap` (não preparar `02_` / `06_` automaticamente).

**Por quê?** O artigo separa Tabela VII (merged, tiers 1–2) de Tabela IX (fine, LOAO). A fase 11 só precisa saber **qual família** de modelo venceu na Tabela VII (RF/XGB/DT/ET) para treinar B₁/B₂ — não um supervisionado re-treinado em 14 classes no fine. Detalhes: [PASTAS_E_BOOTSTRAP.md](PASTAS_E_BOOTSTRAP.md#por-que-o-bootstrap-é-assim-decisão-de-design).

## Notebook (`--protocol notebook`)

Alinhado ao `MTH_IDS_IoTJ.ipynb`:

| Item | Valor |
|------|-------|
| Amostragem | k-means `frac=0.008` |
| Split | 80/20, HPO no teste |
| Fase 4 | Z-score só da fase 1 (`scale-mode phase1`), FCBF no dataset completo |
| Stacking (tier 2) | Meta-learner **XGBoost** + HPO (`meta-learner xgb`) |
| Anomaly fase 8 | IG/FCBF/KPCA no **combinado**, Z-score **per_split** (df1/df2), `--benign-target 1255` |

```powershell
python -m mth_ids_pipeline.run_all --protocol notebook --from 1 --to 6
python -m mth_ids_pipeline.run_anomaly --protocol notebook --from 7 --to 11
```

## Dependências e compatibilidade

Ver `requirements.txt`. Versões recentes de `imbalanced-learn` (≥ 0.12) **não** aceitam `n_jobs` no `SMOTE`; o pipeline detecta isso em `phase05_smote` e `anomaly_io.apply_notebook_anomaly_smote`.

BO-GP (`scikit-optimize`): `n_calls` mínimo **10** — o protocolo paper usa **15** (`PAPER_HPO_N_CALLS`).

## Índice da documentação

- [docs/README.md](README.md) — índice geral e troubleshooting resumido
- [PIPELINE_PHASES.md](PIPELINE_PHASES.md) — guia completo das fases e CLI
- [GUIA_ARQUITETURA_MTH_IDS.md](GUIA_ARQUITETURA_MTH_IDS.md) — estrutura do pacote
- [PASTAS_E_BOOTSTRAP.md](PASTAS_E_BOOTSTRAP.md) — pastas merged/fine e bootstrap
