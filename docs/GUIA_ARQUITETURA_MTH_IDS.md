# Arquitetura do pipeline MTH-IDS

## Estrutura de pastas

```
mth_ids_pipeline/
├── config.py              # caminhos, constantes, artefatos
├── protocol.py            # presets paper vs notebook
├── label_profiles.py      # merged (7 classes) vs fine (LOAO)
├── cli.py                 # argumentos CLI compartilhados
├── run_supervised.py      # entrada: fases 1–6
├── run_anomaly.py         # entrada: fases 7–12
├── run_all.py             # entrada genérica
├── report_paper_tables.py # Tabela VII / IX vs artigo
│
├── core/                  # algoritmos ML reutilizáveis
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── dimensionality_reduction.py
│   ├── clustering.py
│   ├── hyperparameter_optimization.py
│   ├── biased_classifiers.py
│   ├── validation.py
│   └── evaluation.py
│
├── io/                    # persistência e relatórios
│   ├── anomaly_io.py      # splits LOAO, SMOTE anomaly
│   ├── loao_reporting.py  # agregação Tabela IX (loao_summary.json)
│   ├── run_log.py         # supervised_run.log (fases 1–6) e loao_run.log (fase 12)
│   ├── reporting.py
│   └── reproducibility.py
│
├── phases/                # scripts executáveis (1 fase = 1 módulo)
│   ├── phase01_load_preprocess.py
│   ├── phase02_sample_kmeans.py
│   ├── phase04_feature_engineering.py
│   ├── phase05_smote.py
│   ├── phase06_supervised_models.py
│   ├── phase07_anomaly_datasets.py
│   ├── phase08_anomaly_features.py
│   ├── phase09_anomaly_cluster.py
│   ├── phase10_anomaly_cluster_hpo.py
│   ├── phase11_anomaly_biased.py
│   └── phase12_anomaly_loao.py
│
├── orchestration/
│   └── experiment_runner.py
│
└── utils/
    ├── bootstrap.py       # sys.path para FCBF
    ├── merge_cicids.py    # gera CICIDS2017.csv / _fine.csv
    └── FCBF_module.py
```

## Ramos de execução

| Ramo | Fases | Perfil paper | CSV | Pasta padrão |
|------|-------|--------------|-----|--------------|
| Supervisionado | 1–6 | `merged` | `CICIDS2017.csv` | `data/pipeline_mth_ids_merged/` |
| Anomaly LOAO | 7–12 | `fine` | `CICIDS2017_fine.csv` | `data/pipeline_mth_ids_fine/` |

Ver [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) para parâmetros do artigo e [PASTAS_E_BOOTSTRAP.md](PASTAS_E_BOOTSTRAP.md) para separação de pastas e bootstrap automático.

### Bootstrap no ramo anomaly

`run_anomaly` chama `ensure_anomaly_prerequisites()` em `orchestration/experiment_runner.py` antes das fases 7–12:

- Falta `02_sampled_kmeans.parquet` → fases **1–2** em `pipeline_mth_ids_fine`.
- Falta `06_supervised_metrics.json` no fine → Tabela VII em `pipeline_mth_ids_merged` (fases **1–6** se necessário) e **cópia** do JSON para o fine (biased tier 4).

Flag `--skip-bootstrap` desativa esse comportamento.

Motivação completa (artigo × bootstrap antigo × implementação atual): [PASTAS_E_BOOTSTRAP.md — Por que o bootstrap é assim?](PASTAS_E_BOOTSTRAP.md#por-que-o-bootstrap-é-assim-decisão-de-design).

## Como executar

| Necessidade | Onde ler |
|-------------|----------|
| Comandos por fase (1–12), LOAO manual, retomar após falha | [PIPELINE_PHASES.md — Rodar cada fase manualmente](PIPELINE_PHASES.md#rodar-cada-fase-manualmente) |
| Protocolo paper vs notebook | [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) |
| Índice geral da documentação | [README.md](README.md) |

**Orquestradores:** `run_supervised` (1–6), `run_anomaly` (7–12), `run_all` / `experiment_runner` (qualquer intervalo `--from` / `--to`).

**Supervisionado (fases 1–6):** `experiment_runner` grava `supervised_run.log` na pasta `intermediate-dir` via `RunLog` (comandos + stdout de cada fase).

**LOAO (fase 12):** para cada ataque `N`, executa subprocessos das fases 7–11 em `anomaly/loao/attack_N/` e grava `loao_run.log` via `RunLog`. Fases rodadas manualmente **não** entram nesses logs.

**Relatórios:** `report_paper_tables` (Tabela VII / IX); JSON por fase em `phase_reports/` ou `attack_N/reports/`.

## Documentação histórica

Relatórios de auditoria e refatoração antigos estão em [archive/](archive/README.md).
