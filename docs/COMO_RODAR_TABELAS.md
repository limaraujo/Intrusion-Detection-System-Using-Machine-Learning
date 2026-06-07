# Como rodar as Tabelas VII, IX e X (MTH-IDS)

Guia prático para reproduzir as tabelas do artigo **MTH-IDS** (Yang et al., IEEE IoT Journal 2022) com o pacote `mth_ids_pipeline`.

> **Comandos separados?** Sim — cada tabela tem seu próprio fluxo. Ver [TABELAS_COMANDOS_SEPARADOS.md](TABELAS_COMANDOS_SEPARADOS.md).

Documentos relacionados: [TABELAS_COMANDOS_SEPARADOS.md](TABELAS_COMANDOS_SEPARADOS.md) · [MERGED_VS_FINE_E_TABELAS.md](MERGED_VS_FINE_E_TABELAS.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [PIPELINE_PHASES.md](PIPELINE_PHASES.md)

---

## Pré-requisitos

Execute todos os comandos na **raiz do repositório**, com o ambiente virtual ativo:

```powershell
cd C:\caminho\para\Intrusion-Detection-System-Using-Machine-Learning
.venv\Scripts\Activate.ps1
```

### 1. Gerar os CSVs (uma vez)

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
```

| CSV | Perfil | Classes | Usado em |
|-----|--------|---------|----------|
| `data/CICIDS2017.csv` | merged | 7 (BENIGN + 6 famílias) | Tabelas **VII** e **X** |
| `data/CICIDS2017_fine.csv` | fine | ~15 subtipos | Tabela **IX** (LOAO) |

### 2. Protocolo

Use `--protocol paper` para seguir o artigo (HPO, SMOTE notebook, split 80/20).  
Use `--protocol notebook` apenas se quiser reproduzir o notebook IoTJ publicado.

---

## Visão geral

| Tabela | O que mede | Perfil | Pasta principal | Script(s) |
|--------|------------|--------|-----------------|-----------|
| **VII** | Ataques **conhecidos** (tiers 1–2: DT/RF/ET/XGB + stacking) | merged | `data/pipeline_mth_ids_merged/` | `run_supervised` |
| **IX** | **Zero-day** LOAO (1 ataque excluído do treino por rodada) | fine | `data/pipeline_mth_ids_fine/anomaly/loao/` | `run_anomaly --loao` |
| **X** | **Sistema completo** (cascata tiers 1→4 no hold-out) | merged | `.../anomaly/global/` | `run_global_anomaly` + `run_eval` |

**Regra prática:** Tabela VII e X → **merged**. Tabela IX → **fine**. São experimentos distintos.

---

## Pasta `results/` (tabelas, logs e configs)

O `report_paper_tables` **lê** métricas em `data/pipeline_*` e **grava as tabelas formatadas** na raiz do repositório, **fora de `data/`**. Execuções do pipeline (`run_supervised`, `run_anomaly`, LOAO, `run_global_anomaly`, `run_eval`) gravam **logs** e **configs** na mesma árvore:

```text
results/
├── paper_comparison.json       # métricas estruturadas (VII, IX, X)
├── tables_report.txt           # tabelas legíveis (terminal)
├── logs/
│   ├── merged_paper_phases1-6_YYYYMMDD_HHMMSS.log   # run_supervised / experiment_runner
│   ├── fine_paper_phases7-12_YYYYMMDD_HHMMSS.log    # run_anomaly
│   ├── global_anomaly_paper_YYYYMMDD_HHMMSS.log     # run_global_anomaly
│   ├── eval_phase13_YYYYMMDD_HHMMSS.log             # run_eval
│   ├── report_tables_all_YYYYMMDD_HHMMSS.log        # report_paper_tables
│   └── loao/
│       ├── attack_1.log       # espelho de cada rodada LOAO (fase 12)
│       └── attack_14.log
└── config/
    └── *_config.json          # espelho dos JSONs de config em phase_reports/
```

Comando (salva automaticamente em `results/`):

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

| Flag | Efeito |
|------|--------|
| `--results-dir results/cicids2017` | Outra pasta de saída |
| `--save-json results/custom.json` | JSON em caminho específico (sobrescreve o JSON de `--results-dir`) |
| `--no-save` | Só imprime no terminal, sem gravar arquivos |

> **Artefatos de treino** (parquets, modelos, `phase_reports/`) continuam em `data/`. A pasta `results/` concentra **relatórios comparativos**, **logs de execução** e **configs espelhados**. Cópias locais de LOAO (`attack_<N>/loao_run.log`) permanecem em `data/` para depuração.

Para **IDS2018**, use pastas separadas: `--results-dir results/ids2018` (ver [IDS2018_TABELAS_VII_IX_X.md](IDS2018_TABELAS_VII_IX_X.md)).

---

## Tabela VII — supervisionado

### O que é

Avaliação multi-classe no hold-out 80/20: acurácia e F1 dos modelos base e do stacking (tier 2).

### Comandos

```powershell
# Treino completo (fases 1–6)
python -m mth_ids_pipeline.run_supervised --protocol paper

# Imprimir tabela comparativa vs artigo
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_mth_ids_merged
```

### O que cada comando faz

| Comando | Fases | Saída principal |
|---------|-------|-----------------|
| `run_supervised` | 1–6 | `06_supervised_metrics.json`, `models/supervised/` |
| `report_paper_tables --table vii` | — | Tabela no terminal + `results/` (JSON + TXT) |

### Retomar por fase

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper --from 4 --to 6
```

### Tempo estimado

Fases 1–2 (k-means sampling): dezenas de minutos. Fase 6 (HPO): ~10–15 min.

---

## Tabela IX — LOAO (zero-day)

### O que é

**Leave-One-Attack-Out:** em cada uma de ~14 rodadas, um tipo de ataque fica de fora do treino e entra só no teste. Métricas agregadas: média de F1, DR e FAR.

### Comandos

```powershell
# Amostra fine (fases 1–2) — pule se 02_sampled_kmeans.parquet já existir
python -m mth_ids_pipeline.run_all --label-profile fine --protocol paper --from 1 --to 2

# LOAO completo (fases 7–12 × 14 ataques — muitas horas)
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao

# Tabela IX
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

### Bootstrap automático

`run_anomaly --loao` garante:

- fases 1–2 no **fine** (se faltar `02_sampled_kmeans.parquet`);
- `06_supervised_metrics.json` no **merged** (fases 4–6) e **cópia** para o fine (fase 11 biased).

Se faltar métricas supervisionadas:

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper
```

### Um ataque só (ex.: Bot, label 1)

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap `
  --attack-label 1
```

### Retomar fases 9–11 de um ataque

Se a fase 8 já terminou (`a04_after_kpca.parquet` existe):

```powershell
python -m mth_ids_pipeline.phases.phase09_anomaly_cluster `
  --work-dir data/pipeline_mth_ids_fine/anomaly/loao/attack_1 `
  --intermediate-dir data/pipeline_mth_ids_fine `
  --report-dir data/pipeline_mth_ids_fine/anomaly/loao/attack_1/reports
# ... fases 10 e 11 manualmente, ou via run_all --from 12
```

Ver [PIPELINE_PHASES.md — Retomar LOAO](PIPELINE_PHASES.md#retomar-um-ataque-loao-fases-911-manuais).

### Tempo estimado

~1 h na fase 8 (KPCA + HPO) **por ataque**; LOAO completo: **muitas horas a dias**.

### Saídas

| Artefato | Descrição |
|----------|-----------|
| `anomaly/loao/attack_<N>/` | Artefatos de uma rodada |
| `anomaly/loao/loao_summary.json` | Médias agregadas (Tabela IX) |
| `anomaly/loao/attack_<N>/loao_run.log` | Cópia local da rodada (espelho em `results/logs/loao/attack_<N>.log`) |

---

## Tabela X — sistema completo

> No artigo, a numeração romana **X** corresponde ao “10”. Não confundir com a Tabela IX (LOAO).

### O que é

Avaliação **end-to-end** no hold-out 20%: para cada fluxo de teste, a cascata MTH-IDS decide:

1. **Tier 2 (stacking):** se ataque conhecido → classe multi-classe;
2. **Tiers 3–4 (anomaly):** se “Normal” → KPCA + CL-k-means + B₁/B₂ → benigno ou ataque genérico.

Métricas reportadas: **Acc (%), DR (%), FAR (%), F1** binário.

### Pré-requisitos (ordem obrigatória)

| # | Etapa | Comando |
|---|--------|---------|
| 1 | Amostra k-means (merged) | `run_supervised --from 1 --to 2` |
| 2 | Modelos supervisionados | `run_supervised --from 4 --to 6` |
| 3 | Detector anomaly global (fases 7–11) | `run_global_anomaly` |
| 4 | Avaliação cascata (fase 13) | `run_eval` |
| 5 | Imprimir tabela | `report_paper_tables --table x` |

### Comandos completos

```powershell
# Passo 1–2: supervisionado (Tabela VII também)
python -m mth_ids_pipeline.run_supervised --protocol paper

# Passo 3: anomaly global (fases 7–11) — pode levar 2–3 h com HPO
python -m mth_ids_pipeline.run_global_anomaly --protocol paper

# Passo 4: avaliação end-to-end (~segundos a poucos minutos)
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global

# Passo 5: Tabela X
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_mth_ids_merged
```

### O que cada comando faz

| Comando | Fases | Função |
|---------|-------|--------|
| `run_supervised` | 1–6 | Treina stacking; gera `05_test_unchanged.parquet` (hold-out) |
| `run_global_anomaly` | 7–11 | Um detector binário global em `anomaly/global/` |
| `run_eval` | 13 | Cascata no hold-out; salva métricas e confusion matrices |
| `report_paper_tables --table x` | — | Compara reprodução vs valores do artigo |

### Detalhe das fases 7–11 (anomaly global)

| Fase | Conteúdo |
|------|----------|
| 7 | Dataset binário (80% treino; hold-out reservado à fase 13) |
| 8 | Z-score → IG → FCBF → KernelPCA (+ BO-GP no paper) |
| 9 | SMOTE + CL-k-means baseline |
| 10 | BO-GP: melhor `n_clusters` e métrica de distância |
| 11 | Biased B₁/B₂ + otimização de `p*` → modelos finais |

Fases 9–11 usam **validação interna 20%** do treino anomaly. O hold-out real só entra na fase 13.

### Retomar após falha

```powershell
# Ex.: fases 8–11 incompletas
python -m mth_ids_pipeline.run_global_anomaly --protocol paper --from-phase 8

# Ex.: só fases 10–11
python -m mth_ids_pipeline.run_global_anomaly --protocol paper --from-phase 10

# Modo rápido (sem HPO — menos fiel ao artigo)
python -m mth_ids_pipeline.run_global_anomaly --protocol paper --no-hpo --from-phase 8
```

### Tempo estimado

| Etapa | Tempo típico |
|-------|----------------|
| Fase 8 (KPCA HPO, ~21k linhas) | ~30–90 min |
| Fases 10–11 (HPO clustering + p*) | ~30–60 min |
| Fase 13 (`run_eval`) | ~10 s |

### Saídas

| Artefato | Descrição |
|----------|-----------|
| `phase_reports/phase13_full_system_eval.json` | Métricas Acc, DR, FAR, F1 |
| `figures/fig_multiclass_cm.png` | Confusion matrix multi-classe |
| `figures/fig_binary_cm.png` | Confusion matrix binária |
| `anomaly/global/models/anomaly/` | Modelos CL-k-means + B₁/B₂ |

---

## Comparativo completo (VII + IX + X)

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

Grava em `results/paper_comparison.json` e `results/tables_report.txt`. Opções de `--table`: `vii` | `ix` | `x` | `notebook` | `all`.

---

## Mapa de pastas

```
results/                              # tabelas exportadas (fora de data/)
├── paper_comparison.json             # VII + IX + X (estruturado)
└── tables_report.txt                 # tabelas formatadas

data/
├── CICIDS2017.csv                    # merged
├── CICIDS2017_fine.csv               # fine
│
├── pipeline_mth_ids_merged/          # Tabela VII + X (artefatos de treino)
│   ├── 06_supervised_metrics.json
│   ├── 05_test_unchanged.parquet     # hold-out fase 13
│   ├── anomaly/global/               # fases 7–11
│   ├── phase_reports/
│   │   └── phase13_full_system_eval.json
│   └── figures/
│       ├── fig_multiclass_cm.png
│       └── fig_binary_cm.png
│
└── pipeline_mth_ids_fine/            # Tabela IX (artefatos de treino)
    └── anomaly/loao/
        ├── attack_1/ … attack_14/
        └── loao_summary.json
```

---

## Erros comuns

| Sintoma | Causa | Solução |
|---------|-------|---------|
| `Tabela X: relatório não encontrado em phase13_…` | Fase 13 não rodou | `run_global_anomaly` + `run_eval` |
| `02_sampled_kmeans.parquet` ausente | Fases 1–2 merged não rodaram | `run_supervised --from 1 --to 2` |
| Fase 8/9: array com 0 amostras (modo global) | Teste interno vazio na Tabela X | Atualizar código; retomar da fase afetada |
| Fase 8 demora muito sem output | BO-GP do KernelPCA (15 trials) | Normal (~1 h); ou `--no-hpo` |
| `06_supervised_metrics.json` ausente no fine | Tabela VII não rodou no merged | `run_supervised --protocol paper` |
| Tabela IX vazia | LOAO incompleto ou `loao_summary.json` ausente | Concluir LOAO ou reconstruir resumo |
| Comparativo X muito diferente do artigo | Pipeline usa split **80/20**; artigo CICIDS2017 usa **70/30** | Comparativo é aproximado |

---

## Referência rápida de scripts

| Script | Default | Tabela |
|--------|---------|--------|
| `run_supervised` | merged, fases 1–6 | VII |
| `run_anomaly --loao` | fine, fases 7–12 | IX |
| `run_global_anomaly` | merged, fases 7–11 global | X (pré-requisito) |
| `run_eval` | fase 13 | X |
| `report_paper_tables` | grava em `results/` | VII / IX / X |

---

## Nota sobre fidelidade ao artigo

- **Split:** reprodução usa 80/20 (paper protocol); Tabela X do artigo reporta 70/30.
- **HPO:** protocolo `paper` ativa BO-GP em IG, KPCA, CL-k-means e p* — runs longos, mas alinhados ao método.
- **LOAO vs global:** Tabela IX e Tabela X medem coisas diferentes; rodar um não substitui o outro.
