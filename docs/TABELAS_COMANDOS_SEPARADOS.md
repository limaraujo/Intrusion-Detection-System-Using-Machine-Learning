# Tabelas VII, IX e X — comandos separados

Este documento explica **por que cada tabela do artigo MTH-IDS tem comandos próprios**, como elas **não se atrapalham** e como funciona o **conjunto de teste** na Tabela X (Table X / “Tabela 10”).

Guia completo passo a passo: [COMO_RODAR_TABELAS.md](COMO_RODAR_TABELAS.md)

---

## Resposta curta

**Sim — são comandos separados.** Não existe um único script que treina as três tabelas de uma vez.

| Tabela (artigo) | Treino | Impressão |
|-----------------|--------|-----------|
| **VII** | `run_supervised` | `report_paper_tables --table vii` |
| **IX** | `run_anomaly --loao` | `report_paper_tables --table ix` |
| **X** | `run_global_anomaly` + `run_eval` | `report_paper_tables --table x` |

O script `report_paper_tables` **só lê** resultados já salvos; não treina modelos.

---

## Três experimentos, três pastas

```text
data/pipeline_mth_ids_merged/          ← Tabela VII + Tabela X
├── 06_supervised_metrics.json         ← Tabela VII
├── models/supervised/                 ← Tabela VII (tiers 1–2)
├── anomaly/global/                    ← Tabela X (tiers 3–4, detector global)
└── phase_reports/phase13_…json        ← Tabela X (fase 13)

data/pipeline_mth_ids_fine/            ← Tabela IX
└── anomaly/loao/attack_1 … attack_14/ ← LOAO (1 detector por ataque)
```

| | Tabela VII | Tabela IX | Tabela X |
|---|------------|-----------|----------|
| Perfil CSV | **merged** (7 classes) | **fine** (~15 classes) | **merged** |
| Ramo ML | Supervisionado | Anomaly LOAO | Cascata completa |
| Quantos modelos anomaly | — | **14** (um por ataque) | **1** (global) |
| Pasta anomaly | — | `fine/…/loao/attack_*` | `merged/…/global/` |

Rodar a Tabela X **não apaga** o LOAO da Tabela IX (pastas diferentes).  
Rodar LOAO **não gera** a Tabela X.

---

## Comandos por tabela

### Tabela VII — *Performance evaluation of classifiers on the CICIDS2017 dataset*

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.run_supervised --protocol paper
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_mth_ids_merged
```

### Tabela IX — *Performance evaluation on each type of unknown attack*

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile fine
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

### Tabela X — *Performance evaluation on the untouched test set*

```powershell
# Pré-requisito: Tabela VII (run_supervised) já concluída no merged
python -m mth_ids_pipeline.run_global_anomaly --protocol paper
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_mth_ids_merged
```

### Imprimir as três (somente leitura)

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

---

## Ordem recomendada

```text
1. merge_cicids (merged + fine)     ← uma vez
2. run_supervised                   ← Tabela VII  (obrigatório antes da X)
3. run_anomaly --loao               ← Tabela IX   (independente, pode ser em paralelo)
4. run_global_anomaly + run_eval    ← Tabela X
5. report_paper_tables              ← terminal
```

A Tabela **X depende da VII** (modelos stacking + hold-out). A Tabela **IX é independente**.

---

## Isolamento: o que cada comando altera

| Comando | Altera merged | Altera fine |
|---------|---------------|-------------|
| `run_supervised` | fases 1–6, modelos sup. | — |
| `run_global_anomaly` | `anomaly/global/`, reports fase 7–11 | — |
| `run_eval` | `phase13_…json`, `figures/` | — |
| `run_anomaly --loao` | copia `06_…json` (bootstrap) | `loao/attack_*` |
| `report_paper_tables` | nada (só lê) | nada (só lê) |

### Args que isolam a Tabela X

| Arg | Função |
|-----|--------|
| `run_global_anomaly` (em vez de `run_anomaly --loao`) | Só treina detector **global** |
| `--intermediate-dir data/pipeline_mth_ids_merged` | Perfil merged |
| `--work-dir …/anomaly/global` | Artefatos separados do LOAO |
| `--from-phase N` | Retoma fases 7–11 sem refazer as anteriores |
| `report_paper_tables --table x` | Imprime **só** a Tabela X |

Para um segundo experimento global sem sobrescrever o primeiro:

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol paper `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global_v2
# use o mesmo --work-dir no run_eval
```

---

## Tabela X e o “test set intacto”

No artigo, a Tabela X avalia no **test set não utilizado no treino**. No pipeline isso funciona em **duas etapas**:

### Durante o treino anomaly (fases 7–11)

- O hold-out **20% supervisionado** fica **reservado** (não entra nas fases 7–8).
- Fases 9–11 usam **validação interna 20%** do treino anomaly (só para HPO/clustering).
- Isso **não** é o test set final da Tabela X.

### Na avaliação final (fase 13 — `run_eval`)

- Usa `05_test_unchanged.parquet` — o **hold-out real** da Tabela VII.
- Cascata: stacking → se “Normal”, passa pelo anomaly global.
- Métricas Acc, DR, FAR, F1 → `phase13_full_system_eval.json`.

```text
Treino anomaly (80%)  ──► fases 7–11  (teste interno vazio ou val. 20% interna)
Hold-out (20%)      ──► run_eval     (test set intacto — Tabela X)
```

### Ajustes no código (modo global)

O pipeline quebrava ao tentar transformar/clusterizar **0 amostras** de teste nas fases 8–9. Correções:

| Arquivo | O que faz |
|---------|-----------|
| `phase08_anomaly_features.py` | Com teste vazio, pula transform no teste |
| `phase09_anomaly_cluster.py` | Usa `load_anomaly_splits` → validação interna 20% |
| `anomaly_io.py` | Modo global ignora cache `a05` inconsistente |

**Isso não remove o teste da Tabela X** — só permite treinar as fases 7–11 sem vazar o hold-out. O teste final continua na fase 13.

---

## Mapa script → tabela

| Script | Fases | Tabela |
|--------|-------|--------|
| `run_supervised` | 1–6 | VII |
| `run_anomaly --loao` | 7–12 (×14 ataques) | IX |
| `run_global_anomaly` | 7–11 (global) | X (pré-requisito) |
| `run_eval` | 13 | X |
| `report_paper_tables --table vii\|ix\|x\|all` | — | imprime |

---

## Erros comuns

| Pergunta / problema | Resposta |
|---------------------|----------|
| “Um comando gera as 3 tabelas?” | **Não.** Três fluxos de treino + `report_paper_tables` para imprimir. |
| “LOAO gera Tabela X?” | **Não.** LOAO → IX. Global + eval → X. |
| “Tabela X sem teste?” | Treino 7–11 sem hold-out; **teste real só no `run_eval`**. |
| `phase13_…json` ausente | Rodar `run_global_anomaly` + `run_eval` antes de `--table x`. |
| Comparativo muito diferente do artigo | Pipeline usa split **80/20**; artigo CICIDS2017 na Tabela X usa **70/30**. |

---

## Referências

- [COMO_RODAR_TABELAS.md](COMO_RODAR_TABELAS.md) — passo a passo detalhado, tempos, retomada
- [MERGED_VS_FINE_E_TABELAS.md](MERGED_VS_FINE_E_TABELAS.md) — merged vs fine, LOAO vs global
- [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) — parâmetros paper vs notebook
