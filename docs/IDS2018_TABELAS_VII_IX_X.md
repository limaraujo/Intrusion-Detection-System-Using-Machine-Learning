# CSE-CIC-IDS2018 — como rodar cada tabela

Guia **por tabela** (VII, IX, X) para o dataset **`CSE-CIC-IDS2018.csv`**, sem sobrescrever artefatos do **CICIDS2017**.

| Documento | Conteúdo |
|-----------|----------|
| Este arquivo | Comandos **por tabela** (2018) |
| [PROTOCOLO_CSE_CIC_IDS2018.md](PROTOCOLO_CSE_CIC_IDS2018.md) | Isolamento de pastas, bootstrap, limitações |
| [COMO_RODAR_TABELAS.md](COMO_RODAR_TABELAS.md) | Mesmas tabelas no **CICIDS2017** |

---

## Antes de qualquer tabela

### 1. Ambiente

```powershell
cd C:\caminho\para\Intrusion-Detection-System-Using-Machine-Learning
.venv\Scripts\Activate.ps1
```

### 2. Variáveis (sempre use estas no 2018)

```powershell
$RAW2018   = "data/CSE-CIC-IDS2018.csv"
$MERGED18  = "data/pipeline_ids2018_merged"
$FINE18    = "data/pipeline_ids2018_fine"
$GLOBAL18  = "$MERGED18/anomaly/global"
$LOAO18    = "$FINE18/anomaly/loao"
```

### 3. CSV

- Coloque `CSE-CIC-IDS2018.csv` em `data/`.
- Coluna **`Label`**, classe benigna **`BENIGN`**.
- **Não** use `merge_cicids` (só funciona para CICIDS2017).

### 4. Regra de ouro

**Nunca** omita `--intermediate-dir` e `--raw-csv` / `--input`.  
Sem eles, o pipeline usa `pipeline_mth_ids_merged` / `pipeline_mth_ids_fine` (**2017**).

---

## Visão geral

| Tabela (artigo) | O que mede | Pasta | Treino | Impressão |
|-----------------|------------|-------|--------|-----------|
| **VII** — *Performance evaluation of classifiers on the CICIDS2017 dataset* | Ataques **conhecidos** (DT/RF/ET/XGB + stacking) | `$MERGED18` | `run_supervised` | `--table vii` |
| **IX** — *Performance evaluation on each type of unknown attack* | **Zero-day** LOAO (1 ataque fora do treino por rodada) | `$FINE18` / `$LOAO18` | `run_anomaly --loao` | `--table ix` |
| **X** — *Performance evaluation on the untouched test set* | **Sistema completo** (cascata tiers 1→4) | `$MERGED18` + `$GLOBAL18` | `run_global_anomaly` + `run_eval` | `--table x` |

**Ordem sugerida:** VII → X (X depende de VII). IX é **independente**, mas precisa de VII 2018 copiada para o fine (ver Tabela IX).

---

# Tabela VII — classificadores (IDS2018)

## Objetivo

Avaliar modelos supervisionados (tiers 1–2) no hold-out 80/20: acurácia e F1 multi-classe.

## Onde grava

```text
data/pipeline_ids2018_merged/
├── 01_preprocessed.parquet
├── 02_sampled_kmeans.parquet
├── 06_supervised_metrics.json          ← principal
├── models/supervised/
└── phase_reports/phase06_supervised_models.json
```

## Passo a passo

### 1) Fase 1 — load + Z-score

```powershell
python -m mth_ids_pipeline.phases.phase01_load_preprocess `
  --input $RAW2018 `
  --intermediate-dir $MERGED18
```

Inspecione rótulos:

```text
$MERGED18/phase_reports/phase01_load_preprocess.json
```

### 2) Fase 2 — amostragem k-means

Use **`--auto-minority`** (defaults `6,1,4` são do CICIDS2017):

```powershell
python -m mth_ids_pipeline.phases.phase02_sample_kmeans `
  --intermediate-dir $MERGED18 `
  --auto-minority
```

> IDS2018 é grande; a fase 2 pode demorar **muito** mais que no sample 2017.

### 3) Fases 4–6 — features, SMOTE, modelos + stacking

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper `
  --from 4 --to 6 `
  --intermediate-dir $MERGED18 `
  --raw-csv $RAW2018
```

Alternativa (fases 1–6 num comando, **sem** `--auto-minority` na fase 2):

```powershell
# Só use se já rodou fases 1–2 com --auto-minority antes
python -m mth_ids_pipeline.run_supervised --protocol paper `
  --intermediate-dir $MERGED18 `
  --raw-csv $RAW2018
```

### 4) Imprimir Tabela VII

```powershell
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir $MERGED18
```

Salvar só métricas 2018 (ignore coluna **Diff** vs artigo 2017):

```powershell
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir $MERGED18 `
  --save-json $MERGED18/phase_reports/table_vii_ids2018.json
```

## Retomar

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper `
  --from 4 --to 6 `
  --intermediate-dir $MERGED18 `
  --raw-csv $RAW2018
```

## Tempo estimado

Fase 2 + fase 6 (HPO): depende do tamanho do CSV 2018 — pode levar **horas**.

---

# Tabela X — sistema completo no test set intacto (IDS2018)

## Objetivo

Avaliar a **cascata MTH-IDS** no hold-out 20%: Acc, DR, FAR, F1 binário.

Fluxo por fluxo de teste:

1. Stacking classifica → se ataque conhecido, usa classe multi-classe.
2. Se “Normal” → detector anomaly global (KPCA + CL-k-means + B₁/B₂).

## Pré-requisito

**Tabela VII 2018** concluída em `$MERGED18` (`06_supervised_metrics.json`, `models/supervised/`, `05_test_unchanged.parquet`).

## Onde grava

```text
data/pipeline_ids2018_merged/
├── anomaly/global/                     ← fases 7–11
│   └── models/anomaly/
├── phase_reports/phase13_full_system_eval.json
└── figures/
    ├── fig_multiclass_cm.png
    └── fig_binary_cm.png
```

## Passo a passo

### 1) Fases 7–11 — detector anomaly global

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol paper `
  --intermediate-dir $MERGED18
```

| Fase | Conteúdo |
|------|----------|
| 7 | Dataset binário (80% treino; hold-out reservado) |
| 8 | Z-score + IG + FCBF + KPCA (+ HPO no paper) |
| 9 | SMOTE + CL-k-means baseline |
| 10 | BO-GP: `n_clusters` + métrica |
| 11 | Biased B₁/B₂ + `p*` |

Fases 9–11 usam validação interna 20% do treino anomaly. O **test set intacto** só entra no passo 2 abaixo.

### 2) Fase 13 — avaliação end-to-end

```powershell
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir $MERGED18 `
  --work-dir $GLOBAL18
```

### 3) Imprimir Tabela X

```powershell
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir $MERGED18
```

```powershell
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir $MERGED18 `
  --save-json $MERGED18/phase_reports/table_x_ids2018.json
```

## Retomar

```powershell
# Só fases 10–11
python -m mth_ids_pipeline.run_global_anomaly --protocol paper `
  --intermediate-dir $MERGED18 --from-phase 10

# Só reavaliar (segundos)
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir $MERGED18 `
  --work-dir $GLOBAL18
```

## Modo rápido (sem HPO)

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol paper --no-hpo `
  --intermediate-dir $MERGED18
```

## Tempo estimado

Fase 8 (KPCA HPO): ~30–90 min ou mais no 2018. Pipeline completo 7–11: **2–3+ horas**.

---

# Tabela IX — LOAO / zero-day por tipo de ataque (IDS2018)

## Objetivo

Simular **ataques desconhecidos**: em cada rodada, um tipo de ataque fica **fora** do treino e entra só no teste. Métricas agregadas: média F1, DR, FAR.

## Onde grava

```text
data/pipeline_ids2018_fine/
├── 02_sampled_kmeans.parquet
├── 06_supervised_metrics.json          ← cópia da VII 2018
└── anomaly/loao/
    ├── attack_1/ … attack_N/
    └── loao_summary.json               ← principal (Tabela IX)
```

## Pré-requisitos

| # | O quê | Comando / ação |
|---|--------|----------------|
| 1 | Fases 1–2 no **fine** | Ver abaixo |
| 2 | Tabela VII no **merged** 2018 | Secção Tabela VII |
| 3 | Copiar métricas VII → fine | `Copy-Item` abaixo |
| 4 | LOAO com **`--skip-bootstrap`** | Evita puxar Tabela VII de **2017** |

### 1) Fases 1–2 (fine)

```powershell
python -m mth_ids_pipeline.phases.phase01_load_preprocess `
  --input $RAW2018 `
  --intermediate-dir $FINE18

python -m mth_ids_pipeline.phases.phase02_sample_kmeans `
  --intermediate-dir $FINE18 `
  --auto-minority
```

### 2) Tabela VII 2018

Conclua a secção **Tabela VII** em `$MERGED18`.

### 3) Copiar `06_supervised_metrics.json`

```powershell
Copy-Item `
  "$MERGED18/06_supervised_metrics.json" `
  "$FINE18/06_supervised_metrics.json"
```

## Passo a passo — LOAO

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao `
  --skip-bootstrap `
  --intermediate-dir $FINE18 `
  --raw-csv $RAW2018
```

> **`--skip-bootstrap` é obrigatório** no 2018. Sem ele, o pipeline pode bootstrapar a Tabela VII de `pipeline_mth_ids_merged` (**CICIDS2017**).

### Um ataque só

Descubra o `Label` no relatório da fase 1 ou no `value_counts` e rode:

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap `
  --intermediate-dir $FINE18 `
  --raw-csv $RAW2018 `
  --attack-label <N>
```

### Imprimir Tabela IX

```powershell
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root $LOAO18
```

```powershell
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root $LOAO18 `
  --save-json $FINE18/phase_reports/table_ix_ids2018.json
```

## Retomar um ataque (fases 9–11)

Se `a04_after_kpca.parquet` já existe em `attack_<N>/`:

```powershell
python -m mth_ids_pipeline.run_all --label-profile fine `
  --protocol paper --from 12 --to 12 --skip-bootstrap `
  --intermediate-dir $FINE18 `
  --attack-label <N>
```

## Tempo estimado

~1 h na fase 8 **por ataque** (KPCA HPO). LOAO completo: **muitas horas a dias**, conforme número de classes de ataque no 2018.

---

# Imprimir as três tabelas (somente leitura)

Depois de treinar VII, IX e X no 2018:

```powershell
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir $MERGED18 `
  --loao-root $LOAO18 `
  --save-json $MERGED18/phase_reports/paper_comparison_ids2018.json
```

| Flag `--table` | Tabela |
|----------------|--------|
| `vii` | VII only |
| `ix` | IX only |
| `x` | X only |
| `all` | VII + IX + X |

---

# Checklist — não misturar com CICIDS2017

- [ ] `$MERGED18` = `pipeline_ids2018_merged` (não `pipeline_mth_ids_merged`)
- [ ] `$FINE18` = `pipeline_ids2018_fine` (não `pipeline_mth_ids_fine`)
- [ ] `--raw-csv` / `--input` = `CSE-CIC-IDS2018.csv`
- [ ] LOAO: `--skip-bootstrap` + cópia manual de `06_supervised_metrics.json`
- [ ] Log mostra `intermediate-dir: data/pipeline_ids2018_*`
- [ ] Pastas `pipeline_mth_ids_*` do 2017 intactas

---

# Limitações (IDS2018)

| Item | Nota |
|------|------|
| Coluna **Diff** em `report_paper_tables` | Compara com artigo **CICIDS2017**, não 2018 |
| Nomes de ataques no LOAO | Mapas hardcoded para CICIDS2017 |
| `merge_cicids` | Não gera CSV 2018 |
| Bootstrap LOAO default | Aponta para Tabela VII **2017** — use `--skip-bootstrap` |
| Fase 2 `--auto-minority` | Recomendado; defaults 2017 não servem para 2018 |

Mais detalhes: [PROTOCOLO_CSE_CIC_IDS2018.md](PROTOCOLO_CSE_CIC_IDS2018.md)
