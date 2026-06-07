# Protocolo CSE-CIC-IDS2018 — sem sobrescrever CICIDS2017

Este documento descreve como rodar o pipeline MTH-IDS no **`CSE-CIC-IDS2018.csv`** usando **pastas de artefatos separadas**, para **não alterar** os resultados já gerados em `data/pipeline_mth_ids_merged/` e `data/pipeline_mth_ids_fine/` (CICIDS2017).

Documentos relacionados: [IDS2018_TABELAS_VII_IX_X.md](IDS2018_TABELAS_VII_IX_X.md) · [COMO_RODAR_TABELAS.md](COMO_RODAR_TABELAS.md) · [TABELAS_COMANDOS_SEPARADOS.md](TABELAS_COMANDOS_SEPARADOS.md) · [PASTAS_E_BOOTSTRAP.md](PASTAS_E_BOOTSTRAP.md)

---

## Princípio

| CICIDS2017 (não tocar) | CSE-CIC-IDS2018 (novo) |
|------------------------|-------------------------|
| `data/CICIDS2017.csv` | `data/CSE-CIC-IDS2018.csv` |
| `data/pipeline_mth_ids_merged/` | `data/pipeline_ids2018_merged/` |
| `data/pipeline_mth_ids_fine/` | `data/pipeline_ids2018_fine/` |

**Regra:** em **todo** comando de treino, passe explicitamente:

- `--intermediate-dir` → pasta **2018**
- `--raw-csv` → CSV **2018** (fase 1 / `run_supervised`)

Se omitir esses argumentos, o pipeline usa os defaults do **CICIDS2017** e pode sobrescrever ou misturar artefatos.

---

## Variáveis (PowerShell)

Defina uma vez por sessão:

```powershell
$RAW2018   = "data/CSE-CIC-IDS2018.csv"
$MERGED18  = "data/pipeline_ids2018_merged"
$FINE18    = "data/pipeline_ids2018_fine"
$GLOBAL18  = "$MERGED18/anomaly/global"
$LOAO18    = "$FINE18/anomaly/loao"
```

Opcional — cópia “fine” se quiser LOAO com os mesmos rótulos do CSV único:

```powershell
$RAW2018_FINE = "data/CSE-CIC-IDS2018_fine.csv"
# Copy-Item $RAW2018 $RAW2018_FINE   # se for o mesmo conteúdo
```

---

## Pré-requisitos do CSV

1. Arquivo em `data/CSE-CIC-IDS2018.csv` (ou caminho indicado em `$RAW2018`).
2. Coluna de classe **`Label`** (mesmo nome que o pipeline espera).
3. Classe benigna como **`BENIGN`** (texto).
4. **`merge_cicids` não serve para 2018** — ele lê `data/MachineLearningCSV/` do CICIDS2017. Use o CSV 2018 **diretamente** com `--raw-csv` / `--input`.

### Perfil merged vs fine no 2018

| Perfil | CSV | Uso |
|--------|-----|-----|
| **merged** (Tabelas VII / X) | Famílias agregadas **ou** rótulos originais + `--auto-minority` na fase 2 | Supervisionado + global |
| **fine** (Tabela IX / LOAO) | Subtipos de ataque | LOAO |

Se você só tem **um** CSV, pode usar o mesmo nos dois perfis; ajuste a fase 2 conforme a inspeção abaixo.

---

## Passo 0 — Inspecionar rótulos (obrigatório)

Os defaults de minoritárias (`6,1,4`) são do **CICIDS2017**. No 2018 você **deve** inspecionar antes da fase 2.

```powershell
python -m mth_ids_pipeline.phases.phase01_load_preprocess `
  --input $RAW2018 `
  --intermediate-dir $MERGED18
```

Abra:

```text
data/pipeline_ids2018_merged/phase_reports/phase01_load_preprocess.json
```

Depois rode a fase 2 com **`--auto-minority`** (preserva todos os tipos de ataque na amostra) **ou** `--minority-labels` manual:

```powershell
python -m mth_ids_pipeline.phases.phase02_sample_kmeans `
  --intermediate-dir $MERGED18 `
  --auto-minority
```

Para LOAO (fine), repita fases 1–2 em `$FINE18`:

```powershell
python -m mth_ids_pipeline.phases.phase01_load_preprocess `
  --input $RAW2018 `
  --intermediate-dir $FINE18

python -m mth_ids_pipeline.phases.phase02_sample_kmeans `
  --intermediate-dir $FINE18 `
  --auto-minority
```

---

## Tabela VII — IDS2018 (supervisionado)

Com fases 1–2 já concluídas (Passo 0):

```powershell
python -m mth_ids_pipeline.run_supervised --protocol paper `
  --from 4 --to 6 `
  --intermediate-dir $MERGED18 `
  --raw-csv $RAW2018
```

Relatório (métricas 2018; comparação **Diff** é vs artigo CICIDS2017):

```powershell
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir $MERGED18 `
  --save-json $MERGED18/phase_reports/table_vii_ids2018.json
```

---

## Tabela X — IDS2018 (sistema completo)

**Pré-requisito:** Tabela VII 2018 em `$MERGED18`.

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol paper `
  --intermediate-dir $MERGED18

python -m mth_ids_pipeline.run_eval `
  --intermediate-dir $MERGED18 `
  --work-dir $GLOBAL18

python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir $MERGED18 `
  --save-json $MERGED18/phase_reports/table_x_ids2018.json
```

Retomar sem refazer KPCA:

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol paper `
  --intermediate-dir $MERGED18 --from-phase 10
```

Teste rápido (sem HPO):

```powershell
python -m mth_ids_pipeline.run_global_anomaly --protocol paper --no-hpo `
  --intermediate-dir $MERGED18
```

---

## Tabela IX — IDS2018 (LOAO)

### Bootstrap automático e CICIDS2017

`run_anomaly --loao` **sem flags extras** pode bootstrapar a Tabela VII a partir de **`pipeline_mth_ids_merged` (2017)**. Para 2018 use **`--skip-bootstrap`** e prepare manualmente:

1. Fases 1–2 no `$FINE18` (Passo 0).
2. Tabela VII 2018 no `$MERGED18` (secção anterior).
3. Copiar métricas:

```powershell
Copy-Item `
  "$MERGED18/06_supervised_metrics.json" `
  "$FINE18/06_supervised_metrics.json"
```

4. LOAO:

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao `
  --skip-bootstrap `
  --intermediate-dir $FINE18 `
  --raw-csv $RAW2018
```

5. Relatório:

```powershell
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --loao-root $LOAO18 `
  --save-json $FINE18/phase_reports/table_ix_ids2018.json
```

---

## Mapa de pastas

```text
data/
├── CSE-CIC-IDS2018.csv
├── pipeline_mth_ids_merged/          # CICIDS2017 — NÃO TOCAR
├── pipeline_mth_ids_fine/
├── pipeline_ids2018_merged/          # VII + X (2018)
│   ├── anomaly/global/
│   └── phase_reports/phase13_full_system_eval.json
└── pipeline_ids2018_fine/            # IX (2018)
    └── anomaly/loao/
```

---

## Checklist — não sobrescrever 2017

- [ ] Sempre `--intermediate-dir` apontando para `pipeline_ids2018_*`.
- [ ] Sempre `--raw-csv` / `--input` apontando para CSV 2018.
- [ ] LOAO 2018: **`--skip-bootstrap`** + cópia manual de `06_supervised_metrics.json`.
- [ ] Confirmar no log: `intermediate-dir:` → `pipeline_ids2018_*`.
- [ ] Não rodar `merge_cicids` esperando gerar 2018.

---

## Limitações (IDS2018)

| Item | Situação |
|------|----------|
| Referências em `report_paper_tables` | Valores do artigo são **CICIDS2017** |
| `merge_cicids` | Apenas CICIDS2017 |
| Nomes LOAO | Hardcoded CICIDS2017 |
| Bootstrap LOAO | Default puxa Tabela VII de 2017 — use `--skip-bootstrap` |
| Agrupamento merged | `CICIDS_LABEL_MERGE` é 2017 — prepare CSV ou use `--auto-minority` |

---

## Fluxo completo (copiar e colar)

```powershell
$RAW2018  = "data/CSE-CIC-IDS2018.csv"
$MERGED18 = "data/pipeline_ids2018_merged"
$FINE18   = "data/pipeline_ids2018_fine"
$GLOBAL18 = "$MERGED18/anomaly/global"
$LOAO18   = "$FINE18/anomaly/loao"

# VII
python -m mth_ids_pipeline.phases.phase01_load_preprocess --input $RAW2018 --intermediate-dir $MERGED18
python -m mth_ids_pipeline.phases.phase02_sample_kmeans --intermediate-dir $MERGED18 --auto-minority
python -m mth_ids_pipeline.run_supervised --protocol paper --from 4 --to 6 --intermediate-dir $MERGED18 --raw-csv $RAW2018

# X
python -m mth_ids_pipeline.run_global_anomaly --protocol paper --intermediate-dir $MERGED18
python -m mth_ids_pipeline.run_eval --intermediate-dir $MERGED18 --work-dir $GLOBAL18

# IX (opcional)
python -m mth_ids_pipeline.phases.phase01_load_preprocess --input $RAW2018 --intermediate-dir $FINE18
python -m mth_ids_pipeline.phases.phase02_sample_kmeans --intermediate-dir $FINE18 --auto-minority
Copy-Item "$MERGED18/06_supervised_metrics.json" "$FINE18/06_supervised_metrics.json"
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao --skip-bootstrap --intermediate-dir $FINE18 --raw-csv $RAW2018

# Relatórios
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir $MERGED18 `
  --loao-root $LOAO18 `
  --save-json $MERGED18/phase_reports/paper_comparison_ids2018.json
```
