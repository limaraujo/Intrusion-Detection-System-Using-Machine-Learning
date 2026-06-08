# Protocolo CAN — índice (MTH-IDS Tabelas VI / VIII / X)

Este documento cobre o **pipeline** MTH-IDS para benchmarks **intra-veiculares** (`--protocol can_paper` / `can_notebook`). A preparação dos dados depende de **qual dataset CAN** você usa — são dois benchmarks distintos.

Documentos relacionados: [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [PROTOCOLO_UNSW_NB15.md](PROTOCOLO_UNSW_NB15.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [EXECUCAO.md](EXECUCAO.md) · [PIPELINE_PHASES.md](PIPELINE_PHASES.md) · [ARQUITETURA.md](ARQUITETURA.md)

---

## Dois datasets CAN (pastas separadas)

Cada estudo tem **CSV, pipeline e resultados próprios** — rodar um **não sobrescreve** o outro.

| | [Car-Hacking](PROTOCOLO_CAN_INTRUSION.md) | [OTIDS repack](PROTOCOLO_CAN_OTIDS.md) |
|---|-------------------------------------------|----------------------------------------|
| `--protocol` | **`can`** / `can_paper` | **`can_otids`** |
| `merge_can` | `--source original` | `--source otids` |
| CSV | `data/CAN_intrusion_Dataset.csv` | `data/CAN_OTIDS_Dataset.csv` |
| Pipeline VI | `data/pipeline_can_intrusion_merged/` | `data/pipeline_can_otids_merged/` |
| Pipeline VIII | `data/pipeline_can_intrusion_fine/` | `data/pipeline_can_otids_fine/` |
| Resultados | `results/can_intrusion/` | `results/can_otids/` |
| Classes | 5 (Gear + RPM separados) | 4 (Impersonation) |
| LOAO | 4 rodadas | 3 rodadas |

```powershell
# Car-Hacking (artigo)
python -m mth_ids_pipeline.utils.merge_can --source original
python -m mth_ids_pipeline.run_supervised --protocol can

# OTIDS
python -m mth_ids_pipeline.utils.merge_can --source otids
python -m mth_ids_pipeline.run_supervised --protocol can_otids
```

`--source auto` (default) usa **OTIDS** se `CAN_OTIDS_DATA/CAN_OTIDS_*.txt` existir; caso contrário, Car-Hacking em `data/`.

Detalhes de parse, contagens e troubleshooting: protocolos específicos acima.

---

## Princípio (vs CICIDS2017)

| CICIDS2017 (externo) | CAN (intra-veicular) |
|----------------------|----------------------|
| `data/CICIDS2017.csv` | `CAN_intrusion_Dataset.csv` ou `CAN_OTIDS_Dataset.csv` |
| `data/pipeline_mth_ids_merged/` | `pipeline_can_intrusion_merged/` ou `pipeline_can_otids_merged/` |
| `data/pipeline_mth_ids_fine/` | `pipeline_can_intrusion_fine/` ou `pipeline_can_otids_fine/` |
| `--protocol paper` / `notebook` | **`can`** / **`can_otids`** (+ variantes notebook) |
| Tabela VII (7 famílias) | **Tabela VI** (5 classes Car-Hacking / 4 OTIDS) |
| Tabela IX (14 LOAO) | **Tabela VIII** (4 LOAO Car-Hacking / 3 OTIDS) |

**Regra:** `--protocol can` → Car-Hacking; `--protocol can_otids` → repack OTIDS. Não misture com `pipeline_mth_ids_*`.

### Classes após `LabelEncoder` (ordem alfabética)

**Car-Hacking** (`CAN_INTRUSION_LABEL_NAMES`): BENIGN, DoS, Fuzzy, **Gear**, **RPM** (IDs 0–4).

**OTIDS** (`CAN_OTIDS_LABEL_NAMES`): BENIGN, DoS, Fuzzy, **Impersonation** (IDs 0–3).

Meta: `can_intrusion_meta.json` ou `can_otids_meta.json`. Rótulos LOAO via `resolve_can_label_names()`.

---

## Dois presets de pipeline (`can_paper` vs `can_notebook`)

Preset metodológico compartilhado: k-means **0,8%** em **todas** as classes, **sem SMOTE**. Pastas de artefato vêm do protocolo (`can` vs `can_otids`).

| Parâmetro | `can_paper` (alias `can`) | `can_notebook` (IoTJ) |
|-----------|---------------------------|------------------------|
| Split | **80/20** | **80/20** |
| α IG | **BO-GP** (15 calls, CV 10-fold; ex. α≈0,81) | **0,9 fixo** (seleção dinâmica MI) |
| Features supervisionadas | IG acumulado + FCBF (~N features) | IG acumulado (~N features) |
| FCBF | k=20, **só treino** | k=20, **dataset completo** |
| Normalização sup. | StandardScaler **após split** | Z-score **fase 1** |
| HPO fase 6 | **10-fold CV**, objetivo validação | **Hold-out**, objetivo teste |
| Stacking meta | **`best-base`** | **XGBoost** + HPO |
| Anomaly BO-GP | KPCA, k, p* | KPCA fixo; HPO só `n_clusters` |

```powershell
python -m mth_ids_pipeline.run_supervised --protocol can_paper   # Tabela VI (artigo)
python -m mth_ids_pipeline.run_supervised --protocol can_notebook
```

**Atenção:** `can` e `can_notebook` usam `pipeline_can_intrusion_*`; `can_otids` e `can_otids_notebook` usam `pipeline_can_otids_*`.

---

## Preset `can_paper` — resumo vs CICIDS2017

Implementado em `mth_ids_pipeline/protocol.py` (`ProtocolSettings` CAN).

| Item | CAN (`can_paper`) | Paper (CICIDS2017) |
|------|-------------------|---------------------|
| Perfil supervisionado | `can_merged` | `merged` |
| Perfil LOAO | `can_fine` | `fine` |
| Split | **80/20** | 80/20 |
| Validação HPO (fase 6) | 10-fold CV | 10-fold CV |
| Amostragem fase 2 | k-means **0,8% em todas as classes** | k-means 0,8% + minoritárias preservadas |
| SMOTE supervisionado (fase 5) | **Desligado** (`skip_smote`) | BruteForce + Infiltration → 1000 |
| SMOTE anomaly (fases 9–11) | **Desligado** (`--no-smote`) | Binário no treino |
| BO-GP | α IG + KPCA + p* | α IG BO-GP + KPCA + p* |
| Meta-learner stacking | `best-base` | Igual |

### Amostragem k-means

No CICIDS2017, classes minoritárias ficam **intactas**. No CAN, **todas** as classes passam pelo k-means **0,8%** (`--sample-all-classes`).

Resultado típico após fase 2 (repack OTIDS, ~4,5M CSV): **~36k linhas**. Com Car-Hacking (~17,5M), a amostra k-means é proporcionalmente maior — regenere fases 1–2 após trocar de dataset.

### Sem SMOTE

Com `skip_smote=True`, a fase 5 é **pulada**. O orquestrador copia `04_*` → `05_*` após a fase 4 (`_link_fcbf_to_smote_paths()` em `experiment_runner.py`).

---

## Tabela VI — supervisionado

| Item | Valor |
|------|-------|
| Pasta | `pipeline_can_intrusion_merged` ou `pipeline_can_otids_merged` |
| CSV | `CAN_intrusion_Dataset.csv` ou `CAN_OTIDS_Dataset.csv` |
| Perfil | `can_intrusion_merged` ou `can_otids_merged` |
| Amostragem | k-means `frac=0.008` em **todas** as classes |
| Split | **80% treino / 20% teste** |
| SMOTE | **Nenhum** |
| IG | **BO-GP α** (15 calls, CV 10-fold) |
| Stacking | Meta = clone do melhor base (`best-base`) |

```powershell
# 1) merge — escolha UMA fonte (ver tabela no topo)
python -m mth_ids_pipeline.utils.merge_can --source original   # ou --source otids

# 2) pipeline
python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_can_intrusion_merged
```

**Features pós fase 4:** artigo cita **4** (`CAN_ID`, `DATA_1`, `DATA_3`, `DATA_5`). BO-GP α IG pode selecionar **~7** pós-FCBF. Forçar as 4: `--ig-features CAN_ID,DATA_1,DATA_3,DATA_5` na fase 4.

---

## Tabela VIII — LOAO (zero-day)

**Car-Hacking:** 4 rodadas (DoS, Fuzzy, Gear, RPM). **OTIDS:** 3 rodadas (DoS, Fuzzy, Impersonation). Ver [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) e [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md).

| Item | Valor |
|------|-------|
| Pasta | `pipeline_can_intrusion_fine` ou `pipeline_can_otids_fine` |
| Perfil | `can_intrusion_fine` ou `can_otids_fine` |
| LOAO | teste = zero-day + benignos **1:1** |
| SMOTE binário | **Nenhum** |
| Biased (tier 4) | B₁/B₂ da Tabela VI (pasta merged pareada) |

```powershell
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --merged-dir data/pipeline_can_otids_merged
```

```powershell
# Car-Hacking: DoS=1, Fuzzy=2, Gear=3, RPM=4
python -m mth_ids_pipeline.run_anomaly --protocol can --loao --attack-labels 1,2,3,4
# OTIDS: DoS=1, Fuzzy=2, Impersonation=3
python -m mth_ids_pipeline.run_anomaly --protocol can_otids --loao --attack-labels 1,2,3
```

### Bootstrap automático

1. Falta `02_sampled_kmeans.parquet` no fine → fases **1–2** na pasta fine do protocolo.
2. Falta `06_supervised_metrics.json` no fine → Tabela VI no merged e **cópia** do JSON.

---

## Tabela X — sistema completo

```powershell
python -m mth_ids_pipeline.run_supervised --protocol can --from 1 --to 2
python -m mth_ids_pipeline.run_supervised --protocol can --from 4 --to 6
python -m mth_ids_pipeline.run_global_anomaly --protocol can
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_can_intrusion_merged `
  --work-dir data/pipeline_can_intrusion_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_can_intrusion_merged
```

Referência artigo (CAN): Acc ~99,99%, DR ~100%, FAR ~0,00005%, F1 ~0,9999 (`PAPER_TABLE_X_REFERENCE["can"]`).

---

## Mapa de pastas

```text
data/
├── CAN_intrusion_Dataset.csv          # merge --source original
├── CAN_OTIDS_Dataset.csv              # merge --source otids
├── pipeline_can_intrusion_merged/     # Tabela VI (Car-Hacking)
├── pipeline_can_intrusion_fine/       # Tabela VIII
├── pipeline_can_otids_merged/         # Tabela VI (OTIDS)
└── pipeline_can_otids_fine/

results/
├── can_intrusion/
└── can_otids/
```

---

## Fluxo completo (copiar e colar)

Substitua o merge pela fonte escolhida:

```powershell
# Car-Hacking (artigo)          # OU: merge_can --source otids
python -m mth_ids_pipeline.utils.merge_can --source original

python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_intrusion_merged
```

---

## Limitações

| Item | Situação |
|------|----------|
| Um CSV, duas fontes | Mesmo path `CAN_OTIDS_Dataset.csv`; usar `--source` explícito |
| Nomes LOAO | `a01_without_portscan.parquet` — legado CICIDS2017 |
| Features pós IG-FCBF | BO-GP ~7 vs 4 do artigo (`CAN_PAPER_IG_FEATURES`) |
| LOAO | 4 ataques (Car-Hacking) ou 3 (OTIDS); ver `can_dataset_meta.json` |
| Métricas vs artigo | Contagens Tabela IV só para Car-Hacking original |

---

## Erros comuns

### CSV com escala errada

Confirme `--source` no merge e regenere `pipeline_can_otids_*` após trocar de dataset.

### Bootstrap CAN puxando CICIDS2017

Use **`--protocol can`**. Perfil `can_fine` → `paired_supervised_dir=pipeline_can_otids_merged`.

### Fase 5 ausente

Normal com `--protocol can`: copiar `04_*` → `05_*` ou rodar via orquestrador.

### Regenerar k-means

```powershell
Remove-Item data\pipeline_can_otids_merged\02_sampled_kmeans.parquet
Remove-Item data\pipeline_can_otids_fine\02_sampled_kmeans.parquet
python -m mth_ids_pipeline.run_supervised --protocol can --from 1 --to 2
```

---

## Índice

- [PROTOCOLO_CAN_INTRUSION.md](PROTOCOLO_CAN_INTRUSION.md) — Car-Hacking original (artigo)
- [PROTOCOLO_CAN_OTIDS.md](PROTOCOLO_CAN_OTIDS.md) — repack OTIDS
- [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) — CICIDS2017
- [EXECUCAO.md](EXECUCAO.md) — comandos e bootstrap
- [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) — `paper` vs `notebook` vs `can_*`
