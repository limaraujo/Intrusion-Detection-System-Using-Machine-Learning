# Protocolo CAN-intrusion — Tabelas VI e VIII (MTH-IDS)

Este documento descreve como rodar o pipeline MTH-IDS no **CAN-intrusion-dataset** com **`--protocol can_paper`** (artigo) ou **`--protocol can_notebook`** (IoTJ), usando **pastas de artefatos separadas** do CICIDS2017.

Documentos relacionados: [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) · [EXECUCAO.md](EXECUCAO.md) · [PIPELINE_PHASES.md](PIPELINE_PHASES.md) · [ARQUITETURA.md](ARQUITETURA.md)

---

## Princípio

| CICIDS2017 (externo) | CAN-intrusion (intra-veicular) |
|----------------------|--------------------------------|
| `data/CICIDS2017.csv` | `data/CAN_Intrusion_Dataset.csv` |
| `data/pipeline_mth_ids_merged/` | `data/pipeline_can_merged/` |
| `data/pipeline_mth_ids_fine/` | `data/pipeline_can_fine/` |
| `--protocol paper` / `notebook` | **`--protocol can_paper`** / **`can_notebook`** |
| Tabela VII (7 famílias) | **Tabela VI** (4 classes) |
| Tabela IX (14 LOAO) | **Tabela VIII** (3 LOAO) |

**Regra:** use `can_paper` ou `can_notebook` nos orquestradores. O alias **`--protocol can`** aponta para **`can_paper`**. Não misture artefatos CAN com `pipeline_mth_ids_*`.

---

## Dois protocolos CAN

Ambos usam k-means **0,8%** (`frac=0.008`, igual notebook/CICIDS2017) em **todas** as classes, **sem SMOTE**, pastas `pipeline_can_*`.

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
# Artigo (Tabela VI) — alias: --protocol can
python -m mth_ids_pipeline.run_supervised --protocol can_paper

# Notebook IoTJ adaptado ao CAN
python -m mth_ids_pipeline.run_supervised --protocol can_notebook
```

**Atenção:** os dois protocolos gravam na **mesma pasta** `pipeline_can_merged`. Regenerar ao trocar de preset.

Constantes em `mth_ids_pipeline/config.py`:

- `DEFAULT_RAW_CSV_CAN` → `data/CAN_Intrusion_Dataset.csv`
- `INTERMEDIATE_DIR_CAN_MERGED` → `data/pipeline_can_merged`
- `INTERMEDIATE_DIR_CAN_FINE` → `data/pipeline_can_fine`
- `RESULTS_DIR_CAN` → `results/can` (sugestão para relatórios exportados)

---

## Dataset e preparação do CSV

### Origem

Arquivos `.txt` do [CAN-intrusion-dataset](https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset) (Han et al.), colocados em `data/CAN_DATA/`:

| Arquivo típico | Rótulo |
|----------------|--------|
| `CAN_Attack_free_dataset.txt` | BENIGN |
| `CAN_DoS_attack_dataset.txt` | DoS |
| `CAN_Fuzzy_attack_dataset.txt` | Fuzzy |
| `CAN_Impersonation_attack_dataset.txt` | Impersonation |

### Geração do CSV

```powershell
python -m mth_ids_pipeline.utils.merge_can
# ou
python -m mth_ids_pipeline.utils.merge_can `
  --input-dir data/CAN_DATA `
  --output data/CAN_Intrusion_Dataset.csv
```

O script:

- parseia linhas `Timestamp / ID / DLC / payload`;
- **descarta o timestamp** (conforme artigo);
- gera colunas **`CAN_ID`** + **`DATA_0`…`DATA_7`** (bytes hex do payload, padding com 0);
- concatena os arquivos e embaralha (`random_state=42`).

**Não use `merge_cicids`** — ele é exclusivo do CICIDS2017.

### Classes após `LabelEncoder` (ordem alfabética)

| ID | Rótulo |
|----|--------|
| 0 | BENIGN |
| 1 | DoS |
| 2 | Fuzzy |
| 3 | Impersonation |

Definição em `config.CAN_LABEL_NAMES`.

---

## Preset `--protocol can`

Implementado em `mth_ids_pipeline/protocol.py` (`ProtocolSettings` CAN). Resumo vs `--protocol paper`:

| Item | CAN (`can_paper`) | Paper (CICIDS2017) |
|------|-------------------|---------------------|
| Perfil supervisionado | `can_merged` | `merged` |
| Perfil LOAO | `can_fine` | `fine` |
| Split | **80/20** | 80/20 |
| Validação HPO (fase 6) | 10-fold CV | 10-fold CV |
| Amostragem fase 2 | k-means **0,8% em todas as classes** | k-means 0,8% + minoritárias preservadas |
| SMOTE supervisionado (fase 5) | **Desligado** (`skip_smote`) | BruteForce + Infiltration → 1000 |
| SMOTE anomaly (fases 9–11) | **Desligado** (`--no-smote`) | Binário no treino |
| BO-GP | α IG + KPCA + p* (supervisionado e anomaly) | α IG BO-GP + KPCA + p* |
| Meta-learner stacking | `best-base` (clone do melhor base) | Igual |
| FCBF / Z-score supervisionado | Fit só no treino; scaler pós-split | Igual |

### Amostragem k-means (diferença principal)

No CICIDS2017, BENIGN, DoS, PortScan etc. passam pelo k-means 0,8%, mas Bot/Infiltration/WebAttack ficam **intactos**.

No CAN, **nenhuma classe é preservada**: BENIGN + DoS + Fuzzy + Impersonation passam todas pelo k-means **0,8%** (`--sample-all-classes` via perfil `can_merged` / `can_fine`).

Resultado típico após fase 2: **~36k linhas** (0,8% do CSV completo; ex.: BENIGN ~18k, Impersonation ~8k, DoS ~5k, Fuzzy ~4,7k).

### Sem SMOTE

Com `skip_smote=True`, a fase 5 é **pulada**. Após a fase 4, o orquestrador copia os parquets FCBF para os caminhos da fase 5 (`04_*` → `05_*`) para a fase 6 ler os mesmos dados — ver `_link_fcbf_to_smote_paths()` em `experiment_runner.py`.

---

## Tabela VI — supervisionado (ataques conhecidos)

Equivalente pipeline à Tabela VII do CICIDS2017.

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_can_merged` |
| CSV | `data/CAN_Intrusion_Dataset.csv` |
| Perfil | `can_merged` (4 classes) |
| Amostragem | k-means `frac=0.008` (0,8%) em **todas** as classes |
| Split | **80% treino / 20% teste** |
| Validação | 10-fold CV no treino (HPO fase 6) |
| SMOTE | **Nenhum** |
| IG | **BO-GP α** (`--optimize-ig`, 15 calls, CV 10-fold; α inicial 0,9) |
| Features IG | Seleção dinâmica por IG acumulado + FCBF (k=20, treino) |
| FCBF | k=20, ajuste só no treino |
| Stacking (tier 2) | Meta-learner = clone do melhor base (`best-base`) |

```powershell
python -m mth_ids_pipeline.utils.merge_can
python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.report_paper_tables --table vii `
  --merged-dir data/pipeline_can_merged
```

`run_supervised --protocol can` define automaticamente `--label-profile can_merged` e `--intermediate-dir data/pipeline_can_merged`.

**Features após fase 4:** o artigo cita **4 features** (`CAN ID`, `DATA[1]`, `DATA[3]`, `DATA[5]`). O preset `--protocol can_paper` usa **BO-GP α IG** (como CICIDS2017 `--protocol paper`); na run validada de 2026-06-07 o α ótimo foi **≈0,81** e restaram **~7 features** pós-FCBF. Para forçar as 4 do artigo, use `--ig-features CAN_ID,DATA_1,DATA_3,DATA_5` na fase 4.

---

## Tabela VIII — LOAO anomaly (zero-day)

Equivalente pipeline à Tabela IX do CICIDS2017, com **3 rodadas** (DoS, Fuzzy, Impersonation).

| Item | Valor |
|------|-------|
| Pasta | `data/pipeline_can_fine` |
| CSV | **Mesmo** `CAN_Intrusion_Dataset.csv` (não há CSV fine separado) |
| Perfil | `can_fine` |
| Bootstrap | Fases **1–2** no fine + Tabela VI no merged (`06_…` copiado; automático) |
| Amostragem fase 2 | k-means 0,8% em **todas** as classes (igual merged CAN) |
| LOAO | 3 ataques; teste = zero-day + benignos **1:1** |
| SMOTE binário | **Nenhum** (`--no-smote` fases 9–11) |
| Features | Z-score → IG → FCBF → KPCA no **conjunto combinado** |
| BO-GP | KPCA, n_clusters, p* (anomaly) |
| Biased (tier 4) | Família B₁/B₂ da **Tabela VI** (`pipeline_can_merged`) |

```powershell
python -m mth_ids_pipeline.utils.merge_can
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
python -m mth_ids_pipeline.report_paper_tables --table ix `
  --merged-dir data/pipeline_can_merged
```

**Um ataque ou subset:**

```powershell
python -m mth_ids_pipeline.run_all --label-profile can_fine `
  --protocol can --from 12 --to 12 --skip-bootstrap `
  --attack-label 1

# DoS=1, Fuzzy=2, Impersonation=3
python -m mth_ids_pipeline.run_anomaly --protocol can --loao --attack-labels 1,2,3
```

### Bootstrap automático (CAN)

Mesma lógica do CICIDS2017 fine, mas apontando para pastas CAN:

1. Falta `02_sampled_kmeans.parquet` no fine → fases **1–2** em `pipeline_can_fine` (`--sample-all-classes`).
2. Falta `06_supervised_metrics.json` no fine → Tabela VI em `pipeline_can_merged` (fases **1–6** se necessário) e **cópia** do JSON para o fine.

O bootstrap CAN **não** puxa métricas do CICIDS2017 — usa `paired_supervised_dir=INTERMEDIATE_DIR_CAN_MERGED` e `table_vii_profile="can_merged"`.

Mensagens esperadas:

```text
[anomaly] Bootstrap fases 1–2 (fine) em data/pipeline_can_fine ...
[anomaly] Tabela VII ausente: bootstrap fases 1–6 em data/pipeline_can_merged (perfil can_merged ...)
[anomaly] 06_supervised_metrics.json ← .../pipeline_can_merged/... (Tabela VII → tier 4 biased)
```

(A mensagem diz “Tabela VII” por nomenclatura interna; no artigo CAN corresponde à **Tabela VI**.)

---

## Sistema completo (Tabela X — referência CAN)

O artigo reporta métricas de sistema completo também para CAN (`config.PAPER_TABLE_X_REFERENCE["can"]`). O pipeline reutiliza fases 7–11 global + fase 13:

```powershell
python -m mth_ids_pipeline.run_supervised --protocol can --from 1 --to 2
python -m mth_ids_pipeline.run_supervised --protocol can --from 4 --to 6
python -m mth_ids_pipeline.run_global_anomaly --protocol can
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_can_merged `
  --work-dir data/pipeline_can_merged/anomaly/global
python -m mth_ids_pipeline.report_paper_tables --table x `
  --merged-dir data/pipeline_can_merged
```

Referência do artigo (CAN, Tabela X): Acc ~99,99%, DR ~100%, FAR ~0,00005%, F1 ~0,9999.

---

## Mapa de pastas

```text
data/
├── CAN_DATA/                         # CAN_*.txt (entrada bruta)
├── CAN_Intrusion_Dataset.csv         # merge_can
├── pipeline_mth_ids_merged/          # CICIDS2017 — NÃO TOCAR
├── pipeline_mth_ids_fine/
├── pipeline_can_merged/              # Tabela VI
│   ├── 01_preprocessed.parquet
│   ├── 02_sampled_kmeans.parquet
│   ├── 04_train_after_fcbf.parquet
│   ├── 05_train_after_smote.parquet  # cópia de 04_* (sem SMOTE real)
│   ├── 06_supervised_metrics.json
│   ├── models/supervised/
│   ├── anomaly/global/               # Tabela X (opcional)
│   └── phase_reports/
└── pipeline_can_fine/                # Tabela VIII (LOAO)
    ├── 02_sampled_kmeans.parquet
    ├── 06_supervised_metrics.json    # cópia do can_merged
    └── anomaly/loao/
        ├── attack_1/                 # DoS zero-day
        ├── attack_2/                 # Fuzzy
        ├── attack_3/                 # Impersonation
        └── loao_summary.json

results/
├── paper_comparison.json             # CICIDS2017 (default)
└── can/                              # CAN (auto quando --merged-dir contém pipeline_can)
```

---

## Defaults dos entrypoints (CAN)

| Script | `--protocol` | Perfil | Pasta padrão |
|--------|--------------|--------|--------------|
| `run_supervised` | `can` | `can_merged` | `data/pipeline_can_merged` |
| `run_anomaly` | `can` | `can_fine` | `data/pipeline_can_fine` |
| `run_global_anomaly` | `can` | — | `data/pipeline_can_merged` |
| `experiment_runner` / `run_all` | `can` | conforme ramo | pastas CAN acima |

Aliases aceitos: `can`, `can-intrusion`, `can_intrusion` (`get_protocol_settings`).

---

## Fluxo completo (copiar e colar)

```powershell
# 0) CSV
python -m mth_ids_pipeline.utils.merge_can

# 1) Tabela VI
python -m mth_ids_pipeline.run_supervised --protocol can

# 2) Tabela VIII (LOAO — 3 ataques)
python -m mth_ids_pipeline.run_anomaly --protocol can --loao

# 3) Relatórios (Tabelas VI/VIII/X vs artigo CAN — detecção automática)
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_merged
```

Logs timestampados: `results/logs/supervised_merged_can_phases*.log`, `results/logs/loao/attack_*.log`.

---

## Limitações e diferenças conhecidas

| Item | Situação |
|------|----------|
| Nomes de artefatos LOAO | `a01_without_portscan.parquet` etc. — legado do notebook CICIDS2017 |
| Contagem de features pós IG-FCBF | BO-GP α IG (~7 na run validada); artigo cita 4 (`CAN_PAPER_IG_FEATURES`) |
| Tabela X CAN | Suportada via `run_global_anomaly --protocol can`; referência numérica em `PAPER_TABLE_X_REFERENCE["can"]` |
| LOAO | 3 ataques (não 14); nomes corretos via `CAN_LABEL_NAMES` na fase 12 |

---

## Erros comuns

### `FileNotFoundError: CAN_*.txt`

Coloque os quatro arquivos `.txt` do dataset em `data/CAN_DATA/` antes de `merge_can`.

### Bootstrap CAN puxando CICIDS2017

Use **`--protocol can`** (não `paper`). O perfil `can_fine` define `paired_supervised_dir` → `pipeline_can_merged`.

### Fase 5 ausente / fase 6 sem `05_*`

Normal com `--protocol can`: a fase 5 é omitida e os parquets são ligados automaticamente após a fase 4. Se rodar fases manualmente, execute fase 4 antes da 6 ou copie `04_*` → `05_*`.

### Regenerar amostra k-means CAN

```powershell
Remove-Item data\pipeline_can_merged\02_sampled_kmeans.parquet
Remove-Item data\pipeline_can_fine\02_sampled_kmeans.parquet
python -m mth_ids_pipeline.run_supervised --protocol can --from 1 --to 2
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
```

---

## Índice

- [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) — CICIDS2017
- [EXECUCAO.md](EXECUCAO.md) — comandos e bootstrap
- [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) — comparativo `paper` vs `notebook`
- [PIPELINE_PHASES.md](PIPELINE_PHASES.md) — referência de cada fase
