# Replicacao MTH-IDS

## Comandos Corretos

### Criar ambiente e instalar dependencias

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp
```

### Rodar o pipeline supervisionado a partir do dataset padrao

Usar somente o arquivo:

`/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/data/CICIDS2017.csv`

Comando:

```bash
.venv/bin/python -m mth_ids_pipeline.run_all --to 6
```

### Rodar somente ate a fase 4

```bash
.venv/bin/python -m mth_ids_pipeline.run_all --to 4
```

### Conferir quantas linhas ficaram apos o sampling e os splits

```bash
wc -l \
  data/pipeline_mth_ids/02_sampled_kmeans.csv \
  data/pipeline_mth_ids/03_train.csv \
  data/pipeline_mth_ids/03_test.csv \
  data/pipeline_mth_ids/04_train_after_fcbf.csv \
  data/pipeline_mth_ids/04_test_after_fcbf.csv
```

### Conferir shapes e distribuicao por classe

```bash
python3 - <<'PY'
import pandas as pd
for path in [
    'data/pipeline_mth_ids/02_sampled_kmeans.csv',
    'data/pipeline_mth_ids/03_train.csv',
    'data/pipeline_mth_ids/03_test.csv',
    'data/pipeline_mth_ids/04_train_after_fcbf.csv',
    'data/pipeline_mth_ids/04_test_after_fcbf.csv',
]:
    df = pd.read_csv(path)
    print(path, df.shape)
    print(df['Label'].value_counts().sort_index())
    print('-' * 40)
PY
```

### Conferir metricas da fase supervisionada

```bash
cat data/pipeline_mth_ids/06_supervised_metrics.json
```

## Valores Esperados Com `data/CICIDS2017.csv`

Ao rodar:

```bash
.venv/bin/python -m mth_ids_pipeline.run_all --to 4
```

Os tamanhos esperados ficam aproximadamente em:

- `02_sampled_kmeans.csv`: ~26.8k amostras
- `03_train.csv`: ~21.4k amostras
- `03_test.csv`: ~5.3k amostras
- `04_train_after_fcbf.csv`: mesmo numero de linhas do treino
- `04_test_after_fcbf.csv`: mesmo numero de linhas do teste

Observacao:

- A fase 4 reduz colunas, nao linhas.
- Pode haver pequena variacao no numero exato de linhas do sampling porque o notebook original usa `group.sample(frac=0.008)` sem `random_state` nessa etapa.
