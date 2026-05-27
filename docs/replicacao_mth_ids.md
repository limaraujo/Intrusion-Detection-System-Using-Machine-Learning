# Replicacao MTH-IDS

Este documento compara o notebook original [MTH_IDS_IoTJ_original.ipynb](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/MTH_IDS_IoTJ_original.ipynb) com o pipeline modular em `mth_ids_pipeline/` para verificar se estamos reproduzindo o procedimento do autor principal.

## Objetivo

A comparacao aqui nao avalia apenas a metrica final. Ela verifica:

- ordem das etapas
- entradas e saidas intermediarias
- hiperparametros
- logica de amostragem, split e selecao de atributos
- possiveis divergencias metodologicas

## Legenda

- `igual`: implementacao equivalente ao notebook, com mesma logica central
- `adaptado`: mesma ideia geral, mas com reorganizacao de codigo, IO ou instrumentacao
- `divergente`: fluxo atual nao replica exatamente o notebook e pode alterar os resultados

## Artefatos de referencia

- Notebook original: [MTH_IDS_IoTJ_original.ipynb](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/MTH_IDS_IoTJ_original.ipynb)
- Orquestrador: [run_all.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/run_all.py:24)
- Fase 1: [phase01_load_preprocess.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase01_load_preprocess.py:76)
- Fase 2: [phase02_sample_kmeans.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase02_sample_kmeans.py:33)
- Fase 3: [phase03_train_test_split.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase03_train_test_split.py:24)
- Fase 4: [phase04_feature_engineering.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase04_feature_engineering.py:52)
- Fase 5: [phase05_smote.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase05_smote.py:20)
- Fase 6: [phase06_supervised_models.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase06_supervised_models.py:42)
- Fase 7: [phase07_anomaly_datasets.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase07_anomaly_datasets.py:18)
- Fase 8: [phase08_anomaly_features.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase08_anomaly_features.py:44)
- Fase 9: [phase09_anomaly_cluster.py](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/mth_ids_pipeline/phase09_anomaly_cluster.py:64)

## Matriz De Comparacao

| Notebook | Conteudo original | Pipeline atual | Status | Observacao |
|---|---|---|---|---|
| Cells 9-11 | Normalizacao Z-score nas colunas numericas, `fillna(0)`, `LabelEncoder` | Fase 1 e inicio da Fase 2 | `adaptado` | A logica foi preservada, mas o `LabelEncoder` foi movido da fase de preprocessamento para a fase de sampling. |
| Cells 13-20 | Separa classes minoritarias `6,1,4`, roda `MiniBatchKMeans(n_clusters=1000, random_state=0)` nas majoritarias e amostra `frac=0.008` por cluster | Fase 2 | `igual` | A logica central e os hiperparametros principais foram preservados. |
| Cell 20 | `group.sample(frac=0.008)` sem `random_state` | Fase 2 | `igual` | O pipeline atual replica essa falta de determinismo no sampling por cluster. Isso pode gerar pequenas variacoes entre execucoes. |
| Cell 24 | Salva `./data/CICIDS2017_sample_km.csv` | Fase 2 salva `data/pipeline_mth_ids/02_sampled_kmeans.csv` | `adaptado` | Mesmo papel funcional, mas em outro caminho e com outro nome. |
| Cells 26-28 | Le o dataset amostrado e faz `train_test_split(..., random_state=0, stratify=y)` | Fase 3 | `igual` | O split supervisionado esta alinhado com o notebook. |
| Cells 31-33 | Calcula Information Gain em `X_train` e seleciona atributos ate 90% da importancia acumulada | Fase 4 | `igual` | O pipeline atual usa `mutual_info_classif(..., random_state=0)` e aplica o criterio acumulado de 90%. |
| Cells 37-38 | Executa `FCBFK(k=20)` sobre `X_fs` e `y` do dataset inteiro | Fase 4 | `divergente` | O notebook aplica FCBF sobre o conjunto inteiro apos a selecao IG; o pipeline atual ajusta FCBF no treino e projeta o teste pelas features selecionadas. |
| Cell 41 | Re-faz o `train_test_split` apos FCBF em `X_fss` e `y` | Fase 4 | `divergente` | No notebook ha um segundo split apos feature selection. No pipeline atual o split vem antes, na fase 3. |
| Cells 45-46 | `SMOTE(n_jobs=-1, sampling_strategy={2:1000,4:1000})` apenas no treino | Fase 5 | `igual` | Estrategia e alvo de balanceamento estao alinhados. |
| Cell 51 | XGBoost base com `n_estimators=10` | Fase 6 | `divergente` | O pipeline atual nao executa o baseline simples do notebook; vai direto para a variante HPO quando `--no-hpo` esta ativo. |
| Cell 54 | XGBoost HPO com `learning_rate=0.7340229699980686`, `n_estimators=70`, `max_depth=14` | Fase 6 | `igual` | Hiperparametros replicados. |
| Cell 57 | RandomForest base com `random_state=0` | Fase 6 | `divergente` | O baseline simples nao foi mantido como experimento separado no pipeline. |
| Cell 60 | RF HPO com `n_estimators=71`, `min_samples_leaf=1`, `max_depth=46`, `min_samples_split=9`, `max_features=20`, `criterion='entropy'` | Fase 6 | `igual` | Hiperparametros replicados. |
| Cell 63 | DecisionTree base com `random_state=0` | Fase 6 | `divergente` | O baseline simples nao foi mantido como experimento separado no pipeline. |
| Cell 66 | DT HPO com `min_samples_leaf=2`, `max_depth=47`, `min_samples_split=3`, `max_features=19`, `criterion='gini'` | Fase 6 | `igual` | Hiperparametros replicados. |
| Cell 69 | ExtraTrees base com `random_state=0` | Fase 6 | `divergente` | O baseline simples nao foi mantido como experimento separado no pipeline. |
| Cell 72 | ET HPO com `n_estimators=53`, `min_samples_leaf=1`, `max_depth=31`, `min_samples_split=5`, `max_features=20`, `criterion='entropy'` | Fase 6 | `igual` | Hiperparametros replicados. |
| Cells 75-79 | Concatena predicoes dos quatro modelos e treina `xgb.XGBClassifier()` para stacking | Fase 6 | `igual` | A mesma construcao de meta-features foi implementada. |
| Cell 82 | Meta-XGBoost com `learning_rate=0.19229249758051492`, `n_estimators=30`, `max_depth=36` | Fase 6 | `igual` | Hiperparametros replicados. |
| Cells 87-90 | Gera datasets do ramo anomaly: sem PortScan e so PortScan; converte classes para rotulo binario | Fase 7 | `igual` | A logica geral esta alinhada. |
| Cell 96 | `df = df1.append(df2)` | Fase 8 | `igual` | O pipeline usa `pd.concat`, que e a forma moderna equivalente. |
| Cell 100 | IG no ramo anomaly com `mutual_info_classif(X, y)` sobre o dataset combinado | Fase 8 | `igual` | Mesmo principio metodologico. |
| Cell 108 | FCBF no ramo anomaly com `FCBFK(k=20)` sobre o dataset combinado | Fase 8 | `igual` | O fluxo do ramo anomaly esta mais proximo do notebook que o fluxo supervisionado. |
| Cell 112 | `KernelPCA(n_components=10, kernel='rbf')` | Fase 8 | `igual` | Parametrizacao replicada. |
| Cell 114 | Particiona treino e teste por fatiamento: treino = `df1`, teste = `df2` | Fase 9 | `igual` | O pipeline usa o metadata `a06_test_slice.json` para reconstruir exatamente esse fatiamento. |
| Cell 117 | `SMOTE(n_jobs=-1, sampling_strategy={1:18225})` no treino anomaly | Fase 9 | `igual` | Alvo de oversampling replicado. |
| Cell 122 | `CL_kmeans(..., n, b=100)` com `MiniBatchKMeans` e mapeamento de clusters para binario | Fase 9 | `adaptado` | A logica de contagem e mapeamento foi mantida, com encapsulamento em funcao e `random_state=0` adicionado. |
| Cell 127 | Executa `CL_kmeans(..., 16)` na otimizacao mostrada | Fase 9 | `adaptado` | O pipeline atual usa `--n-clusters` configuravel e default `8`. Para replicar a ultima chamada do notebook, deve-se executar com `--n-clusters 16`. |

## Divergencias Prioritarias

Estas sao as divergencias que mais provavelmente afetam a comparacao com os resultados do autor:

1. O fluxo supervisionado do notebook faz um segundo split apos o FCBF.
   No notebook:
   `sampled dataset -> split -> IG em X_train -> FCBF em X_fs do dataset inteiro -> novo split`

   No pipeline atual:
   `sampled dataset -> split -> IG no treino -> FCBF no treino -> projecao do teste`

2. O notebook inclui baselines simples para XGBoost, RF, DT e ET antes dos modelos HPO.
   O pipeline atual registra apenas as variantes HPO e os modelos de stacking.

3. O sampling por cluster nao fixa `random_state` dentro de `group.sample`.
   Isso afeta a reproducibilidade fina do arquivo amostrado.

4. O ramo anomaly no notebook termina com uma chamada explicita `CL_kmeans(..., 16)`.
   O pipeline atual deixa isso parametrizavel e usa `8` como default.

## Recomendacao De Validacao

Para afirmar replicacao com boa documentacao, comparar por checkpoints:

1. `sampling`
   Registrar shape final e `Label.value_counts()` apos o arquivo amostrado.

2. `split supervisionado`
   Registrar shape de treino e teste e distribuicao de classes em cada conjunto.

3. `feature selection`
   Registrar:
   - quantidade de features apos IG
   - nomes das features apos IG
   - quantidade final apos FCBF

4. `smote`
   Registrar `value_counts()` antes e depois do oversampling.

5. `modelos`
   Registrar hiperparametros efetivos, accuracy, precision, recall, F1 e matriz de confusao.

6. `anomaly branch`
   Registrar:
   - shape de `df1` e `df2`
   - quantidade de benignos misturados ao conjunto PortScan
   - dimensao apos KPCA
   - accuracy e matriz de confusao do CL-k-means

## Como Usar Este Documento

- Se o objetivo for `replicacao fiel`, as linhas marcadas como `divergente` devem ser corrigidas antes de comparar metricas finais.
- Se o objetivo for `pipeline reprodutivel da equipe`, as divergencias devem ser mantidas, mas explicitadas no relatorio como adaptacoes metodologicas.

## Proximo Passo Recomendado

Criar um modo de execucao `--faithful-notebook` no pipeline supervisionado para reproduzir exatamente:

- leitura de `CICIDS2017_sample_km.csv`
- primeiro split
- IG em `X_train`
- FCBF sobre `X_fs` do dataset inteiro
- segundo split apos FCBF
- treinamento dos baselines simples e das variantes HPO

Assim voces ficam com dois trilhos:

- `faithful`: replicacao do notebook do autor
- `modular`: versao mais limpa e reprodutivel da equipe
