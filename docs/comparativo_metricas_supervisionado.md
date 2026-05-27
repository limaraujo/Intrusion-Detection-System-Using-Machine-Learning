# Comparativo De Metricas Supervisionadas

Comparacao entre o notebook original [MTH_IDS_IoTJ_original.ipynb](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/MTH_IDS_IoTJ_original.ipynb) e o pipeline atual do projeto para o ramo supervisionado do MTH-IDS.

## Observação Importante

- No notebook original, o conjunto de teste mostrado nas saidas tem `5360` amostras.
- No pipeline atual, o conjunto de teste usado nesta execucao tem `5358` amostras.
- Essa pequena diferenca e compativel com a etapa de sampling por cluster do notebook, que usa `group.sample(frac=0.008)` sem `random_state`.
- Por isso, as metricas devem ser comparadas como `muito proximas`, e nao como iguais bit a bit.

## Tabela Comparativa

| Modelo | Notebook original Accuracy | Pipeline atual Accuracy | Diferenca | Notebook original F1 weighted | Pipeline atual F1 weighted | Diferenca |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (HPO) | 0.995709 | 0.996827 | +0.001118 | 0.995690 | 0.996800 | +0.001110 |
| RandomForest (HPO) | 0.995149 | 0.995707 | +0.000558 | 0.995122 | 0.995720 | +0.000599 |
| DecisionTree (HPO) | 0.993657 | 0.993654 | -0.000002 | 0.993818 | 0.993658 | -0.000160 |
| ExtraTrees (HPO) | 0.995522 | 0.995894 | +0.000372 | 0.995493 | 0.995822 | +0.000329 |
| Stacking (XGB meta) | 0.995522 | 0.996267 | +0.000745 | 0.995504 | 0.996196 | +0.000692 |
| Meta-XGBoost (HPO) | 0.995709 | 0.996081 | +0.000372 | 0.995690 | 0.996009 | +0.000319 |

## Leitura Rapida

- Os resultados do pipeline atual estao muito proximos dos do notebook original.
- Em quase todos os modelos, o pipeline atual ficou levemente acima do notebook.
- A maior diferenca observada foi no XGBoost, ainda assim pequena: cerca de `+0.11` ponto percentual em accuracy.
- O DecisionTree ficou praticamente identico.
- Os valores exibidos como `1.00` no `classification_report` sao efeito de arredondamento para duas casas decimais, nao significam acerto perfeito real.

## Comparativo Dos Conjuntos Intermediarios

### Ramo supervisionado

| Etapa | Notebook original | Pipeline atual | Observacao |
|---|---|---|---|
| Dataset apos sampling k-means | `26800` linhas | `26788` linhas | Diferenca pequena, compativel com `group.sample(frac=0.008)` sem `random_state`. |
| Treino apos split | `21435 x 20` | `21430 x 21` | O notebook mostra shape apos FCBF; no pipeline a fase 4 tambem termina com 20 features + `Label`. |
| Teste apos split | `5360` linhas | `5358` linhas | Diferenca residual do sampling. |
| Treino apos SMOTE | classes `2` e `4` elevadas para `1000` | classes `2` e `4` elevadas para `1000` | Comportamento alinhado. |

### Distribuicao por classe no sampling k-means

| Classe | Notebook original | Pipeline atual | Diferenca |
|---|---:|---:|---:|
| 0 | 18185 | 18185 | 0 |
| 1 | 1966 | 1966 | 0 |
| 2 | 118 | 111 | -7 |
| 3 | 3029 | 3051 | +22 |
| 4 | 36 | 36 | 0 |
| 5 | 1280 | 1259 | -21 |
| 6 | 2180 | 2180 | 0 |
| Total | 26794* | 26788 | -6 |

\* A soma das contagens mostradas nas saidas do notebook e `26794`, embora o arquivo `data/CICIDS2017_sample_km.csv` presente no repositorio tenha `26800` linhas. Isso reforca que houve mais de uma execucao com sampling nao deterministico.

### Distribuicao por classe no treino antes do SMOTE

| Classe | Notebook original | Pipeline atual | Diferenca |
|---|---:|---:|---:|
| 0 | 14548 | 14547 | -1 |
| 1 | 1573 | 1573 | 0 |
| 2 | 94 | 89 | -5 |
| 3 | 2423 | 2441 | +18 |
| 4 | 29 | 29 | 0 |
| 5 | 1024 | 1007 | -17 |
| 6 | 1744 | 1744 | 0 |

### Distribuicao por classe no treino apos SMOTE

| Classe | Notebook original | Pipeline atual | Diferenca |
|---|---:|---:|---:|
| 0 | 14548 | 14547 | -1 |
| 1 | 1573 | 1573 | 0 |
| 2 | 1000 | 1000 | 0 |
| 3 | 2423 | 2441 | +18 |
| 4 | 1000 | 1000 | 0 |
| 5 | 1024 | 1007 | -17 |
| 6 | 1744 | 1744 | 0 |

### Ramo anomaly

| Etapa | Notebook original | Pipeline atual | Observacao |
|---|---|---|---|
| Dataset sem PortScan | classe `0: 18225`, classe `1: 7320` | classe `0: 18185`, classe `1: 7344` | Muito proximo; diferenca herdada do sampling supervisionado. |
| Dataset so PortScan | classe `1: 1255` | classe `1: 1259` | Diferenca pequena do sampling. |
| Dataset combinado apos mistura | total `28055` linhas | total `28040` linhas | Diferenca residual por conta dos totais anteriores. |
| Apos IG | `(28055, 50)` | nao registramos separadamente no arquivo final | O pipeline atual registra o resultado depois do KPCA. |
| Apos FCBF | `(28055, 20)` | nao registramos separadamente no arquivo final | O pipeline segue direto para KPCA. |
| Apos KPCA | nao mostrado na saida capturada | `(28040, 11)` | `10` componentes + `Label`. |
| Treino anomaly antes do SMOTE | classe `0: 18225`, classe `1: 7320` | classe `0: 18185`, classe `1: 7344` | Muito proximo. |
| Treino anomaly apos SMOTE | classe `0: 18225`, classe `1: 18225` | classe `0: 18185`, classe `1: 18225` | O alvo de oversampling foi preservado. |

## Suporte Do Conjunto De Teste

### Notebook original

| Classe | Suporte |
|---|---:|
| 0 | 3645 |
| 1 | 393 |
| 2 | 19 |
| 3 | 609 |
| 4 | 7 |
| 5 | 251 |
| 6 | 436 |
| Total | 5360 |

### Pipeline atual

| Classe | Suporte |
|---|---:|
| 0 | 3638 |
| 1 | 393 |
| 2 | 22 |
| 3 | 610 |
| 4 | 7 |
| 5 | 252 |
| 6 | 436 |
| Total | 5358 |

## Texto Sugerido Para O Relatório

Os resultados obtidos pelo pipeline supervisionado da equipe foram altamente consistentes com os resultados reportados no notebook original do artigo MTH-IDS. As metricas de accuracy e F1-weighted apresentaram diferencas muito pequenas entre as duas implementacoes, tipicamente abaixo de 0.12 ponto percentual. Essa pequena variacao e esperada, pois a etapa de amostragem por clusters no notebook original nao fixa `random_state` durante o `group.sample(frac=0.008)`, o que pode produzir subconjuntos ligeiramente diferentes entre execucoes. Ainda assim, a proximidade entre os resultados indica que o comportamento geral da solucao supervisionada foi replicado com sucesso.
