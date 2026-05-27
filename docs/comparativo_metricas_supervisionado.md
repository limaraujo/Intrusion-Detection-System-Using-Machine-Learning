# Comparativo De Metricas Supervisionadas

Comparacao entre o notebook original [MTH_IDS_IoTJ_original.ipynb](/Users/fabrielyluana/Projects/CIn/Intrusion-Detection-System-Using-Machine-Learning/MTH_IDS_IoTJ_original.ipynb) e o pipeline atual do projeto para o ramo supervisionado do MTH-IDS.

## Observacao Importante

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

## Texto Sugerido Para O Relatorio

Os resultados obtidos pelo pipeline supervisionado da equipe foram altamente consistentes com os resultados reportados no notebook original do artigo MTH-IDS. As metricas de accuracy e F1-weighted apresentaram diferencas muito pequenas entre as duas implementacoes, tipicamente abaixo de 0.12 ponto percentual. Essa pequena variacao e esperada, pois a etapa de amostragem por clusters no notebook original nao fixa `random_state` durante o `group.sample(frac=0.008)`, o que pode produzir subconjuntos ligeiramente diferentes entre execucoes. Ainda assim, a proximidade entre os resultados indica que o comportamento geral da solucao supervisionada foi replicado com sucesso.
