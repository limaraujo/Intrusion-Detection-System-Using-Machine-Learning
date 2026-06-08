# Documentação MTH-IDS

Índice do pacote `mth_ids_pipeline` — reprodução Yang et al. (IEEE IoT Journal 2022).

## Documentos

| Documento | Conteúdo |
|-----------|----------|
| [ARQUITETURA.md](ARQUITETURA.md) | 4 tiers, 3 experimentos, layout do código, perfis merged/fine |
| [EXECUCAO.md](EXECUCAO.md) | Comandos por tabela, pastas, bootstrap, troubleshooting |
| [PIPELINE_PHASES.md](PIPELINE_PHASES.md) | Referência das fases 1–13 + [CLI manual](PIPELINE_PHASES.md#rodar-cada-fase-manualmente) |
| [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) | CICIDS2017 (Tabelas VII/IX/X) |
| [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md) | CAN-intrusion (Tabelas VI/VIII) |
| [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md) | Comparativo `paper` vs `notebook` |
| [NOTA_REPRODUCAO.md](NOTA_REPRODUCAO.md) | **Limitações artigo × notebook** (texto para relatório) |
| [REPRODUCAO_CICIDS2017_VALIDACAO.md](REPRODUCAO_CICIDS2017_VALIDACAO.md) | Validação vs artigo e notebook |

Histórico de auditorias: [archive/](archive/README.md).

---

## Início rápido (CICIDS2017)

```powershell
python -m mth_ids_pipeline.utils.merge_cicids --profile merged
python -m mth_ids_pipeline.utils.merge_cicids --profile fine

python -m mth_ids_pipeline.run_supervised --protocol paper          # Tabela VII
python -m mth_ids_pipeline.run_anomaly --protocol paper --loao     # Tabela IX
python -m mth_ids_pipeline.run_global_anomaly --protocol paper       # Tabela X (treino)
python -m mth_ids_pipeline.run_eval `
  --intermediate-dir data/pipeline_mth_ids_merged `
  --work-dir data/pipeline_mth_ids_merged/anomaly/global

python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_mth_ids_merged `
  --loao-root data/pipeline_mth_ids_fine/anomaly/loao
```

## Início rápido (CAN)

```powershell
python -m mth_ids_pipeline.utils.merge_can
python -m mth_ids_pipeline.run_supervised --protocol can
python -m mth_ids_pipeline.run_anomaly --protocol can --loao
python -m mth_ids_pipeline.report_paper_tables --table all `
  --merged-dir data/pipeline_can_merged `
  --loao-root data/pipeline_can_fine/anomaly/loao `
  --results-dir results/can
```

Detalhes: [PROTOCOLO_CICIDS.md](PROTOCOLO_CICIDS.md) · [EXECUCAO.md](EXECUCAO.md) · CAN: [PROTOCOLO_CAN.md](PROTOCOLO_CAN.md).
