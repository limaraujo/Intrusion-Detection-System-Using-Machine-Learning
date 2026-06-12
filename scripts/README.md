# Scripts auxiliares

Scripts **fora** do pacote `mth_ids_pipeline`. Não fazem parte do pipeline modular nem dos testes automatizados.

| Script | Descrição |
|--------|-----------|
| `can_mth_ids_baseline.py` | Replicação simplificada MTH-IDS (Tiers 1–2) no CAN-intrusion-dataset — protótipo/experimento local |

Executar a partir da raiz do repositório:

```powershell
python scripts/can_mth_ids_baseline.py
```

Pré-requisito: `data/CAN_intrusion_Dataset.csv` (ver `data/README.md` e [docs/PROTOCOLO_CAN_INTRUSION.md](../docs/PROTOCOLO_CAN_INTRUSION.md)).
