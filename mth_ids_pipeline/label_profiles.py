"""
Perfis de rótulos CICIDS2017 para o pipeline MTH-IDS.

- merged: famílias agregadas (Tabela VII) — ~7 classes
- fine: rótulos originais do dataset — ~14 ataques para LOAO (Tabela IX)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import DATA_DIR, DEFAULT_MINORITY_LABELS

# Separador corrompido quando UTF-8 (U+FFFD) é lido com encoding="latin1"
_WEB_ATTACK_SEP_MOJIBAKE = "\xef\xbf\xbd"

# Mapeamento notebook / artigo (subtipos → família)
CICIDS_LABEL_MERGE: dict[str, str] = {
    "DoS Hulk": "DoS",
    "DDoS": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \ufffd Brute Force": "WebAttack",
    "Web Attack \ufffd XSS": "WebAttack",
    "Web Attack \ufffd Sql Injection": "WebAttack",
    f"Web Attack {_WEB_ATTACK_SEP_MOJIBAKE} Brute Force": "WebAttack",
    f"Web Attack {_WEB_ATTACK_SEP_MOJIBAKE} XSS": "WebAttack",
    f"Web Attack {_WEB_ATTACK_SEP_MOJIBAKE} Sql Injection": "WebAttack",
    "Web Attack – Brute Force": "WebAttack",
    "Web Attack – XSS": "WebAttack",
    "Web Attack – Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
}


def normalize_cicids_label_text(labels) -> "pd.Series":
    """
    Normaliza strings de Label do CICIDS2017 antes do agrupamento merged.

    O CSV oficial usa um separador não-ASCII entre 'Web Attack' e o subtipo.
    Com ``encoding='latin1'`` (notebook / merge_cicids), U+FFFD vira três
    caracteres ``\\xef\\xbf\\xbd``, que não casam com ``\\ufffd`` no dict.
    """
    import pandas as pd

    s = labels.astype(str).str.strip()
    s = s.str.replace(_WEB_ATTACK_SEP_MOJIBAKE, "\ufffd", regex=False)
    return s


class LabelProfileKind(str, Enum):
    MERGED = "merged"
    FINE = "fine"


@dataclass(frozen=True)
class LabelProfile:
    """Caminhos e defaults por perfil de rótulo."""

    kind: LabelProfileKind
    raw_csv: Path
    intermediate_dir: Path
    minority_labels: tuple[int, ...] | None
    description: str

    @property
    def auto_minority(self) -> bool:
        return self.minority_labels is None

    def minority_labels_csv(self) -> str | None:
        if self.minority_labels is None:
            return None
        return ",".join(str(x) for x in self.minority_labels)


def apply_cicids_label_merge(df, *, inplace: bool = False):
    """Agrupa subtipos de ataque em famílias (perfil merged)."""
    out = df if inplace else df.copy()
    labels = normalize_cicids_label_text(out["Label"])
    labels = labels.replace(CICIDS_LABEL_MERGE)
    # Qualquer variante restante de Web Attack → WebAttack (notebook)
    web = labels.str.startswith("Web Attack", na=False)
    labels = labels.where(~web, "WebAttack")
    out["Label"] = labels
    return out


def get_label_profile(name: str) -> LabelProfile:
    key = name.strip().lower()
    if key in (LabelProfileKind.MERGED.value, "merge", "7"):
        return MERGED_PROFILE
    if key in (LabelProfileKind.FINE.value, "fine-grained", "finegrained", "14"):
        return FINE_PROFILE
    raise ValueError(
        f"Perfil desconhecido: {name!r}. Use 'merged' ou 'fine'."
    )


MERGED_PROFILE = LabelProfile(
    kind=LabelProfileKind.MERGED,
    raw_csv=DATA_DIR / "CICIDS2017.csv",
    intermediate_dir=DATA_DIR / "pipeline_mth_ids_merged",
    minority_labels=DEFAULT_MINORITY_LABELS,
    description="Famílias agregadas (Tabela VII): BENIGN + 6 ataques; LOAO com 6 rodadas.",
)

FINE_PROFILE = LabelProfile(
    kind=LabelProfileKind.FINE,
    raw_csv=DATA_DIR / "CICIDS2017_fine.csv",
    intermediate_dir=DATA_DIR / "pipeline_mth_ids_fine",
    minority_labels=None,
    description="Rótulos originais CICIDS2017; LOAO ~14 ataques (Tabela IX — padrão anomaly).",
)

ALL_PROFILES: dict[str, LabelProfile] = {
    LabelProfileKind.MERGED.value: MERGED_PROFILE,
    LabelProfileKind.FINE.value: FINE_PROFILE,
}


def minority_labels_all_attacks(
    df,
    *,
    label_col: str = "Label",
    benign_label: int = 0,
) -> tuple[int, ...]:
    """
    Rótulos inteiros de ataque preservados no k-means sampling (fase 2).

    Assume BENIGN → 0 após LabelEncoder (ordem alfabética no CICIDS2017).
    """
    from .preprocessing import encode_labels

    encoded, _ = encode_labels(df, label_col=label_col)
    return tuple(
        sorted(int(x) for x in encoded[label_col].unique() if int(x) != benign_label)
    )
