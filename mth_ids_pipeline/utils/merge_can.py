"""
Concatena arquivos CAN-intrusion (.txt) → ``data/CAN_Intrusion_Dataset.csv``.

Features (artigo): CAN_ID + DATA[0]…DATA[7] (bytes do payload); sem timestamp.
Rótulos: BENIGN (attack-free) + DoS / Fuzzy / Impersonation.

Uso:
  python -m mth_ids_pipeline.utils.merge_can
  python -m mth_ids_pipeline.utils.merge_can --input-dir data/CAN_DATA --output data/CAN_Intrusion_Dataset.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from mth_ids_pipeline.config import DATA_DIR, DEFAULT_RAW_CSV_CAN

_LINE_RE = re.compile(
    r"Timestamp:\s+([\d.]+)\s+ID:\s+(\w+)\s+(\d+)\s+DLC:\s+(\d+)\s+(.*)"
)

_CAN_LABEL_BY_STEM: dict[str, str] = {
    "CAN_Attack_free_dataset": "BENIGN",
}


def _attack_label_from_stem(stem: str) -> str:
    if stem in _CAN_LABEL_BY_STEM:
        return _CAN_LABEL_BY_STEM[stem]
    # CAN_DoS_attack_dataset → DoS; CAN_Fuzzy_attack_dataset → Fuzzy; …
    return stem.split("_")[1]


def _parse_payload_bytes(payload: str, dlc: int) -> list[int]:
    parts = payload.strip().split()
    out: list[int] = []
    for token in parts[: max(0, dlc)]:
        out.append(int(token, 16))
    while len(out) < 8:
        out.append(0)
    return out[:8]


def _parse_can_file(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as infile:
        for line in infile:
            match = _LINE_RE.match(line.strip())
            if not match:
                continue
            _ts, can_id, _flag, dlc_s, payload = match.groups()
            dlc = int(dlc_s)
            data = _parse_payload_bytes(payload, dlc)
            row: dict = {
                "CAN_ID": int(can_id, 16),
                "Label": label,
            }
            for i, val in enumerate(data):
                row[f"DATA_{i}"] = val
            rows.append(row)
    return rows


def build_can_dataset(
    input_dir: Path,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    txt_files = sorted(input_dir.glob("CAN_*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"Nenhum CAN_*.txt em {input_dir}. "
            "Coloque os arquivos do CAN-intrusion-dataset em data/CAN_DATA/."
        )

    frames: list[pd.DataFrame] = []
    for path in txt_files:
        stem = path.stem
        label = _attack_label_from_stem(stem)
        rows = _parse_can_file(path, label)
        if not rows:
            print(f"Aviso: nenhuma linha parseada em {path.name}")
            continue
        frames.append(pd.DataFrame(rows))
        print(f"{path.name}: {len(rows):,} linhas -> Label={label}")

    if not frames:
        raise ValueError(f"Nenhum registro CAN válido em {input_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera CSV CAN_Intrusion_Dataset (features + Label)")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DATA_DIR / "CAN_DATA",
        help="Pasta com CAN_*.txt (default: data/CAN_DATA)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RAW_CSV_CAN,
        help="CSV de saída (default: data/CAN_Intrusion_Dataset.csv)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = build_can_dataset(args.input_dir, random_state=args.random_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nShape: {df.shape}")
    print(df["Label"].value_counts())
    print(f"Salvo em: {args.output}")


if __name__ == "__main__":
    main()
