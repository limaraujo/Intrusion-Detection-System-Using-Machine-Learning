"""Utilitários compartilhados do ramo anomaly (fases 7–11)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from mth_ids_pipeline.config import A04_AFTER_KPCA, A05_TRAIN_SMOTE, A06_TEST_SLICE_INFO

FITTED_SCALER = "fitted_scaler.joblib"
FITTED_IG_FEATURES = "fitted_ig_features.txt"
FITTED_FCBF = "fitted_fcbf.joblib"
FITTED_KPCA = "fitted_kpca.joblib"


def _json_safe(value: Any) -> Any:
    """Converte tipos numpy/pandas para tipos nativos serializáveis em JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def require_path(path: Path, *, hint: str) -> Path:
    """Valida existência de artefato e sugere o comando de correção."""
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}\n{hint}")
    return path


def build_anomaly_binary_split(
    df: pd.DataFrame,
    attack_label: int,
    *,
    label_col: str = "Label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conjunto sem o ataque zero-day (binário) e conjunto só com esse ataque (rótulo 1).

    Alinhado ao artigo: leave-one-attack-out — o ataque escolhido é o desconhecido.
    """
    df1 = df[df[label_col] != attack_label].copy()
    df1.loc[df1[label_col] > 0, label_col] = 1

    df2 = df[df[label_col] == attack_label].copy()
    df2.loc[:, label_col] = 1
    return df1, df2


def discover_attack_labels(df: pd.DataFrame, *, label_col: str = "Label") -> list[int]:
    """Rótulos de ataque (>0) presentes no dataframe."""
    return sorted(int(x) for x in df[label_col].unique() if int(x) > 0)


def benign_sample_size_for_zero_day(n_zero_day: int, available_benign: int) -> int:
    """
    Tabela IX (MTH-IDS): amostrar tantos BENIGN quanto fluxos do ataque zero-day.

    Se não houver benignos suficientes em df1, usa o máximo disponível.
    """
    if n_zero_day <= 0 or available_benign <= 0:
        return 0
    return min(int(n_zero_day), int(available_benign))


def label_value_counts_dict(series: pd.Series) -> dict[str, int]:
    """Contagens de rótulo serializáveis em JSON."""
    counts = series.value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def loao_original_label_report(
    df: pd.DataFrame,
    attack_label: int,
    *,
    label_col: str = "Label",
) -> dict[str, Any]:
    """Contagens por rótulo original antes do colapso binário (fase 7)."""
    train_orig = df[df[label_col] != attack_label]
    zero_day = df[df[label_col] == attack_label]
    attack_labels_in_train = sorted(int(x) for x in train_orig[label_col].unique() if int(x) > 0)
    return {
        "zero_day_label": int(attack_label),
        "zero_day_samples": int(len(zero_day)),
        "train_original_label_counts": label_value_counts_dict(train_orig[label_col]),
        "train_attack_labels_present": attack_labels_in_train,
        "zero_day_fully_excluded_from_train": int(attack_label) not in {
            int(x) for x in train_orig[label_col].unique()
        },
    }


def log_loao_partition(
    *,
    stage: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: dict[str, Any],
    label_col: str = "Label",
) -> None:
    """Imprime rótulos binários e tamanhos do particionamento LOAO."""
    train_counts = label_value_counts_dict(train_df[label_col])
    test_counts = label_value_counts_dict(test_df[label_col])
    print(f"\n=== LOAO [{stage}] ===")
    print(f"  treino: rows={len(train_df)} labels={train_counts}")
    print(f"  teste:  rows={len(test_df)} labels={test_counts}")
    print(
        f"  zero-day={meta.get('zero_day_samples')} benign_test={meta.get('benign_sampled')} "
        f"regra={meta.get('benign_pairing_rule')}"
    )
    if meta.get("train_original_label_counts"):
        print(f"  rótulos originais no treino (pré-binário): {meta['train_original_label_counts']}")
    if meta.get("zero_day_label") is not None:
        print(
            f"  zero-day label={meta['zero_day_label']} "
            f"excluído_do_treino={meta.get('zero_day_fully_excluded_from_train')}"
        )


def validate_loao_partition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: dict[str, Any],
    *,
    label_col: str = "Label",
) -> None:
    """Valida invariantes do protocolo LOAO / Tabela IX."""
    n_zero_day = int(meta["zero_day_samples"])
    sample_n = int(meta["benign_sampled"])
    if int(len(test_df)) != n_zero_day + sample_n:
        raise ValueError(
            f"Teste deve ter {n_zero_day + sample_n} linhas, obteve {len(test_df)}"
        )
    test_counts = test_df[label_col].value_counts()
    if int(test_counts.get(1, 0)) != n_zero_day:
        raise ValueError("Teste deve conter exatamente todas as amostras zero-day como classe 1")
    if int(test_counts.get(0, 0)) != sample_n:
        raise ValueError("Teste deve conter exatamente sample_n benignos como classe 0")

    if meta.get("zero_day_fully_excluded_from_train") is False:
        raise ValueError(
            f"Zero-day label={meta.get('zero_day_label')} não foi excluído do treino"
        )
    zero_day_label = meta.get("zero_day_label")
    train_orig = meta.get("train_original_label_counts") or {}
    if zero_day_label is not None and str(zero_day_label) in train_orig:
        raise ValueError(
            f"Treino contém {train_orig[str(zero_day_label)]} amostras do zero-day "
            f"(label={zero_day_label})"
        )

    available_benign = meta.get("benign_available_in_train")
    if meta.get("benign_pairing_rule") == "paper_table_ix_1_to_1":
        if available_benign is not None and int(available_benign) >= n_zero_day:
            if sample_n != n_zero_day:
                raise ValueError(
                    f"Emparelhamento 1:1 (Tabela IX): esperado {n_zero_day} benignos no teste, "
                    f"obteve {sample_n} (disponíveis={available_benign})"
                )
        elif sample_n < n_zero_day:
            print(
                f"Aviso LOAO: emparelhamento 1:1 limitado por benignos disponíveis "
                f"({sample_n} < {n_zero_day})"
            )


def numeric_feature_columns(df: pd.DataFrame, *, label_col: str = "Label") -> list[str]:
    return [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]


def build_loao_train_test_split(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    label_col: str = "Label",
    benign_target: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Particiona LOAO em treino (df1) e teste (zero-day + benignos 1:1) antes de IG/FCBF/KPCA.
    """
    train_df = df1.copy()
    df2_attack = df2.copy()
    n_zero_day = int(len(df2_attack))

    benign_pool = train_df[train_df[label_col] == 0]
    available_benign = int(len(benign_pool))

    if benign_target is not None:
        sample_n = int(benign_target)
        pairing_rule = "benign_target_override"
    else:
        sample_n = benign_sample_size_for_zero_day(n_zero_day, available_benign)
        pairing_rule = "paper_table_ix_1_to_1"

    sample_n = max(0, min(sample_n, available_benign))
    benign_test_positions: list[int] = []

    if sample_n > 0:
        sampled = benign_pool.sample(n=sample_n, replace=False, random_state=random_state)
        benign_test_positions = [int(i) for i in sampled.index.tolist()]
        test_df = pd.concat([df2_attack, sampled.reset_index(drop=True)], ignore_index=True)
    else:
        test_df = df2_attack.reset_index(drop=True)

    n_train = int(len(train_df))
    n_test = int(len(test_df))
    meta: dict[str, Any] = {
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "n_df1_rows": n_train,
        "train_row_start": 0,
        "train_row_end": n_train,
        "test_row_start": n_train,
        "test_row_end": n_train + n_test,
        "zero_day_samples": n_zero_day,
        "benign_sampled": sample_n,
        "benign_available_in_train": available_benign,
        "benign_pairing_rule": pairing_rule,
        "benign_test_indices_in_df1": benign_test_positions,
        "benign_overlap_train_test": sample_n,
        "train_binary_label_counts": label_value_counts_dict(train_df[label_col]),
        "test_binary_label_counts": label_value_counts_dict(test_df[label_col]),
        "random_state": random_state,
        "protocol": "loao_train_test_then_fit_transform",
    }
    validate_loao_partition(train_df, test_df, meta, label_col=label_col)
    return train_df, test_df, meta


def save_anomaly_fitted_artifacts(
    work_dir: Path,
    *,
    scaler: StandardScaler,
    ig_features: list[str],
    fcbf,
    kpca: KernelPCA,
    partition_meta: dict[str, Any],
) -> dict[str, str]:
    """Persiste scaler, IG, FCBF, KPCA e metadados de partição."""
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "scaler": str(work_dir / FITTED_SCALER),
        "ig_features": str(work_dir / FITTED_IG_FEATURES),
        "fcbf": str(work_dir / FITTED_FCBF),
        "kpca": str(work_dir / FITTED_KPCA),
        "partition": str(work_dir / A06_TEST_SLICE_INFO),
    }
    joblib.dump(scaler, paths["scaler"])
    Path(paths["ig_features"]).write_text("\n".join(ig_features), encoding="utf-8")
    joblib.dump(fcbf, paths["fcbf"])
    joblib.dump(kpca, paths["kpca"])

    meta = {**partition_meta, "artifact_paths": paths}
    Path(paths["partition"]).write_text(
        json.dumps(_json_safe(meta), indent=2),
        encoding="utf-8",
    )
    return paths


def load_anomaly_fitted_artifacts(work_dir: Path) -> dict[str, Any]:
    """Carrega artefatos da fase 8."""
    meta_path = work_dir / A06_TEST_SLICE_INFO
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifact_paths = meta.get("artifact_paths", {})

    def _resolve(key: str, filename: str) -> Path:
        if key in artifact_paths:
            return Path(artifact_paths[key])
        return work_dir / filename

    return {
        "meta": meta,
        "scaler": joblib.load(_resolve("scaler", FITTED_SCALER)),
        "ig_features": _resolve("ig_features", FITTED_IG_FEATURES).read_text(encoding="utf-8").splitlines(),
        "fcbf": joblib.load(_resolve("fcbf", FITTED_FCBF)),
        "kpca": joblib.load(_resolve("kpca", FITTED_KPCA)),
    }


def resolve_notebook_anomaly_smote_target(
    y_train: np.ndarray,
    smote_target: int | None,
) -> int | None:
    """
    Notebook MTH_IDS_IoTJ.ipynb (anomaly):
      ``SMOTE(n_jobs=-1, sampling_strategy={1: 18225})``
    com 18225 = contagem de BENIGN (label 0) no treino df1 antes do SMOTE.
    """
    counts = pd.Series(y_train).value_counts()
    if 0 not in counts.index or 1 not in counts.index:
        return None
    attack_n = int(counts[1])
    benign_n = int(counts[0])
    target = benign_n if smote_target is None else int(smote_target)
    if attack_n >= target:
        return None
    return target


def apply_notebook_anomaly_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    smote_target: int | None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, bool, int | None]:
    """SMOTE binário no treino anomaly — alinha ao notebook (classe 1 → nº benignos)."""
    target = resolve_notebook_anomaly_smote_target(y_train, smote_target)
    if target is None:
        return X_train, y_train, False, None
    kw: dict = {"sampling_strategy": {1: target}}
    smote_params = inspect.signature(SMOTE.__init__).parameters
    if "n_jobs" in smote_params:
        kw["n_jobs"] = -1
    if "random_state" in smote_params:
        kw["random_state"] = random_state
    smote = SMOTE(**kw)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res, True, target


def load_anomaly_splits(
    input_dir: Path,
    *,
    smote_target: int | None,
    random_state: int,
    label_col: str = "Label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Carrega KPCA + meta de slice; aplica SMOTE no treino (notebook IoTJ)."""
    df = pd.read_parquet(input_dir / A04_AFTER_KPCA)
    meta = json.loads((input_dir / A06_TEST_SLICE_INFO).read_text(encoding="utf-8"))
    n_train = int(meta.get("n_train_rows", meta["n_df1_rows"]))

    X_all = df.drop(columns=[label_col]).values
    y_all = np.ravel(df[label_col].values)
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_test = X_all[n_train:]
    y_test = y_all[n_train:]

    train_counts = label_value_counts_dict(pd.Series(y_train))
    test_counts = label_value_counts_dict(pd.Series(y_test))
    print(
        f"Partição KPCA: treino={X_train.shape} labels={train_counts} | "
        f"teste={X_test.shape} labels={test_counts}"
    )

    train_path = input_dir / A05_TRAIN_SMOTE
    if train_path.exists():
        tr = pd.read_parquet(train_path)
        X_train = tr.drop(columns=[label_col]).values
        y_train = np.ravel(tr[label_col].values)
        return X_train, X_test, y_train, y_test, True

    X_train, y_train, did_smote, resolved = apply_notebook_anomaly_smote(
        X_train,
        y_train,
        smote_target=smote_target,
        random_state=random_state,
    )
    if did_smote and resolved is not None:
        print(f"SMOTE notebook: classe 1 -> {resolved} (alvo = n benignos no treino)")
    return X_train, X_test, y_train, y_test, did_smote
