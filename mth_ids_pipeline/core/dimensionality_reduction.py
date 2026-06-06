"""Redução de dimensionalidade: Kernel PCA (ramo anomaly)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA


def _prepare_kpca_input(X: np.ndarray, *, dtype: type = np.float32) -> tuple[np.ndarray, int, float]:
    X32 = np.ascontiguousarray(X, dtype=dtype)
    n = X32.shape[0]
    est_gb = (n * n * np.dtype(dtype).itemsize) / (1024**3)
    if est_gb > 3.0:
        print(
            f"KernelPCA: matriz kernel estimada ~{est_gb:.1f} GiB "
            f"({n} amostras). Feche outros programas se ocorrer MemoryError."
        )
    return X32, n, est_gb


def fit_kpca(
    X_train: np.ndarray,
    *,
    n_components: int = 10,
    kernel: str = "rbf",
    dtype: type = np.float32,
) -> KernelPCA:
    """Ajusta KernelPCA somente no treino."""
    X32, n, est_gb = _prepare_kpca_input(X_train, dtype=dtype)
    print(f"  [KPCA fit treino] shape={X32.shape}, n_components={n_components}")
    kpca = KernelPCA(n_components=n_components, kernel=kernel, copy_X=False)
    try:
        kpca.fit(X32)
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        raise MemoryError(
            f"KernelPCA esgotou memória (~{est_gb:.1f} GiB necessários para {n} amostras). "
            "Opções: fechar outros apps, usar máquina com mais RAM, ou reduzir amostras na fase 2."
        ) from e
    return kpca


def transform_kpca(
    X: np.ndarray,
    kpca: KernelPCA,
    *,
    dtype: type = np.float32,
    split: str = "teste",
) -> np.ndarray:
    """Projeta amostras com KernelPCA já ajustado no treino."""
    X32, _, _ = _prepare_kpca_input(X, dtype=dtype)
    out = kpca.transform(X32)
    print(f"  [KPCA transform {split}] shape={out.shape}")
    return out


def apply_kpca(
    X: np.ndarray,
    *,
    n_components: int = 10,
    kernel: str = "rbf",
    y: np.ndarray | None = None,
    dtype: type = np.float32,
) -> tuple[np.ndarray, KernelPCA]:
    """Ajusta e transforma no mesmo conjunto (legado). `y` é ignorado."""
    del y
    kpca = fit_kpca(X, n_components=n_components, kernel=kernel, dtype=dtype)
    return transform_kpca(X, kpca, dtype=dtype, split="fit+transform"), kpca
