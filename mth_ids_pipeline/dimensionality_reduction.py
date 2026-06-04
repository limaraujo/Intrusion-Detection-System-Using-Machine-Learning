"""Redução de dimensionalidade: Kernel PCA (ramo anomaly)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import KernelPCA


def apply_kpca(
    X: np.ndarray,
    *,
    n_components: int = 10,
    kernel: str = "rbf",
    y: np.ndarray | None = None,
    dtype: type = np.float32,
) -> tuple[np.ndarray, KernelPCA]:
    """
    Aplica KernelPCA conforme notebook (y é passado mas ignorado pela API sklearn).

    Usa float32 para reduzir RAM (~n²×4 bytes). Para ~28k amostras, reserve ≥4 GiB livres.
    """
    X32 = np.ascontiguousarray(X, dtype=dtype)
    n = X32.shape[0]
    est_gb = (n * n * np.dtype(dtype).itemsize) / (1024**3)
    if est_gb > 3.0:
        print(
            f"KernelPCA: matriz kernel estimada ~{est_gb:.1f} GiB "
            f"({n} amostras). Feche outros programas se ocorrer MemoryError."
        )

    kpca = KernelPCA(n_components=n_components, kernel=kernel, copy_X=False)
    try:
        if y is not None:
            kpca.fit(X32, y)
        else:
            kpca.fit(X32)
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        raise MemoryError(
            f"KernelPCA esgotou memória (~{est_gb:.1f} GiB necessários para {n} amostras). "
            "Opções: fechar outros apps, usar máquina com mais RAM, ou reduzir amostras na fase 2."
        ) from e

    return kpca.transform(X32), kpca
