"""
Presets paper (artigo Yang et al. 2022) vs notebook (IoTJ).

Paper:
  - Supervisionado: merged, k-means frac=0.008, 80/20, 10-fold CV, SMOTE notebook, FCBF no treino, BO-GP α IG
  - Anomaly LOAO: fine, IG/FCBF/KPCA no conjunto combinado, BO-GP α/KPCA, tier 3–4
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mth_ids_pipeline.core.clustering import CL_KMEANS_METRICS
from mth_ids_pipeline.config import (
    DEFAULT_ANOMALY_SMOTE_TARGET,
    DEFAULT_BIASED_MODE,
    DEFAULT_CL_P_STAR,
    DEFAULT_META_LEARNER,
    PAPER_META_LEARNER,
    DEFAULT_TEST_SIZE,
    NOTEBOOK_SMOTE_TARGETS,
    PAPER_CV_FOLDS,
    PAPER_FCBF_K,
    PAPER_FCBF_SCOPE,
    PAPER_FEATURE_FIT_SCOPE,
    PAPER_HPO_N_CALLS,
    PAPER_HPO_ON_VALIDATION,
    PAPER_IG_CUMULATIVE,
    PAPER_KPCA_COMPONENTS,
    PAPER_KPCA_KERNEL,
    PAPER_SUPERVISED_SCALE,
    PAPER_ANOMALY_ZSCORE,
    NOTEBOOK_SUPERVISED_SCALE,
    NOTEBOOK_ANOMALY_ZSCORE,
)


class MthIdsProtocol(str, Enum):
    PAPER = "paper"
    NOTEBOOK = "notebook"


@dataclass(frozen=True)
class ProtocolSettings:
    name: str
    description: str
    supervised_profile: str
    anomaly_profile: str
    test_size: float
    cv_folds: int
    hpo_on_validation: bool
    smote_targets: dict[int, int]
    kmeans_frac: float
    skip_kmeans_sampling: bool
    anomaly_benign_target: int | None
    anomaly_smote_target: int | None
    biased_mode: str
    force_biased: bool
    optimize_p_star: bool
    cl_hpo_metric: str
    cl_hpo_n_calls: int
    optimize_ig: bool
    optimize_kpca: bool
    ig_cumulative: float
    fcbf_k: int
    fcbf_scope: str
    supervised_scale_mode: str
    feature_fit_scope: str
    anomaly_zscore_scope: str
    kpca_components: int
    kpca_kernel: str
    hpo_n_calls: int
    meta_learner: str = DEFAULT_META_LEARNER
    cl_p_star: float = DEFAULT_CL_P_STAR
    cl_kmeans_metrics: tuple[str, ...] = CL_KMEANS_METRICS


PAPER = ProtocolSettings(
    name="paper",
    description=(
        "Artigo MTH-IDS: sup k-means 0.8% + 80/20 + 10-fold CV + SMOTE notebook; "
        "HPO em validação; stacking meta best-base; anomaly LOAO fine + BO-GP tier 3–4."
    ),
    supervised_profile="merged",
    anomaly_profile="fine",
    test_size=DEFAULT_TEST_SIZE,
    cv_folds=PAPER_CV_FOLDS,
    hpo_on_validation=PAPER_HPO_ON_VALIDATION,
    smote_targets=dict(NOTEBOOK_SMOTE_TARGETS),
    kmeans_frac=0.008,
    skip_kmeans_sampling=False,
    anomaly_benign_target=None,
    anomaly_smote_target=DEFAULT_ANOMALY_SMOTE_TARGET,
    biased_mode="both",
    force_biased=True,
    optimize_p_star=True,
    cl_hpo_metric="f1",
    cl_hpo_n_calls=PAPER_HPO_N_CALLS,
    optimize_ig=True,
    optimize_kpca=True,
    ig_cumulative=PAPER_IG_CUMULATIVE,
    fcbf_k=PAPER_FCBF_K,
    fcbf_scope=PAPER_FCBF_SCOPE,
    supervised_scale_mode=PAPER_SUPERVISED_SCALE,
    feature_fit_scope=PAPER_FEATURE_FIT_SCOPE,
    anomaly_zscore_scope=PAPER_ANOMALY_ZSCORE,
    kpca_components=PAPER_KPCA_COMPONENTS,
    kpca_kernel=PAPER_KPCA_KERNEL,
    hpo_n_calls=PAPER_HPO_N_CALLS,
    meta_learner=PAPER_META_LEARNER,
)

NOTEBOOK = ProtocolSettings(
    name="notebook",
    description="Notebook IoTJ: merged, hold-out 20%, PortScan demo, parâmetros fixos.",
    supervised_profile="merged",
    anomaly_profile="merged",
    test_size=0.2,
    cv_folds=0,
    hpo_on_validation=False,
    smote_targets=dict(NOTEBOOK_SMOTE_TARGETS),
    kmeans_frac=0.008,
    skip_kmeans_sampling=False,
    anomaly_benign_target=1255,
    anomaly_smote_target=DEFAULT_ANOMALY_SMOTE_TARGET,
    biased_mode=DEFAULT_BIASED_MODE,
    force_biased=False,
    optimize_p_star=False,
    cl_hpo_metric="accuracy",
    cl_hpo_n_calls=20,
    optimize_ig=False,
    optimize_kpca=False,
    ig_cumulative=0.9,
    fcbf_k=20,
    fcbf_scope="full",
    supervised_scale_mode=NOTEBOOK_SUPERVISED_SCALE,
    feature_fit_scope="combined",
    anomaly_zscore_scope=NOTEBOOK_ANOMALY_ZSCORE,
    kpca_components=10,
    kpca_kernel="rbf",
    hpo_n_calls=20,
    meta_learner=DEFAULT_META_LEARNER,
    cl_kmeans_metrics=("euclidean", "manhattan", "cosine"),
)


def get_protocol_settings(name: str) -> ProtocolSettings:
    key = name.strip().lower()
    if key in ("paper", "article", "artigo"):
        return PAPER
    if key in ("notebook", "nb", "ioj"):
        return NOTEBOOK
    raise ValueError(f"Protocolo desconhecido: {name!r}. Use 'paper' ou 'notebook'.")
