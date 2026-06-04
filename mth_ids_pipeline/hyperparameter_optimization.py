"""Otimização de hiperparâmetros: BO-TPE (Hyperopt) e BO-GP (skopt)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score


def _criterion_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ["gini", "entropy"][int(value)]


def optimize_xgb_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_evals: int = 20,
    use_validation_cv: bool = False,
    cv_folds: int = 10,
    random_state: int = 0,
) -> dict[str, Any]:
    """BO-TPE para XGBoost — test set (notebook) ou CV no treino (artigo)."""
    from hyperopt import STATUS_OK, fmin, hp, tpe
    import xgboost as xgb

    from .validation import hpo_objective_on_validation

    def objective(params):
        p = {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": abs(float(params["learning_rate"])),
        }
        if use_validation_cv:

            def build_estimator(pp: dict[str, Any]):
                return xgb.XGBClassifier(**pp, random_state=random_state)

            score = hpo_objective_on_validation(
                build_estimator,
                p,
                X_train,
                y_train,
                n_splits=cv_folds,
                random_state=random_state,
            )
        else:
            clf = xgb.XGBClassifier(**p, random_state=random_state)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_test, clf.predict(X_test))
        return {"loss": -score, "status": STATUS_OK}

    space = {
        "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
        "max_depth": hp.quniform("max_depth", 4, 100, 1),
        "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
    }
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    return {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": abs(float(best["learning_rate"])),
    }


def optimize_sklearn_tree_hyperparams(
    model_factory: Callable[..., Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_evals: int = 20,
    include_n_estimators: bool = True,
    use_validation_cv: bool = False,
    cv_folds: int = 10,
    random_state: int = 0,
) -> dict[str, Any]:
    """BO-TPE para RF/ET/DT — test set (notebook) ou CV no treino (artigo)."""
    from hyperopt import STATUS_OK, fmin, hp, tpe

    from .validation import hpo_objective_on_validation

    def objective(params):
        p2: dict[str, Any] = {
            "max_depth": int(params["max_depth"]),
            "max_features": int(params["max_features"]),
            "min_samples_split": int(params["min_samples_split"]),
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "criterion": _criterion_value(params["criterion"]),
        }
        if include_n_estimators:
            p2["n_estimators"] = int(params["n_estimators"])
        if use_validation_cv:

            def build_estimator(pp: dict[str, Any]):
                kw = dict(pp)
                if include_n_estimators:
                    return model_factory(**kw, random_state=random_state)
                return model_factory(**kw, random_state=random_state)

            score = hpo_objective_on_validation(
                build_estimator,
                p2,
                X_train,
                y_train,
                n_splits=cv_folds,
                random_state=random_state,
            )
            return {"loss": -score, "status": STATUS_OK}
        clf = model_factory(**p2, random_state=random_state)
        clf.fit(X_train, y_train)
        return {"loss": -clf.score(X_test, y_test), "status": STATUS_OK}

    space: dict = {
        "max_depth": hp.quniform("max_depth", 5, 50, 1),
        "max_features": hp.quniform("max_features", 1, 20, 1),
        "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
    }
    if include_n_estimators:
        space["n_estimators"] = hp.quniform("n_estimators", 10, 200, 1)

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    out: dict[str, Any] = {
        "max_depth": int(best["max_depth"]),
        "max_features": int(best["max_features"]),
        "min_samples_split": int(best["min_samples_split"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "criterion": _criterion_value(best["criterion"]),
    }
    if include_n_estimators:
        out["n_estimators"] = int(best["n_estimators"])
    return out


def optimize_cl_kmeans_clusters(
    objective_fn: Callable[..., float],
    *,
    n_calls: int = 20,
    low: int = 2,
    high: int = 50,
    random_state: int = 0,
    optimize_metric: bool = False,
) -> tuple[int, float, str]:
    """BO-GP (skopt) para n_clusters e opcionalmente métrica de distância (artigo)."""
    from skopt import gp_minimize
    from skopt.space import Categorical, Integer

    if optimize_metric:
        space = [Integer(low, high, name="n_clusters"), Categorical(["euclidean", "manhattan"], name="metric")]

        def wrapped(params):
            n, metric = int(params[0]), str(params[1])
            return 1.0 - objective_fn(n, metric)

        result = gp_minimize(wrapped, space, n_calls=n_calls, random_state=random_state)
        return int(result.x[0]), 1.0 - float(result.fun), str(result.x[1])

    space = [Integer(low, high, name="n_clusters")]

    def wrapped(params):
        n = int(params[0])
        return 1.0 - objective_fn(n)

    result = gp_minimize(wrapped, space, n_calls=n_calls, random_state=random_state)
    best_n = int(result.x[0])
    best_acc = 1.0 - float(result.fun)
    return best_n, best_acc, "euclidean"


def optimize_cl_kmeans_tpe(
    objective_fn: Callable[[int], float],
    *,
    max_evals: int = 20,
    low: int = 2,
    high: int = 50,
) -> tuple[int, float]:
    """BO-TPE alternativo para n_clusters (notebook também testa)."""
    from hyperopt import STATUS_OK, fmin, hp, tpe

    best_acc = -1.0
    best_n = low

    def objective(params):
        nonlocal best_acc, best_n
        n = int(params["n_clusters"])
        acc = objective_fn(n)
        if acc > best_acc:
            best_acc = acc
            best_n = n
        return {"loss": -acc, "status": STATUS_OK}

    space = {"n_clusters": hp.quniform("n_clusters", low, high, 1)}
    fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    return best_n, best_acc
