"""
Fase 6: modelos supervisionados (XGBoost, RF, DT, ET), HPO opcional (hyperopt),
stacking com meta XGBoost.

Entradas: CSVs da fase 5.

Uso rápido (sem HPO, usa hiperparâmetros fixos do notebook):
  python -m mth_ids_pipeline.phase06_supervised_models --no-hpo --no-plots
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.tree import DecisionTreeClassifier

from .config import INTERMEDIATE_DIR, P05_TEST, P05_TRAIN_SMOTE, ensure_intermediate_dirs


def _evaluate(name: str, clf, X_test, y_test) -> dict:
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    print(f"\n=== {name} ===\nAccuracy: {acc}\nPrecision: {p}\nRecall: {r}\nF1: {f}")
    print(classification_report(y_test, y_pred))
    return {"model": name, "accuracy": acc, "precision": float(p), "recall": float(r), "f1_weighted": float(f)}


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Fase 6 — treino supervisionado + stacking")
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--no-hpo", action="store_true", help="Usa hiperparâmetros fixos do notebook")
    parser.add_argument("--no-plots", action="store_true", help="Não exibe heatmaps")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Opcional: salvar métricas em JSON")
    args = parser.parse_args()
    ensure_intermediate_dirs()

    tr = pd.read_csv(args.train or (args.output_dir / P05_TRAIN_SMOTE))
    te = pd.read_csv(args.test or (args.output_dir / P05_TEST))
    label_col = "Label"
    X_train = tr.drop(columns=[label_col]).values
    y_train = tr[label_col].values
    X_test = te.drop(columns=[label_col]).values
    y_test = te[label_col].values

    metrics: list[dict] = []

    if not args.no_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns

        def heatmap(y_true, y_pred, title: str) -> None:
            cm = confusion_matrix(y_true, y_pred)
            _, ax = plt.subplots(figsize=(5, 5))
            sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)
            ax.set_xlabel("y_pred")
            ax.set_ylabel("y_true")
            ax.set_title(title)
            plt.show()
    else:

        def heatmap(y_true, y_pred, title: str) -> None:  # noqa: ARG001
            pass

    # ----- XGBoost -----
    if args.no_hpo:
        xg = xgb.XGBClassifier(learning_rate=0.7340229699980686, n_estimators=70, max_depth=14, random_state=0)
    else:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

        def xg_objective(params):
            params = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "learning_rate": abs(float(params["learning_rate"])),
            }
            clf = xgb.XGBClassifier(**params, random_state=0)
            clf.fit(X_train, y_train)
            score = accuracy_score(y_test, clf.predict(X_test))
            return {"loss": -score, "status": STATUS_OK}

        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
            "max_depth": hp.quniform("max_depth", 4, 100, 1),
            "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
        }
        Trials()
        best = fmin(fn=xg_objective, space=space, algo=tpe.suggest, max_evals=20)
        print("XGBoost HPO best:", best)
        xg = xgb.XGBClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            learning_rate=float(best["learning_rate"]),
            random_state=0,
        )

    xg.fit(X_train, y_train)
    metrics.append(_evaluate("XGBoost (base)", xg, X_test, y_test))
    heatmap(y_test, xg.predict(X_test), "XGBoost")
    xg_train_p = xg.predict(X_train).reshape(-1, 1)
    xg_test_p = xg.predict(X_test).reshape(-1, 1)

    # ----- Random Forest -----
    if args.no_hpo:
        rf_hpo = RandomForestClassifier(
            n_estimators=71,
            min_samples_leaf=1,
            max_depth=46,
            min_samples_split=9,
            max_features=20,
            criterion="entropy",
            random_state=0,
        )
    else:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

        def rf_objective(params):
            p2 = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "max_features": int(params["max_features"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "criterion": ["gini", "entropy"][int(params["criterion"])],
            }
            clf = RandomForestClassifier(**p2, random_state=0)
            clf.fit(X_train, y_train)
            return {"loss": -clf.score(X_test, y_test), "status": STATUS_OK}

        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, 20, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }
        Trials()
        best = fmin(fn=rf_objective, space=space, algo=tpe.suggest, max_evals=20)
        print("RF HPO best:", best)
        crit = ["gini", "entropy"][int(best["criterion"])]
        rf_hpo = RandomForestClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            max_features=int(best["max_features"]),
            min_samples_split=int(best["min_samples_split"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            criterion=crit,
            random_state=0,
        )

    rf_hpo.fit(X_train, y_train)
    metrics.append(_evaluate("RandomForest (HPO)", rf_hpo, X_test, y_test))
    heatmap(y_test, rf_hpo.predict(X_test), "RF")
    rf_train_p = rf_hpo.predict(X_train).reshape(-1, 1)
    rf_test_p = rf_hpo.predict(X_test).reshape(-1, 1)

    # ----- Decision Tree -----
    if args.no_hpo:
        dt_hpo = DecisionTreeClassifier(
            min_samples_leaf=2, max_depth=47, min_samples_split=3, max_features=19, criterion="gini", random_state=0
        )
    else:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

        def dt_objective(params):
            p2 = {
                "max_depth": int(params["max_depth"]),
                "max_features": int(params["max_features"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "criterion": ["gini", "entropy"][int(params["criterion"])],
            }
            clf = DecisionTreeClassifier(**p2, random_state=0)
            clf.fit(X_train, y_train)
            return {"loss": -clf.score(X_test, y_test), "status": STATUS_OK}

        space = {
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, 20, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }
        Trials()
        best = fmin(fn=dt_objective, space=space, algo=tpe.suggest, max_evals=50)
        print("DT HPO best:", best)
        crit = ["gini", "entropy"][int(best["criterion"])]
        dt_hpo = DecisionTreeClassifier(
            max_depth=int(best["max_depth"]),
            max_features=int(best["max_features"]),
            min_samples_split=int(best["min_samples_split"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            criterion=crit,
            random_state=0,
        )

    dt_hpo.fit(X_train, y_train)
    metrics.append(_evaluate("DecisionTree (HPO)", dt_hpo, X_test, y_test))
    heatmap(y_test, dt_hpo.predict(X_test), "DT")
    dt_train_p = dt_hpo.predict(X_train).reshape(-1, 1)
    dt_test_p = dt_hpo.predict(X_test).reshape(-1, 1)

    # ----- Extra Trees -----
    if args.no_hpo:
        et_hpo = ExtraTreesClassifier(
            n_estimators=53,
            min_samples_leaf=1,
            max_depth=31,
            min_samples_split=5,
            max_features=20,
            criterion="entropy",
            random_state=0,
        )
    else:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

        def et_objective(params):
            p2 = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "max_features": int(params["max_features"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"]),
                "criterion": ["gini", "entropy"][int(params["criterion"])],
            }
            clf = ExtraTreesClassifier(**p2, random_state=0)
            clf.fit(X_train, y_train)
            return {"loss": -clf.score(X_test, y_test), "status": STATUS_OK}

        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
            "max_depth": hp.quniform("max_depth", 5, 50, 1),
            "max_features": hp.quniform("max_features", 1, 20, 1),
            "min_samples_split": hp.quniform("min_samples_split", 2, 11, 1),
            "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 11, 1),
            "criterion": hp.choice("criterion", ["gini", "entropy"]),
        }
        Trials()
        best = fmin(fn=et_objective, space=space, algo=tpe.suggest, max_evals=20)
        print("ET HPO best:", best)
        crit = ["gini", "entropy"][int(best["criterion"])]
        et_hpo = ExtraTreesClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            max_features=int(best["max_features"]),
            min_samples_split=int(best["min_samples_split"]),
            min_samples_leaf=int(best["min_samples_leaf"]),
            criterion=crit,
            random_state=0,
        )

    et_hpo.fit(X_train, y_train)
    metrics.append(_evaluate("ExtraTrees (HPO)", et_hpo, X_test, y_test))
    heatmap(y_test, et_hpo.predict(X_test), "ET")
    et_train_p = et_hpo.predict(X_train).reshape(-1, 1)
    et_test_p = et_hpo.predict(X_test).reshape(-1, 1)

    # ----- Stacking (meta features = predições dos 4 modelos) -----
    x_train_meta = np.concatenate((dt_train_p, et_train_p, rf_train_p, xg_train_p), axis=1)
    x_test_meta = np.concatenate((dt_test_p, et_test_p, rf_test_p, xg_test_p), axis=1)

    stk = xgb.XGBClassifier(random_state=0)
    stk.fit(x_train_meta, y_train)
    metrics.append(_evaluate("Stacking (XGB meta)", stk, x_test_meta, y_test))
    heatmap(y_test, stk.predict(x_test_meta), "Stacking")

    if args.no_hpo:
        meta = xgb.XGBClassifier(learning_rate=0.19229249758051492, n_estimators=30, max_depth=36, random_state=0)
    else:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

        def stk_objective(params):
            p2 = {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "learning_rate": abs(float(params["learning_rate"])),
            }
            clf = xgb.XGBClassifier(**p2, random_state=0)
            clf.fit(x_train_meta, y_train)
            return {"loss": -accuracy_score(y_test, clf.predict(x_test_meta)), "status": STATUS_OK}

        space = {
            "n_estimators": hp.quniform("n_estimators", 10, 100, 5),
            "max_depth": hp.quniform("max_depth", 4, 100, 1),
            "learning_rate": hp.normal("learning_rate", 0.01, 0.9),
        }
        Trials()
        best = fmin(fn=stk_objective, space=space, algo=tpe.suggest, max_evals=20)
        print("Stacking meta HPO best:", best)
        meta = xgb.XGBClassifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            learning_rate=float(best["learning_rate"]),
            random_state=0,
        )

    meta.fit(x_train_meta, y_train)
    metrics.append(_evaluate("Stacking meta (HPO XGB)", meta, x_test_meta, y_test))
    heatmap(y_test, meta.predict(x_test_meta), "Stacking HPO")

    out_metrics = args.metrics_json or (args.output_dir / "06_supervised_metrics.json")
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nMétricas salvas em: {out_metrics}")


if __name__ == "__main__":
    main()
