"""
Replicação do MTH-IDS (Yang et al., 2021) para o CAN-intrusion-dataset.
Tier 1: DT, RF, ET, XGBoost  
Tier 2: Stacking + BO-TPE
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

DATA_CSV = Path("data/CAN_intrusion_Dataset.csv")

# ------------------------------------------------------------
# 1. FEATURES — exatamente as 10 do paper (sem timestamp)
# ------------------------------------------------------------
RAW_FEATURES = ["CAN_ID", "DLC", "DATA_0", "DATA_1", "DATA_2", "DATA_3",
                 "DATA_4", "DATA_5", "DATA_6", "DATA_7"]


def load_and_preprocess(csv_path: Path, sample_frac: float = 0.10) -> tuple:
    """
    Segue o paper:
    1. Carrega CSV
    2. Amostragem estratificada (paper usa k-means clustering sampling;
       aqui usamos stratified random como proxy razoável)
    3. Label encoding
    4. Z-score normalization
    """
    print("Carregando dataset...")
    df = pd.read_csv(csv_path)
    print(f"  Total: {len(df):,} linhas | Classes: {df['Label'].value_counts().to_dict()}")

    # Amostragem estratificada por classe
    if sample_frac < 1.0:
        df = (df.groupby("Label", group_keys=False)
                .apply(lambda g: g.sample(frac=sample_frac, random_state=42)))
        print(f"  Amostra ({sample_frac:.0%}): {len(df):,} linhas")

    # Encode CAN_ID (hex string → int)
    df["CAN_ID"] = pd.to_numeric(df["CAN_ID"], errors="coerce").fillna(0).astype(int)

    X = df[RAW_FEATURES].fillna(0).astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])

    return X, y, le


def select_features_ig_fcbf(X: pd.DataFrame, y: np.ndarray,
                              threshold: float = 0.9) -> list[str]:
    """
    IG-FCBF do paper:
    1. Calcula Information Gain de cada feature
    2. Ordena por importância decrescente
    3. Seleciona features até acumular `threshold` da importância total
    
    Resultado esperado pelo paper: ["CAN_ID", "DATA_5", "DATA_3", "DATA_1"]
    """
    ig_scores = mutual_info_classif(X, y, random_state=42)
    ig_norm = ig_scores / ig_scores.sum()

    ranked = sorted(zip(X.columns, ig_norm), key=lambda x: x[1], reverse=True)
    print("\nInformation Gain (normalizado):")
    for feat, score in ranked:
        print(f"  {feat}: {score:.4f}")

    selected, cumulative = [], 0.0
    for feat, score in ranked:
        selected.append(feat)
        cumulative += score
        if cumulative >= threshold:
            break

    print(f"\nFeatures selecionadas (α={threshold}): {selected}")
    return selected


def build_models(random_state: int = 42) -> dict:
    """Tier 1 — 4 tree-based learners (hiperparâmetros default para baseline)."""
    return {
        "DT":      DecisionTreeClassifier(random_state=random_state),
        "RF":      RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
        "ET":      ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            tree_method="hist", eval_metric="mlogloss",
            n_jobs=-1, random_state=random_state
        ),
    }


def optimize_xgboost_botpe(X_train, y_train, max_evals: int = 30) -> dict:
    """
    Tier 2 — BO-TPE para otimizar hiperparâmetros (paper usa Hyperopt).
    Otimiza XGBoost como meta-learner do stacking.
    """
    space = {
        "n_estimators":  hp.choice("n_estimators", [100, 200, 300, 500]),
        "max_depth":     hp.choice("max_depth", [4, 6, 8, 10]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "subsample":     hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "min_child_weight": hp.choice("min_child_weight", [1, 3, 5, 7]),
    }

    def objective(params):
        model = xgb.XGBClassifier(
            **params, tree_method="hist", eval_metric="mlogloss",
            n_jobs=-1, random_state=42
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="f1_weighted", n_jobs=-1).mean()
        return {"loss": -score, "status": STATUS_OK}

    print(f"\nBO-TPE: otimizando XGBoost ({max_evals} iterações)...")
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest,
                max_evals=max_evals, trials=trials, verbose=False)

    # Converte índices de volta para valores
    choices = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [4, 6, 8, 10],
        "min_child_weight": [1, 3, 5, 7],
    }
    for k, v in choices.items():
        if k in best:
            best[k] = v[best[k]]

    print(f"  Melhores params: {best}")
    return best


def build_stacking(base_models: dict, best_xgb_params: dict) -> StackingClassifier:
    """
    Tier 2 — Stacking ensemble.
    Base learners: DT, RF, ET, XGBoost
    Meta-learner: melhor modelo (XGBoost otimizado, conforme paper)
    """
    estimators = [(name, model) for name, model in base_models.items()]
    meta_learner = xgb.XGBClassifier(
        **best_xgb_params, tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=42
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )


def evaluate(model, X_test, y_test, le: LabelEncoder, label: str = "") -> None:
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"Resultados: {label}")
    print(f"{'='*50}")
    print(classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred)
    ))


def main():
    # 1. Carregar e pré-processar
    X, y, le = load_and_preprocess(DATA_CSV, sample_frac=0.10)

    # 2. IG-FCBF feature selection (paper: threshold α=0.9)
    selected_features = select_features_ig_fcbf(X, y, threshold=0.9)
    X = X[selected_features]

    # 3. Z-score normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train/test split (70/30 como no paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.30, stratify=y, random_state=42
    )

    # 5. Tier 1 — treinar e avaliar cada modelo individualmente
    models = build_models()
    print("\n" + "="*50)
    print("TIER 1 — Modelos individuais (10-fold CV)")
    print("="*50)
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"  {name}: F1={scores.mean():.4f} ± {scores.std():.4f}")
        model.fit(X_train, y_train)
        evaluate(model, X_test, y_test, le, label=f"Tier 1 — {name}")

    # 6. Tier 2 — BO-TPE + Stacking
    best_params = optimize_xgboost_botpe(X_train, y_train, max_evals=30)
    stacking = build_stacking(models, best_params)
    print("\nTreinando Stacking ensemble...")
    stacking.fit(X_train, y_train)
    evaluate(stacking, X_test, y_test, le, label="Tier 2 — Stacking + BO-TPE")


if __name__ == "__main__":
    main()