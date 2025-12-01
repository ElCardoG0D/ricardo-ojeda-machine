import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ============================================================
# 1) PREPARACIÓN DE DATOS (usa outcome_type del CSV)
# ============================================================
def preparar_datos_clasificacion(df: pd.DataFrame):
    """
    Prepara datos para clasificación de adopción.
    Retorna: X_train, X_test, y_train, y_test
    """

    df = df.copy()

    # Columnas posibles para el outcome
    possible_cols = ["Outcome Type_outcome", "Outcome Type", "outcome_type"]
    outcome_col = None
    for col in possible_cols:
        if col in df.columns:
            outcome_col = col
            break

    if outcome_col is None:
        raise KeyError(
            "No encuentro columna de outcome. "
            f"Busqué: {possible_cols}. "
            f"Columnas disponibles: {list(df.columns)[:20]}"
        )

    # Etiqueta binaria: 1 = adopción, 0 = resto
    df[outcome_col] = df[outcome_col].astype(str).str.lower()
    y = (df[outcome_col] == "adoption").astype(int)

    # ==========
    # FEATURES
    # ==========
    available = df.columns.tolist()

    feature_map = {
        # tipo de animal
        "animal_type": (
            "animal_type_intake"
            if "animal_type_intake" in available
            else "animal_type" if "animal_type" in available
            else None
        ),
        # sexo
        "sex": (
            "sex_upon_intake"
            if "sex_upon_intake" in available
            else "sex_intake" if "sex_intake" in available
            else None
        ),
        # condición de ingreso
        "status": (
            "intake_condition"
            if "intake_condition" in available
            else "status_intake" if "status_intake" in available
            else None
        ),
        # temporalidad
        "intake_month": (
            "intake_month"
            if "intake_month" in available
            else "monthyear_intake" if "monthyear_intake" in available
            else None
        ),
        "intake_year": (
            "intake_year"
            if "intake_year" in available
            else "year_intake" if "year_intake" in available
            else None
        ),
    }

    features = [
        col for col in feature_map.values()
        if col is not None and col in df.columns
    ]

    if not features:
        raise ValueError(
            "No se encontraron columnas adecuadas para features. "
            f"Columnas disponibles: {list(df.columns)[:20]}"
        )

    X = df[features].copy()

    # Codificación categórica
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 2) MODELADO + GRIDSEARCHCV
# ============================================================
def entrenar_modelos_clasificacion(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Entrena LogisticRegression, RandomForest y XGBoost con GridSearchCV.
    Devuelve un DataFrame de métricas listo para exportar como CSV.
    """

    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=300),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss"),
    }

    grids = {
        "LogisticRegression": {"C": [0.1, 1, 10]},
        "RandomForest": {"n_estimators": [100, 300]},
        "XGBoost": {
            "n_estimators": [200, 300],
            "learning_rate": [0.05, 0.1],
        },
    }

    filas = []

    for nombre, modelo in modelos.items():
        grid = GridSearchCV(modelo, grids[nombre], cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)

        # Probabilidades para ROC-AUC
        y_prob = (
            grid.predict_proba(X_test)[:, 1]
            if hasattr(grid, "predict_proba")
            else None
        )

        filas.append({
            "Modelo": nombre,
            "BestParams": str(grid.best_params_),
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "ROC_AUC": (
                round(roc_auc_score(y_test, y_prob), 3)
                if y_prob is not None
                else ""
            ),
        })

    return pd.DataFrame(filas)


# ============================================================
# 3) EXPORTACIÓN PARA KEDRO (CSV + artefactos)
# ============================================================
def exportar_metricas_clasificacion(resultados: pd.DataFrame) -> Dict[str, Any]:
    """
    Deja resultados listos para:
    - resultados_clasificacion (CSV)
    - artefactos_clf (MemoryDataset)
    """

    mejor = resultados.sort_values("F1", ascending=False).iloc[0]

    artefactos = {
        "mejor_modelo": mejor["Modelo"],
        "mejores_hiperparametros": mejor["BestParams"],
        "metricas": resultados.to_dict(orient="records"),
    }

    return artefactos
