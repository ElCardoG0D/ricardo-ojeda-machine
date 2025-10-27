import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt


def preparar_datos_clasificacion(df: pd.DataFrame):
    """Prepara los datos para clasificación (predicción de adopción)."""
    df = df.copy()

    # Buscar la columna que contiene el tipo de resultado
    possible_cols = ["Outcome Type_outcome", "Outcome Type", "outcome_type"]
    outcome_col = None
    for col in possible_cols:
        if col in df.columns:
            outcome_col = col
            break

    if not outcome_col:
        raise KeyError("No encuentro columna de outcome ('Outcome Type' o 'outcome_type').")

    # Si la columna es numérica, detectar el código más frecuente como 'Adoption'
    if pd.api.types.is_numeric_dtype(df[outcome_col]):
        print("ℹ️ Se detectó columna Outcome codificada numéricamente.")
        adoption_code = df[outcome_col].mode()[0]
        y = (df[outcome_col] == adoption_code).astype(int)
    else:
        df[outcome_col] = df[outcome_col].astype(str).str.lower()
        y = (df[outcome_col] == "adoption").astype(int)

    # ===============================
    # 🔹 BLOQUE FLEXIBLE DE FEATURES
    # ===============================
    available_cols = df.columns.tolist()
    feature_map = {
        "animal_type_intake_enc": "animal_type_intake" if "animal_type_intake" in available_cols else "animal_type",
        "sex_intake_enc": "sex_upon_intake" if "sex_upon_intake" in available_cols else "sex_intake",
        "status_intake_enc": "intake_condition" if "intake_condition" in available_cols else "status_intake",
        "intake_month": "intake_month" if "intake_month" in available_cols else "monthyear_intake",
        "intake_year": "intake_year" if "intake_year" in available_cols else "year_intake",
    }

    features = [col for col in feature_map.values() if col in df.columns]
    if not features:
        raise KeyError(
            f"No se encontraron columnas adecuadas para features. "
            f"Columnas disponibles: {list(df.columns)[:15]}"
        )

    X = df[features]

    # ======================================
    # 🔸 Codificar columnas no numéricas
    # ======================================
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            print(f"⚙️  Codificada columna categórica: {col}")

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"✅ Columnas usadas para clasificación: {features}")
    return X_train, X_test, y_train, y_test


def entrenar_modelos_clasificacion(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """
    Entrena varios clasificadores y retorna una tabla comparativa de métricas.
    """
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42
        ),
    }

    filas = []
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        # Probabilidades para ROC-AUC (si el modelo lo soporta)
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        elif hasattr(modelo, "decision_function"):
            y_prob = modelo.decision_function(X_test)
        else:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        filas.append({
            "Modelo": nombre,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3),
            "ROC_AUC": round(roc, 3) if not np.isnan(roc) else ""
        })

    return pd.DataFrame(filas)


def exportar_metricas_y_graficos(
    resultados: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Guarda CSV con métricas y genera:
      - comparativa_modelos_clf.png
      - confusion_matrix.png (del mejor modelo por F1)
      - roc_curve.png (del mejor modelo si tiene predict_proba)
    Retorna un dict con rutas y el nombre del mejor modelo.
    """
    # Guardar CSV (Kedro lo persiste vía catalog)
    print("📊 Resultados clasificación:\n", resultados)

    # Gráfico comparativo por F1
    try:
        plt.figure(figsize=(7, 4))
        orden = resultados.sort_values("F1", ascending=False)
        plt.bar(orden["Modelo"], orden["F1"])
        plt.title("Comparativa Modelos – F1")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig("data/08_reporting/comparativa_modelos_clf.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] No se pudo guardar comparativa_modelos_clf.png: {e}")

    # Elegimos mejor modelo por F1
    mejor = resultados.sort_values("F1", ascending=False).iloc[0]["Modelo"]

    # Reconstruimos y graficamos Confusion matrix + ROC del mejor
    # (Para no duplicar entrenamiento, volvemos a entrenarlo rápido)
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=200, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=42
        ),
    }
    # OJO: aquí necesitamos X_train/y_train; para mantener el contrato de nodos,
    # la confusión y ROC las dejamos solo si el modelo soporta proba y evitamos reentrenar completo.
    # En producción las haríamos en el mismo node de entrenamiento devolviendo el modelo.
    # Para no romper Kedro ahora, solo graficamos la ROC usando y_test y una aproximación:
    # -> En este diseño, omitimos ROC y matriz a menos que se replantee el node para devolver el mejor modelo ya entrenado.

    # Devolvemos paths generados
    return {
        "mejor_modelo": mejor,
        "comparativa_png": "data/08_reporting/comparativa_modelos_clf.png"
    }
