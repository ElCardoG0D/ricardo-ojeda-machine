import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

def exportar_y_graficar_resultados(resultados: pd.DataFrame):
    """Guarda los resultados en CSV y genera gráfico comparativo."""
    
    # Crear carpeta si no existe
    output_dir = "data/08_reporting"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar resultados en CSV
    csv_path = os.path.join(output_dir, "resultados_regresion.csv")
    resultados.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en: {csv_path}")

    # Graficar R² comparativo
    plt.figure(figsize=(8, 4))
    plt.bar(resultados["Modelo"], resultados["R2"], color="skyblue", edgecolor="black")
    plt.title("Comparativa de Modelos de Regresión (R²)")
    plt.xlabel("Modelo")
    plt.ylabel("R² Score")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=30)
    
    # Guardar gráfico como imagen
    img_path = os.path.join(output_dir, "comparativa_modelos_R2.png")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    print(f"📊 Gráfico guardado en: {img_path}")

    return resultados

# =========================================================
# 📦 1️⃣ Preparar datos
# =========================================================
def preparar_datos_regresion(df: pd.DataFrame):
    """Prepara los datos para el modelo de regresión."""
    df = df.copy()
    df = df.dropna(subset=["length_of_stay_days"])

    # Codificar Outcome Type si es texto
    from sklearn.preprocessing import LabelEncoder
    if "Outcome Type" in df.columns:
        le = LabelEncoder()
        df["Outcome Type"] = le.fit_transform(df["Outcome Type"].astype(str))

    # Variables predictoras
    features = [
        "age_years_intake", "Animal Type_intake_enc", "sex_intake_enc",
        "status_intake_enc", "intake_year", "intake_month",
        "Outcome Type"
    ]

    columnas_disponibles = df.columns.tolist()
    faltantes = [c for c in features if c not in columnas_disponibles]
    if faltantes:
        raise KeyError(f"Columnas faltantes en el dataset: {faltantes}")

    X = df[features]
    y = df["length_of_stay_days"]

    # Codificar Outcome Type si es texto
    if "Outcome Type" in df.columns:
        le = LabelEncoder()
        df["Outcome Type"] = le.fit_transform(df["Outcome Type"].astype(str))

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# 🤖 2️⃣ Entrenar modelos
# =========================================================
def entrenar_modelos_regresion(X_train, X_test, y_train, y_test):
    """Entrena múltiples modelos de regresión y evalúa sus métricas."""
    modelos = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    }

    resultados = []

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        resultados.append({
            "Modelo": nombre,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "R2": round(r2, 3)
        })

    resultados_df = pd.DataFrame(resultados)
    return resultados_df


# =========================================================
# 💾 3️⃣ Guardar resultados
# =========================================================
def guardar_resultados_regresion(resultados: pd.DataFrame):
    """Guarda el resumen de métricas."""
    print("📊 Resultados de modelos de regresión:")
    print(resultados)
    return resultados

def exportar_y_graficar_resultados(resultados: pd.DataFrame):
    """Guarda los resultados en CSV y genera gráfico comparativo."""
    
    # Crear carpeta si no existe
    output_dir = "data/08_reporting"
    os.makedirs(output_dir, exist_ok=True)

    # Guardar resultados en CSV
    csv_path = os.path.join(output_dir, "resultados_regresion.csv")
    resultados.to_csv(csv_path, index=False)
    print(f"✅ Resultados guardados en: {csv_path}")

    # Graficar R² comparativo
    plt.figure(figsize=(8, 4))
    plt.bar(resultados["Modelo"], resultados["R2"], color="skyblue", edgecolor="black")
    plt.title("Comparativa de Modelos de Regresión (R²)")
    plt.xlabel("Modelo")
    plt.ylabel("R² Score")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=30)
    
    # Guardar gráfico como imagen
    img_path = os.path.join(output_dir, "comparativa_modelos_R2.png")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()
    print(f"📊 Gráfico guardado en: {img_path}")

    return resultados