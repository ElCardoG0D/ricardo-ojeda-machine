import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# 1) PREPARACIÓN BASE PARA CLUSTERING
# ============================================================
def preparar_datos_clustering(df: pd.DataFrame):
    """
    Prepara el dataset para clustering:
    - Muestra máximo N filas (para que el pipeline no se demore una eternidad).
    - Convierte columnas datetime a números.
    - Codifica categóricas como enteros.
    - Escala todas las columnas numéricas.
    Devuelve:
        X_scaled: np.ndarray con los features escalados.
        num_cols: lista de nombres de columnas usadas.
    """
    df = df.copy()

    # 🔹 0) MUESTREO PARA QUE SEA MÁS RÁPIDO EN AIRFLOW
    MAX_MUESTRA = 10000
    if len(df) > MAX_MUESTRA:
        df = df.sample(MAX_MUESTRA, random_state=42)

    # 1) Convertir fechas a números (segundos desde época)
    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns
    for col in datetime_cols:
        # pasamos datetime64 a segundos (float)
        df[col] = df[col].view("int64") / 1e9

    # 2) Codificar variables categóricas a enteros
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col], _ = pd.factorize(df[col])

    # 3) Quedarnos con todas las columnas numéricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        raise ValueError(
            "No existen columnas numéricas para clustering ni siquiera tras la conversión."
        )

    X = df[num_cols].fillna(0)

    # 4) Escalado estándar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, num_cols


# ============================================================
# 2) EJECUCIÓN DE CLUSTERING
# ============================================================
def ejecutar_clustering(X_scaled: np.ndarray):
    """
    Ejecuta 3 algoritmos de clustering sobre X_scaled:
    - KMeans (con gráfico del codo).
    - DBSCAN.
    - OPTICS.
    Devuelve un diccionario con labels y métricas por algoritmo.
    """
    resultados: dict[str, dict] = {}

    # -----------------------
    # 1) KMEANS
    # -----------------------
    # Modelo principal con k=5 (se puede ajustar)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    resultados["kmeans"] = {
        "labels": labels_kmeans,
        "inertia": kmeans.inertia_,
        "silhouette": silhouette_score(X_scaled, labels_kmeans),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels_kmeans),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels_kmeans),
    }

    # --- Método del codo (versión corta para Airflow) ---
    inertias = []
    k_values = range(3, 7)  # antes: range(2, 10) → menos k = menos tiempo
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(5, 4))
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("Elbow Method - KMeans")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig("data/08_reporting/clustering/elbow_kmeans.png", dpi=120)
    plt.close()

    # -----------------------
    # 2) DBSCAN
    # -----------------------
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    labels_dbscan = dbscan.fit_predict(X_scaled)
    clusters_db = set(labels_dbscan)

    resultados["dbscan"] = {
        "labels": labels_dbscan,
        "silhouette": (
            silhouette_score(X_scaled, labels_dbscan)
            if len(clusters_db) > 1
            else None
        ),
        "davies_bouldin": (
            davies_bouldin_score(X_scaled, labels_dbscan)
            if len(clusters_db) > 1
            else None
        ),
        "calinski_harabasz": (
            calinski_harabasz_score(X_scaled, labels_dbscan)
            if len(clusters_db) > 1
            else None
        ),
    }

    # -----------------------
    # 3) OPTICS
    # -----------------------
    optics = OPTICS(min_samples=10)
    labels_optics = optics.fit_predict(X_scaled)
    clusters_opt = set(labels_optics)

    resultados["optics"] = {
        "labels": labels_optics,
        "silhouette": (
            silhouette_score(X_scaled, labels_optics)
            if len(clusters_opt) > 1
            else None
        ),
        "davies_bouldin": (
            davies_bouldin_score(X_scaled, labels_optics)
            if len(clusters_opt) > 1
            else None
        ),
        "calinski_harabasz": (
            calinski_harabasz_score(X_scaled, labels_optics)
            if len(clusters_opt) > 1
            else None
        ),
    }

    return resultados
