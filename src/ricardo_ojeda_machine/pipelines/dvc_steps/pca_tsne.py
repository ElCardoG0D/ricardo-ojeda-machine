import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def main():
    print("=== RUNNING PCA + t-SNE ===")

    input_path = "data/05_model_input/features_dataset.csv"
    df = pd.read_csv(input_path)

    # ============================
    # 1) Seleccionar columnas numéricas
    # ============================
    df_num = df.select_dtypes(include=[np.number]).copy()

    # Si NO hay columnas numéricas → convertir categóricas a códigos
    if df_num.shape[1] == 0:
        print("⚠️ No había columnas numéricas → convirtiendo categóricas a códigos")
        df_num = df.apply(lambda col: col.astype("category").cat.codes)

    # ============================
    # 2) Imputar NaN con la media
    # ============================
    df_num = df_num.fillna(df_num.mean())

    # ============================
    # 3) Limitar a máximo 5000 muestras para t-SNE
    # ============================
    if len(df_num) > 5000:
        print("⚠️ Dataset muy grande → tomando solo 5000 muestras para t-SNE")
        df_num = df_num.sample(5000, random_state=42)

    # ============================
    # 4) Escalar
    # ============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)

    # ============================
    # 5) PCA (2 componentes)
    # ============================
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    os.makedirs("08_reporting/pca_tsne", exist_ok=True)

    pd.DataFrame(pca_result, columns=["PC1", "PC2"]).to_csv(
        "08_reporting/pca_tsne/pca_2d.csv", index=False
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=5, alpha=0.5)
    plt.title("PCA (2D)")
    plt.savefig("08_reporting/pca_tsne/pca_plot.png")
    plt.close()

    print("Saved PCA outputs.")

    # ============================
    # 6) t-SNE (corregido)
    # ============================
    print("Running t-SNE (esto tarda unos segundos)...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=500,   # <-- CORRECTO PARA TU VERSIÓN
        random_state=42
    )

    tsne_result = tsne.fit_transform(X_scaled)

    pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"]).to_csv(
        "08_reporting/pca_tsne/tsne_2d.csv", index=False
    )

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, alpha=0.5, c="purple")
    plt.title("t-SNE (2D)")
    plt.savefig("08_reporting/pca_tsne/tsne_plot.png")
    plt.close()

    print("Saved t-SNE outputs.")

if __name__ == "__main__":
    main()
