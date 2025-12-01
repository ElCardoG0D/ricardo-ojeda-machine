import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def aplicar_pca(X_scaled: np.ndarray):

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)

    varianza = pca.explained_variance_ratio_

    plt.figure(figsize=(5, 4))
    plt.bar(["PC1", "PC2"], varianza)
    plt.title("Varianza Explicada PCA")
    plt.tight_layout()
    plt.savefig("data/08_reporting/pca_varianza.png", dpi=140)
    plt.close()

    return comps


def aplicar_tsne(X_scaled: np.ndarray):
    tsne = TSNE(n_components=2, random_state=42, learning_rate="auto")
    comps = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(5, 4))
    plt.scatter(comps[:, 0], comps[:, 1], s=5)
    plt.title("t-SNE 2D")
    plt.tight_layout()
    plt.savefig("data/08_reporting/tsne_plot.png", dpi=140)
    plt.close()

    return comps
