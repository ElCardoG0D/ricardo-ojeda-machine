from kedro.pipeline import Pipeline, node
from .nodes_clustering import preparar_datos_clustering, ejecutar_clustering
from .nodes_pca_tsne import aplicar_pca, aplicar_tsne


def create_pipeline(**kwargs):
    return Pipeline([

        node(
            preparar_datos_clustering,
            inputs="intakes_outcomes_merged",
            outputs=["X_scaled_unsup", "unsup_columns"],
            name="prepare_clustering"
        ),

        node(
            ejecutar_clustering,
            inputs="X_scaled_unsup",
            outputs="resultados_clustering",
            name="run_clustering"
        ),

        node(
            aplicar_pca,
            inputs="X_scaled_unsup",
            outputs="pca_2d",
            name="apply_pca"
        ),

        node(
            aplicar_tsne,
            inputs="X_scaled_unsup",
            outputs="tsne_2d",
            name="apply_tsne"
        ),

    ])
