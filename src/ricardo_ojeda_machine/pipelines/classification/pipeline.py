from kedro.pipeline import Pipeline, node
from .nodes import (
    preparar_datos_clasificacion,
    entrenar_modelos_clasificacion,
    exportar_metricas_clasificacion,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preparar_datos_clasificacion,
                inputs="intakes_outcomes_merged",
                outputs=["X_train_clf", "X_test_clf", "y_train_clf", "y_test_clf"],
                name="prepare_and_split_clf_data",
            ),
            node(
                func=entrenar_modelos_clasificacion,
                inputs=["X_train_clf", "X_test_clf", "y_train_clf", "y_test_clf"],
                outputs="resultados_clasificacion",
                name="train_classifiers",
            ),
            node(
                func=exportar_metricas_clasificacion,
                inputs="resultados_clasificacion",
                outputs="artefactos_clf",
                name="export_metrics",
            ),
        ]
    )
