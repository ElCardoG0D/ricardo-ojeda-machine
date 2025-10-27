from kedro.pipeline import Pipeline, node
from .nodes import (
    preparar_datos_clasificacion,
    entrenar_modelos_clasificacion,
    exportar_metricas_y_graficos
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preparar_datos_clasificacion,
                inputs="intakes_outcomes_merged",     # dataset de entrada
                outputs=["X_train_clf", "X_test_clf", "y_train_clf", "y_test_clf"],
                name="preparar_datos_clf_node",
            ),
            node(
                func=entrenar_modelos_clasificacion,
                inputs=["X_train_clf", "X_test_clf", "y_train_clf", "y_test_clf"],
                outputs="resultados_clasificacion",
                name="entrenar_modelos_clf_node",
            ),
            node(
                func=exportar_metricas_y_graficos,
                inputs=["resultados_clasificacion", "X_test_clf", "y_test_clf"],
                outputs="artefactos_clf",
                name="exportar_metricas_y_graficos_clf_node",
            ),
        ]
    )
