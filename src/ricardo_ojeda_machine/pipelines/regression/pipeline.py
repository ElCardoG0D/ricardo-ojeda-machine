from kedro.pipeline import Pipeline, node
from .nodes import exportar_y_graficar_resultados, preparar_datos_regresion, entrenar_modelos_regresion, guardar_resultados_regresion,exportar_y_graficar_resultados


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=preparar_datos_regresion,
            inputs="features_dataset",
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="preparar_datos_regresion_node"
        ),
        node(
            func=entrenar_modelos_regresion,
            inputs=["X_train", "X_test", "y_train", "y_test"],
            outputs="resultados_regresion",
            name="entrenar_modelos_regresion_node"
        ),
        node(
            func=guardar_resultados_regresion,
            inputs="resultados_regresion",
            outputs=None,
            name="guardar_resultados_regresion_node"
        ),
         node(
            func=exportar_y_graficar_resultados,
            inputs="resultados_regresion",
            outputs="resultados_finales",
            name="exportar_y_graficar_resultados"
        )
    ])
