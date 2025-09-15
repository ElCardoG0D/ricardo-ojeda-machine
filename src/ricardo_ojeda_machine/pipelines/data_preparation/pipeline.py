from kedro.pipeline import Pipeline, node
from .nodes import limpiar_datos, transformar_fechas, unir_datasets

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(limpiar_datos, inputs="intakes", outputs="intakes_clean"),
            node(transformar_fechas, inputs="outcomes", outputs="outcomes_clean"),
            node(unir_datasets, inputs=["intakes_clean", "outcomes_clean", "licenses"], outputs="intakes_outcomes_merged"),
        ]
    )
