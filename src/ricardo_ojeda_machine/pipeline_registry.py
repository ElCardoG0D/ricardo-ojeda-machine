from kedro.pipeline import Pipeline
from ricardo_ojeda_machine.pipelines import data_engineering, data_preparation

def register_pipelines() -> dict[str, Pipeline]:
    """
    Registra los pipelines disponibles en el proyecto.
    """
    data_eng_pipeline = data_engineering.create_pipeline()
    data_prep_pipeline = data_preparation.create_pipeline()

    return {
        "__default__": data_eng_pipeline + data_prep_pipeline,  # corre ambos si no especificas
        "data_engineering": data_eng_pipeline,
        "data_preparation": data_prep_pipeline,
    }
