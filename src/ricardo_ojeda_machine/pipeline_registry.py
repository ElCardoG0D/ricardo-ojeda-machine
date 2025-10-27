from kedro.pipeline import Pipeline
from ricardo_ojeda_machine.pipelines import data_engineering, data_preparation, regression, classification

def register_pipelines() -> dict[str, Pipeline]:
    """Registrar todos los pipelines del proyecto."""
    data_eng = data_engineering.create_pipeline()
    data_prep = data_preparation.create_pipeline()
    regression_pipe = regression.create_pipeline()
    classification_pipe = classification.create_pipeline()

    return {
        "__default__": data_eng + data_prep + regression_pipe + classification_pipe,
        "data_engineering": data_eng,
        "data_preparation": data_prep,
        "regression": regression_pipe,
        "classification": classification_pipe,
    }
