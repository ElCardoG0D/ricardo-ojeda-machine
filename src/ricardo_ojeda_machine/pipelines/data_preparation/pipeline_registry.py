from __future__ import annotations
from kedro.pipeline import Pipeline
from ricardo_ojeda_machine.pipelines import data_preparation

def register_pipelines() -> dict[str, Pipeline]:
    data_prep = data_preparation.create_pipeline()
    return {
        "__default__": data_prep,          # corre por defecto
        "data_preparation": data_prep,     # corre con --pipeline=data_preparation
    }
