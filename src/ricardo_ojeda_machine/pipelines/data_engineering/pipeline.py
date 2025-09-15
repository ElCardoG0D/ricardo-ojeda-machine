from kedro.pipeline import Pipeline, node
from .nodes import validate_dataset

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=validate_dataset,
            inputs=dict(df="intakes", name="params:intakes_name"),
            outputs="validated_intakes",
            name="validate_intakes_node"
        ),
        node(
            func=validate_dataset,
            inputs=dict(df="outcomes", name="params:outcomes_name"),
            outputs="validated_outcomes",
            name="validate_outcomes_node"
        ),
        node(
            func=validate_dataset,
            inputs=dict(df="licenses", name="params:licenses_name"),
            outputs="validated_licenses",
            name="validate_licenses_node"
        ),
    ])
