# airflow/dags/ml_pipeline_dag.py
from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.operators.bash import BashOperator

# ---------------------------
# Configuración base del DAG
# ---------------------------
default_args = {
    "owner": "yasna",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Ajusta si quieres otra hora/frecuencia
SCHEDULE = "0 3 * * *"    # todos los días a las 03:00

# Ruta del proyecto dentro del contenedor (móntala con -v)
PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")

# Tip: variables de entorno útiles para Kedro/DVC
ENV_EXPORT = (
    f'export PROJECT_DIR="{PROJECT_DIR}" && '
    'export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH" && '
    'export KEDRO_ENV="local" && '
    'cd "$PROJECT_DIR" && '
    'pwd && ls -la'
)

with DAG(
    dag_id="ml_bank_kedro_dvc_pipeline",
    default_args=default_args,
    description="Orquestación con Airflow de data_preparation -> regression -> classification (Kedro + DVC)",
    schedule_interval=SCHEDULE,
    start_date=datetime(2025, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["kedro", "dvc", "ml", "automation"],
) as dag:

    # 1) Pre-chequeos: estado de DVC/Git
    dvc_status = BashOperator(
        task_id="dvc_status",
        bash_command=ENV_EXPORT + " && dvc status && git status -s || true"
    )

    # 2) Repro etapa de preparación de datos (DVC apunta a tu stage)
    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command=ENV_EXPORT + " && dvc repro data_preparation"
    )

    # 3) Repro regresión
    regression = BashOperator(
        task_id="regression",
        bash_command=ENV_EXPORT + " && dvc repro regression"
    )

    # 4) Repro clasificación
    classification = BashOperator(
        task_id="classification",
        bash_command=ENV_EXPORT + " && dvc repro classification"
    )

    # 5) Publicar artefactos (ej: mostrar rutas/sha y listar reportes)
    publish_artifacts = BashOperator(
        task_id="publish_artifacts",
        bash_command=ENV_EXPORT + (
            ' && echo "=== Artefactos ==="'
            ' && dvc list . data/08_reporting || true'
            ' && echo "--- SHA de datos versionados ---"'
            ' && dvc status -c'
        ),
    )

    # Dependencias (grafo):
    dvc_status >> data_preparation >> regression >> classification >> publish_artifacts
