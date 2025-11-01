# ============================================================
# 🌀 DAG: ricardo_ml_dag.py
# Orquesta los pipelines de Kedro (regresión + clasificación)
# ============================================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ============================================================
# 1️⃣ Configuración general del DAG
# ============================================================
default_args = {
    "owner": "Ricardo Ojeda",
    "depends_on_past": False,
    "email": ["ml_project@airflow.local"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    dag_id="ricardo_ml_dag",
    default_args=default_args,
    description="Orquestación de pipelines Kedro para Regresión y Clasificación",
    schedule_interval=None,  # Ejecución manual
    start_date=datetime(2025, 10, 27),
    catchup=False,
    tags=["kedro", "machine_learning", "ricardo_ojeda_machine"],
)

# ============================================================
# 2️⃣ Tareas del pipeline
# ============================================================

# --- 1️⃣ Ejecución del pipeline de preparación de datos ---
prepare_task = BashOperator(
    task_id="data_preparation",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=data_preparation",
    dag=dag,
)

# --- 2️⃣ Ejecución del pipeline de regresión ---
regression_task = BashOperator(
    task_id="regression_pipeline",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=regression",
    dag=dag,
)

# --- 3️⃣ Ejecución del pipeline de clasificación ---
classification_task = BashOperator(
    task_id="classification_pipeline",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=classification",
    dag=dag,
)

# ============================================================
# 🚫 DVC Push (desactivado para entrega)
# ============================================================
# Si deseas activarlo más adelante, quita los comentarios
# y asegúrate de tener un remote configurado en DVC.
# dvc_push_task = BashOperator(
#     task_id="dvc_push",
#     bash_command="cd /opt/airflow/project && dvc push",
#     dag=dag,
# )

# ============================================================
# 3️⃣ Dependencias del flujo
# ============================================================
# prepare_task >> [regression_task, classification_task] >> dvc_push_task
prepare_task >> [regression_task, classification_task]
