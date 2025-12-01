# ============================================================
# 🌀 DAG: ricardo_ml_dag.py
# Orquesta los pipelines de Kedro:
# - data_engineering
# - data_preparation
# - regression
# - classification
# - unsupervised (clustering)
# ============================================================

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# ============================================================
# 1️⃣ CONFIGURACIÓN GENERAL
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
    description=(
        "Orquestación completa de pipelines Kedro: "
        "data_engineering, data_preparation, regression, classification y unsupervised"
    ),
    schedule_interval=None,    # ejecución manual
    start_date=datetime(2025, 11, 28),  # evita backfill
    catchup=False,
    tags=["kedro", "ml_project", "Ricardo"],
)

# ============================================================
# 2️⃣ TAREAS DEL PIPELINE (AIRFLOW → KEDRO)
# ============================================================

# --- 0️⃣ DATA ENGINEERING (validación datasets crudos) ---
data_engineering_task = BashOperator(
    task_id="data_engineering_pipeline",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=data_engineering",
    dag=dag,
)

# --- 1️⃣ DATA PREPARATION (limpieza + merge) ---
prepare_task = BashOperator(
    task_id="data_preparation",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=data_preparation",
    dag=dag,
)

# --- 2️⃣ REGRESIÓN ---
regression_task = BashOperator(
    task_id="regression_pipeline",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=regression",
    dag=dag,
)

# --- 3️⃣ CLASIFICACIÓN ---
classification_task = BashOperator(
    task_id="classification_pipeline",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=classification",
    dag=dag,
)

# --- 4️⃣ NO SUPERVISADO / CLUSTERING ---
unsupervised_task = BashOperator(
    task_id="unsupervised_pipeline",
    bash_command="cd /opt/airflow/project && kedro run --pipeline=unsupervised",
    dag=dag,
)

# (Opcional) --- 5️⃣ REPORTING GLOBAL ---
# Si algún día tienes un pipeline `reporting`, lo agregas aquí:
# reporting_task = BashOperator(
#     task_id="reporting_pipeline",
#     bash_command="cd /opt/airflow/project && kedro run --pipeline=reporting",
#     dag=dag,
# )

# ============================================================
# 3️⃣ DEPENDENCIAS ENTRE TAREAS
# ============================================================

# Primero validar datasets crudos
data_engineering_task >> prepare_task

# Luego de preparar el dataset unificado, puedes lanzar:
# - modelos supervisados (regresión y clasificación)
# - análisis no supervisado (clustering + PCA + t-SNE)
prepare_task >> [regression_task, classification_task, unsupervised_task]

# Si tuvieras reporting global:
# [regression_task, classification_task, unsupervised_task] >> reporting_task
